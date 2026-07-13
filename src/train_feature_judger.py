#!/usr/bin/env python3
"""Train feature-based bolt loosening judgers from SAM3 geometry outputs."""

from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = {
    "id",
    "image",
    "true_label",
    "pred_label",
    "confidence",
    "error",
    "candidates",
    "components",
    "judgment_reason",
    "judgment_reason_code",
    "interface_loose_evidence",
}

PROJECT_ROOT = Path(os.environ.get("BOLT_MARKING_ROOT", Path(__file__).resolve().parent.parent))


def collage_group(image: str) -> str:
    match = re.search(r"collage_(\d+)", str(image))
    if match:
        return match.group(1)
    return str(image)


def load_exclude_ids(path: Path) -> set[int]:
    if not path or not path.exists():
        return set()
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} must contain an 'id' column")
    return set(pd.to_numeric(df["id"], errors="coerce").dropna().astype(int).tolist())


def prepare_frame(features_csv: Path, exclude_ids_csv: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    excluded = load_exclude_ids(exclude_ids_csv)
    if excluded:
        df = df[~df["id"].astype(int).isin(excluded)].copy()
    if task == "binary":
        df = df[df["true_label"].isin(["normal", "loose"])].copy()
    elif task == "three_class":
        df = df[df["true_label"].isin(["normal", "loose", "unknown"])].copy()
    else:
        raise ValueError(f"Unknown task: {task}")
    df["group"] = df["image"].map(collage_group)
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in DROP_COLUMNS and c != "group"]
    numeric_cols = []
    categorical_cols = []
    for col in cols:
        if pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 80:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols


def build_models(numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Pipeline]:
    numeric_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre_tree = ColumnTransformer([
        ("num", numeric_tree, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

    numeric_linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pre_linear = ColumnTransformer([
        ("num", numeric_linear, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

    return {
        "random_forest": Pipeline([
            ("pre", pre_tree),
            ("model", RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("pre", pre_tree),
            ("model", ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=43,
                n_jobs=-1,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("pre", pre_linear),
            ("model", LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=44,
            )),
        ]),
    }


def cross_validate(df: pd.DataFrame, model_name: str, model: Pipeline, labels: List[str], output_dir: Path) -> Dict:
    numeric_cols, categorical_cols = feature_columns(df)
    x = df[numeric_cols + categorical_cols]
    y = df["true_label"].astype(str)
    groups = df["group"].astype(str)
    n_splits = min(5, y.value_counts().min(), groups.nunique())
    if n_splits < 2:
        raise ValueError("Not enough samples/groups for cross validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=123)

    pred = pd.Series(index=df.index, dtype=object)
    proba_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[test_idx] = model.predict(x.iloc[test_idx])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x.iloc[test_idx])
            classes = list(model.named_steps["model"].classes_)
            for row_i, sample_idx in enumerate(test_idx):
                item = {"id": int(df.loc[sample_idx, "id"]), "fold": fold}
                for cls_i, cls in enumerate(classes):
                    item[f"prob_{cls}"] = float(probs[row_i, cls_i])
                proba_rows.append(item)

    pred = pred.astype(str)
    cm = confusion_matrix(y, pred, labels=labels)
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    pred_df = df[["id", "image", "true_label", "group"]].copy()
    pred_df["pred_label"] = pred.values
    if proba_rows:
        pred_df = pred_df.merge(pd.DataFrame(proba_rows), on="id", how="left")
    pred_df.to_csv(output_dir / f"{model_name}_cv_predictions.csv", index=False)
    return {
        "model": model_name,
        "n_splits": int(n_splits),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_feature_importance(model: Pipeline, numeric_cols: List[str], categorical_cols: List[str], path: Path) -> None:
    clf = model.named_steps["model"]
    if not hasattr(clf, "feature_importances_"):
        return
    pre = model.named_steps["pre"]
    names = list(numeric_cols)
    if categorical_cols:
        cat_pipe = pre.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]
        names.extend(onehot.get_feature_names_out(categorical_cols).tolist())
    imp = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(path, index=False)


def train_final(df: pd.DataFrame, model_name: str, model: Pipeline, output_dir: Path) -> None:
    numeric_cols, categorical_cols = feature_columns(df)
    x = df[numeric_cols + categorical_cols]
    y = df["true_label"].astype(str)
    model.fit(x, y)
    joblib.dump(
        {
            "model": model,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "labels": sorted(y.unique().tolist()),
        },
        output_dir / f"{model_name}_final.joblib",
    )
    save_feature_importance(model, numeric_cols, categorical_cols, output_dir / f"{model_name}_feature_importance.csv")


def run_task(args: argparse.Namespace, task: str) -> Dict:
    df = prepare_frame(args.features_csv, args.exclude_ids, task)
    output_dir = args.output_dir / task
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "training_table.csv", index=False)
    numeric_cols, categorical_cols = feature_columns(df)
    models = build_models(numeric_cols, categorical_cols)
    labels = ["loose", "normal"] if task == "binary" else ["loose", "normal", "unknown"]
    metrics = {
        "task": task,
        "features_csv": str(args.features_csv),
        "exclude_ids": str(args.exclude_ids),
        "sample_count": int(len(df)),
        "label_counts": df["true_label"].value_counts().to_dict(),
        "group_count": int(df["group"].nunique()),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "models": [],
    }
    for name, model in models.items():
        item = cross_validate(df, name, model, labels, output_dir)
        metrics["models"].append(item)
    best = max(metrics["models"], key=lambda m: m["macro_f1"])
    metrics["best_model"] = best["model"]
    train_final(df, best["model"], models[best["model"]], output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, default=PROJECT_ROOT / "runs" / "sam3_marking" / "detailed_results.csv")
    parser.add_argument("--exclude-ids", type=Path, default=PROJECT_ROOT / "data" / "exclude_ids.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "feature_judger")
    parser.add_argument("--task", choices=["binary", "three_class", "both"], default="both")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["binary", "three_class"] if args.task == "both" else [args.task]
    all_metrics = {}
    for task in tasks:
        metrics = run_task(args, task)
        all_metrics[task] = {
            "sample_count": metrics["sample_count"],
            "label_counts": metrics["label_counts"],
            "best_model": metrics["best_model"],
            "models": [
                {
                    "model": m["model"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "confusion_matrix": m["confusion_matrix"],
                }
                for m in metrics["models"]
            ],
        }
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
