# 螺栓防松标记线 SAM3 最小交接包

这个文件夹只保留一个能跑通的交接脚本和一份说明文档。接手人先看本文件，然后执行 `run_smoke.sh`。

## 任务

对单螺栓/紧固件 crop 图像判断防松标记线状态：

| 标签 | 含义 |
| --- | --- |
| `normal` | 不松动，标记线连续或对齐 |
| `loose` | 松动，标记线断开、错位或连接关系被破坏 |
| `unknown` | 无有效标记线，无法判断 |

## 目录结构

```text
.
├── README.md              # 本说明文档
├── run_smoke.sh           # 一键跑通脚本
├── src/                   # 必要代码
├── data/                  # 图片、标签、manifest
└── models/                # SAM3 权重和最终特征判别器
```

关键文件：

```text
src/sam3_marking_detector.py
src/train_feature_judger.py
data/images/
data/labels.csv
models/sam3.pt
models/feature_judger/random_forest_final.joblib
```

当前默认最终判定方案：

```text
SAM3 标记线分割 -> 几何/骨架/界面特征 -> RandomForest 三分类判别器
```

其中 `LooseningJudger` 的规则结果会保存在输出表的 `rule_pred_label` 字段中；最终 `pred_label` 使用 `models/feature_judger/random_forest_final.joblib`。

最优三分类交叉验证指标：

| 指标 | 数值 |
| --- | ---: |
| sample_count | 1813 |
| accuracy | 0.929950 |
| macro_f1 | 0.914441 |
| `loose -> normal` | 47 |
| `normal -> loose` | 52 |

当前实际图片数：

| 类别 | 数量 |
| --- | ---: |
| `normal` | 1028 |
| `loose` | 319 |
| `unknown` | 466 |
| 合计 | 1813 |

说明：`data/labels.csv` 原始记录为 1843 条，其中 30 条 `loose` 图片在当前包中不存在；脚本遇到缺失图片会记录为读取失败。

## 环境

推荐在 198 服务器原 SAM3 环境运行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sam3
```

默认 SAM3 代码路径：

```text
/home/cvailab/zhaoza/sam3
```

如果 SAM3 代码在其他位置，先设置：

```bash
export SAM3_ROOT=/path/to/sam3
```

依赖至少包括：

```text
torch
opencv-python
numpy
pandas
pillow
scikit-learn
joblib
scipy
scikit-image
matplotlib
```

## 一键跑通

在本目录执行：

```bash
./run_smoke.sh
```

默认跑前 20 条样本，输出到：

```text
runs/smoke_YYYYMMDD_HHMMSS/
```

调整样本数或设备：

```bash
MAX_SAMPLES=50 DEVICE=cuda ./run_smoke.sh
```

跑通后应生成：

```text
detailed_results.csv
metrics.json
confusion_matrix.png
visualizations/
errors/
```

输出里的 `pred_label` 是 RandomForest 最终判定；`rule_pred_label` 是规则判定，便于排查。

## 全量运行

```bash
python src/sam3_marking_detector.py \
  --data-dir data \
  --labels data/labels.csv \
  --checkpoint models/sam3.pt \
  --device cuda \
  --output-dir runs/full_sam3_$(date +%Y%m%d) \
  --save-vis \
  --use-interface-rule
```

如需只看规则判定，可额外加：

```bash
--rule-only
```

## 训练特征判别器

全量 SAM3 运行结束后，用输出的特征表训练模型：

```bash
python src/train_feature_judger.py \
  --features-csv runs/full_sam3_YYYYMMDD/detailed_results.csv \
  --exclude-ids data/exclude_ids.csv \
  --output-dir runs/feature_judger_YYYYMMDD \
  --task both
```

重新训练后的三分类结果在：

```text
runs/feature_judger_YYYYMMDD/three_class/metrics.json
runs/feature_judger_YYYYMMDD/three_class/*_final.joblib
```

## 交接原则

这个包不要再加入历史实验目录、旧可视化、代理配置文件或临时测试图。新实验统一输出到 `runs/`。


## web部分

# Install API deps (once)
pip install fastapi uvicorn requests

# Start server
cd /Users/liudi/sam3_skill/src
python web_api.py --host 0.0.0.0 --port 8000 --device cuda

# Or with env vars
DEVICE=cuda CHECKPOINT=/path/to/sam3.pt python web_api.py

#Response shape (each detection)

{
  "image_id": "bolt_001",
  "prediction": "normal",        // normal | loose | unknown
  "confidence": 0.92,
  "features": { ... },           // all SAM3 + geometry + color features
  "candidates": [ ... ],         // top-8 SAM3 prompt candidates
  "error": null,
  "elapsed_ms": 1230.5
}

#Usage
  # Quick demo with a local image
  python examples/api_client_example.py \
    --base-url http://localhost:8000 \
    --image data/images/loose/sample.jpg

  # Full demo with batch
  python examples/api_client_example.py \
    --base-url http://localhost:8000 \
    --image data/images/loose/sample.jpg \
    --batch-dir data/images/normal

#Embed in your agent

from examples.api_client_example import Sam3Client

client = Sam3Client("http://localhost:8000")
if client.is_ready():
    result = client.detect_file("bolt.jpg")
    print(result["prediction"], result["confidence"])

    # batch
    results = client.detect_batch(["a.jpg", "b.jpg", "c.jpg"])
    for r in results:
        print(r["image_id"], r["prediction"])