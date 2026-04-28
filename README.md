# SheepTrack
Official repo for the paper **"SheepTrack: Occlusion-Robust Detection and Tracking for Dense Sheep Monitoring"**, published in *Electronics* (2026).

This repository contains the full PyTorch implementation of the proposed detection and tracking framework, including training and inference scripts.

SheepTrack is an integrated pipeline for robust detection and tracking of sheep in dense and occluded indoor scenes. It improves YOLOv8 with SheepNMS and FL-Loss, and enhances BoT-SORT with FAM and TCM to reduce ID switches and trajectory drift.

## Abstract
Automated detection and tracking of individual sheep are essential for precision livestock farming. However, existing approaches face significant challenges: (1) Limited dataset diversity with predominant aerial perspectives; (2) Detection failures under severe occlusions; (3) Frequent ID switches due to high appearance similarity. To address these challenges, we present an integrated framework. Firstly, we construct a multi-scene indoor sheep dataset with diverse environmental conditions. Secondly, for detection, we propose an improved YOLOv8 incorporating SheepNMS and Flock-aware Localization Loss (FL-Loss) to handle crowded scenarios and occlusion. Finally, for tracking, we enhance BoT-SORT with a Flock Appearance Module (FAM) and Trajectory Correction Module (TCM) for robust association and drift mitigation. Extensive experiments demonstrate measurable improvements in detection accuracy, tracking consistency, and reductions in ID switches and fragmentations across diverse monitoring scenarios.

## Dataset
The corresponding multi-scene indoor sheep dataset is publicly available to support further research:
- Zenodo: https://zenodo.org/records/19567803

## Setup
This project is built on the Ultralytics framework.

```bash
pip install ultralytics
pip install torch torchvision opencv-python numpy scipy tqdm
```

## Training

```python
from ultralytics import YOLO

# 加载 YOLOv8 预训练模型
model = YOLO("yolov8n.pt")

# 启动训练
train_results = model.train(
    data="/home/data/datasets/data.yaml",  # 数据集配置文件路径
    epochs=10,                             # 训练轮数
    imgsz=640,                             # 输入图片尺寸
    device=[0],                            # 训练设备（0 表示第1块GPU，cpu 表示CPU）
)

# 可选：导出训练后的模型为 ONNX 格式
path = model.export(format="onnx")  # 导出模型路径返回
```

## Evaluation

```python
# 验证集评估
val_metrics = model.val()
print(f"验证集 mAP@0.5: {val_metrics.box.map:.4f}")
# 测试集评估
test_metrics = model.val(
    split='test',                          # 指定评估测试集
    data="/home/data/datasets/data.yaml",  # 数据集配置文件
    imgsz=640                              # 保持与训练一致的图片尺寸
)

# 输出测试集核心指标
print("--- 测试集评估结果 ---")
print(f"测试集 mAP@0.5: {test_metrics.box.map:.4f}")          # IoU=0.5 时的 mAP
print(f"测试集 mAP@0.5:0.95: {test_metrics.box.map50_95:.4f}")# IoU=0.5~0.95 时的 mAP
print(f"测试集 Precision: {test_metrics.box.p:.4f}")          # 精确率
print(f"测试集 Recall: {test_metrics.box.r:.4f}")             # 召回率
print(f"测试集 F1-Score: {test_metrics.box.f1:.4f}")          # F1 分数

from ultralytics import YOLO

# 加载训练后的最优模型
model = YOLO("/home/runyolo/runs/detect/train/weights/best.pt")

# 执行目标跟踪
results = model.track(
    source="/root/autodl-tmp/datasets/test/test1/222.mp4",  # 测试视频路径
    tracker="/home/ultralytics/ultralytics/cfg/trackers/botsort.yaml"  # 跟踪器配置
)

# 保存跟踪结果（MOTChallenge 格式）
with open('/home/runyolo/1.txt', 'w') as f:
    for frame_id, result in enumerate(results):
        for box in result.boxes:
            if box.id is None:
                continue
            bbox = box.xyxy[0].tolist()
            track_id = box.id.item()
            conf = box.conf.item()
            # 格式：帧号,跟踪ID,左上角x,左上角y,宽,高,-1,-1,置信度
            f.write(f'{frame_id+1},{int(track_id)},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')
```

## BibTeX
If you find this work useful for your research, please cite:

```bibtex
@article{feng2026sheeptrack,
  title={SheepTrack: Occlusion-Robust Detection and Tracking for Dense Sheep Monitoring},
  author={Feng, Xiaomu and Li, Jiping and Yi, Jiacheng and Wang, Zhenhua},
  journal={Electronics},
  year={2026},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
