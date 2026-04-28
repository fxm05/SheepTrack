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
