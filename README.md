# SheepTrack
Official repo for the paper **"SheepTrack: Occlusion-Robust Detection and Tracking for Dense Sheep Monitoring"**, published in *Electronics* (2026).

This repository contains the full PyTorch implementation of the proposed detection and tracking framework, including training and inference scripts.

SheepTrack is an integrated pipeline for robust detection and tracking of sheep in dense and occluded indoor scenes. It improves YOLOv8 with SheepNMS and FL-Loss, and enhances BoT-SORT with FAM and TCM to reduce ID switches and trajectory drift.

## Dataset
The corresponding multi-scene indoor sheep dataset is publicly available to support further research:
- Zenodo: https://zenodo.org/records/19567803
