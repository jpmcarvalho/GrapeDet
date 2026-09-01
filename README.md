# Grapevine Disease Detection Dataset

<img width="1600" height="900" alt="WhatsApp Image 2026-07-10 at 14 59 51" src="https://github.com/user-attachments/assets/e22eddf7-cfef-496a-a523-229e5129f4ba" />

This repository contains the source code, datasets, and experimental material developed for the research project **"Deep Learning Models for Early Detection of Diseases in Grapevines"**.

The main objective of this work is to investigate deep learning approaches for the early detection of grapevine diseases through object detection and image classification. To achieve this, a new object detection dataset was created from the publicly available **Grapevine Disease Dataset (Original)** (https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original) by manually annotating the visible disease symptoms with bounding boxes.

The repository also provides the complete experimental pipeline used to compare **one-stage** and **two-stage** detection architectures. The one-stage approach is based on **YOLOv12**, while the two-stage architecture combines a YOLOv12 detector with three image classifiers: **MobileNetV3**, **EfficientNetV2**, and **DINOv2**. All datasets follow the standard **Ultralytics ****`config.yaml`** directory structure.

## Repository Structure

The repository is organized according to the chronological workflow of the research.

### 1. Dataset Creation

Contains the manually annotated dataset created using **CVAT.ai** from the images of the **Grapevine Disease Dataset (Original)**. This stage includes the raw annotations before preprocessing and dataset splitting for training and validation.

### 2. Dataset Analysis

Includes all scripts used to analyse the dataset, including:

* Image and annotation statistics;
* Bounding box size and area analysis;
* Color distribution analysis;
* Feature visualization using **UMAP**;
* Sample similarity analysis using **K-Nearest Neighbors (KNN)**.

### 3. Dataset Preprocessing for Training

Contains the complete preprocessing pipeline used to generate the final datasets for both one-stage and two-stage architectures.

This stage also investigates the influence of annotation outliers by generating alternative datasets where bounding boxes exceeding the upper whisker limits of **1.2 × IQR** and **1.5 × IQR** are removed using box plot analysis.

### 4. One-Stage YOLOv12 Training

Provides the training scripts and experimental configurations for the one-stage detection architecture using **YOLOv12**.

The experiments evaluate the influence of bounding box size and annotation area on training performance and detection accuracy.

### 5. Two-Stage Object Detection Training

Contains the training pipeline for the first stage of the two-stage architecture, where **YOLOv12** is trained exclusively as a disease detector using a single **Disease** class.

### 6. Classification Models Training

Includes the training scripts for the second-stage classifiers:

* MobileNetV3
* EfficientNetV2
* DINOv2

All classification models are implemented using the **Hugging Face Transformers** library.

### 7. Final Validation

Contains the complete evaluation framework used to compare the one-stage and two-stage architectures.

The scripts generate the final performance metrics, comparative analyses, and experimental results.

## Citation

Vasco M. Marmelo, João P. Matos-Carvalho and Sérgio D. Correia. "YOLO-Based Architectures for Grapevine Disease Detection." 2026 10th International Young Engineers Forum on Electrical and Computer Engineering (YEF-ECE). IEEE, 2026.

