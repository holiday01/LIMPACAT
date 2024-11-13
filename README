# LIMPACAT Tutorial 🚀

Welcome to the **LIMPACAT (Liver Immune Prediction Attention Transformer)** tutorial! This guide will walk you through the setup, usage, and features of LIMPACAT – a deep learning model designed to predict immune cell composition in liver cancer using multi-omics data and whole-slide imaging (WSI). 🎨🧬

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Step 1: Download the Software](#step-1-download-the-software)
  - [Step 2: Analyze Immune Cell Composition](#step-2-analyze-immune-cell-composition)
  - [Step 3: Perform Survival Analysis](#step-3-perform-survival-analysis)
  - [Step 4: Prepare Image Data for Training](#step-4-prepare-image-data-for-training)
  - [Step 5: Train WSI Images for Prediction](#step-5-train-wsi-images-for-prediction)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Introduction

LIMPACAT (Liver Immune Prediction Attention Transformer) combines **whole-slide imaging** and **multi-omics analysis** with a **transformer-based attention mechanism**. It focuses on detecting relevant immune cell patterns, aiding personalized liver cancer treatment strategies. The model allows for configurable parameters in **input data**, **cell type specification**, and **training**.

## Requirements 📋
Ensure the following software is installed:
- **Operating Systems**: Windows, macOS, or Linux
- **Software**: Python, Conda, Docker, Git

---

## Setup Instructions

### Step 1: Download the Software 🐋
Pull the Docker image for LIMPACAT:
```
docker pull yenjungchiu/limpacat
```

### Step 2: Analyze Immune Cell Composition 🧬

This section guides you through analyzing immune cell composition from gene expression data using the **LIMPACAT** Docker image. Follow the steps below to get started.

### Command
Navigate to your project directory and run the following command:
```
docker run --rm -v .:/app/output -e SCRIPT_ALIAS=CCD limpacat
```

### Step 3: Perform Survival Analysis 📊

This step describes how to perform a survival analysis based on immune cell composition data using **LIMPACAT**.

### Command
In your project directory, use the following command:
```
docker run --rm -v .:/app/output -e SCRIPT_ALIAS=SUR limpacat
```

### Step 4: Prepare Image Data for Training 🖼️

This step will guide you in preparing image data for training based on cell type composition using **LIMPACAT**.

### Command
To create the necessary image data for training, run:
```
docker run --rm -v .:/app/output -e SCRIPT_ALIAS=IMG_JSON limpacat -f image_data_source.txt -c Monocyte
```

### Step 5: Train WSI Images for Immune Cell Content Prediction 📈

This step will walk you through training the model on Whole Slide Images (WSI) to predict immune cell content using **LIMPACAT**.

### Command
Use the following command to initiate training on WSI images:
```
docker run --rm --gpus all -v /data/liver_image_all:/app/image -v ./test:/app/output -e SCRIPT_ALIAS=LIMPACAT limpacat
```
