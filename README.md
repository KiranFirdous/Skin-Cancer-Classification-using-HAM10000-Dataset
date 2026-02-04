# Skin-Cancer-Classification-using-HAM10000-Dataset
This repository implements a deep learning system for classifying skin lesion images into 8 different categories using the HAM10000 dataset.


## 📋 Project Overview

The HAM10000 ("Human Against Machine with 10000 training images") dataset contains 10,015 dermatoscopic images of common pigmented skin lesions. This project implements transfer learning with ResNet152 and EfficientNetB0 for multi-class classification of skin cancer.

## 🎯 Classes

The model classifies images into 8 categories:
- **AK** (Actinic keratosis)
- **BCC** (Basal cell carcinoma)
- **BKL** (Benign keratosis)
- **DF** (Dermatofibroma)
- **MEL** (Melanoma)
- **NV** (Melanocytic nevus)
- **SCC** (Squamous cell carcinoma)
- **VASC** (Vascular lesion)

## 🏗️ Architecture

The project implements two state-of-the-art models:
1. **ResNet152** - 152-layer residual network
2. **EfficientNetB0** - EfficientNet B0 architecture



## 🚀 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.4+
- CUDA-compatible GPU (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt


