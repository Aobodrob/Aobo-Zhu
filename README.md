# Aobo-Zhu

# Automatically Classifying and Extracting Ice Layers and Bedrock Interfaces from IPR Data

This repository contains the source code, pre-trained models, and sample data for our research on the simultaneous extraction of internal ice layers and ice-bedrock interfaces at Dome A, Antarctica.

## 1. Project Overview
This project proposes a U-Net-based pipeline that integrates **Instantaneous Phase** analysis to enhance the extraction of low-contrast englacial structures. 

Key features:
- Phase-driven feature enhancement.
- Roughness-based discriminator for feature classification.
- Morphological post-processing for layer continuity.

## 2. Repository Structure
- `models/`: Contains the U-Net and Discriminator architecture definitions.
- `weights/`: Pre-trained weights for the U-Net and Discriminator (.pth files).
- `data/`: Sample synthetic and field IPR profiles for testing.
- `utils/`: Scripts for instantaneous phase computation and morphological filtering.
- `predict_demo.py`: A minimal runnable script to reproduce extraction results on sample data.

## 3. Operational Reproducibility
### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- NumPy, Scikit-image, Matplotlib

### Quick Start
To run the extraction on the provided sample data:
```bash
python predict_demo.py --input ./data/sample_profile.png --output ./results/
