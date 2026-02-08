# FOSR-DA  

FOSR-DA is a feature-enhanced framework for jamming open-set recognition from raw I/Q data, with a DA module designed for the Mahalanobis-distance classifier.

This repository provides the official PyTorch implementation of:

**"Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation."**


## Setup

Install the required dependencies:

    pip install -r requirements.txt

---

## Dataset

Please place the dataset files in the data/ directory. The datasets include:

- JPR2024_gaussian.mat

- JPR2024_rayleigh.mat

- JPR2024_rician.mat

The dataset is provided via GitHub Releases: https://github.com/jhl3676-hub/FOSR-OpenSet-JPR/releases/tag/dataset

---

## Usage

### 1. Basic Mode (Source-Only)

Train on the source domain and evaluate on the source domain:

    python main.py --source_data ./data/JPR2024_gaussian.mat    

---

### 2. Domain Adaptation Mode (Domain Adaptation)

Train on the source domain and then perform test-time adaptation on the target domain: 

    python main.py --source_data ./data/JPR2024_gaussian.mat --target_data ./data/JPR2024_rayleigh.mat --mode da

---

## Arguments

- `--source_data`: Path to the source-domain dataset  
- `--target_data`: Path to the target-domain dataset (required for DA mode)  
- `--mode`: Running mode  
  - `basic`: Source-only evaluation  
  - `da`: Training with Source and testing with Domain Adaptation. 

---

## Citation

    @article{fosrda2025,
      title     = {Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation},
      author    = {Jiahao Li, Ziming Du, Bo Zhou, Wei Wang, Qihui Wu and Walid Saad},
      journal   = {Under Review},
      year      = {2025}
    }
    
---

## License/许可证

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
