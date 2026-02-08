# FOSR-DA  

FOSR-DA is a feature-enhanced deep learning framework for open-set recognition of communication jamming directly from raw complex-valued I/Q data.  
The domain adaptation (DA) module is specifically designed to improve the robustness of the Mahalanobis-distance-based open-set classifier under domain shifts.

This repository provides the official PyTorch implementation of:

**"Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation."**


## Setup

Install the required dependencies:

    pip install -r requirements.txt

---

## Usage

### 1. Basic Mode (Source-Only / No Adaptation)

Train on the source domain and evaluate on the source domain:

    python main.py --source_data ./data/JPR2024_gaussian.mat    

---

### 2. Domain Adaptation Mode (Test-Time Adaptation)

Train on the source domain and then perform test-time adaptation on the target domain: 

    python main.py --source_data ./data/JPR2024_gaussian.mat --target_data ./data/JPR2024_rayleigh.mat --mode da

---

## Arguments

- `--source_data`: Path to the source-domain dataset  
- `--target_data`: Path to the target-domain dataset (required for DA mode)  
- `--mode`: Running mode  
  - `basic`: Source-only evaluation  
  - `da`: Source trained model and test in target using DA mode 

---

## Dataset

数据集可以从data路径中下载，包含JPR2024_rayleigh.mat,JPR2024_rician.mat and JPR2024_gaussian.mat 数据集详情见论文.

---

## Citation

    @article{fosrda2025,
      title     = {Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation},
      author    = {Jiahao Li, Ziming Du, Bo Zhou, Wei Wang, Qihui Wu and Walid Saad},
      journal   = {Under Review},
      year      = {2025}
    }
## License/许可证
本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
