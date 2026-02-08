# FOSR-DA  
**Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation**

FOSR-DA is a feature-enhanced deep learning framework for open-set recognition of communication jamming directly from raw complex-valued I/Q data.  
The domain adaptation (DA) module is specifically designed to improve the robustness of the Mahalanobis-distance-based open-set classifier under domain shifts.

This repository provides the official PyTorch implementation of:

**"Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation."**

---

## Features

- Open-set recognition of communication jamming patterns  
- End-to-end learning from raw complex-valued I/Q samples  
- Mahalanobis-distance-based open-set classifier  
- Domain adaptation module for improved robustness under channel shifts  
- Support for source-only testing and test-time adaptation

---

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
