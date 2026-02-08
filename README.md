# FOSR-DA  

FOSR-DA is a feature-enhanced deep learning framework for open-set recognition of communication jamming directly from raw complex-valued I/Q data, with a domain adaptation (DA) module designed for the Mahalanobis-distance classifier.

This repository provides the official PyTorch implementation of:

**"Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation."**

---

## Features

- Open-set recognition of communication jamming patterns  
- End-to-end learning from raw complex-valued I/Q samples  
- Mahalanobis-distance-based open-set classifier  
- Domain adaptation module for improved robustness under channel shifts  
- Support for test-time adaptation

---

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt

---

## Setup

1. Basic Mode (Source-Only / No Adaptation)

Train on the source domain and evaluate on the target domain
