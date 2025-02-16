# Guide for running the project

# Installation
**Reference:** https://mmpose.readthedocs.io/en/latest/installation.html
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

CHeck our version of CUDA
```bash
nvidia-smi
#  CUDA Version: 12.8
```

Install PyTorch
**Reference:** https://pytorch.org/get-started/locally/
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
 Install MMEngine and MMCV using MIM.
 ```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
 ```
Install MMPose
```bash
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```
Clone the MMPose repository with Pig pose implementation
 ```bash
git clone git@github.com:Francis-Ferri-personal/mmpose.git
  ```