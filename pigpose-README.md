# Guide for running the project

## Installation
**Reference:** https://mmpose.readthedocs.io/en/latest/installation.html
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

Check our version of CUDA
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

Clone the MMPose repository with Pig pose implementation
 ```bash
git clone git@github.com:Francis-Ferri-personal/mmpose.git
  ```

Install MMPose
```bash
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

## Use in Digital Research Alliance Canada (DRAC)

Upload your dataset and compress it

You can use (modify if needed) the script

```bash
# tools\drac-train.sh
cd $project/Workspace/mmpose
# sbatch tools\drac-train.sh ${CONFIG_FILE} ${NUM_GPUS} ${SEED}
sbatch tools\drac-train.sh configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py 2 42
```

Modification for weights:
/home/kzn518/cmc/.conda/envs/openmmlabpose/lib/python3.9/site-packages/torch/serialization.py
```python
def load(
    f: FileLike,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    # weights_only: Optional[bool] = None,
    weights_only: Optional[bool] = False,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
```