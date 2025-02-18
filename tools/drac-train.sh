#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6-23  
#SBATCH --mail-user=kzn518@usask.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:2

# CODE AND DATA SETUP
cd $SLURM_TMPDIR
cp /home/kzn518/.ssh/id_drac .
eval "$(ssh-agent -s)"
ssh-add id_drac
git clone git@github.com:Francis-Ferri-personal/mmpose.git


cd ./mmpose
mkdir -p data/
tar -xv --use-compress-program="pigz -d -p 4" -f $projects/Datasets/pigpose.tar.gz -C ./data/


# VIRTUAL ENVIRONMENT SETUP
module purge
module load gcc cuda/12.6 arrow opencv/4.10.0 python/3.10
virtualenv --no-download $SLURM_TMPDIR/.venv
source $SLURM_TMPDIR/.venv/bin/activate
pip install --upgrade pip


# INSTALL DEPENDENCIES
pip install -r drac-requirements.txt
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
pip install -r requirements.txt
pip install -v -e .


# GET PARAMETERS
CONFIG_FILE=$1
GPU_NUM=$2
SEED=$3


# RUN TRAINING
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --cfg-options randomness.seed=${SEED}


# RUN EVALUATION
CONFIG_BASENAME=$(basename "$CONFIG_FILE")
EXPERIMENT_NAME="${CONFIG_BASENAME%.py}"

MODEL_FILE=$(ls work_dirs/${EXPERIMENT_NAME}/best_coco_AP_epoch_*.pth 2>/dev/null | head -n 1)

python tools/test.py ${CONFIG_FILE} work_dirs/${EXPERIMENT_NAME}/${MODEL_FILE} --work-dir work_dirs/${EXPERIMENT_NAME}/eval

python tools/test.py configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py work_dirs/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256/best_coco_AP_epoch_3.pth --work-dir work_dirs/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256/eval

# SAVE RESULTS
mkdir -p $projects/Outputs/pigpose
tar -cvf $projects/Outputs/pigpose/${EXPERIMENT_NAME}-${SEED}.tar work_dirs/${EXPERIMENT_NAME}


# RUN YOUR JOB
# cd $project/Workspace/mmpose
# sbatch tools\drac-train.sh configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py 2 42

# CHECK YOUR JOB
# sq

# INTERACTIVE JOBS
# salloc --time=0:30:0 --ntasks=8 --gres=gpu:a100:1 --mem=16G --nodes=1

# CANCEL JOB
# scancel <jobid>