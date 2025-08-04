#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6-23  
#SBATCH --mail-user=kzn518@usask.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:2


export CUBLAS_WORKSPACE_CONFIG=:4096:8

# GET PARAMETERS AND VALIDATION
usage() {
    echo "Usage: $0 --config CONFIG_FILE --gpus NUM_GPUS --dataset DATASET"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" || -z "$NUM_GPUS" || -z "$DATASET" ]]; then
    echo "Missing required parameters."
    usage
fi


# CODE AND DATA SETUP
cd $SLURM_TMPDIR
cp /home/kzn518/.ssh/id_drac .
eval "$(ssh-agent -s)"
ssh-add id_drac
git clone git@github.com:Francis-Ferri-personal/mmpose.git


cd ./mmpose
mkdir -p data/
tar -xv --use-compress-program="pigz -d -p 4" -f $projects/Datasets/${DATASET}.tar.gz -C ./data/


# VIRTUAL ENVIRONMENT SETUP
module load scipy-stack gcc cuda/12.6 arrow opencv/4.10.0 python/3.10
virtualenv --no-download $SLURM_TMPDIR/.venv
source $SLURM_TMPDIR/.venv/bin/activate
pip install --upgrade pip


# INSTALL DEPENDENCIES
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --no-binary torch==2.1.2 torchvision==0.16.2
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
pip install -r requirements.txt
python setup.py install
pip install -v -e .


# RUN TRAINING
PORT=29501 bash ./tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS}


# RUN EVALUATION
CONFIG_BASENAME=$(basename "$CONFIG_FILE")
EXPERIMENT_NAME="${CONFIG_BASENAME%.py}"

MODEL_FILE=$(ls work_dirs/${EXPERIMENT_NAME}/best_coco_AP_epoch_*.pth 2>/dev/null | head -n 1)

# Metrics evaluation
python tools/test.py ${CONFIG_FILE} work_dirs/${EXPERIMENT_NAME}/${MODEL_FILE} --work-dir work_dirs/${EXPERIMENT_NAME}/eval > work_dirs/${EXPERIMENT_NAME}/model-evaluation.txt

# Performance evaluation
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} > work_dirs/${EXPERIMENT_NAME}/model-analysis.txt


# SAVE RESULTS
mkdir -p $projects/Outputs/${DATASET}
tar -cvf $projects/Outputs/${DATASET}/${EXPERIMENT_NAME}.tar work_dirs/${EXPERIMENT_NAME}


# RUN YOUR JOB
# cd $project/Workspace/mmpose
# sbatch tools/drac-train.sh --config configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py --gpus 2 --dataset pigpose

# CHECK YOUR JOB
# sq

# INTERACTIVE JOBS
# salloc --time=1:30:0 --ntasks=8 --gres=gpu:p100:4 --mem=64G --nodes=1
# salloc --time=1:30:0 --ntasks=8 --gres=gpu:v100:2 --mem=64G --nodes=1

# CANCEL JOB
# scancel <jobid>