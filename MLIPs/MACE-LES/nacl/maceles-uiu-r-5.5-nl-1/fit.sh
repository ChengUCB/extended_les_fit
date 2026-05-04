#!/bin/bash
#
#----------------------------------
# single GPU + single CPU example
#----------------------------------
#
#SBATCH --job-name=test
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=24:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=48G
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning,
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Define the "gpu" partition for GPU-accelerated jobs
#SBATCH --partition=gpu
#
#Define the number of GPUs used by your job
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[113,114,118,119,123-125,127,136-137,138,139,144,146,148,150]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#
#Define the GPU architecture (GTX980 in the example, other options are GTX1080Ti, K40)
##SBATCH --constraint=GTX980
#
#SBATCH --exclusive
#Do not export the local environment to the compute nodes
##SBATCH --export=NONE
##unset SLURM_EXPORT_ENV
#

#source ~/.bashrc
module load miniforge3/24.3.0
conda activate myenv
export PYTHONPATH=/nfs/scistore23/chenggrp/bcheng/les-dipole/les/src/:$PYTHONPATH
export PYTHONPATH=/nfs/scistore23/chenggrp/bcheng/les-dipole/mace/:$PYTHONPATH
module load cuda/12.0.0

export PYTHONWARNINGS="ignore"

python /nfs/scistore23/chenggrp/bcheng/les-dipole/mace/scripts/run_train.py \
    --name="nacl" \
    --train_file="../NaCl.xyz" \
    --valid_fraction=0.05 \
    --test_file="../NaCl.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s='average' \
    --model="MACELES" \
    --les_arguments='les.yaml' \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.5 \
    --num_interactions=1 \
    --batch_size=16 \
    --max_num_epochs=650 \
    --stage_two \
    --start_stage_two=400 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --default_dtype="float64"\
    --save_cpu \
