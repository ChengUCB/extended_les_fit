#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=pc_lestest
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Processors per task (on `es1` please always specify the total number of processors at least twice the number of GPUs):
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:V100:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=48:00:00

source ~/.bashrc

#load an CUDA software module
module load gcc/11.4.0 cuda/12.8.0

export PYTHONWARNINGS="ignore"

srun python ~/repositories/mace/scripts/run_train.py \
    --name="mapi" \
    --train_file="../MaPbI3.xyz" \
    --valid_fraction=0.05 \
    --test_file="../MaPbI3.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s='average' \
    --model="MACELES" \
    --les_arguments='les.yaml' \
    --hidden_irreps='64x0e + 64x1o' \
    --r_max=5.5 \
    --num_interactions=1 \
    --batch_size=4 \
    --max_num_epochs=1200 \
    --stage_two \
    --start_stage_two=700 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --default_dtype="float32"\
    --save_cpu \
