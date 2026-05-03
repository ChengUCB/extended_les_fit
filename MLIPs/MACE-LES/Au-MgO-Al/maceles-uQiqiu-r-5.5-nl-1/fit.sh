#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=pc_lestest
#SBATCH --partition=es2                # Einsteinium 2
#SBATCH --qos=es2_normal               # QoS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=9200M
#SBATCH --time=48:00:00

source ~/.bashrc

#load an CUDA software module
module load gcc/11.4.0 cuda/12.8.0

export PYTHONWARNINGS="ignore"

srun python ~/repositories/mace/scripts/run_train.py \
    --name="Au2-MgO" \
    --train_file="../train-Au-MgO-Al.xyz" \
    --valid_fraction=0.1 \
    --test_file="../test-Au-MgO-Al.xyz" \
    --energy_key="energy" \
    --forces_key="forces" \
    --swa_forces_weight=100.0 \
    --swa_energy_weight=2000.0 \
    --E0s='average' \
    --model="MACELES" \
    --les_arguments='les.yaml' \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.5 \
    --num_interactions=1 \
    --forces_weight=100 \
    --energy_weight=0.1 \
    --batch_size=8 \
    --max_num_epochs=800 \
    --stage_two \
    --start_stage_two=500 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
