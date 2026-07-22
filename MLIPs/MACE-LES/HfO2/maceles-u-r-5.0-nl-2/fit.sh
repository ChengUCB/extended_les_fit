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
#
## Command(s) to run (example):

source ~/.bashrc

#load an CUDA software module
module load gcc/11.4.0 cuda/12.8.0

export PYTHONWARNINGS="ignore"

srun python ~/repositories/mace/scripts/run_train.py \
    --name="Hf02" \
    --train_file="../HfO2_train.xyz" \
    --valid_fraction=0.05 \
    --test_file="../HfO2_test.xyz" \
    --energy_key="Force_consistent_energy" \
    --forces_key="new_forces" \
    --E0s='average' \
    --model="MACELES" \
    --les_arguments='les.yaml' \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --num_interactions=2 \
    --batch_size=4 \
    --max_num_epochs=700 \
    --stage_two \
    --start_stage_two=350 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --default_dtype="float32"\
    --save_cpu \

