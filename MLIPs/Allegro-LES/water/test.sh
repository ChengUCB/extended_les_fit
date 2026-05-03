#!/bin/bash
file=$1
#radius=$2
name=$2
nequip-train -cn water_bec_test.yaml training_module.model.checkpoint_path="$file"

mv ./predictions/test_dataset0.xyz ./predictions/"$name".xyz

#nequip-train -cn water_bec_test.yaml cutoff_radius=4.5 training_module.model.checkpoint_path=/global/scratch/users/dongjinkim/NequIP-LES/water/outputs/2025-05-15/15-05-48/best.ckpt

echo "End: $(date)"
