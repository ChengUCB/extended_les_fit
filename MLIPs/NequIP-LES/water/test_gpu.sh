file=$1
radius=$2
name=$3
nequip-train -cn water_bec_test.yaml cutoff_radius="$radius" training_module.model.checkpoint_path="$file"

mv ./predictions/test_dataset0.xyz ./predictions/"$name".xyz
