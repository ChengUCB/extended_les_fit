for i in */; do

echo $i
cd $i
python /global/scratch/users/dongjinkim/md_jobs/u_MACE/mace/scripts/eval_configs.py --configs /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/MAPbI3_BEC.xyz --output test-bec.xyz --model mapi_stagetwo.model --batch_size 2 --default_dtype float32 --compute_bec

cd ..
done
