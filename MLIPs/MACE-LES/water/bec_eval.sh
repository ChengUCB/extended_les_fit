for i in */; do
echo $i
cd $i
#python /global/scratch/users/dongjinkim/md_jobs/u_MACE/mace/scripts/eval_configs.py --configs /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz --output test-bec.xyz --model H20.model --batch_size 2 --default_dtype float32 --compute_bec

python /global/scratch/users/dongjinkim/md_jobs/u_MACE/mace/scripts/eval_configs.py --configs /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz --output test-bec.xyz --model H2O_stagetwo.model --batch_size 2 --default_dtype float32 --compute_bec
cd ..
done

