#!/bin/bash


#for i in ./t-*; do
#
#echo $i
#python ./test.py /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz $i T
#python ./test.py /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/test-H2O_RPBE-D3.xyz $i F

for i in ./*uiu*; do

echo $i
python ./test.py /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz $i T T
done

for i in ./*uiqiu*; do

echo $i
python ./test.py /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz $i T T
done
