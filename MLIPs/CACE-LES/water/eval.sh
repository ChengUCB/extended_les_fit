#!/bin/bash


python ./test.py /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/test-H2O_RPBE-D3.xyz $i F F
python ./test.py /global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz $i T T
