#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append('/global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/cace/')
sys.path.append('/global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/les/src/')

import numpy as np
import torch
import torch.nn as nn
import logging

import cace
from cace.tools import torch_geometric
from cace.data import AtomicData


root = './best_model.pth'
#average_E0 ={1: -5.853064337340629, 8: -2.926532168670322} # avg # This should be changed to be the same as atomic_energies set in the training script
average_E0= {8: -18599.43617104475, 12: -8721.75974245582, 13: -9877.676428588728, 79: -688.8680063349827} # avg

cace_nnp = torch.load(root, map_location='cuda')

if len(sys.argv) < 2:
    print("Usage: python test.py path/to/test_SYSTEM.xyz")
    sys.exit(1)


to_read = sys.argv[1]
base = os.path.basename(to_read)       # 'water.xyz'
system = os.path.splitext(base)[0] # 'water'



os.makedirs('test', exist_ok=True)
test_write = f"./test/{system}_test_cace.xyz"

print(f"[INFO] Input file : {to_read}")
print(f"[INFO] System     : {system}")
print(f"[INFO] Output xyz : {test_write}")



evaluator = cace.tasks.EvaluateTask(model_path=cace_nnp, device='cuda',
                                    energy_key = 'CACE_energy',
                                    forces_key = 'CACE_forces',
                                    #other_keys=['q', 'tot_q'], # only for lr model
                                    atomic_energies = None,
                                    data_key={'energy': 'energy', 'forces':'forces'}, 
                                    )
from ase.io import read
dataset = read(to_read, index=':')

model_cutoff = cace_nnp.representation.cutoff




result = evaluator(data=dataset, batch_size=1, compute_stress=False, xyz_output=test_write)


import numpy as np
dataset = read(test_write, index=':')

pred_residual_energy = np.array([atoms.info['CACE_energy'] for atoms in dataset])
pred_forces = np.vstack([atoms.get_array('CACE_forces') for atoms in dataset])

ref_total_energy = np.array([atoms.info['energy'] for atoms in dataset])
ref_forces = np.vstack([atoms.get_array('forces') for atoms in dataset])

def get_baseline(atoms, e0_dict):
    return sum(e0_dict.get(z, 0.0) for z in atoms.get_atomic_numbers())
baseline_energies = np.array([get_baseline(atoms, average_E0) for atoms in dataset])
ref_residual_energy = ref_total_energy - baseline_energies

def rmse(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return np.sqrt(np.mean((a - b)**2))

atoms_list = read(to_read, index=":")
n_atoms = np.array([len(atoms) for atoms in atoms_list])

ref_res_e_per_atom = ref_residual_energy / n_atoms
pred_res_e_per_atom = pred_residual_energy / n_atoms

rmse_per_atom = rmse(ref_res_e_per_atom, pred_res_e_per_atom)
force_rmse  = rmse(ref_forces, pred_forces)


print(f"Test file: {to_read}")
print(f"Per-atom energy RMSE: {rmse_per_atom*1000:.4f} meV/atom")
print(f"Force  RMSE: {force_rmse*1000:.3f} meV/Å")

# summary_file = "cace_test_summary.txt"
# header = "# system  per_atom_RMSE_meV_per_atom  force_RMSE_eV_per_A\n"

# line = f"{system}  {rmse_per_atom*1000:.3f}  {force_rmse:.5f}\n"

# write_header = not os.path.exists(summary_file)

# with open(summary_file, "a") as f:
#     if write_header:
#         f.write(header)
#     f.write(line)
