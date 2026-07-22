import pickle
import sys
sys.path.insert(0, "/global/home/users/yoonjaepark/Program/repositories/les/src")
sys.path.insert(0, "/global/home/users/yoonjaepark/Program/repositories/mace/")

import os
import numpy as np
import torch
import torch.nn as nn
import les
import mace
import seekpath
from mace.calculators import MACECalculator
from ase.md import MDLogger
from ase.io import read, write
from ase import Atoms
from ase.data import atomic_numbers
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import PwscfToTHz
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp
from phonopy.units import Hartree, Bohr, Rydberg
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


variant = "uiu-isoa-mp1"

MODEL = f"/global/scratch/users/yoonjaepark/MLIP/fit-hfo2-selected/t-mace-{variant}/Hf02_stagetwo.model"
print('\nmodel=',MODEL)
DEVICE = 'cpu'
calculator = MACECalculator(model_paths=MODEL, device=DEVICE)

directory = f"./"


# ---------- read structure ----------
atoms = read(f"{directory}/ref_cell.extxyz", format="extxyz")
print("INPUT natoms =", len(atoms))
print("INPUT symbols =", atoms.get_chemical_symbols())

unitcell = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell.array,
    scaled_positions=atoms.get_scaled_positions(),)


# ---------- phonopy object ----------
phonon = Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]],)
symmetry = phonon.symmetry


# ---------- read BEC data ----------
pred_xyz = read(f"{directory}/bec_supercell.xyz", index=":")
print("nframes in BECS.xyz =", len(pred_xyz))
for i, at in enumerate(pred_xyz):
    print(i, len(at), at.get_chemical_symbols())

epsilon_infty = 5.365
n_uc = len(atoms)   # number of atoms in unit cell
if n_uc != 12: print('\n NOT ortho-ferro? or check unit cell')

for xyz in pred_xyz:
    epsilon_0 = 0.00552635 
    volume = xyz.get_volume()
    if 'MACE_latent_alphas' in xyz.arrays:
        alpha = xyz.arrays['MACE_latent_alphas']
        if alpha.shape[-1] == 9:
            chi = np.einsum('icc->', alpha.reshape(-1, 3, 3)) / 3.0 / volume / epsilon_0
        else:
            chi = alpha.sum(axis=0) / volume / epsilon_0
        epsilon_r = epsilon_infty / (1.0 + chi)
    else:
        epsilon_r = epsilon_infty
    raw_bec = xyz.get_array('MACE_BEC')

    natoms_sc = len(xyz)
    if natoms_sc % n_uc != 0:
        raise ValueError(f"Supercell natoms={natoms_sc} is not divisible by unit-cell natoms={n_uc}")

    n_rep = natoms_sc // n_uc
    bec_all = raw_bec[:, :9] * epsilon_r ** 0.5 
    if xyz.arrays['MACE_BEC'].shape[1] > 9:
       bec_all += raw_bec[:, 9:18] * epsilon_r ** 0.5
    bec_all = np.array(bec_all)
    bec_all = bec_all.reshape(n_rep, n_uc, 9)
    bec_avg = bec_all.mean(axis=0)  
    bec_avg = bec_avg.reshape(n_uc, 3, 3)
born = bec_avg.tolist()


# ---------- generate displaced supercells ----------
delta = 0.01
print('\n displacement =', delta)
phonon.generate_displacements(distance=delta)


# ---------- NAC turn on ----------
epsilon = [[epsilon_infty, 0, 0],
           [0, epsilon_infty, 0],
           [0, 0, epsilon_infty]]

phonon._nac_params = {
    "factor": 14.399652,
    "born": born,
    "dielectric": epsilon,
}
if not phonon.nac_params: print('check NAC para:', phonon.nac_params)


# ---------- compute forces ----------
forces = []
for supercell in phonon.supercells_with_displacements:
    scellcalc = Atoms(symbols=supercell.symbols, 
                      scaled_positions=supercell.scaled_positions, 
                      cell=supercell.cell, 
                      pbc =True)
    scellcalc.calc = calculator
    f = scellcalc.get_forces()
    f -= f.mean(axis=0, keepdims=True)
    forces.append(f)
phonon.produce_force_constants(forces=forces)
phonon.symmetrize_force_constants()


# ---------- band struc. calc. ----------
G = [0.0,   0.0,   0.0]
X = [0.5,   0.0,   0.0]
S = [0.5,   0.5,   0.0]
Y = [0.0,   0.5,   0.0]
Z = [0.0,   0.0,   0.5]
U = [0.5,   0.0,   0.5]
R = [0.5,   0.5,   0.5]
T = [0.0,   0.5,   0.5]

path = [ [G, X, S, Y, G, Z, U, R, T, Z] ]
labels = ["$\\Gamma$", "X", "S", "Y", "$\\Gamma$", "Z", "U", "R", "T", "Z"]

qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

band_dict = phonon.get_band_structure_dict()
q_points = band_dict["qpoints"]
distances = band_dict["distances"]
frequencies = band_dict["frequencies"]
eigvecs = band_dict["eigenvectors"]

freq_array = np.vstack(frequencies)   
np.savetxt("frequenciesNAC.txt", freq_array)

dist_array = np.hstack(distances)
np.savetxt("distancesNAC.txt", dist_array, fmt="%.6f")

for q_path, d_path, freq_path in zip(q_points, distances, frequencies):
    for q, d, freq in zip(q_path, d_path, freq_path):
        print(
            ("%10.5f  %5.2f %5.2f %5.2f " + (" %7.3f" * len(freq)))
            % ((d, q[0], q[1], q[2]) + tuple(freq))
        )


# ---------- plot ----------
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
phonon.plot_band_structure()
plt.savefig("phonon_dispersion_nac_labeled.png", bbox_inches="tight")
plt.show()


