import os
import numpy as np
import sys
sys.path.insert(0, "/global/home/users/yoonjaepark/Program/repositories/les/src")
sys.path.insert(0, "/global/home/users/yoonjaepark/Program/repositories/mace/")

import les
import mace
from mace.calculators import MACECalculator
from ase import units
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import mol
from ase.filters import StrainFilter, UnitCellFilter
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet


if len(sys.argv) != 2:
    print("Usage: python npt-ase.py [cpu|cuda]")
    sys.exit(1)

device = sys.argv[1]
atemp = 300
init_file = "./init_hfo2.xyz"
variant = "uiu-isoa-mp1"


# -----------------------------
# Simulation parameters
# -----------------------------
TEMPERATURE = float(atemp)  # K
PRESSURE_BAR = 1.01325  # bar
TIMESTEP = 0.5
steps_equi = int(1000//TIMESTEP * 20)
steps_sample = int(1000//TIMESTEP * 100)
N_STEPS = steps_sample
LOG_INTERVAL = 10
TRAJ_INTERVAL = 1
TTIME_FS = 50.0


# -----------------------------
# Load initial structure
# -----------------------------
atoms = read(init_file, index=0)
print("\nOriginal cell:", flush=True)
print("Cell:", atoms.cell, flush=True)
print("PBC:", atoms.pbc, flush=True)


# -----------------------------
# Setup MACE calculator
# -----------------------------
MODEL = f"/global/scratch/users/yoonjaepark/MLIP/fit-hfo2-selected/t-mace-{variant}/Hf02_stagetwo.model"
calculator = MACECalculator(
    model_path=MODEL, 
    device=device,
    compute_bec=True,
    default_dtype="float32"
)
atoms.calc = calculator


# -----------------------------
# Minimization
# -----------------------------
ucf = UnitCellFilter(atoms, hydrostatic_strain=True, scalar_pressure=0.0)
opt = FIRE(ucf, logfile=f"min.log")
opt.run(fmax=0.02)


# -----------------------------
# Set initial velocities
# -----------------------------
MaxwellBoltzmannDistribution(atoms, TEMPERATURE * units.kB)
Stationary(atoms)
ZeroRotation(atoms)


# -----------------------------
# Define functions
# -----------------------------
def save_mace_md_properties(atoms=atoms):
  results = atoms.calc.results
  n_atoms = len(atoms)
   
  velocity = atoms.get_velocities() # shape: (N_atoms, 3)
  if velocity is not None:
    atoms.arrays['velocities'] = velocity 
   
  bec = results.get('bec', results.get('BEC', results.get('LES_BEC', None)))
  if bec is not None:
    if bec.ndim == 4:
      if bec.shape[1] == 2:   
        bec_summed = np.sum(bec, axis=1)
      elif bec.shape[0] == 2:  
        bec_summed = np.sum(bec, axis=0)
      else:
        bec_summed = bec
    else:
      bec_summed = bec  
       
    dP = np.einsum('nij,nj->ni', bec_summed, velocity)
    total_dP = np.sum(dP, axis=0) 
    atoms.arrays['dP'] = dP        
    atoms.info['total_dP'] = tuple(float(x) for x in total_dP) 
    atoms.arrays['LES_BEC'] = bec.reshape(n_atoms, -1)

  alphas = results.get('LES_alphas', results.get('latent_alphas', results.get('alphas', None)))
  if alphas is not None:
    alphas_reshaped = alphas.reshape(n_atoms, -1)
    if alphas_reshaped.shape[1] == 1:
      alphas_reshaped = alphas_reshaped.flatten()
       
    atoms.arrays['latent_alphas'] = alphas_reshaped
    atoms.info['total_alpha'] = np.sum(alphas)

  xyz_out_file = './md_out.xyz'
  write(xyz_out_file, atoms, format='extxyz', append=True)


# -----------------------------
# Run MD
# -----------------------------
dyn_nvt = Langevin(
    atoms,
    timestep=TIMESTEP * units.fs,
    temperature_K=TEMPERATURE,
    friction=0.02  
)
dyn_nvt.run(steps_equi)   

dyn_nve = VelocityVerlet(
    atoms,
    timestep=TIMESTEP * units.fs
)
dyn_nve.run(1000)

dyn_nve.attach(
    MDLogger(
        dyn_nve, atoms, 
        f'log_{atemp}K_mace.log', 
        header=True, stress=False, peratom=False, mode="w"
    ), 
    interval=LOG_INTERVAL
)
dyn_nve.attach(save_mace_md_properties, interval=1)
dyn_nve.run(N_STEPS)


# END

