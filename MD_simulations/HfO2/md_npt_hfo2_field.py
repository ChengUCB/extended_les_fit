import os
import numpy as np
import sys
import les
import mace
from mace.calculators import MACECalculator
from ase import units
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nose_hoover_chain import IsotropicMTKNPT, MTKNPT
from ase.units import mol
from ase.filters import StrainFilter, UnitCellFilter
from ase.optimize import FIRE


if len(sys.argv) != 2:
    print("Usage: python npt-ase.py [cpu|cuda]")
    sys.exit(1)

device = sys.argv[1]
atemp = 300
y_field = 0.03

print(les.__file__)
print(mace.__file__)

 
# -----------------------------
# Simulation parameters
# -----------------------------
TEMPERATURE = float(atemp)  # K
y_field = float(y_field)
PRESSURE_BAR = 1.01325  # bar
TIMESTEP = 1.0 
steps_equi = int(1000//TIMESTEP * 20)
steps_sample = int(1000//TIMESTEP * 20)
N_STEPS = steps_sample
LOG_INTERVAL = 100
TRAJ_INTERVAL = 200
TTIME_FS = 50.0              
PTIME_FS = 200.0             


# -----------------------------
# Load initial structure
# -----------------------------
init_file = "./init_hfo2.xyz"
atoms = read(init_file, index=0)
print("\nOriginal cell:", flush=True)
print("Cell:", atoms.cell, flush=True)
print("PBC:", atoms.pbc, flush=True)


# -----------------------------
# Setup MACE calculator
# -----------------------------
e_field = [0.0, y_field, 0.0]
MODEL = "/global/scratch/users/yoonjaepark/MLIP/fit-hfo2-selected/t-mace-uiu-isoa-mp1/Hf02_stagetwo.model"

calculator = MACECalculator(
    model_paths=MODEL, 
    device=device,
    compute_bec=True,
    eps_infty=5.365,
    external_field=e_field,
    default_dtype="float32"
)
atoms.calc = calculator


# -----------------------------
# Minimization
# -----------------------------
ucf = UnitCellFilter(atoms, hydrostatic_strain=True, scalar_pressure=0.0)
opt = FIRE(ucf, logfile=f"min.log")
opt.run(fmax=0.03)
write("minimized.extxyz", atoms, format="extxyz") 


# -----------------------------
# Set initial velocities
# -----------------------------
MaxwellBoltzmannDistribution(atoms, TEMPERATURE * units.kB)
Stationary(atoms)
ZeroRotation(atoms)


# -----------------------------
# Run MD
# -----------------------------
dyn = MTKNPT(
             atoms,
             timestep=TIMESTEP * units.fs,
             temperature_K=TEMPERATURE,
             pressure_au=PRESSURE_BAR * units.bar,
             tdamp=TTIME_FS * units.fs,
             pdamp=PTIME_FS * units.fs,
             tchain=3,
             pchain=3,
             tloop=1,
             ploop=1,
             trajectory=None,
             logfile=f"md.log",
             loginterval=LOG_INTERVAL,
)
dyn.run(steps_equi)


# -----------------------------
# Define functions
# -----------------------------  
def save_mace_md_properties(atoms=atoms):
  results = atoms.calc.results
  n_atoms = len(atoms)
   
  velocity = atoms.get_velocities() 
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

  xyz_out_file = './npt-mace-field.xyz'
  write(xyz_out_file, atoms, format='extxyz', append=True)


dyn.attach(
    MDLogger(
        dyn, atoms, 
        f'log_{atemp}K_1bar_mace.log', 
        header=True, stress=False, peratom=False, mode="w"
    ), 
    interval=LOG_INTERVAL
)

dyn.attach(save_mace_md_properties, interval=TRAJ_INTERVAL)
dyn.run(N_STEPS)


# END

