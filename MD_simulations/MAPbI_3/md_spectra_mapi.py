import os
import numpy as np
import sys
import les
import mace
from mace.calculators import MACECalculator
from ase import units
from ase.md import MDLogger
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import mol
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet


# Simulation parameters
TEMPERATURE = 300.0  # K
PRESSURE_BAR = 1.01325  # bar
TIMESTEP = 0.5
steps_equi = int(1000//TIMESTEP * 20)
steps_sample = int(1000//TIMESTEP * 100)
N_STEPS = steps_sample
LOG_INTERVAL = 10
TRAJ_INTERVAL = 1


# Thermostat / barostat parameters
TTIME_FS = 50.0              
PTIME_FS = 200.0             
BULK_MODULUS_GPA = 15.0
BULK_MODULUS = BULK_MODULUS_GPA * 0.006241509  
PRESSURE_ASE = PRESSURE_BAR * 1.0e5 / 1.602176634e11  
ptime = PTIME_FS * units.fs
pfactor = ptime**2 * BULK_MODULUS


# Load initial structure
atoms = read('init.xyz', index=0)


# Setup MACE calculator
MODEL = "mapi_stagetwo.model"
device = 'cuda'
calculator = MACECalculator(model_path=MODEL, device=device, compute_bec=True)
atoms.calc = calculator


# Set initial velocities
MaxwellBoltzmannDistribution(atoms, TEMPERATURE * units.kB)
Stationary(atoms)
ZeroRotation(atoms)


# Initial NVT equilibration
dyn_nvt = Langevin(
    atoms,
    timestep=TIMESTEP * units.fs,
    temperature_K=TEMPERATURE,
    friction=0.02   # small friction
)
dyn_nvt.run(steps_equi)


# Main NVE simulation
dyn_nve = VelocityVerlet(
    atoms,
    timestep=TIMESTEP * units.fs
)
dyn_nve.run(1000)


# Define functions for saving properties
def save_mace_md_properties(atoms=atoms):
  results = atoms.calc.results
  n_atoms = len(atoms)
   
  velocity = atoms.get_velocities() # shape: (N_atoms, 3)
  if velocity is not None:
    atoms.arrays['velocities'] = velocity 
   
  bec = results.get('bec', results.get('BEC', results.get('LES_BEC', None)))
   
  if bec is not None:
    if bec.ndim == 4:
      if bec.shape[1] == 2:   # (N_atoms, 2, 3, 3)
        bec_summed = np.sum(bec, axis=1)
      elif bec.shape[0] == 2:  # (2, N_atoms, 3, 3)
        bec_summed = np.sum(bec, axis=0)
      else:
        bec_summed = bec
    else:
      bec_summed = bec #(N_atoms, 3, 3) 
       
    dP = np.einsum('nij,nj->ni', bec_summed, velocity)
    total_dP = np.sum(dP, axis=0) # shape: (3,)
     
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


# Attach observers
dyn_nve.attach(
    MDLogger(
        dyn_nve, atoms, 
        f'log_1bar.log', 
        header=True, stress=False, peratom=False, mode="w"
    ), 
    interval=LOG_INTERVAL
)

dyn_nve.attach(save_mace_md_properties, interval=1)


# Run simulation
dyn_nve.run(N_STEPS)


