import os
import numpy as np
import sys
import les
import mace
from mace.calculators import MACECalculator
from ase import units
from ase.md.npt import NPT
from ase.md.melchionna import MelchionnaNPT
from ase.md import MDLogger
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.units import mol
from ase.optimize import FIRE


# Simulation parameters
TEMPERATURE = 300.  # K
PRESSURE_BAR = 1.01325  # bar
TIMESTEP = 1.0 
steps_equi = int(1000//TIMESTEP * 20)
steps_sample = int(1000//TIMESTEP * 300)
N_STEPS = steps_sample
LOG_INTERVAL = 100
TRAJ_INTERVAL = 200


# Thermostat / barostat parameters
TTIME_FS = 50.0              
PTIME_FS = 200.0             
BULK_MODULUS_GPA = 15.0
BULK_MODULUS = BULK_MODULUS_GPA * 0.006241509  
compressibility_au = 1.0 / BULK_MODULUS
PRESSURE_ASE = PRESSURE_BAR * 1.0e5 / 1.602176634e11  
ptime = PTIME_FS * units.fs
pfactor = ptime**2 * BULK_MODULUS


# Load initial structure
init_file = "init_phase.xyz"
atoms = read(init_file, index=0)


# Setup MACE calculator
MODEL = "mapi_stagetwo.model"
device = 'cuda'
calculator = MACECalculator(model_path=MODEL, device=device)
atoms.calc = calculator


# Minimization
opt = FIRE(atoms)
opt.run(fmax=0.03)


# Set initial velocities
MaxwellBoltzmannDistribution(atoms, TEMPERATURE * units.kB)
Stationary(atoms)
ZeroRotation(atoms)


# Initial NPT equilibration
dyn_npt1 = NPT(
    atoms,
    timestep=TIMESTEP * units.fs,
    temperature_K=TEMPERATURE,
    ttime=TTIME_FS * units.fs,
    pfactor=None,
    externalstress=0.0,
    logfile= f'md_npt1.log',
    loginterval= LOG_INTERVAL
)
dyn_npt1.run(steps_equi)


# Initial NPT equilibration
dyn_npt2 = IsotropicMTKNPT(
    atoms,
    timestep=TIMESTEP * units.fs,
    temperature_K=TEMPERATURE,
    pressure_au=PRESSURE_BAR * units.bar,
    logfile= f'md_npt2.log',
    loginterval= LOG_INTERVAL,
    tdamp=TTIME_FS * units.fs,
    pdamp=PTIME_FS * units.fs,
    tchain=5,
    pchain=5,
    tloop=2,
    ploop=2
)
dyn_npt2.run(steps_equi)


# Main NPT simulation
dyn = MelchionnaNPT(
    atoms,
    timestep=TIMESTEP * units.fs,
    temperature_K=TEMPERATURE,
    externalstress=PRESSURE_ASE,
    ttime=TTIME_FS * units.fs,
    pfactor=pfactor,
    trajectory=None,      
    logfile=f'md.log',
    loginterval= LOG_INTERVAL
)


# Attach lattice parameter function
with open(f'lattice_1bar.log', 'w') as f:
    f.write('#  step  a  b  c  alpha  beta  gamma  volume \n')

def log_lattice():
    step = dyn.nsteps
    a, b, c, alpha, beta, gamma = dyn.atoms.cell.cellpar()
    vol = dyn.atoms.get_volume()

    with open(f'lattice_1bar.log', 'a') as f:
        f.write(f"{step}   {a:.6f}   {b:.6f}   {c:.6f}   "
                f"{alpha:.4f}   {beta:.4f}   {gamma:.4f}   "
                f"{vol:.6f}\n")
dyn.attach(log_lattice, interval=LOG_INTERVAL)


# Attach observers
dyn.attach(
    MDLogger(
        dyn, atoms, 
        f'log_1bar.log', 
        header=True, stress=False, peratom=False, mode="w"
    ), 
    interval=LOG_INTERVAL
)
dyn.attach(lambda: dyn.atoms.write('npt-mace.xyz', append=True), interval=TRAJ_INTERVAL)


# Run simulation
dyn.run(N_STEPS)


