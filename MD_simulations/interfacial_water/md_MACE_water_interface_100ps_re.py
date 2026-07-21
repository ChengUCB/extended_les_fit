import sys
import numpy as np
import torch
from ase.io import read

import os
from mace.calculators import MACECalculator

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

###### variables ######
model_path = 'H20_slab.model'
DEVICE = 'cuda'
temperature = 300 
timestep = 0.25 # to ensure to capture the fast O-H vibrations
nsteps = 400000 
trajectory_file = f'md_h2o.traj'
logfile = f'md_h2o.log'
xyz_out_file = './md_out.xyz'

###### load model ######
#z_field = float(sys.argv[1])
#e_field = [0.0, 0.0, z_field]

#density_table = {
#    280: 0.9999,
#    300: 0.9965,
#    320: 0.9894,
#    340: 0.9796,
#    360: 0.9674,
#}

def get_ase_density(atoms):
    """g/cm^3 """
    mass = atoms.get_masses().sum()
    vol = atoms.get_volume()
    return (mass / vol) * 1.660539

def get_scale_factor(T, current_rho):
    T_int = int(T)
    if T_int not in density_table:
        raise ValueError(f"No density data for T={T_int} in density_table.")
    
    rho_target = density_table[T_int]
    scale = (current_rho / rho_target) ** (1/3)
    return scale, rho_target


calculator = MACECalculator(model_paths=model_path, device=DEVICE, 
                            compute_bec=True,
#                            eps_infty=1.78,
#                           external_field=e_field,
                            )

def save_mace_md_properties(atoms):
    results = atoms.calc.results
    n_atoms = len(atoms)
    
    velocity = atoms.get_velocities() # shape: (N_atoms, 3)
    if velocity is not None:
        atoms.arrays['velocities'] = velocity 
    
    bec = results.get('bec', results.get('BEC', results.get('LES_BEC', None)))
    
    if bec is not None:
        if bec.ndim == 4:
            if bec.shape[1] == 2:     # (N_atoms, 2, 3, 3)
                bec_summed = np.sum(bec, axis=1)
            elif bec.shape[0] == 2:   # (2, N_atoms, 3, 3)
                bec_summed = np.sum(bec, axis=0)
            else:
                bec_summed = bec
        else:
            bec_summed = bec #(N_atoms, 3, 3) 
            
        # dP  bec_summed (N, 3, 3) * velocity (N, 3) -> dP (N, 3)
        dP = np.einsum('nij,nj->ni', bec_summed, velocity)
        total_dP = np.sum(dP, axis=0) # shape: (3,)
        
        atoms.arrays['dP'] = dP              
        atoms.info['total_dP'] = total_dP     
        
        atoms.arrays['LES_BEC'] = bec.reshape(n_atoms, -1)

    alphas = results.get('LES_alphas', results.get('latent_alphas', results.get('alphas', None)))
    if alphas is not None:
        alphas_reshaped = alphas.reshape(n_atoms, -1)
        if alphas_reshaped.shape[1] == 1:
            alphas_reshaped = alphas_reshaped.flatten()
            
        atoms.arrays['latent_alphas'] = alphas_reshaped
        atoms.info['total_alpha'] = np.sum(alphas)

    write(xyz_out_file, atoms, format='extxyz', append=True)

###### load init_config ######
###### load init_config / restart ######
if os.path.exists(xyz_out_file):
    print("[RESTART] Found md_out.xyz → restarting from last frame")
    atoms = read(xyz_out_file, index=-1)
    v = atoms.get_velocities()
    print("[DEBUG] velocity shape:", None if v is None else v.shape)
    #done_steps = len(read(xyz_out_file, index=":"))
    with open(xyz_out_file) as f:
        done_steps = sum(1 for line in f if "Lattice" in line)
    print(f"[RESTART] Completed steps: {done_steps}")

else:
    print("[INIT] Fresh start")
    if os.path.exists("optimized.xyz"):
        print("[INIT] Using existing optimized.xyz")
        atoms = read("optimized.xyz")
    else:
        init_config = read('thin-slab.xyz', index=0)
        atoms = init_config.copy()
#        initial_rho = get_ase_density(atoms)
#        print(f"\n[DENSITY] Initial Density: {initial_rho:.5f} g/cm^3")
#        scale, target_rho = get_scale_factor(temperature, initial_rho)
#        print(f"[DENSITY] Target Density for {temperature}K: {target_rho:.5f} g/cm^3")
#        atoms.set_cell(atoms.get_cell() * scale, scale_atoms=True)
        atoms.calc = calculator
        from ase.optimize import BFGS
        optimizer = BFGS(atoms)
        optimizer.run(fmax=0.05)
        write('optimized.xyz', atoms)
    done_steps = 0

#final_rho = get_ase_density(atoms)
#print(f"[DENSITY] Final Adjusted Density: {final_rho:.5f} g/cm^3\n")

atoms.calc = calculator

###### set NVT velocity ######
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.logger import MDLogger
from ase import Atoms, units
from ase.io.trajectory import Trajectory

if done_steps == 0:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

if done_steps == 0:
    print("Starting equilibration...")
    equil_dyn = NPT(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        ttime=100 * units.fs,
        pfactor=None,
        externalstress=0.0
    )
    logger = MDLogger(equil_dyn, atoms, logfile='equil.log',
                         header=not os.path.exists('equil.log'),
                         stress=False, mode='a')
    equil_dyn.attach(logger, interval=100)
    equil_dyn.run(80000)

remaining_steps = nsteps - done_steps
print(f"[MD] Remaining steps: {remaining_steps}")

if remaining_steps <= 0:
    print("[MD] Already finished. Skipping MD.")
else:
    dyn = NPT(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        ttime=100 * units.fs,
        pfactor=None,
        externalstress=0.0
    )

    md_logger = MDLogger(dyn, atoms, logfile=logfile,
                         header=not os.path.exists(logfile),
                         stress=False, mode='a')
    dyn.attach(md_logger, interval=100)
    dyn.attach(save_mace_md_properties, interval=1, atoms=atoms)

    print("Starting production MD...")
    dyn.run(remaining_steps)
    print("complete.")
