import numpy as np
import pickle
from ase.io import iread

# --- Configuration ---
xyz_file = './md_out.xyz'
dp_pkl_file = 'total_dp.pkl'
alpha_pkl_file = 'total_alpha_epsilon.pkl'
dp_txt_file = 'total_dp.txt'
alpha_txt_file = 'total_alpha_epsilon.txt'

# --- Constants for Alpha Correction ---
inv_4pie0 = 90.4756 / (2 * np.pi)
epsilon_infty = 1.78 
epsilon_0 = 0.00552635   # e^2 eV^{-1} A^{-1}
epsilon_r = None 

# Lists to store extracted data
total_dP_list = []
total_alpha_list = []

print(f"Reading {xyz_file}...")

for i, atoms in enumerate(iread(xyz_file)):
    

        
    # 2. Extract and process latent_alphas (from atoms.arrays)
    if 'latent_alphas' in atoms.arrays:
        try:
            volume = atoms.get_volume()
            alpha = atoms.arrays['latent_alphas'] # shape: (N_atoms, 9) or (N_atoms,)

            # Calculate susceptibility (chi)
            if alpha.shape[-1] == 9:
                # Sum of traces of all 3x3 matrices: np.einsum('icc->')
                chi = np.einsum('icc->', alpha.reshape(-1, 3, 3)) / 3.0 / volume / epsilon_0
            else:
                # Isotropic scalar case
                chi = np.sum(alpha) / volume / epsilon_0

            # Calculate dielectric constant
            epsilon_r = epsilon_infty / (1.0 + chi)

        except Exception:
            # Fallback if volume calculation fails
            epsilon_r = epsilon_infty
        
        # Calculate corrected frame total alpha
        frame_total_alpha = np.sum(alpha, axis=0) * inv_4pie0 * epsilon_r
        total_alpha_list.append(frame_total_alpha)
        
    # 1. Extract total_dP (from atoms.info)
    if epsilon_r is not None:
        epsilon = epsilon_r
    else:
        epsilon = epsilon_infty
    if 'total_dP' in atoms.info:
        total_dP_list.append(atoms.info['total_dP'] * (epsilon ** 0.5))

    # Print progress
    if (i + 1) % 10000 == 0:
        print(f"  ... {i + 1} frames processed.")

# Convert lists to numpy arrays
total_dP_stack = np.array(total_dP_list)       # Expected shape: (N_frames, 3)
total_alpha_stack = np.array(total_alpha_list) # Expected shape: (N_frames, 9) or (N_frames,)

print(f"\nExtraction complete! Total frames: {len(total_dP_stack)}")

# ==========================================
# Save to Pickle (.pkl) files
# ==========================================
print("Saving to Pickle files...")
if len(total_dP_stack) > 0:
    with open(dp_pkl_file, 'wb') as f:
        pickle.dump({'total_dp': total_dP_stack}, f)

if len(total_alpha_stack) > 0:
    with open(alpha_pkl_file, 'wb') as f:
        pickle.dump({'total_alpha': total_alpha_stack}, f)

# ==========================================
# Save to Text (.txt) files
# ==========================================
print("Saving to TXT files...")

# Save dP
if len(total_dP_stack) > 0:
    np.savetxt(dp_txt_file, total_dP_stack, fmt='%.8e', header='total_dP_x total_dP_y total_dP_z')

# Save Alpha
if len(total_alpha_stack) > 0:
    # Adjust header based on tensor shape (3x3=9 or scalar=1)
    if total_alpha_stack.ndim == 2 and total_alpha_stack.shape[1] == 9:
        alpha_header = 'alpha_xx alpha_xy alpha_xz alpha_yx alpha_yy alpha_yz alpha_zx alpha_zy alpha_zz'
    else:
        alpha_header = 'alpha_isotropic'
        
    np.savetxt(alpha_txt_file, total_alpha_stack, fmt='%.8e', header=alpha_header)

print("All done! Successfully saved data.")

