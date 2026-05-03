import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.io import read

# =========================
# user settings
# =========================
REF_XYZ = '/global/scratch/users/dongjinkim/potential_training/CACE-LES-dipole/data-sets/h2o_bec.xyz'
TARGET_FILENAME = 'test-bec.xyz'
SAVE_DIR = 'bec_figures'

TWO_DIM = True
USE_ALPHA = True
APPLY_TRANSPOSE = False
PARITY_INIT = 1.0
EPSILON_INFTY = 1.78

# Vacuum permittivity (epsilon_0) in e^2 eV^-1 A^-1
EPSILON_0 = 5.52635e-3

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# helper functions
# =========================
def calc_metrics(ref, pred):
    errors = ref - pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    variance = np.sum((ref - np.mean(ref))**2)
    if variance == 0:
        r2 = float('nan')
    else:
        r2 = 1 - np.sum(errors**2) / variance

    return mae, rmse, r2


def load_bec_data(ref_xyz_path, pred_xyz_path, epsilon_infty, parity, two_dim, use_alpha, apply_transpose, epsilon_0):
    ref_xyz = read(ref_xyz_path, index=':')
    pred_xyz = read(pred_xyz_path, index=':')

    bec_refs = []
    bec_preds = []
    atomic_nums = []

    for xyz in ref_xyz:
        bec_ref = xyz.get_array('BEC').reshape(-1, 3, 3)
        bec_refs.append(bec_ref)
        atomic_nums.append(xyz.arrays['numbers'])

    for xyz in pred_xyz:
        les_bec = xyz.get_array('MACE_BEC')
        bec_pred = les_bec.reshape(les_bec.shape[0], -1, 9).sum(axis=1)
        
        if use_alpha:
            volume = xyz.get_volume()
            if 'MACE_latent_alphas' in xyz.arrays:
                
                alpha_arr = xyz.arrays['MACE_latent_alphas']
                if alpha_arr.shape[-1] == 9:
                    chi = np.einsum('icc->', alpha_arr.reshape(-1,3,3)) / 3 / volume / epsilon_0
                else:
                    chi =  alpha_arr.sum(axis=0) / volume / epsilon_0
                epsilon_r = epsilon_infty / (1. + chi)
            else:
                epsilon_r = epsilon_infty
        else:
            epsilon_r = epsilon_infty

        bec_pred = bec_pred * parity * (epsilon_r ** 0.5)

        bec_pred_3x3 = bec_pred.reshape(-1, 3, 3)
        
        if apply_transpose:
            bec_pred_final = np.transpose(bec_pred_3x3, (0, 2, 1))
        else:
            bec_pred_final = bec_pred_3x3
            
        bec_preds.append(bec_pred_final)

    all_refs_3x3 = np.concatenate(bec_refs, axis=0)
    all_preds_3x3 = np.concatenate(bec_preds, axis=0)
    all_an = np.concatenate([x.flatten() for x in atomic_nums])

    return all_refs_3x3, all_preds_3x3, all_an

def evaluate_bec(all_refs_3x3, all_preds_3x3):
    diag_mask = np.eye(3, dtype=bool)
    off_diag_mask = ~diag_mask

    ref_diag = all_refs_3x3[:, diag_mask].flatten()
    pred_diag = all_preds_3x3[:, diag_mask].flatten()

    ref_off_diag = all_refs_3x3[:, off_diag_mask].flatten()
    pred_off_diag = all_preds_3x3[:, off_diag_mask].flatten()

    all_ref = all_refs_3x3.flatten()
    all_pred = all_preds_3x3.flatten()

    mae_all, rmse_all, r2_all = calc_metrics(all_ref, all_pred)
    mae_diag, rmse_diag, r2_diag = calc_metrics(ref_diag, pred_diag)
    mae_off, rmse_off, r2_off = calc_metrics(ref_off_diag, pred_off_diag)

    return {
        'all_ref': all_ref,
        'all_pred': all_pred,
        'ref_diag': ref_diag,
        'pred_diag': pred_diag,
        'ref_off_diag': ref_off_diag,
        'pred_off_diag': pred_off_diag,
        'mae_all': mae_all,
        'rmse_all': rmse_all,
        'r2_all': r2_all,
        'mae_diag': mae_diag,
        'rmse_diag': rmse_diag,
        'r2_diag': r2_diag,
        'mae_off': mae_off,
        'rmse_off': rmse_off,
        'r2_off': r2_off,
    }

def plot_bec_result(folder_name, all_ref, all_pred, all_an, save_path):
    djvfont = {'fontname': 'DejaVu Sans'}
    colors   = {1:"white", 8:"red", 6:"black", 7:"blue"}
    eltrans  = {1:"H",     8:"O",   7:"N",     6:"C"}
    el_list  = [8, 7, 6, 1]

    an_rep = np.repeat(all_an, 9)

    idx = np.arange(all_ref.size) % 9
    diag_mask    = np.isin(idx, [0, 4, 8])
    offdiag_mask = ~diag_mask

    xr = np.linspace(-6, 6, 100)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)

    # main plot: diagonal
    for el in el_list:
        m = (an_rep == el) & diag_mask
        if np.sum(m) == 0:
            continue
        if el == 1:
            ax.scatter(all_ref[m], all_pred[m],
                       s=10, c=colors[el], marker='.', alpha=0.3,
                       edgecolor='black', linewidth=0.1,
                       rasterized=True, label=eltrans[el])
        else:
            ax.scatter(all_ref[m], all_pred[m],
                       s=10, c=colors[el], marker='.', alpha=0.3,
                       rasterized=True, label=eltrans[el])

    ax.plot(xr, xr, 'k--', alpha=0.2)
    ax.set_box_aspect(1)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r'DFT $Z^*_{\alpha\alpha}$ [e]', fontsize=14, **djvfont, labelpad=0)
    ax.set_ylabel(r'LES $Z^*_{\alpha\alpha}$ [e]', fontsize=14, **djvfont, labelpad=-5)
    ax.tick_params(axis='both', labelsize=14)

    # inset: off-diagonal
    ins = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
    for el in el_list:
        m = (an_rep == el) & offdiag_mask
        if np.sum(m) == 0:
            continue
        if el == 1:
            ins.scatter(all_ref[m], all_pred[m],
                        s=3, c=colors[el], marker='.', alpha=0.1,
                        edgecolor='black', linewidth=0.05,
                        rasterized=True)
        else:
            ins.scatter(all_ref[m], all_pred[m],
                        s=3, c=colors[el], marker='.', alpha=0.1,
                        rasterized=True)

    ins.plot(xr, xr, 'k--', alpha=0.2)
    ins.set_box_aspect(1)
    ins.set_xlim(-1, 1)
    ins.set_ylim(-1, 1)
    ins.set_xticks([-1, 0, 1])
    ins.set_yticks([-1, 0, 1])
    ins.set_ylabel(r'$Z^*_{\alpha\beta}$ [e]', fontsize=12, **djvfont, labelpad=-10)
    ins.tick_params(axis='both', labelsize=12, length=1.5, pad=1)

    proxy_handles = []
    proxy_labels  = []
    for el in [1, 8]:
        if el == 1:
            proxy_handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       color=colors[el], markersize=2,
                       alpha=1.0, markeredgecolor='black',
                       markeredgewidth=0.5)
            )
        else:
            proxy_handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       color=colors[el], markersize=2,
                       alpha=1.0)
            )
        proxy_labels.append(eltrans[el])

    ax.legend(proxy_handles, proxy_labels,
              loc='upper left',
              frameon=False,
              bbox_to_anchor=(-0.05, 1.02),
              fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()

# =========================
# main loop
# =========================
subdirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and 'mp0' in d])

for folder in subdirs:
    pred_xyz_path = os.path.join(folder, TARGET_FILENAME)

    if not os.path.exists(pred_xyz_path):
        continue

    print(f'\n{"="*60}')
    print(f'{folder}')
    print(f'{"="*60}')

    current_parity = PARITY_INIT

    try:
        all_refs_3x3, all_preds_3x3, all_an = load_bec_data(
            REF_XYZ, pred_xyz_path, EPSILON_INFTY, current_parity, TWO_DIM, USE_ALPHA, APPLY_TRANSPOSE, EPSILON_0
        )
        result = evaluate_bec(all_refs_3x3, all_preds_3x3)

        # if r2_all < 0, flip parity sign and recalc
        if result['r2_all'] < 0:
            print("Negative R2 detected. Flipping parity sign...")
            current_parity *= -1.0
            all_refs_3x3, all_preds_3x3, all_an = load_bec_data(
                REF_XYZ, pred_xyz_path, EPSILON_INFTY, current_parity, TWO_DIM, USE_ALPHA, APPLY_TRANSPOSE, EPSILON_0
            )
            result = evaluate_bec(all_refs_3x3, all_preds_3x3)

        print(f'Parity used: {current_parity}')

        print('--- (All Terms) ---')
        print(f"MAE:  {result['mae_all']:.3f}")
        print(f"RMSE: {result['rmse_all']:.3f}")
        print(f"R2:   {result['r2_all']:.3f}\n")

        print('--- (Diagonal Terms) ---')
        print(f"MAE:  {result['mae_diag']:.3f}")
        print(f"RMSE: {result['rmse_diag']:.3f}")
        print(f"R2:   {result['r2_diag']:.3f}\n")

        print('--- (Off-Diagonal Terms) ---')
        print(f"MAE:  {result['mae_off']:.3f}")
        print(f"RMSE: {result['rmse_off']:.3f}")
        print(f"R2:   {result['r2_off']:.3f}")

        save_path = os.path.join(SAVE_DIR, f'{folder}_bec_result.pdf')
        plot_bec_result(
            folder_name=folder,
            all_ref=result['all_ref'],
            all_pred=result['all_pred'],
            all_an=all_an,
            save_path=save_path
        )

        print(f'\nSaved plot: {save_path}')

    except Exception as e:
        print(f'Error in {folder}: {e}')
