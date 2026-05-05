import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


temps = [100, 120, 150, 160, 175, 200, 228, 250, 280, 300, 320, 350]

# phase by temperature
def get_phase(T):
    if T <= 150:
        return "ortho"
    elif T < 320:
        return "tetra"
    else:
        return "cubic"

# per-phase analysis settings
phase_settings = {
    "ortho": {
        "repx": 3,
        "repy": 2,
        "repz": 3,
        "facx": np.sqrt(2.0),
        "facy": 2.0,
        "facz": np.sqrt(2.0),
    },
    "tetra": {
        "repx": 3,
        "repy": 3,
        "repz": 2,
        "facx": np.sqrt(2.0),
        "facy": np.sqrt(2.0),
        "facz": 2.0,
    },
    "cubic": {
        "repx": 4,
        "repy": 4,
        "repz": 4,
        "facx": 1.0,
        "facy": 1.0,
        "facz": 1.0,
    },
}


# collect data from NPT MD output
T_list = []

a_mean_list, b_mean_list, c_mean_list = [], [], []
a_sem_list,  b_sem_list,  c_sem_list  = [], [], []

discard_lines = 500

for T in temps:

    phase = get_phase(T)
    s = phase_settings[phase]

    repx = s["repx"]
    repy = s["repy"]
    repz = s["repz"]
    facx = s["facx"]
    facy = s["facy"]
    facz = s["facz"]

    f = Path(f"I-{T}K") / f"lattice_1bar.log"

    data = np.loadtxt(f, comments="#")
    i0 = min(discard_lines, len(data))
    data_eq = data[i0:]
    if len(data_eq) == 0:
        continue

    a = data_eq[:, 1] / repx / facx
    b = data_eq[:, 2] / repy / facy
    c = data_eq[:, 3] / repz / facz
    alpha = data_eq[:, 4]
    beta  = data_eq[:, 5]
    gamma = data_eq[:, 6]

    N = len(a)

    a_mean = np.mean(a)
    b_mean = np.mean(b)
    c_mean = np.mean(c)

    a_sem = np.std(a, ddof=1) / np.sqrt(N) if N > 1 else 0.0
    b_sem = np.std(b, ddof=1) / np.sqrt(N) if N > 1 else 0.0
    c_sem = np.std(c, ddof=1) / np.sqrt(N) if N > 1 else 0.0

    T_list.append(T)

    a_mean_list.append(a_mean)
    b_mean_list.append(b_mean)
    c_mean_list.append(c_mean)

    a_sem_list.append(a_sem)
    b_sem_list.append(b_sem)
    c_sem_list.append(c_sem)



# plotting
T_arr = np.array(T_list)

a_mean_arr = np.array(a_mean_list)
b_mean_arr = np.array(b_mean_list)
c_mean_arr = np.array(c_mean_list)

a_sem_arr = np.array(a_sem_list)
b_sem_arr = np.array(b_sem_list)
c_sem_arr = np.array(c_sem_list)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.5))

ax1.errorbar(
    T_arr, a_mean_arr, yerr=a_sem_arr,
    fmt='s-', color='blue', ecolor=(0, 0, 1, 0.35),
    capsize=4, label='a'
)
ax1.errorbar(
    T_arr, b_mean_arr, yerr=b_sem_arr,
    fmt='o-', color='red', ecolor=(1, 0, 0, 0.35),
    capsize=4, label='b'
)
ax1.errorbar(
    T_arr, c_mean_arr, yerr=c_sem_arr,
    fmt='^-', color='limegreen', ecolor=(0.2, 0.8, 0.2, 0.35),
    capsize=4, label='c'
)

ax1.set_xlabel("T (K)")
ax1.set_ylabel("Lattice parameter (Å)")
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(90, 360)

plt.tight_layout()
plt.show()


 

