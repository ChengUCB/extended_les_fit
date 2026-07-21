import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # set BEFORE torch/CUDA init

import sys
from pathlib import Path

import torch

MODEL = "uiu-aniso"

PROJECT_ROOT = Path("/global/scratch/users/namdao2404/surface-spectroscopy").resolve()
REPO_ROOT = PROJECT_ROOT 

RUN_DIR = PROJECT_ROOT / MODEL
RUN_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_XYZ = PROJECT_ROOT / "data" / "train-water_slab_REVPBE-D3.xyz"
TEST_XYZ = PROJECT_ROOT / "data" / "test-water_slab_REVPBE-D3.xyz"

# yaml file contains les settings for a maceles-uiu
LES_YAML = RUN_DIR / "les.yaml"

for p, label in [(TRAIN_XYZ, "train"), (TEST_XYZ, "test"), (LES_YAML, "les.yaml")]:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {label}: {p}")

os.chdir(PROJECT_ROOT)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mace import tools
from mace.cli.run_train import run

device = "cuda" if torch.cuda.is_available() else "cpu"

argv = [
    "--name=H20_slab",
    f"--work_dir={RUN_DIR}",
    f"--train_file={TRAIN_XYZ}",
    "--valid_fraction=0.05",
    f"--test_file={TEST_XYZ}",
    "--energy_key=energy",
    "--forces_key=forces",
    "--E0s=average",
    "--model=MACELES",
    f"--les_arguments={LES_YAML}",
    "--hidden_irreps=128x0e + 128x1o + 128x2e",
    "--r_max=5.5",
    "--num_interactions=1",
    "--batch_size=2",
    "--max_num_epochs=1000",
    "--ema",
    "--ema_decay=0.99",
    "--amsgrad",
    f"--device={device}",
    "--default_dtype=float32",
    "--save_cpu",
    "--num_workers=0",
    "--restart_latest"
]

args = tools.build_default_arg_parser().parse_args(argv)
run(args)