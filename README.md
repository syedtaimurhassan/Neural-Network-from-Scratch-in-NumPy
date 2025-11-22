# Neural Network from Scratch in NumPy

This project trains a simple feed-forward network on Fashion-MNIST or CIFAR-10 using only NumPy.

## Setup
1. Create and activate venv:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install packages:
   - `pip install -r requirements.txt`

## Download data
Run the helper to get datasets into `data/`:
- `python scripts/download_datasets.py --dataset fashion`
- `python scripts/download_datasets.py --dataset cifar10`
- Or both: `python scripts/download_datasets.py --dataset all`

## Train
- XOR sanity: `python train.py --dataset xor --epochs 20 --hidden-layers 8 --activation tanh`
- Fashion-MNIST: `python train.py --dataset fashion --epochs 5 --batch-size 128 --hidden-layers 256,128 --activation relu --init he`
- CIFAR-10: `python train.py --dataset cifar10 --epochs 5 --batch-size 256 --hidden-layers 512,256 --activation relu --init he`

Adjust flags for learning rate (`--lr`), optimizer (`--optimizer sgd|momentum|adam|nadam`), L2 (`--l2`), and validation split (`--val-ratio`).

## Optional features
- Dropout / BatchNorm: `--dropout 0.1 --batch-norm`
- Grad clip / scheduler / early stop: `--clip-grad-norm 1.0 --scheduler step --step-size 3 --gamma 0.5 --patience 5`
- Logging/checkpoints: `--log-dir logs --checkpoint-dir checkpoints`
- W&B: `--wandb-project your_project --wandb-mode online` (default is disabled)

## Analysis helpers
- Misclassified examples: `MPLBACKEND=Agg python scripts/misclassified_examples.py --dataset fashion --checkpoint checkpoints/fashion_best.npz --hidden-layers 256,128 --batch-norm --out outputs/plots/misclassified.png` (match arch to checkpoint; set backend if headless)

## W&B sweeps (Bayesian template)
- Use `sweeps/wandb_sweep.yaml` as a starter. Launch with W&B CLI (requires network and W&B auth):  
  `wandb sweep sweeps/wandb_sweep.yaml` then `wandb agent <sweep_id>`
