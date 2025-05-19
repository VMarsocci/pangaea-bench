#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --partition=gpu.4h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=4:00:00
#SBATCH --job-name=agbd_test
#SBATCH --mem-per-cpu=4096

# module load stack/2024-06
# module load python/3.11.6
source ~/pangaea-bench-venv/bin/activate

# Debugging: print python and hydra info
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "hydra-core version: $(pip show hydra-core | grep Version)"
echo "PYTHONPATH: $PYTHONPATH"

cd /cluster/home/anogueira/pangaea-bench/

export HYDRA_FULL_ERROR=1

# Print GPU info for debugging
echo "GPU info:" && nvidia-smi

# --- FULL RUN (commented out for now) ---
# /cluster/home/anogueira/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
#   dataset=agbd \
#   encoder=prithvi \
#   decoder=reg_upernet \
#   preprocessing=agbd_reg_resize \
#   criterion=mse \
#   task=regression \
#   task.trainer.n_epochs=2 \
#   task.trainer.eval_interval=999 \
#   use_wandb=True \
#   task.trainer.use_wandb=True \
#   test_batch_size=32 \
#   test_num_workers=8

# --- FAST SUBSET RUN (should finish in <4h) ---
/cluster/home/anogueira/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
  dataset=agbd \
  encoder=prithvi \
  decoder=reg_upernet \
  preprocessing=agbd_reg_resize \
  criterion=mse \
  task=regression \
  task.trainer.n_epochs=2 \
  task.trainer.eval_interval=999 \
  use_wandb=True \
  task.trainer.use_wandb=True \
  test_batch_size=32 \
  test_num_workers=8 \
  limited_label_train=0.0001 \
  limited_label_val=0.0001 \
  limited_label_strategy=random