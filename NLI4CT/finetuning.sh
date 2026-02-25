#!/bin/bash
#SBATCH --job-name=ft_NLI4PR
#SBATCH --output=logs_finetuning/ft_%j.out
#SBATCH --error=logs_finetuning/ft_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        
#SBATCH --mem=64G 
#SBATCH --nodes=1

# --- 1. Préparation ---
cd /mnt/beegfs/home/kebdi/ftctinfer/NLI4CT
mkdir -p logs_finetuning

# --- 2. Environnement & Correction LibC++ ---
PYTHON_EXE="/mnt/beegfs/projects/ftctinfer/env_stagiaires/bin/python"
CONDA_PREFIX="/mnt/beegfs/projects/ftctinfer/env_stagiaires"

LIB_STD=$(find $CONDA_PREFIX -name libstdc++.so.6 | head -n 1)
export LD_PRELOAD=$LIB_STD

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$($PYTHON_EXE -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Optimisation PyTorch : limite la fragmentation de la mémoire GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 3. Configuration des arguments ---
# FINETUNING_MODE: normal = sans Optuna (anciens hyperparamètres) ; job1 = Optuna search ; job2 = train avec HP chargés
FINETUNING_MODE="${FINETUNING_MODE:-normal}"
BEST_HP_FILE="${BEST_HP_FILE:-outputs/best_optuna_hp.json}"

MODEL_PATH="/mnt/beegfs/home/kebdi/ftctinfer/model/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_FILE="train_formatted.jsonl"
EVAL_FILE="dev_formatted.jsonl"
OUTPUT_NAME="outputs/qwen2_5_7b_finetuned_optuna"

# --- 4. Lancement ---
if [ "$FINETUNING_MODE" = "normal" ]; then
  # Sans Optuna : 5 epochs, lr 2e-4, batch 2, grad_accum 4 (anciens hyperparamètres)
  echo "Finetuning sans Optuna (anciens hyperparamètres)"
  $PYTHON_EXE -u finetuning.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_NAME"
elif [ "$FINETUNING_MODE" = "job1" ]; then
  # Job 1 (48h) : Optuna uniquement, ~11 essais possibles, sauvegarde des meilleurs HP
  echo "Job 1: Optuna search only (best HP will be saved to $BEST_HP_FILE)"
  $PYTHON_EXE -u finetuning.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_NAME" \
    --use_optuna \
    --n_trials 11 \
    --optuna_epochs_max 4 \
    --final_epochs 5 \
    --optuna_save_best_hp "$BEST_HP_FILE"
elif [ "$FINETUNING_MODE" = "job2" ]; then
  # Job 2 (48h) : charge les HP du Job 1, un seul entraînement final (5 epochs)
  echo "Job 2: Loading HP from $BEST_HP_FILE, running final training only"
  $PYTHON_EXE -u finetuning.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_NAME" \
    --load_optuna_hp "$BEST_HP_FILE" \
    --final_epochs 5
else
  echo "FINETUNING_MODE must be 'normal', 'job1' or 'job2'. Current: $FINETUNING_MODE"
  exit 1
fi

echo "Fin du job."
