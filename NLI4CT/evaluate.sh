#!/bin/bash
#SBATCH --job-name=eval_NLI4CT
#SBATCH --output=logs_evaluate/eval_%j.out
#SBATCH --error=logs_evaluate/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 1. Navigation
cd "/mnt/beegfs/home/kebdi/ftctinfer/NLI4CT"
mkdir -p logs_evaluate

# --- CONFIGURATION (À MODIFIER SELON BESOIN) ---

# A. Le modèle de base (Toujours le Qwen original)
BASE_MODEL="/mnt/beegfs/home/kebdi/ftctinfer/model/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# B. Le modèle à évaluer (Soit le même que Base, soit ton dossier outputs)
# --> Pour évaluer le FINETUNING, décommente cette ligne :
#MODEL_TO_EVAL="/mnt/beegfs/home/kebdi/ftctinfer/NLI4CT/outputs/qwen2_5_7b_nli4ct"

# --> Pour évaluer la BASELINE, tu décommenterais plutôt celle-ci :
MODEL_TO_EVAL="$BASE_MODEL"

# -----------------------------------------------

TEST_FILE="Gold_test_formatted.jsonl"
PYTHON_EXE="/mnt/beegfs/projects/ftctinfer/env_stagiaires/bin/python"

echo "Démarrage de l'évaluation..."
echo "Base: $BASE_MODEL"
echo "Eval: $MODEL_TO_EVAL"
echo "Job ID: $SLURM_JOB_ID"

$PYTHON_EXE -u evaluate.py \
    --base_model_path "$BASE_MODEL" \
    --model_path "$MODEL_TO_EVAL" \
    --test_file "$TEST_FILE"

echo "Fin du job."
