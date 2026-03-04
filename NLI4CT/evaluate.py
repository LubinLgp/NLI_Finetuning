import json
import argparse
import csv
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

def load_jsonl(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    return samples

def predict(model, tokenizer, messages, max_new_tokens=50, max_context_tokens=8192):
    """Prédit le label et retourne (label_extrait, texte_brut_généré).
    max_context_tokens: limite du prompt (augmenter pour few-shot, sinon la fin est coupée)."""
    text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    model_max = getattr(tokenizer, "model_max_length", 32768)
    max_len = min(max_context_tokens, model_max)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Extraction simple du label
    text_lower = generated_text.strip().lower()
    if "entailment" in text_lower[:50]: 
        extracted_label = "Entailment"
    elif "contradiction" in text_lower[:50]: 
        extracted_label = "Contradiction"
    elif "entail" in text_lower: 
        extracted_label = "Entailment"
    elif "contradict" in text_lower: 
        extracted_label = "Contradiction"
    else:
        extracted_label = "UNKNOWN"  # Si on ne peut pas extraire un label clair
    
    return extracted_label, generated_text.strip()

def extract_premise_hypothesis(user_content):
    """Extrait la PREMISE et HYPOTHESIS du contenu utilisateur"""
    if "PREMISE:" in user_content and "HYPOTHESIS:" in user_content:
        parts = user_content.split("HYPOTHESIS:")
        premise = parts[0].replace("PREMISE:", "").strip()
        hypothesis = parts[1].strip() if len(parts) > 1 else ""
        return premise, hypothesis
    return user_content, ""  # Fallback si format inattendu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Chemin du modèle à évaluer (Output ou Base)")
    parser.add_argument("--base_model_path", type=str, required=True, help="Chemin du modèle original (Qwen)")
    parser.add_argument("--test_file", type=str, required=True, help="Fichier de test")
    parser.add_argument("--output_csv", type=str, default=None, 
                        help="Chemin du fichier CSV de sortie pour les prédictions (par défaut: predictions_<model_name>.csv)")
    parser.add_argument("--max_context_tokens", type=int, default=8192,
                        help="Nombre max de tokens du prompt (défaut 8192 pour few-shot; réduire si pas de few-shot)")
    args = parser.parse_args()

    EVAL_PATH = Path(args.model_path)
    BASE_PATH = Path(args.base_model_path)
    TEST_PATH = Path(args.test_file)

    print("="*60)
    print(f"BASE MODEL : {BASE_PATH.name}")
    print(f"EVAL TARGET: {EVAL_PATH.name}")
    print("="*60)

    # 1. On charge toujours le modèle de BASE d'abord
    print("Chargement du modèle de base...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", 
        trust_remote_code=True
    )

    # 2. Logique intelligente : Est-ce un finetuning LoRA ?
    # Si le dossier d'évaluation est différent de la base ET contient une config d'adapter
    if EVAL_PATH != BASE_PATH and (EVAL_PATH / "adapter_config.json").exists():
        print(f"🔄 Détection d'adapters LoRA. Fusion depuis : {EVAL_PATH}")
        model = PeftModel.from_pretrained(model, EVAL_PATH)
    else:
        print("✅ Mode Baseline : Aucune fusion d'adapters nécessaire.")

    # Inférence
    test_samples = load_jsonl(TEST_PATH)
    print(f"\nInférence sur {len(test_samples)} exemples...")
    
    # Préparer les données pour le CSV
    csv_rows = []
    predictions, true_labels = [], []
    
    for i, sample in enumerate(test_samples):
        if (i + 1) % 50 == 0: 
            print(f"  Progrès : {i + 1}/{len(test_samples)}")
        
        # Extraire les informations : dernier message = assistant (gold), dernier "user" = la question de test
        true_label = sample["messages"][-1]["content"]
        user_messages = [m for m in sample["messages"] if m.get("role") == "user"]
        user_content = user_messages[-1]["content"] if user_messages else sample["messages"][0].get("content", "")
        premise, hypothesis = extract_premise_hypothesis(user_content)
        
        # Faire la prédiction (max_context_tokens pour que tout le prompt few-shot soit vu)
        pred_label, raw_generated = predict(model, tokenizer, sample["messages"], max_context_tokens=args.max_context_tokens)
        
        # Stocker pour les métriques
        true_labels.append(true_label)
        predictions.append(pred_label)
        
        # Préparer la ligne CSV
        is_correct = pred_label == true_label
        csv_rows.append({
            "index": i,
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": true_label,
            "predicted_label": pred_label,
            "is_correct": is_correct,
            "raw_generated": raw_generated
        })
    
    # Déterminer le chemin de sortie du CSV
    if args.output_csv:
        csv_path = Path(args.output_csv)
        # Créer le dossier parent si nécessaire
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Générer un nom automatique basé sur le modèle et le job ID Slurm si disponible
        model_name = EVAL_PATH.name if EVAL_PATH != BASE_PATH else BASE_PATH.name
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        
        # Créer le dossier logs_evaluate/ dans le répertoire courant
        output_dir = Path.cwd() / "logs_evaluate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if slurm_job_id:
            csv_path = output_dir / f"predictions_{model_name}_job{slurm_job_id}.csv"
        else:
            csv_path = output_dir / f"predictions_{model_name}.csv"
    
    # Sauvegarder le CSV
    csv_path_absolute = csv_path.resolve()
    print(f"\n💾 Sauvegarde des prédictions dans : {csv_path_absolute}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "premise", "hypothesis", "true_label", "predicted_label", "is_correct", "raw_generated"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"✓ {len(csv_rows)} prédictions sauvegardées")

    # Compter les UNKNOWN (sorties non reconnues comme Entailment/Contradiction)
    n_unknown = sum(1 for p in predictions if p == "UNKNOWN")
    if n_unknown > 0:
        print(f"\n⚠️  Prédictions non reconnues (UNKNOWN) : {n_unknown} / {len(predictions)}")
        print("    -> Cause probable : prompt tronqué (--max_context_tokens trop petit pour le few-shot)")
        print("    -> Essayer : --max_context_tokens 8192 ou 16384")

    # Métriques (macro F1 pour l'article ; UNKNOWN compte comme faux)
    acc = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average="macro", zero_division=0)
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted", zero_division=0)
    
    print("\n" + "="*30)
    print(f"ACCURACY      : {acc:.4f} ({acc*100:.2f}%)")
    print(f"MACRO F1      : {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    print("="*30)

    print("\n--- Matrice de Confusion ---")
    cm = confusion_matrix(true_labels, predictions, labels=["Entailment", "Contradiction"])
    print(f"                Prédit Entailment | Prédit Contradiction")
    print(f"Vrai Entail.  : {cm[0][0]:17d} | {cm[0][1]:18d}")
    print(f"Vrai Contrad. : {cm[1][0]:17d} | {cm[1][1]:18d}")
    if n_unknown > 0:
        print(f"(Les {n_unknown} réponses UNKNOWN ne figurent pas dans cette matrice)")
    print("="*60)
if __name__ == "__main__":
    main()
