import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


###############################################################################
# Configuration générale
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "results" / "Fewshot_Kate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers d'entrée (adaptés à ton setup NLI4PR)
TRAIN_PATH = DATA_DIR / "train-00000-of-00001.parquet"
TEST_PATH = DATA_DIR / "test-00000-of-00001.parquet"

# Hyperparamètres KATE / RoBERTa (faciles à changer)
ROBERTA_MODEL_NAME = "roberta-base"
N_PER_LABEL = 1
MAX_LENGTH = 512
USE_SYSTEM_MESSAGE = True

# Fichiers de sortie
OUT_POL = OUT_DIR / "fewshot_test_pol.jsonl"
OUT_MEDICAL = OUT_DIR / "fewshot_test_medical.jsonl"


###############################################################################
# Formatage des prompts (même logique que NLI4CT Prompt 1)
###############################################################################

SYSTEM_MSG_FEWSHOT = (
    "You will see several examples of premise-hypothesis pairs with their "
    "classification (Entailment or Contradiction). Use these examples to "
    "understand the task. Then classify the relationship between the last "
    "premise and hypothesis. Respond with only one word: 'Entailment' or "
    "'Contradiction'."
)


def format_input_text_prompt1(premise: str, statement: str) -> str:
    return f"PREMISE: {premise}\n\nHYPOTHESIS: {statement}"


def df_to_examples(df: pd.DataFrame) -> List[Dict]:
    """
    Convertit un DataFrame (colonnes: premise, statement, label) en liste d'exemples
    avec champ user_content (comme dans le notebook NLI4CT).
    """
    examples: List[Dict] = []
    for _, row in df.iterrows():
        premise = str(row["premise"])
        statement = str(row["statement"])
        label = str(row["label"])
        user_content = format_input_text_prompt1(premise, statement)
        examples.append({"user_content": user_content, "label": label})
    return examples


def make_fewshot_messages(
    test_ex: Dict, shot_examples: List[Dict], use_system_message: bool = True
) -> List[Dict]:
    messages: List[Dict] = []
    if use_system_message:
        messages.append({"role": "system", "content": SYSTEM_MSG_FEWSHOT})
    for shot in shot_examples:
        messages.append({"role": "user", "content": shot["user_content"]})
        messages.append({"role": "assistant", "content": shot["label"]})
    messages.append({"role": "user", "content": test_ex["user_content"]})
    messages.append({"role": "assistant", "content": test_ex["label"]})
    return messages


###############################################################################
# Embeddings RoBERTa + KATE
###############################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
model = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME, add_pooling_layer=False).to(device)
model.eval()


def get_roberta_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Moyenne des états cachés (comme dans le notebook NLI4CT)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def embed_examples(examples: List[Dict]) -> List[Dict]:
    for ex in tqdm(examples, desc="Embedding (RoBERTa)"):
        ex["embedding"] = get_roberta_embedding(ex["user_content"])
    return examples


def select_kate_shots(
    test_embedding: np.ndarray,
    train_examples_with_emb: List[Dict],
    n_per_label: int = 1,
) -> List[Dict]:
    """
    Sélectionne n_per_label exemples les plus proches pour chaque label
    (Entailment, Contradiction) en similarité cosine.
    """
    embeddings = np.array([ex["embedding"] for ex in train_examples_with_emb])
    sims = cosine_similarity([test_embedding], embeddings)[0]
    shots: List[Dict] = []
    for label in ["Entailment", "Contradiction"]:
        indices = [i for i, ex in enumerate(train_examples_with_emb) if ex["label"] == label]
        if not indices:
            continue
        best_idx = indices[np.argmax(sims[indices])]
        shots.append(train_examples_with_emb[best_idx])
    return shots


###############################################################################
# Pipeline principal : génération des JSONL few-shot pour pol et médical
###############################################################################

def build_fewshot_jsonl(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: Path,
    task_name: str,
) -> None:
    """
    Construit un fichier JSONL few-shot (format KATE) pour un sous-ensemble donné
    (pol ou medical), à partir des DataFrame train/test déjà filtrés.
    """
    print(f"\n=== Construction few-shot KATE pour {task_name} ===")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    train_examples = df_to_examples(train_df)
    test_examples = df_to_examples(test_df)

    # Embedding du train
    train_with_emb = embed_examples(train_examples)

    # Construction JSONL
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for test_ex in tqdm(test_examples, desc=f"Building few-shot JSONL ({task_name})"):
            test_emb = get_roberta_embedding(test_ex["user_content"])
            shots = select_kate_shots(
                test_emb, train_with_emb, n_per_label=N_PER_LABEL
            )
            messages = make_fewshot_messages(
                test_ex, shots, use_system_message=USE_SYSTEM_MESSAGE
            )
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    print(f"{out_path.name} : {count} lignes -> {out_path}")


def main() -> None:
    # Chargement Parquet
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Fichiers Parquet introuvables.\n"
            f"Train attendu: {TRAIN_PATH}\n"
            f"Test attendu: {TEST_PATH}\n"
            f"Adapte les chemins en haut du script si nécessaire."
        )

    print("Chargement des fichiers Parquet...")
    train_df_full = pd.read_parquet(TRAIN_PATH)
    test_df_full = pd.read_parquet(TEST_PATH)

    print("Shape train:", train_df_full.shape)
    print("Shape test :", test_df_full.shape)

    # On sait déjà quelles colonnes utiliser:
    # - premise
    # - statement_pol
    # - statement_medical
    # - label

    def make_pol_med(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        rows_pol: List[Dict] = []
        rows_med: List[Dict] = []
        for _, row in df.iterrows():
            premise = row["premise"]
            label = row["label"]

            if pd.notna(row.get("statement_pol", None)):
                rows_pol.append(
                    {
                        "premise": premise,
                        "statement": row["statement_pol"],
                        "label": label,
                    }
                )
            if pd.notna(row.get("statement_medical", None)):
                rows_med.append(
                    {
                        "premise": premise,
                        "statement": row["statement_medical"],
                        "label": label,
                    }
                )
        return pd.DataFrame(rows_pol), pd.DataFrame(rows_med)

    train_pol, train_med = make_pol_med(train_df_full)
    test_pol, test_med = make_pol_med(test_df_full)

    print(
        f"Taille train_pol={len(train_pol)}, train_medical={len(train_med)}, "
        f"test_pol={len(test_pol)}, test_medical={len(test_med)}"
    )

    # Génération des fichiers few-shot
    if len(train_pol) > 0 and len(test_pol) > 0:
        build_fewshot_jsonl(train_pol, test_pol, OUT_POL, task_name="POL")
    else:
        print("Aucun exemple POL détecté, fichier fewshot_test_pol.jsonl non généré.")

    if len(train_med) > 0 and len(test_med) > 0:
        build_fewshot_jsonl(train_med, test_med, OUT_MEDICAL, task_name="MEDICAL")
    else:
        print("Aucun exemple MEDICAL détecté, fichier fewshot_test_medical.jsonl non généré.")


if __name__ == "__main__":
    main()


