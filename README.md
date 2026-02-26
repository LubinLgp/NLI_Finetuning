# NLI_Finetuning — Projet NLI pour essais cliniques

Ce document décrit le projet **NLI_Finetuning** : objectifs, structure des dossiers, rôle des fichiers.

---

## 1. Objectif du projet

- **NLI (Natural Language Inference)** appliqué aux **essais cliniques** : déterminer si une **hypothèse** est en **Entailment** ou en **Contradiction** par rapport à une **prémisse** extraite des protocoles d’essais (ClinicalTrials.gov, format NLI4CT).
- **Fine-tuning** d’un modèle de langage (Qwen2.5-7B-Instruct) avec **LoRA** sur des paires (prémisse, hypothèse) annotées, puis **évaluation** sur un jeu de test gold.
- Comparaison **baseline (modèle non finetuné)** vs **modèle finetuné**, et analyse des erreurs (notebook).

### Stratégie à deux datasets (NLI4CT + NLI4PR, à terme)

- **Runs séparés** : un fine-tuning par dataset (NLI4CT aujourd’hui ; NLI4PR sera ajouté plus tard).
- **Analyse d’erreurs par dataset** : pour chaque jeu, comprendre *pourquoi* le modèle se trompe et sur *quels types de tâches* il est le moins bon (ex. Single vs Comparison, par section, etc.).
- **Évaluation croisée** : tester chaque jeu de test sur le modèle finetuné sur *l’autre* dataset (ex. test NLI4CT → modèle finetuné sur NLI4PR, et inversement) pour évaluer si le modèle généralise lorsque le prompt ou les données diffèrent un peu.

---

## 2. Structure du dépôt

```
NLI_Finetuning/
├── README.md                 ← Ce fichier
├── literature/               ← Littérature (PDFs, revue)
│   ├── README.md
│   ├── datasets/             ← Articles sur les jeux de données NLI4CT / NLI4PR
│   ├── finetuning/           ← Articles sur le fine-tuning
│   ├── fewshot/              ← Articles few-shot
│   └── lit_review.pdf
└── NLI4CT/                   ← Code et données du projet NLI4CT
    ├── finetuning.py         ← Script de fine-tuning (LoRA, Optuna optionnel)
    ├── evaluate.py           ← Script d’évaluation (baseline ou finetuné)
    ├── finetuning.sh         ← Lancement SLURM du fine-tuning (cluster)
    ├── evaluate.sh           ← Lancement SLURM de l’évaluation
    ├── error_analysis.ipynb  ← Analyse des erreurs (CSV prédictions + Gold test)
    ├── CT_json/              ← JSON des essais cliniques (NCT*.json)
    ├── *.json/              ← JSON du dataset NLI4CT qui lie des hypothèses avec des extraits de CT
    └── results/              ← Données formatées et résultats par prompt
        ├── Prompt 1/         ← Ancien prompt (system + PREMISE / HYPOTHESIS)
        └── Prompt 2/         ← Nouveau prompt (user seul, formulation explicite)
```

---

## 3. Rôle des fichiers principaux

### 3.1 `NLI4CT/finetuning.py`

- **Rôle** : Fine-tuning du modèle Qwen avec **LoRA** (4-bit quantized), à partir de fichiers JSONL au format « messages » (user/assistant).
- **Entrées** : `--model_path`, `--train_file` (JSONL), optionnellement `--eval_file`, `--output_dir`.
- **Modes** :
  - **Normal** : entraînement avec hyperparamètres fixes (epochs, lr, batch, etc.).
  - **Optuna** (`--use_optuna`) : recherche d’hyperparamètres puis entraînement final (ou seulement sauvegarde des meilleurs HP avec `--optuna_save_best_hp`).
  - **Job 2** (`--load_optuna_hp`) : chargement des HP depuis un JSON et entraînement final uniquement.
- **Sortie** : dossier `output_dir` contenant l’adapter LoRA (`adapter_config.json`, poids) et le tokenizer.
Note : dans le projet actuel on ne se sert pas de optuna

### 3.2 `NLI4CT/evaluate.py`

- **Rôle** : Évaluer un modèle (baseline = modèle de base, ou finetuné = base + LoRA) sur un fichier de test JSONL.
- **Entrées** : `--base_model_path`, `--model_path` (base ou dossier contenant les adapters), `--test_file` (JSONL), optionnellement `--output_csv`.
- **Comportement** : Si `model_path != base_model_path` et présence de `adapter_config.json`, fusion LoRA puis inférence. Sinon, inférence directe (baseline).
- **Sortie** : métriques (accuracy, F1, matrice de confusion) + fichier CSV des prédictions (index, premise, hypothesis, true_label, predicted_label, is_correct, raw_generated).

### 3.3 `NLI4CT/finetuning.sh` et `NLI4CT/evaluate.sh`

- **Rôle** : Scripts **SLURM** pour lancer le fine-tuning et l’évaluation sur le cluster (chemins type `/mnt/beegfs/...`).
- **finetuning.sh** : variables `FINETUNING_MODE` (normal / job1 / job2), `BEST_HP_FILE`, chemins vers le modèle Qwen, `train_formatted.jsonl`, `dev_formatted.jsonl`, `outputs/...`.
- **evaluate.sh** : configuration de `BASE_MODEL`, `MODEL_TO_EVAL`, `TEST_FILE` (ex. `Gold_test_formatted.jsonl`). À adapter selon que l’on évalue la baseline ou le modèle finetuné.

### 3.4 `NLI4CT/error_analysis.ipynb`

- **Rôle** : Analyse poussée des erreurs : quand et pourquoi le modèle se trompe, quand il réussit le mieux. Comparaison baseline vs finetuné, par **type** (Single/Comparison), **section**, et **longueur du prompt** (court/moyen/long). Régressions, améliorations, synthèse pour rapport.
- **Configuration** : en tête du notebook, choisir `PROMPT_ID = 1` ou `2` pour analyser les résultats de Prompt 1 ou Prompt 2. Les chemins (CSV, JSONL, `Gold_test.json`) sont déduits automatiquement depuis `NLI4CT/results/Prompt {ID}/`.
- **Figures** : toutes les figures sont enregistrées dans `NLI4CT/results/Prompt {ID}/figures/` (nommées `01_accuracy_globale.png`, `02_accuracy_par_type.png`, etc.) pour réutilisation dans un rapport. Une synthèse texte est aussi sauvegardée (`synthese_erreurs.txt`) dans ce dossier.

### 3.5 `NLI4CT/CT_json/`

- **Rôle** : Fichiers JSON des essais cliniques (identifiants NCT*). Ce sont les sources des textes (prémisses) utilisés pour construire les paires (prémisse, hypothèse) des jeux d’entraînement et de test.

### 3.6 Données utilisées

- **Données brutes** : les JSON dans le dossier `NLI4CT/` (`train.json`, `dev.json`, `Gold_test.json`) et les essais dans `NLI4CT/CT_json/` (NCT*.json). Les JSONL formatés (train / dev / Gold test) sont générés par un script externe (hors repo) et placés dans `results/Prompt 1/` et `results/Prompt 2/`. Ce sont ces JSONL qui servent au fine-tuning et à l’évaluation.
- **NLI4PR** : pour l’instant absent du dépôt ; sera intégré plus tard (même type de pipeline : JSONL « messages » dans des dossiers dédiés).

### 3.7 `NLI4CT/results/` — Les deux prompts

Les données **formatées** (train / dev / Gold test) et les **résultats** (prédictions, logs) sont organisés par **prompt**.

| Élément | Prompt 1 (ancien) | Prompt 2 (nouveau) |
|--------|--------------------|---------------------|
| **Format** | Message **system** : « Classify… Respond with only one word: 'Entailment' or 'Contradiction'. » + **user** : `PREMISE: ... HYPOTHESIS: ...` | **User** seul : `PREMISE: ... Is this premise in agreement with the following hypothesis? HYPOTHESIS: ... Answer only with: Entailment or Contradiction.` |
| **Train** | `train_formatted_old_prompt.jsonl` | `train_formatted.jsonl` |
| **Dev** | `dev_formatted_old_prompt.jsonl` | `dev_formatted.jsonl` |
| **Test** | `Gold_test_formatted_old_prompt.jsonl` | `Gold_test_formatted.jsonl` |
| **Prédictions** | `pred_bl_*_prompt1.csv`, `pred_ft_*_prompt1.csv` | `pred_bl_*_prompt2.csv`, `pred_ft_*_prompt2.csv` |
| **Logs** | `resultats_bl_*`, `resultats_ft_*` | `resultats_bl_*_prompt2.out`, `resultats_ft_*_prompt2.out` |

- **Prompt 1** : plus court, une seule instruction dans le system.
- **Prompt 2** : tout dans le user, avec la phrase explicite « Is this premise in agreement with the following hypothesis? » et « Answer only with: Entailment or Contradiction. »
Note: Dans NLI4CT, on peut avoir une ou deux premisses pour une hypothèse (type Single ou Comparison)
---

## 4. Workflow typique

1. **Données** : Les JSONL formatés (train / dev / Gold test) sont dans `results/Prompt 1/` et `results/Prompt 2/`, générés à partir des JSON (script externe, hors repo). Les fichiers JSON du dataset (train, dev, Gold_test) et `CT_json/` sont dans `NLI4CT/`.
2. **Fine-tuning** : Depuis `NLI4CT/`, lancer `finetuning.py` (ou `finetuning.sh` sur le cluster) en pointant `--train_file` vers le JSONL voulu (ex. `results/Prompt 2/train_formatted.jsonl`) et `--output_dir` vers un dossier de sortie (ex. `outputs/qwen2_5_7b_nli4ct_prompt2`).
3. **Évaluation** : Lancer `evaluate.py` (ou `evaluate.sh`) avec le même `--test_file` (ex. `results/Prompt 2/Gold_test_formatted.jsonl`), une fois avec le modèle de base (baseline), une fois avec le dossier du modèle finetuné. Les CSV de prédictions peuvent être sauvegardés dans le même sous-dossier `results/Prompt 2/` ou dans `logs_evaluate/`.
4. **Analyse** : Ouvrir `error_analysis.ipynb`, adapter les chemins (voir § 3.4) vers les CSV et le Gold test du prompt utilisé, puis exécuter les cellules. Objectif : identifier les types d’erreurs et les tâches les plus difficiles pour chaque dataset.
5. **Évaluation croisée** (quand NLI4PR sera en place) : lancer `evaluate.py` avec le test set d’un dataset et le modèle finetuné sur l’autre (ex. `--test_file` NLI4CT, `--model_path` dossier du modèle finetuné NLI4PR) pour mesurer la généralisation.

---


En résumé : **NLI_Finetuning** contient le cœur du projet NLI4CT (code, données formatées par prompt, résultats par prompt, littérature).

---

## 6. Dépendances et environnement

- **Python** : environnement utilisé sur le cluster (voir `finetuning.sh` / `evaluate.sh` : `PYTHON_EXE`, `CONDA_PREFIX`).
- **Librairies** : `torch`, `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes` (optionnel, avec mock si absent), `sklearn` pour l’évaluation. Optuna si `--use_optuna`.
- Pour exécuter en local, adapter les chemins dans les `.sh` (modèle, fichiers train/test, répertoires de sortie).

---

## 7. Résumé pour un chat Cursor

- **Objectif** : NLI sur essais cliniques (Entailment / Contradiction), fine-tuning LoRA de Qwen, évaluation baseline vs finetuné. À terme : deux datasets (NLI4CT + NLI4PR), deux runs de finetuning, analyse d’erreurs par dataset et évaluation croisée (test A sur modèle finetuné B et inversement) pour la généralisation.
- **Où est le code** : `NLI4CT/finetuning.py`, `evaluate.py` ; scripts cluster `finetuning.sh`, `evaluate.sh`.
- **Données** : JSON dans `NLI4CT/` (train, dev, Gold_test) + `CT_json/` ; JSONL formatés (générés en externe) dans `NLI4CT/results/Prompt 1/` et `Prompt 2/` + CSV de prédictions et logs. NLI4PR à venir.
- **Analyse d’erreurs** : `NLI4CT/error_analysis.ipynb` (adapter les chemins vers `results/Prompt 1/` ou `Prompt 2/`).


