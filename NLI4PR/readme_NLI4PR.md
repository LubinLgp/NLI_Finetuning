# NLI4PR — Natural Language Inference for Protocol Retrieval

Projet de **fine-tuning** de modèles de langage pour la tâche NLI (Entailment / Contradiction) appliquée au **matching patient–critères d’essais cliniques**. Les données sont en deux formulations : **POL** (vulgarisée) et **MEDICAL** (experte). Trois types de prompts sont utilisés : **NLI** (baseline), **Clinical Matching** et **Chain-of-Thought (CoT)**. Les résultats sont évalués en **baseline** (modèle non finetuné) et **finetune** (modèle finetuné), puis analysés et comparés.

---

## Prérequis

- **Données** : fichiers Parquet dans un dossier `data/` à la racine du projet :
  - `data/train-00000-of-00001.parquet`
  - `data/validation-00000-of-00001.parquet`
  - `data/test-00000-of-00001.parquet`
- Chaque Parquet doit contenir au minimum : `premise`, `statement_pol`, `statement_medical`, `label` (Entailment / Contradiction).
- **Python 3** avec les bibliothèques suivantes :
  - Pour les scripts de génération : `pandas`, `pathlib`
  - Pour `save_fewshot.py` : `pandas`, `numpy`, `torch`, `transformers`, `scikit-learn`, `tqdm`
  - Pour les notebooks d’analyse : `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`

---

## Arborescence du projet

```
NLI4PR/
├── README_NLI4PR.md          # Ce fichier
├── data/                     # Données Parquet (train / validation / test) — à fournir
├── graphes/                  # Graphiques de l'analyse linguistique (créé à l'exécution)
├── save_nli.py               # Génère les JSONL d’entraînement NLI
├── save_clinical_matching.py # Génère les JSONL Clinical Matching
├── save_CoT.py               # Génère les JSONL Chain-of-Thought
├── save_fewshot.py           # Génère les JSONL few-shot KATE (RoBERTa + similarité cosine)
│
└── results/                  # Tous les résultats, évals et analyses
    ├── graphes/              # Graphiques sauvegardés par les notebooks (créé à l’exécution)
    ├── fusionne_result.py    # Fusionne eval POL + MEDICAL en un CSV combined
    │
    ├── indices_pol_all_baseline_wrong.csv      # Indices POL où les 3 baselines se trompent
    ├── indices_medical_all_baseline_wrong.csv
    ├── indices_pol_all_finetune_wrong.csv      # Indices POL où les 3 finetunes se trompent
    ├── indices_medical_all_finetune_wrong.csv
    ├── comparison_linguistic_pol.csv          # Comparaison métriques linguistiques (candidats vs autres) - POL
    ├── comparison_linguistic_medical.csv      # Comparaison métriques linguistiques (candidats vs autres) - MEDICAL
    │
    ├── compare_all_baseline.ipynb    # Comparaison des 3 prompts baseline (scores + indices “tous wrong”)
    ├── compare_all_finetuning.ipynb  # Comparaison des 3 prompts finetune (scores + indices “tous wrong”)
    ├── analyse_linguistique_candidats.ipynb  # Analyse linguistique automatique : métriques (Jaccard, numeric, négations, etc.) pour comparer candidats vs autres indices
    │
    ├── Fewshot_kate/
    │   ├── fewshot_test_pol.jsonl, fewshot_test_medical.jsonl  # JSONL few-shot KATE pour test
    │   ├── eval_fewshot_combined.csv, eval_fewshot_pol.csv, eval_fewshot_medical.csv
    │   ├── analyse_fewshot.ipynb
    │   └── graphes/
    │
    ├── Prompt_NLI/
    │   ├── nli_train_pol.jsonl, nli_train_medical.jsonl
    │   ├── nli_validation_pol.jsonl, nli_validation_medical.jsonl
    │   ├── nli_test_pol.jsonl, nli_test_medical.jsonl
    │   ├── eval_nli_combined_baseline.csv, eval_nli_combined_finetune.csv
    │   ├── eval_nli_pol_baseline.csv, eval_nli_pol_finetune.csv
    │   ├── eval_nli_medical_baseline.csv, eval_nli_medical_finetune.csv
    │   ├── analyse_nli_baseline.ipynb, analyse_nli_finetune.ipynb
    │   └── compare_nli_baseline_finetune.ipynb
    │
    ├── Prompt_clinical_matching/
    │   ├── clinical_matching_{train,validation,test}_{pol,medical}.jsonl
    │   ├── eval_clinical_matching_combined_baseline.csv, eval_clinical_matching_combined_finetune.csv
    │   ├── eval_clinical_matching_{pol,medical}_{baseline,finetune}.csv
    │   ├── analyse_clinical_matching_baseline.ipynb, analyse_clinical_matching_finetune.ipynb
    │   └── compare_clinical_matching_baseline_finetune.ipynb
    │
    └── Prompt_cot/
        ├── cot_{train,validation,test}_{pol,medical}.jsonl
        ├── eval_cot_combined_baseline.csv, eval_cot_combined_finetune.csv
        ├── eval_cot_{pol,medical}_{baseline,finetune}.csv
        ├── analyse_cot_baseline.ipynb, analyse_cot_finetune.ipynb
        └── compare_cot_baseline_finetune.ipynb
```

---

## Rôle des fichiers principaux

### À la racine de NLI4PR

| Fichier | Rôle |
|--------|------|
| **save_nli.py** | Lit les Parquet dans `data/`, produit 6 JSONL au format OpenAI (train/val/test × pol/medical) avec un prompt **NLI** (Premise + Hypothesis, réponse en un mot). Les sorties sont du type `nli_*_pol.jsonl` / `nli_*_medical.jsonl` (à placer dans `results/Prompt_NLI/` si besoin). |
| **save_clinical_matching.py** | Même principe pour le prompt **Clinical Matching** : question patient / critères, réponse en un mot. Produit `clinical_matching_*_pol.jsonl` et `clinical_matching_*_medical.jsonl`. |
| **save_CoT.py** | Même principe pour le prompt **Chain-of-Thought** : même question + consigne de raisonnement étape par étape puis un mot. Produit `cot_*_pol.jsonl` et `cot_*_medical.jsonl`. |
| **save_fewshot.py** | Génère des JSONL few-shot pour le test en utilisant la méthode **KATE** (kNN-Augmented in-context Example selection) : sélectionne les exemples d'entraînement les plus sémantiquement proches de chaque instance de test via embeddings RoBERTa + similarité cosine. Pour chaque test, sélectionne 1 exemple Entailment + 1 exemple Contradiction les plus proches. Produit `fewshot_test_pol.jsonl` et `fewshot_test_medical.jsonl` dans `results/Fewshot_kate/`. |

À exécuter depuis la racine **NLI4PR** (avec le dossier `data/` présent). Les JSONL générés peuvent être déplacés dans `results/Prompt_NLI/`, `results/Prompt_clinical_matching/`, `results/Prompt_cot/`, `results/Fewshot_kate/` selon l’usage (entraînement / évaluation).

### Dans results/

| Fichier | Rôle |
|--------|------|
| **fusionne_result.py** | Fusionne deux CSV d’évaluation (POL + MEDICAL) en un seul **combined**, en ajoutant une colonne `statement_type` (pol / medical). Les chemins en tête du script sont à adapter (ex. CoT baseline : `Prompt_cot/eval_cot_pol_baseline.csv` + `eval_cot_medical_baseline.csv` → `eval_cot_combined_baseline.csv`). À lancer depuis `results/`. |

### Notebooks dans results/

| Notebook | Rôle |
|----------|------|
| **compare_all_baseline.ipynb** | Charge les CSV d’évaluation **baseline** des 3 prompts (NLI, Clinical Matching, CoT). Calcule les scores (accuracy) Global / POL / MEDICAL, affiche les proportions Contradiction vs Entailment quand tous se trompent, sauvegarde les graphiques dans `graphes/` et exporte les listes d’indices : `indices_pol_all_baseline_wrong.csv`, `indices_medical_all_baseline_wrong.csv`. |
| **compare_all_finetuning.ipynb** | Même logique pour les modèles **finetunés** : scores, graphiques, et export des indices où les 3 finetunes se trompent (`indices_pol_all_finetune_wrong.csv`, `indices_medical_all_finetune_wrong.csv`). |
| **analyse_linguistique_candidats.ipynb** | Analyse linguistique automatique des indices problématiques. Définit les **candidats** = intersection des indices présents dans les deux listes “tous finetune wrong” (POL et MEDICAL). Calcule des métriques linguistiques (chevauchement lexical Jaccard/coverage, densité numérique, négations, longueur, mots-clés trial) pour tous les indices du test, puis compare les **moyennes des candidats** vs les **moyennes des autres indices** (séparément pour POL et MEDICAL). Génère des visualisations (barres, boxplots) sauvegardées dans `graphes/` à la racine et exporte les comparaisons en CSV (`comparison_linguistic_pol.csv`, `comparison_linguistic_medical.csv`). Permet aussi l’inspection manuelle cas par cas (choix d’un `idx`, affichage prompts + eval finetune/baseline). |

### Par type de prompt (Prompt_NLI, Prompt_clinical_matching, Prompt_cot)

Dans chaque dossier on trouve :

- **JSONL** : `*_train_*.jsonl`, `*_validation_*.jsonl`, `*_test_*.jsonl` (pol / medical). Ce sont les jeux d’entraînement / validation / test au format messages OpenAI.
- **CSV d’évaluation** :
  - `eval_*_combined_*.csv` : POL + MEDICAL concaténés (avec colonne `statement_type`).
  - `eval_*_pol_*.csv` et `eval_*_medical_*.csv` : évaluations séparées par type d’énoncé.
  - Pour CoT, les CSV contiennent en plus une colonne **reasoning** (chaîne de raisonnement du modèle).
- **Notebooks d’analyse** :
  - **analyse_*_baseline.ipynb** / **analyse_*_finetune.ipynb** : métriques (accuracy, F1), matrices de confusion, analyses par taille de prompt, critères, chevauchement des erreurs POL/MEDICAL ; sauvegarde des graphiques dans un sous-dossier `graphes/` du prompt.
  - **compare_*_baseline_finetune.ipynb** : comparaison directe baseline vs finetune (scores, accord/désaccord, erreurs corrigées ou nouvelles), avec sauvegarde de graphiques dans `graphes/`.

Les noms des graphiques et des CSV suivent le schéma : préfixe du prompt (nli, clinical_matching, cot) + baseline/finetune + descriptif (ex. `cot_baseline_confusion_combined.png`).

### Few-shot KATE (Fewshot_kate/)

Le dossier `results/Fewshot_kate/` contient :

- **JSONL few-shot** : `fewshot_test_pol.jsonl`, `fewshot_test_medical.jsonl` générés par `save_fewshot.py`. Chaque ligne contient un prompt few-shot avec 1 exemple Entailment + 1 exemple Contradiction sélectionnés via KATE (embeddings RoBERTa + similarité cosine).
- **CSV d’évaluation** : `eval_fewshot_combined.csv`, `eval_fewshot_pol.csv`, `eval_fewshot_medical.csv` (format identique aux autres prompts).
- **Notebooks d’analyse** : `analyse_fewshot.ipynb` pour analyser les performances du few-shot KATE.

---

## Workflow typique

1. **Données** : placer les Parquet dans `data/`.
2. **Génération des JSONL** : exécuter `save_nli.py`, `save_clinical_matching.py`, `save_CoT.py` depuis la racine ; déplacer les JSONL dans les dossiers `results/Prompt_*` si nécessaire.
3. **Génération des JSONL few-shot KATE** (optionnel) : exécuter `save_fewshot.py` depuis la racine pour générer les JSONL few-shot avec sélection KATE. Les fichiers sont créés directement dans `results/Fewshot_kate/`.
4. **Entraînement / évaluation** : effectuer le fine-tuning et l’évaluation en dehors de ce dépôt (ex. OpenAI, script custom), et écrire les CSV d’évaluation dans les bons dossiers `Prompt_*` ou `Fewshot_kate/` (format : `index`, `statement_type` si combined, `gold`, `prediction`, `is_correct`, et éventuellement `reasoning` pour CoT).
5. **Fusion combined** : pour chaque prompt et chaque condition (baseline/finetune), utiliser `fusionne_result.py` (en adaptant les chemins) pour produire les `*_combined_*.csv` à partir des CSV pol et medical.
6. **Analyses** : exécuter les notebooks depuis le dossier `results/` (répertoire de travail = `results/`) pour comparer tous les prompts (compare_all_*.ipynb), analyser chaque prompt (analyse_*_baseline/finetune.ipynb, compare_*_baseline_finetune.ipynb), faire l’analyse linguistique automatique (analyse_linguistique_candidats.ipynb) et l’inspection manuelle cas par cas.

---

## Conventions

- **POL** : énoncé vulgarisé (patient).
- **MEDICAL** : énoncé expert (critères cliniques).
- **Baseline** : modèle non finetuné ; **Finetune** : modèle finetuné sur les JSONL du projet.
- Les **indices** dans les CSV et JSONL sont cohérents : même `index` dans pol et medical désigne la même paire premise/statement dans les deux formulations.

Si tu adaptes les chemins (data/, résultats, graphes), il suffit de mettre à jour les variables en tête des scripts ou des premières cellules des notebooks.
