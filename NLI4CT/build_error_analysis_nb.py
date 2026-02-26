#!/usr/bin/env python3
"""Génère le notebook error_analysis.ipynb avec la nouvelle structure."""
import json
from pathlib import Path

def cell_md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def cell_code(lines):
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in lines], "outputs": [], "execution_count": None}

cells = []

# --- 1. Titre ---
cells.append(cell_md([
    "# Analyse des erreurs — NLI4CT",
    "",
    "Analyse poussée des prédictions (baseline vs finetuné) : quand et pourquoi le modèle se trompe, quand il réussit le mieux. Choix du **Prompt 1** ou **Prompt 2** et sauvegarde de toutes les figures dans un dossier pour rapport."
]))

# --- 2. Configuration ---
cells.append(cell_md(["## 1. Configuration et chemins"]))

cells.append(cell_code([
    "# Choix du prompt : 1 (ancien) ou 2 (nouveau)",
    "PROMPT_ID = 2  # Modifier ici pour analyser Prompt 1 ou Prompt 2",
    "",
    "from pathlib import Path",
    "import os",
    "",
    "# Racine du projet NLI4CT (lancer le notebook depuis le dossier NLI4CT/)",
    "NLI4CT_ROOT = Path(\".\").resolve()",
    "if not (NLI4CT_ROOT / \"results\").exists():",
    "    NLI4CT_ROOT = NLI4CT_ROOT.parent",
    "",
    "RESULTS_DIR = NLI4CT_ROOT / \"results\" / f\"Prompt {PROMPT_ID}\"",
    "FIGURES_DIR = RESULTS_DIR / \"figures\"",
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "# Fichiers JSON (données brutes) — à la racine NLI4CT",
    "GOLD_TEST_JSON = NLI4CT_ROOT / \"Gold_test.json\"",
    "GOLD_TEST_JSONL = RESULTS_DIR / \"Gold_test_formatted.jsonl\"",
    "",
    "# CSV de prédictions : un seul fichier baseline et un seul finetuné par dossier",
    "csv_baseline = list(RESULTS_DIR.glob(\"pred_bl_*.csv\"))",
    "csv_finetuned = list(RESULTS_DIR.glob(\"pred_ft_*.csv\"))",
    "CSV_BASELINE = csv_baseline[0] if csv_baseline else None",
    "CSV_FINETUNED = csv_finetuned[0] if csv_finetuned else None",
    "",
    "print(f\"Prompt analysé : {PROMPT_ID}\")",
    "print(f\"Résultats : {RESULTS_DIR}\")",
    "print(f\"Figures  : {FIGURES_DIR}\")",
    "print(f\"CSV baseline : {CSV_BASELINE}\")",
    "print(f\"CSV finetuné: {CSV_FINETUNED}\")",
    "print(f\"Gold test JSON: {GOLD_TEST_JSON.exists()}\")",
    "print(f\"Gold test JSONL: {GOLD_TEST_JSONL.exists()}\")",
]))

# --- 3. Imports ---
cells.append(cell_code([
    "import json",
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from collections import Counter",
    "",
    "sns.set_style(\"whitegrid\")",
    "plt.rcParams['figure.figsize'] = (10, 6)",
    "plt.rcParams['font.size'] = 10",
]))

# --- 4. Chargement des données ---
cells.append(cell_md(["## 2. Chargement des données"]))

cells.append(cell_code([
    "def extract_hypothesis_from_user_content(text):",
    "    if \"HYPOTHESIS:\" not in text:",
    "        return text[:500] if len(text) > 500 else text",
    "    part = text.split(\"HYPOTHESIS:\", 1)[1]",
    "    for sep in (\".\\n\\nAnswer\", \"?\\n\\nAnswer\", \"? Answer\", \"\\n\\nAnswer\"):",
    "        if sep in part:",
    "            part = part.split(sep)[0]",
    "    return part.strip()",
    "",
    "assert CSV_BASELINE and CSV_FINETUNED, \"CSV baseline ou finetuné introuvables dans le dossier résultats.\"",
    "",
    "df_baseline = pd.read_csv(CSV_BASELINE)",
    "df_finetuned = pd.read_csv(CSV_FINETUNED)",
    "",
    "for df in (df_baseline, df_finetuned):",
    "    if df['is_correct'].dtype == object:",
    "        df['is_correct'] = df['is_correct'].apply(lambda x: str(x).strip().lower() == 'true')",
    "",
    "metadata_list = []",
    "with open(GOLD_TEST_JSONL, 'r', encoding='utf-8') as f:",
    "    for idx, line in enumerate(f):",
    "        if not line.strip():",
    "            continue",
    "        obj = json.loads(line)",
    "        msgs = obj.get(\"messages\", [])",
    "        if len(msgs) < 2:",
    "            continue",
    "        user_content = msgs[0].get(\"content\", \"\")",
    "        label = msgs[1].get(\"content\", \"\").strip()",
    "        statement = extract_hypothesis_from_user_content(user_content)",
    "        metadata_list.append({'index': idx, 'type': 'N/A', 'section_id': 'N/A', 'statement': statement, 'label': label})",
    "",
    "if GOLD_TEST_JSON.exists():",
    "    with open(GOLD_TEST_JSON, 'r', encoding='utf-8') as f:",
    "        gold_data = json.load(f)",
    "    keys = list(gold_data.keys())",
    "    if len(keys) == len(metadata_list):",
    "        for i in range(len(metadata_list)):",
    "            e = gold_data[keys[i]]",
    "            metadata_list[i]['type'] = e.get('Type', 'N/A')",
    "            metadata_list[i]['section_id'] = e.get('Section_id', 'N/A')",
    "",
    "df_metadata = pd.DataFrame(metadata_list)",
    "df_baseline_merged = df_baseline.merge(df_metadata, on='index', how='left')",
    "df_finetuned_merged = df_finetuned.merge(df_metadata, on='index', how='left')",
    "",
    "for df in (df_baseline_merged, df_finetuned_merged):",
    "    if 'hypothesis' in df.columns and 'statement' in df.columns:",
    "        df['hypothesis'] = df['hypothesis'].fillna(df['statement'])",
    "",
    "# Longueurs (caractères) pour analyse prompt long/court",
    "def safe_len(s):",
    "    return len(str(s)) if pd.notna(s) else 0",
    "",
    "df_baseline_merged['premise_len'] = df_baseline_merged['premise'].apply(safe_len)",
    "df_baseline_merged['hypothesis_len'] = df_baseline_merged['hypothesis'].apply(safe_len)",
    "df_baseline_merged['prompt_len'] = df_baseline_merged['premise_len'] + df_baseline_merged['hypothesis_len']",
    "df_finetuned_merged['premise_len'] = df_finetuned_merged['premise'].apply(safe_len)",
    "df_finetuned_merged['hypothesis_len'] = df_finetuned_merged['hypothesis'].apply(safe_len)",
    "df_finetuned_merged['prompt_len'] = df_finetuned_merged['premise_len'] + df_finetuned_merged['hypothesis_len']",
    "",
    "# Bins de longueur (tertiles)",
    "for df in (df_baseline_merged, df_finetuned_merged):",
    "    df['length_bin'] = pd.qcut(df['prompt_len'], q=3, labels=['Court', 'Moyen', 'Long'], duplicates='drop')",
    "",
    "print(f\"Baseline : {len(df_baseline_merged)} exemples\")",
    "print(f\"Finetuné : {len(df_finetuned_merged)} exemples\")",
    "print(f\"Métadonnées (type, section) : {df_metadata['type'].notna().all() and (df_metadata['section_id'] != 'N/A').all()}\")",
]))

# --- 5. Statistiques globales ---
cells.append(cell_md(["## 3. Statistiques globales"]))

cells.append(cell_code([
    "acc_bl = df_baseline_merged['is_correct'].mean()",
    "acc_ft = df_finetuned_merged['is_correct'].mean()",
    "",
    "fig, ax = plt.subplots(figsize=(6, 4))",
    "ax.bar(['Baseline', 'Finetuné'], [acc_bl, acc_ft], color=['#3498db', '#2ecc71'])",
    "ax.set_ylabel('Accuracy')",
    "ax.set_ylim(0, 1)",
    "ax.set_title(f'Accuracy globale (Prompt {PROMPT_ID})')",
    "for i, v in enumerate([acc_bl, acc_ft]):",
    "    ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '01_accuracy_globale.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "print(f\"Baseline : {acc_bl:.2%}\")",
    "print(f\"Finetuné : {acc_ft:.2%}\")",
]))

# --- 6. Analyse par type (Single / Comparison) ---
cells.append(cell_md(["## 4. Analyse par type de tâche (Single vs Comparison)"]))

cells.append(cell_code([
    "def accuracy_by(df, col, model_name='Baseline'):",
    "    g = df.groupby(col)",
    "    n = g.size()",
    "    correct = g['is_correct'].sum()",
    "    return pd.DataFrame({'n': n, 'correct': correct}).assign(",
    "        accuracy=lambda x: x['correct'] / x['n'], model=model_name)",
    "",
    "acc_type_bl = accuracy_by(df_baseline_merged, 'type', 'Baseline')",
    "acc_type_ft = accuracy_by(df_finetuned_merged, 'type', 'Finetuné')",
    "acc_type = pd.concat([acc_type_bl, acc_type_ft])",
    "",
    "fig, ax = plt.subplots(figsize=(8, 4))",
    "x = np.arange(len(acc_type_bl.index))",
    "w = 0.35",
    "ax.bar(x - w/2, acc_type_bl['accuracy'], w, label='Baseline', color='#3498db')",
    "ax.bar(x + w/2, acc_type_ft['accuracy'], w, label='Finetuné', color='#2ecc71')",
    "ax.set_xticks(x)",
    "ax.set_xticklabels(acc_type_bl.index)",
    "ax.set_ylabel('Accuracy')",
    "ax.set_ylim(0, 1)",
    "ax.legend()",
    "ax.set_title(f'Accuracy par type de tâche (Prompt {PROMPT_ID})')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '02_accuracy_par_type.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "display(acc_type)",
]))

# --- 7. Analyse par section ---
cells.append(cell_md(["## 5. Analyse par section du protocole"]))

cells.append(cell_code([
    "acc_sec_bl = accuracy_by(df_baseline_merged, 'section_id', 'Baseline')",
    "acc_sec_ft = accuracy_by(df_finetuned_merged, 'section_id', 'Finetuné')",
    "",
    "fig, ax = plt.subplots(figsize=(10, 4))",
    "x = np.arange(len(acc_sec_bl.index))",
    "w = 0.35",
    "ax.bar(x - w/2, acc_sec_bl['accuracy'], w, label='Baseline', color='#3498db')",
    "ax.bar(x + w/2, acc_sec_ft['accuracy'], w, label='Finetuné', color='#2ecc71')",
    "ax.set_xticks(x)",
    "ax.set_xticklabels(acc_sec_bl.index, rotation=20, ha='right')",
    "ax.set_ylabel('Accuracy')",
    "ax.set_ylim(0, 1)",
    "ax.legend()",
    "ax.set_title(f'Accuracy par section (Prompt {PROMPT_ID})')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '03_accuracy_par_section.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "display(pd.concat([acc_sec_bl, acc_sec_ft]))",
]))

# --- 8. Analyse par longueur du prompt ---
cells.append(cell_md([
    "## 6. Analyse par longueur du prompt (court / moyen / long)",
    "",
    "On regarde si les modèles se trompent davantage sur les **prompts longs** (prémisse + hypothèse plus longues) ou **courts**."
]))

cells.append(cell_code([
    "acc_len_bl = accuracy_by(df_baseline_merged, 'length_bin', 'Baseline')",
    "acc_len_ft = accuracy_by(df_finetuned_merged, 'length_bin', 'Finetuné')",
    "",
    "fig, ax = plt.subplots(figsize=(8, 4))",
    "x = np.arange(len(acc_len_bl.index))",
    "w = 0.35",
    "ax.bar(x - w/2, acc_len_bl['accuracy'], w, label='Baseline', color='#3498db')",
    "ax.bar(x + w/2, acc_len_ft['accuracy'], w, label='Finetuné', color='#2ecc71')",
    "ax.set_xticks(x)",
    "ax.set_xticklabels(acc_len_bl.index)",
    "ax.set_ylabel('Accuracy')",
    "ax.set_ylim(0, 1)",
    "ax.legend()",
    "ax.set_title(f'Accuracy par longueur du prompt (Prompt {PROMPT_ID})')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '04_accuracy_par_longueur.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "print(\"Effectifs par bin (baseline):\")",
    "print(acc_len_bl[['n', 'correct', 'accuracy']])",
    "print(\"\\nEffectifs par bin (finetuné):\")",
    "print(acc_len_ft[['n', 'correct', 'accuracy']])",
]))

# --- 9. Quand le modèle se trompe ---
cells.append(cell_md([
    "## 7. Quand le modèle se trompe",
    "",
    "Distribution des **erreurs** : par type, section, longueur ; et type d'erreur (Entailment prédit Contradiction ou l'inverse)."
]))

cells.append(cell_code([
    "err_bl = df_baseline_merged[~df_baseline_merged['is_correct']]",
    "err_ft = df_finetuned_merged[~df_finetuned_merged['is_correct']]",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
    "for ax, (err, name) in zip(axes, [(err_bl, 'Baseline'), (err_ft, 'Finetuné')]):",
    "    if len(err) > 0:",
    "        c = err.groupby('true_label')['predicted_label'].value_counts().unstack(fill_value=0)",
    "        sns.heatmap(c, annot=True, fmt='d', cmap='Reds', ax=ax)",
    "    ax.set_title(f'Erreurs {name} (vrai → prédit)')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '05_erreurs_confusion.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))",
    "for i, (err, name) in enumerate([(err_bl, 'Baseline'), (err_ft, 'Finetuné')]):",
    "    if len(err) > 0:",
    "        err.groupby('type').size().plot(kind='bar', ax=axes[i, 0], color='coral')",
    "        axes[i, 0].set_title(f'Erreurs {name} par type')",
    "        axes[i, 0].set_ylabel('Nombre')",
    "        err.groupby('section_id').size().plot(kind='bar', ax=axes[i, 1], color='coral')",
    "        axes[i, 1].set_title(f'Erreurs {name} par section')",
    "        axes[i, 1].tick_params(axis='x', rotation=20)",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '06_erreurs_par_type_et_section.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "",
    "# Longueur moyenne : erreurs vs corrects",
    "fig, ax = plt.subplots(figsize=(6, 4))",
    "for df, name in [(df_baseline_merged, 'Baseline'), (df_finetuned_merged, 'Finetuné')]:",
    "    correct_len = df[df['is_correct']]['prompt_len'].mean()",
    "    wrong_len = df[~df['is_correct']]['prompt_len'].mean()",
    "    ax.bar([f'{name} (correct)', f'{name} (erreur)'], [correct_len, wrong_len], label=name)",
    "ax.set_ylabel('Longueur moyenne (car.)')",
    "ax.set_title('Longueur moyenne du prompt : correct vs erreur')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '07_longueur_erreurs_vs_corrects.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 10. Quand le modèle réussit le mieux ---
cells.append(cell_md([
    "## 8. Quand le modèle réussit le mieux",
    "",
    "Profil des **réponses correctes** : types et sections où l'accuracy est la plus élevée ; cas où baseline et finetuné sont d'accord."
]))

cells.append(cell_code([
    "agree_both_correct = df_baseline_merged['is_correct'] & df_finetuned_merged['is_correct']",
    "agree_both_wrong = ~df_baseline_merged['is_correct'] & ~df_finetuned_merged['is_correct']",
    "only_bl = df_baseline_merged['is_correct'] & ~df_finetuned_merged['is_correct']",
    "only_ft = ~df_baseline_merged['is_correct'] & df_finetuned_merged['is_correct']",
    "",
    "fig, ax = plt.subplots(figsize=(6, 4))",
    "ax.bar(['Les deux corrects', 'Les deux en erreur', 'Seul baseline correct', 'Seul finetuné correct'],",
    "       [agree_both_correct.sum(), agree_both_wrong.sum(), only_bl.sum(), only_ft.sum()],",
    "       color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6'])",
    "ax.set_ylabel('Nombre d\\'exemples')",
    "ax.set_title('Accord entre Baseline et Finetuné')",
    "plt.xticks(rotation=15, ha='right')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '08_accord_baseline_finetuned.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "",
    "# Où le modèle (finetuné) est le plus fort : sections avec meilleure accuracy",
    "print('Sections où le finetuné a la meilleure accuracy :')",
    "print(acc_sec_ft.sort_values('accuracy', ascending=False)[['n', 'correct', 'accuracy']])",
]))

# --- 11. Régressions ---
cells.append(cell_md([
    "## 9. Régressions (baseline correct, finetuné incorrect)",
    "",
    "Cas où le fine-tuning a **dégradé** la prédiction."
]))

cells.append(cell_code([
    "baseline_correct = df_baseline_merged['is_correct']",
    "finetuned_wrong = ~df_finetuned_merged['is_correct']",
    "regressions = df_baseline_merged[baseline_correct & finetuned_wrong].copy()",
    "",
    "print(f\"Nombre de régressions : {len(regressions)}\")",
    "if len(regressions) > 0:",
    "    print(regressions.groupby('type').size())",
    "    print(regressions.groupby('section_id').size())",
    "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
    "    regressions.groupby('type').size().plot(kind='bar', ax=axes[0], color='#e74c3c')",
    "    axes[0].set_title('Régressions par type')",
    "    regressions.groupby('section_id').size().plot(kind='bar', ax=axes[1], color='#e74c3c')",
    "    axes[1].set_title('Régressions par section')",
    "    axes[1].tick_params(axis='x', rotation=20)",
    "    plt.tight_layout()",
    "    fig.savefig(FIGURES_DIR / '09_regressions_par_type_section.png', dpi=150, bbox_inches='tight')",
    "    plt.show()",
]))

# --- 12. Améliorations ---
cells.append(cell_md([
    "## 10. Améliorations (baseline incorrect, finetuné correct)",
    "",
    "Cas où le fine-tuning a **amélioré** la prédiction."
]))

cells.append(cell_code([
    "improvements = df_baseline_merged[~df_baseline_merged['is_correct'] & df_finetuned_merged['is_correct']].copy()",
    "",
    "print(f\"Nombre d'améliorations : {len(improvements)}\")",
    "if len(improvements) > 0:",
    "    print(improvements.groupby('type').size())",
    "    print(improvements.groupby('section_id').size())",
    "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
    "    improvements.groupby('type').size().plot(kind='bar', ax=axes[0], color='#2ecc71')",
    "    axes[0].set_title('Améliorations par type')",
    "    improvements.groupby('section_id').size().plot(kind='bar', ax=axes[1], color='#2ecc71')",
    "    axes[1].set_title('Améliorations par section')",
    "    axes[1].tick_params(axis='x', rotation=20)",
    "    plt.tight_layout()",
    "    fig.savefig(FIGURES_DIR / '10_ameliorations_par_type_section.png', dpi=150, bbox_inches='tight')",
    "    plt.show()",
]))

# --- 13. Exemples détaillés ---
cells.append(cell_md([
    "## 11. Exemples d'erreurs et de régressions",
    "",
    "Quelques exemples pour **interpréter** pourquoi le modèle se trompe."
]))

cells.append(cell_code([
    "def get_section_description(sid):",
    "    d = {'Eligibility': \"Critères d'éligibilité\", 'Intervention': 'Traitements',",
    "         'Adverse Events': 'Effets secondaires', 'Results': 'Résultats'}",
    "    return d.get(sid, sid)",
    "",
    "if len(regressions) > 0:",
    "    print('--- Exemples de régressions (baseline correct, finetuné faux) ---')",
    "    for _, row in regressions.head(3).iterrows():",
    "        idx = row['index']",
    "        ft_pred = df_finetuned_merged.loc[df_finetuned_merged['index'] == idx, 'predicted_label'].iloc[0]",
    "        print(f\"\\nIndex {idx} | Type: {row['type']} | Section: {row['section_id']}\")",
    "        print(f\"  Vrai: {row['true_label']} | Baseline: {row['predicted_label']} | Finetuné: {ft_pred}\")",
    "        print(f\"  Statement: {str(row['statement'])[:150]}...\")",
    "",
    "if len(improvements) > 0:",
    "    print('\\n--- Exemples d\\'améliorations (baseline faux, finetuné correct) ---')",
    "    for _, row in improvements.head(3).iterrows():",
    "        idx = row['index']",
    "        ft_pred = df_finetuned_merged.loc[df_finetuned_merged['index'] == idx, 'predicted_label'].iloc[0]",
    "        print(f\"\\nIndex {idx} | Type: {row['type']} | Section: {row['section_id']}\")",
    "        print(f\"  Vrai: {row['true_label']} | Baseline: {row['predicted_label']} | Finetuné: {ft_pred}\")",
]))

# --- 14. Export ---
cells.append(cell_md(["## 12. Export des résultats détaillés"]))

cells.append(cell_code([
    "df_export = df_baseline_merged[['index', 'premise', 'hypothesis', 'true_label', 'premise_len', 'hypothesis_len', 'prompt_len', 'length_bin', 'type', 'section_id']].copy()",
    "df_export['baseline_correct'] = df_baseline_merged['is_correct']",
    "df_export['finetuned_correct'] = df_finetuned_merged['is_correct']",
    "df_export['baseline_pred'] = df_baseline_merged['predicted_label']",
    "df_export['finetuned_pred'] = df_finetuned_merged['predicted_label']",
    "df_export['improvement'] = df_export['finetuned_correct'].astype(int) - df_export['baseline_correct'].astype(int)",
    "",
    "out_csv = RESULTS_DIR / f'error_analysis_detailed_prompt{PROMPT_ID}.csv'",
    "df_export.to_csv(out_csv, index=False)",
    "print(f'Résultats détaillés sauvegardés : {out_csv}')",
    "print(f'Figures sauvegardées dans : {FIGURES_DIR}')",
]))

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(__file__).resolve().parent / "error_analysis.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Notebook écrit :", out_path)
