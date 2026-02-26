#!/usr/bin/env python3
"""Génère compare_error.ipynb : même structure que error_analysis_prompt1 mais avec les deux prompts dans les mêmes tables/graphiques."""
import json
from pathlib import Path

def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(lines):
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in lines], "outputs": [], "execution_count": None}

cells = []

# --- 1. Titre ---
cells.append(md([
    "# Comparaison des erreurs — Prompt 1 vs Prompt 2",
    "",
    "Même structure d’analyse que le notebook par prompt, mais **les deux prompts** sont affichés dans les **mêmes tables** et les **mêmes graphiques** pour comparer directement baseline/finetuné sur Prompt 1 et Prompt 2."
]))

# --- 2. Configuration ---
cells.append(md(["## 1. Configuration et chemins"]))

cells.append(code([
    "from pathlib import Path",
    "",
    "NLI4CT_ROOT = Path(\".\").resolve()",
    "if not (NLI4CT_ROOT / \"results\").exists():",
    "    NLI4CT_ROOT = NLI4CT_ROOT / \"NLI4CT\"",
    "if not (NLI4CT_ROOT / \"results\").exists():",
    "    raise FileNotFoundError(\"Dossier results/ introuvable. Lancer le notebook depuis NLI4CT/ ou racine du dépôt.\")",
    "",
    "RESULTS_DIR_1 = NLI4CT_ROOT / \"results\" / \"Prompt 1\"",
    "RESULTS_DIR_2 = NLI4CT_ROOT / \"results\" / \"Prompt 2\"",
    "FIGURES_DIR = NLI4CT_ROOT / \"results\" / \"compare_figures\"",
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "GOLD_TEST_JSON = NLI4CT_ROOT / \"Gold_test.json\"",
    "GOLD_JSONL_1 = RESULTS_DIR_1 / \"Gold_test_formatted_old_prompt.jsonl\" if (RESULTS_DIR_1 / \"Gold_test_formatted_old_prompt.jsonl\").exists() else RESULTS_DIR_1 / \"Gold_test_formatted.jsonl\"",
    "GOLD_JSONL_2 = RESULTS_DIR_2 / \"Gold_test_formatted.jsonl\"",
    "",
    "csv_bl_1 = list(RESULTS_DIR_1.glob(\"pred_bl_*.csv\"))",
    "csv_ft_1 = list(RESULTS_DIR_1.glob(\"pred_ft_*.csv\"))",
    "csv_bl_2 = list(RESULTS_DIR_2.glob(\"pred_bl_*.csv\"))",
    "csv_ft_2 = list(RESULTS_DIR_2.glob(\"pred_ft_*.csv\"))",
    "CSV_BL_1, CSV_FT_1 = (csv_bl_1[0] if csv_bl_1 else None), (csv_ft_1[0] if csv_ft_1 else None)",
    "CSV_BL_2, CSV_FT_2 = (csv_bl_2[0] if csv_bl_2 else None), (csv_ft_2[0] if csv_ft_2 else None)",
    "",
    "print(\"Prompt 1:\", RESULTS_DIR_1, \"| Baseline:\", CSV_BL_1, \"| Finetuné:\", CSV_FT_1)",
    "print(\"Prompt 2:\", RESULTS_DIR_2, \"| Baseline:\", CSV_BL_2, \"| Finetuné:\", CSV_FT_2)",
    "print(\"Figures:\", FIGURES_DIR)",
]))

# --- 3. Imports ---
cells.append(code([
    "import json",
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "sns.set_style(\"whitegrid\")",
    "plt.rcParams['figure.figsize'] = (10, 6)",
    "plt.rcParams['font.size'] = 10",
]))

# --- 4. Chargement des données ---
cells.append(md(["## 2. Chargement des données (Prompt 1 et Prompt 2)"]))

cells.append(code([
    "def extract_premise_hypothesis(text):",
    "    if \"HYPOTHESIS:\" not in text:",
    "        return text[:800] if len(text) > 800 else text, text[:500] if len(text) > 500 else text",
    "    before, after = text.split(\"HYPOTHESIS:\", 1)",
    "    premise = before.replace(\"PREMISE:\", \"\").strip()",
    "    for sep in (\".\\n\\nAnswer\", \"?\\n\\nAnswer\", \"? Answer\", \"\\n\\nAnswer\"):",
    "        if sep in after: after = after.split(sep)[0]",
    "    return premise, after.strip()",
    "",
    "def load_one_prompt(csv_bl, csv_ft, jsonl_path):",
    "    bl = pd.read_csv(csv_bl)",
    "    ft = pd.read_csv(csv_ft)",
    "    for df in (bl, ft):",
    "        if df['is_correct'].dtype == object:",
    "            df['is_correct'] = df['is_correct'].apply(lambda x: str(x).strip().lower() == 'true')",
    "    meta_list = []",
    "    with open(jsonl_path, 'r', encoding='utf-8') as f:",
    "        for idx, line in enumerate(f):",
    "            if not line.strip(): continue",
    "            obj = json.loads(line)",
    "            msgs = obj.get(\"messages\", [])",
    "            if len(msgs) < 2: continue",
    "            user_content = \"\"",
    "            for m in msgs:",
    "                if m.get(\"role\") == \"user\": user_content = m.get(\"content\", \"\")",
    "            prem, stmt = extract_premise_hypothesis(user_content)",
    "            meta_list.append({'index': idx, 'premise_jsonl': prem, 'statement': stmt})",
    "    df_meta = pd.DataFrame(meta_list)",
    "    bl = bl.merge(df_meta, on='index', how='left')",
    "    ft = ft.merge(df_meta, on='index', how='left')",
    "    for df in (bl, ft):",
    "        df['premise'] = df['premise_jsonl'].astype(str)",
    "        df['hypothesis'] = df['statement'].astype(str)",
    "        df['prompt_len'] = df['premise'].str.len() + df['hypothesis'].str.len()",
    "    def safe_bin(s):",
    "        q = s.quantile([0, 1/3, 2/3, 1]); e = np.sort(q.unique()); n = max(1, len(e)-1)",
    "        if n == 1: return pd.Series('Moyen', index=s.index)",
    "        return pd.cut(s, bins=e, labels=['Court','Moyen','Long'][:n], include_lowest=True)",
    "    for df in (bl, ft): df['length_bin'] = safe_bin(df['prompt_len'])",
    "    return bl, ft",
    "",
    "assert CSV_BL_1 and CSV_FT_1 and CSV_BL_2 and CSV_FT_2, \"Il manque des CSV (baseline ou finetuné) pour au moins un prompt.\"",
    "",
    "df_bl_1, df_ft_1 = load_one_prompt(CSV_BL_1, CSV_FT_1, GOLD_JSONL_1)",
    "df_bl_2, df_ft_2 = load_one_prompt(CSV_BL_2, CSV_FT_2, GOLD_JSONL_2)",
    "",
    "if GOLD_TEST_JSON.exists():",
    "    with open(GOLD_TEST_JSON, 'r', encoding='utf-8') as f: gold = json.load(f)",
    "    keys = list(gold.keys())",
    "    if len(keys) == len(df_bl_1):",
    "        type_sec = pd.DataFrame([{'index': i, 'type': gold[keys[i]].get('Type','N/A'), 'section_id': gold[keys[i]].get('Section_id','N/A')} for i in range(len(keys))])",
    "        for df in (df_bl_1, df_ft_1, df_bl_2, df_ft_2):",
    "            df['type'] = type_sec.set_index('index').reindex(df['index'])['type'].values",
    "            df['section_id'] = type_sec.set_index('index').reindex(df['index'])['section_id'].values",
    "",
    "print(\"Prompt 1 — Baseline:\", len(df_bl_1), \"| Finetuné:\", len(df_ft_1))",
    "print(\"Prompt 2 — Baseline:\", len(df_bl_2), \"| Finetuné:\", len(df_ft_2))",
]))

# --- 5. Statistiques globales ---
cells.append(md(["## 3. Statistiques globales"]))

cells.append(code([
    "acc = pd.DataFrame({",
    "    'Prompt 1 Baseline': [df_bl_1['is_correct'].mean()],",
    "    'Prompt 1 Finetuné': [df_ft_1['is_correct'].mean()],",
    "    'Prompt 2 Baseline': [df_bl_2['is_correct'].mean()],",
    "    'Prompt 2 Finetuné': [df_ft_2['is_correct'].mean()],",
    "})",
    "display(acc)",
    "fig, ax = plt.subplots(figsize=(10, 4))",
    "x = np.arange(4)",
    "vals = [acc['Prompt 1 Baseline'].iloc[0], acc['Prompt 1 Finetuné'].iloc[0], acc['Prompt 2 Baseline'].iloc[0], acc['Prompt 2 Finetuné'].iloc[0]]",
    "colors = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']",
    "ax.bar(x, vals, color=colors)",
    "ax.set_xticks(x)",
    "ax.set_xticklabels(['P1 Baseline', 'P1 Finetuné', 'P2 Baseline', 'P2 Finetuné'], rotation=15)",
    "ax.set_ylabel('Accuracy')",
    "ax.set_ylim(0, 1)",
    "ax.set_title('Accuracy globale — les deux prompts')",
    "for i, v in enumerate(vals): ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '01_accuracy_globale.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 6. Par type ---
cells.append(md(["## 4. Analyse par type de tâche (Single vs Comparison)"]))

cells.append(code([
    "def acc_by(df, col):",
    "    g = df.groupby(col); n = g.size(); c = g['is_correct'].sum()",
    "    return pd.DataFrame({'n': n, 'correct': c}).assign(accuracy=lambda x: x['correct']/x['n'])",
    "",
    "t1_bl = acc_by(df_bl_1, 'type'); t1_ft = acc_by(df_ft_1, 'type')",
    "t2_bl = acc_by(df_bl_2, 'type'); t2_ft = acc_by(df_ft_2, 'type')",
    "tab_type = pd.DataFrame({",
    "    'P1 Baseline': t1_bl['accuracy'], 'P1 Finetuné': t1_ft['accuracy'],",
    "    'P2 Baseline': t2_bl['accuracy'], 'P2 Finetuné': t2_ft['accuracy'],",
    "})",
    "display(tab_type)",
    "fig, ax = plt.subplots(figsize=(10, 4))",
    "x = np.arange(len(tab_type.index))",
    "w = 0.2",
    "ax.bar(x - 1.5*w, tab_type['P1 Baseline'], w, label='P1 Baseline', color='#3498db')",
    "ax.bar(x - 0.5*w, tab_type['P1 Finetuné'], w, label='P1 Finetuné', color='#2ecc71')",
    "ax.bar(x + 0.5*w, tab_type['P2 Baseline'], w, label='P2 Baseline', color='#9b59b6')",
    "ax.bar(x + 1.5*w, tab_type['P2 Finetuné'], w, label='P2 Finetuné', color='#e67e22')",
    "ax.set_xticks(x); ax.set_xticklabels(tab_type.index)",
    "ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1); ax.legend()",
    "ax.set_title('Accuracy par type — les deux prompts')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '02_accuracy_par_type.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 7. Par section ---
cells.append(md(["## 5. Analyse par section du protocole"]))

cells.append(code([
    "s1_bl = acc_by(df_bl_1, 'section_id'); s1_ft = acc_by(df_ft_1, 'section_id')",
    "s2_bl = acc_by(df_bl_2, 'section_id'); s2_ft = acc_by(df_ft_2, 'section_id')",
    "tab_sec = pd.DataFrame({",
    "    'P1 Baseline': s1_bl['accuracy'], 'P1 Finetuné': s1_ft['accuracy'],",
    "    'P2 Baseline': s2_bl['accuracy'], 'P2 Finetuné': s2_ft['accuracy'],",
    "})",
    "display(tab_sec)",
    "fig, ax = plt.subplots(figsize=(12, 4))",
    "x = np.arange(len(tab_sec.index)); w = 0.2",
    "ax.bar(x - 1.5*w, tab_sec['P1 Baseline'], w, label='P1 Baseline', color='#3498db')",
    "ax.bar(x - 0.5*w, tab_sec['P1 Finetuné'], w, label='P1 Finetuné', color='#2ecc71')",
    "ax.bar(x + 0.5*w, tab_sec['P2 Baseline'], w, label='P2 Baseline', color='#9b59b6')",
    "ax.bar(x + 1.5*w, tab_sec['P2 Finetuné'], w, label='P2 Finetuné', color='#e67e22')",
    "ax.set_xticks(x); ax.set_xticklabels(tab_sec.index, rotation=20, ha='right')",
    "ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1); ax.legend()",
    "ax.set_title('Accuracy par section — les deux prompts')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '03_accuracy_par_section.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 8. Par longueur ---
cells.append(md(["## 6. Analyse par longueur du prompt (court / moyen / long)"]))

cells.append(code([
    "l1_bl = acc_by(df_bl_1, 'length_bin'); l1_ft = acc_by(df_ft_1, 'length_bin')",
    "l2_bl = acc_by(df_bl_2, 'length_bin'); l2_ft = acc_by(df_ft_2, 'length_bin')",
    "idx_len = l1_bl.index.union(l2_bl.index).union(l1_ft.index).union(l2_ft.index).unique()",
    "tab_len = pd.DataFrame(index=idx_len, columns=['P1 Baseline','P1 Finetuné','P2 Baseline','P2 Finetuné'])",
    "for col, ser in [('P1 Baseline', l1_bl['accuracy']), ('P1 Finetuné', l1_ft['accuracy']), ('P2 Baseline', l2_bl['accuracy']), ('P2 Finetuné', l2_ft['accuracy'])]:",
    "    tab_len[col] = ser",
    "tab_len = tab_len.fillna(0)",
    "display(tab_len)",
    "fig, ax = plt.subplots(figsize=(10, 4))",
    "x = np.arange(len(tab_len)); w = 0.2",
    "ax.bar(x - 1.5*w, tab_len['P1 Baseline'], w, label='P1 Baseline', color='#3498db')",
    "ax.bar(x - 0.5*w, tab_len['P1 Finetuné'], w, label='P1 Finetuné', color='#2ecc71')",
    "ax.bar(x + 0.5*w, tab_len['P2 Baseline'], w, label='P2 Baseline', color='#9b59b6')",
    "ax.bar(x + 1.5*w, tab_len['P2 Finetuné'], w, label='P2 Finetuné', color='#e67e22')",
    "ax.set_xticks(x); ax.set_xticklabels(tab_len.index)",
    "ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1); ax.legend()",
    "ax.set_title('Accuracy par longueur du prompt — les deux prompts')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '04_accuracy_par_longueur.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 9. Courbe accuracy vs longueur ---
cells.append(code([
    "N_BINS = 15",
    "for name, df in [('P1_bl', df_bl_1), ('P1_ft', df_ft_1), ('P2_bl', df_bl_2), ('P2_ft', df_ft_2)]:",
    "    df['len_bin_idx'] = (df['prompt_len'].rank(pct=True, method='first') * N_BINS).astype(int).clip(0, N_BINS-1)",
    "def curve(df):",
    "    g = df.groupby('len_bin_idx').agg(len_moy=('prompt_len','mean'), n=('prompt_len','count'), correct=('is_correct','sum'))",
    "    g = g[g['n']>=1]; g['accuracy'] = g['correct']/g['n']",
    "    return g.sort_index()",
    "c1_bl, c1_ft = curve(df_bl_1), curve(df_ft_1)",
    "c2_bl, c2_ft = curve(df_bl_2), curve(df_ft_2)",
    "fig, ax = plt.subplots(figsize=(10, 5))",
    "ax.plot(c1_bl['len_moy'], c1_bl['accuracy'], 'o-', color='#3498db', label='P1 Baseline', markersize=4)",
    "ax.plot(c1_ft['len_moy'], c1_ft['accuracy'], 's-', color='#2ecc71', label='P1 Finetuné', markersize=4)",
    "ax.plot(c2_bl['len_moy'], c2_bl['accuracy'], 'o-', color='#9b59b6', label='P2 Baseline', markersize=4)",
    "ax.plot(c2_ft['len_moy'], c2_ft['accuracy'], 's-', color='#e67e22', label='P2 Finetuné', markersize=4)",
    "ax.set_xlabel('Longueur moyenne du prompt (car.)'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.05)",
    "ax.legend(); ax.grid(True, alpha=0.3)",
    "ax.set_title('Performance en fonction de la longueur — les deux prompts')",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '04b_accuracy_vs_longueur_courbe.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 10. Quand le modèle se trompe ---
cells.append(md(["## 7. Quand le modèle se trompe (matrices d'erreurs)"]))

cells.append(code([
    "err_bl_1 = df_bl_1[~df_bl_1['is_correct']]",
    "err_ft_1 = df_ft_1[~df_ft_1['is_correct']]",
    "err_bl_2 = df_bl_2[~df_bl_2['is_correct']]",
    "err_ft_2 = df_ft_2[~df_ft_2['is_correct']]",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))",
    "for ax, (err, title) in zip(axes.flat, [(err_bl_1,'P1 Baseline'), (err_ft_1,'P1 Finetuné'), (err_bl_2,'P2 Baseline'), (err_ft_2,'P2 Finetuné')]):",
    "    if len(err) > 0:",
    "        c = err.groupby('true_label')['predicted_label'].value_counts().unstack(fill_value=0)",
    "        sns.heatmap(c, annot=True, fmt='d', cmap='Reds', ax=ax)",
    "    ax.set_title(title)",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '05_erreurs_confusion.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 8))",
    "for ax, (err, title) in zip(axes.flat, [(err_bl_1,'P1 Baseline'), (err_ft_1,'P1 Finetuné'), (err_bl_2,'P2 Baseline'), (err_ft_2,'P2 Finetuné')]):",
    "    if len(err) > 0: err.groupby('section_id').size().plot(kind='bar', ax=ax, color='coral')",
    "    ax.set_title(title); ax.set_ylabel('Nombre'); ax.tick_params(axis='x', rotation=20)",
    "plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '06_erreurs_par_section.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 11. UNKNOWN ---
cells.append(md(["### 7b. Cas où le modèle n'a pas produit un label reconnu (UNKNOWN)"]))

cells.append(code([
    "unk = []",
    "for df, name in [(df_bl_1,'P1 Baseline'), (df_ft_1,'P1 Finetuné'), (df_bl_2,'P2 Baseline'), (df_ft_2,'P2 Finetuné')]:",
    "    u = df[df['predicted_label']=='UNKNOWN']",
    "    if len(u) > 0: unk.append(u[['index','true_label','type','section_id']].assign(model=name))",
    "if unk:",
    "    unk_df = pd.concat(unk, ignore_index=True)",
    "    display(unk_df.groupby('model').size().to_frame('nb_UNKNOWN'))",
    "    unk_df.to_csv(FIGURES_DIR / 'cas_unknown_compare.csv', index=False)",
    "    print('Exporté:', FIGURES_DIR / 'cas_unknown_compare.csv')",
    "else: print('Aucun cas UNKNOWN.')",
]))

# --- 12. Quand le modèle réussit ---
cells.append(md(["## 8. Quand le modèle réussit le mieux (accord)"]))

cells.append(code([
    "both_ok_p1 = df_bl_1['is_correct'] & df_ft_1['is_correct']",
    "both_ok_p2 = df_bl_2['is_correct'] & df_ft_2['is_correct']",
    "fig, ax = plt.subplots(figsize=(8, 4))",
    "ax.bar(['P1 les deux corrects', 'P1 les deux en erreur', 'P2 les deux corrects', 'P2 les deux en erreur'],",
    "       [both_ok_p1.sum(), (~df_bl_1['is_correct'] & ~df_ft_1['is_correct']).sum(), both_ok_p2.sum(), (~df_bl_2['is_correct'] & ~df_ft_2['is_correct']).sum()],",
    "       color=['#2ecc71','#e74c3c','#2ecc71','#e74c3c'])",
    "ax.set_ylabel(\"Nombre d'exemples\"); ax.set_title('Accord Baseline/Finetuné par prompt')",
    "plt.xticks(rotation=15, ha='right'); plt.tight_layout()",
    "fig.savefig(FIGURES_DIR / '08_accord_par_prompt.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
]))

# --- 13. Régressions ---
cells.append(md(["## 9. Régressions (baseline correct, finetuné incorrect)"]))

cells.append(code([
    "reg1 = df_bl_1[df_bl_1['is_correct'] & ~df_ft_1['is_correct']]",
    "reg2 = df_bl_2[df_bl_2['is_correct'] & ~df_ft_2['is_correct']]",
    "tab_reg = pd.DataFrame({'Prompt 1': [len(reg1)], 'Prompt 2': [len(reg2)]})",
    "display(tab_reg)",
    "if len(reg1) > 0 or len(reg2) > 0:",
    "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
    "    if len(reg1) > 0: reg1.groupby('section_id').size().plot(kind='bar', ax=axes[0], color='#e74c3c'); axes[0].set_title('Régressions P1')",
    "    if len(reg2) > 0: reg2.groupby('section_id').size().plot(kind='bar', ax=axes[1], color='#e74c3c'); axes[1].set_title('Régressions P2')",
    "    axes[1].tick_params(axis='x', rotation=20)",
    "    plt.tight_layout()",
    "    fig.savefig(FIGURES_DIR / '09_regressions.png', dpi=150, bbox_inches='tight')",
    "    plt.show()",
]))

# --- 14. Améliorations ---
cells.append(md(["## 10. Améliorations (baseline incorrect, finetuné correct)"]))

cells.append(code([
    "imp1 = df_bl_1[~df_bl_1['is_correct'] & df_ft_1['is_correct']]",
    "imp2 = df_bl_2[~df_bl_2['is_correct'] & df_ft_2['is_correct']]",
    "tab_imp = pd.DataFrame({'Prompt 1': [len(imp1)], 'Prompt 2': [len(imp2)]})",
    "display(tab_imp)",
    "if len(imp1) > 0 or len(imp2) > 0:",
    "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
    "    if len(imp1) > 0: imp1.groupby('section_id').size().plot(kind='bar', ax=axes[0], color='#2ecc71'); axes[0].set_title('Améliorations P1')",
    "    if len(imp2) > 0: imp2.groupby('section_id').size().plot(kind='bar', ax=axes[1], color='#2ecc71'); axes[1].set_title('Améliorations P2')",
    "    axes[1].tick_params(axis='x', rotation=20)",
    "    plt.tight_layout()",
    "    fig.savefig(FIGURES_DIR / '10_ameliorations.png', dpi=150, bbox_inches='tight')",
    "    plt.show()",
]))

# --- 15. Exemples ---
cells.append(md(["## 11. Exemples de régressions et d'améliorations"]))

cells.append(code([
    "N_EX = 3",
    "for (reg, imp, df_ft, name) in [(reg1, imp1, df_ft_1, 'Prompt 1'), (reg2, imp2, df_ft_2, 'Prompt 2')]:",
    "    print(f'=== {name} ===')",
    "    print('--- Régressions (baseline correct, finetuné faux) ---')",
    "    for _, row in reg.head(N_EX).iterrows():",
    "        idx = row['index']",
    "        ft_pred = df_ft.loc[df_ft['index'] == idx, 'predicted_label'].iloc[0]",
    "        print(f\"  index {idx}: Vrai={row['true_label']} | Baseline={row['predicted_label']} | Finetuné={ft_pred}\")",
    "    print('--- Améliorations (baseline faux, finetuné correct) ---')",
    "    for _, row in imp.head(N_EX).iterrows():",
    "        idx = row['index']",
    "        ft_pred = df_ft.loc[df_ft['index'] == idx, 'predicted_label'].iloc[0]",
    "        print(f\"  index {idx}: Vrai={row['true_label']} | Baseline={row['predicted_label']} | Finetuné={ft_pred}\")",
    "    print()",
]))

# --- 16. Synthèse ---
cells.append(md(["## 12. Synthèse pour le rapport"]))

cells.append(code([
    "lines = [",
    "    '# Synthèse comparaison Prompt 1 vs Prompt 2',",
    "    '',",
    "    '## Accuracy globale',",
    "    f\"P1 Baseline: {df_bl_1['is_correct'].mean():.2%} | P1 Finetuné: {df_ft_1['is_correct'].mean():.2%}\",",
    "    f\"P2 Baseline: {df_bl_2['is_correct'].mean():.2%} | P2 Finetuné: {df_ft_2['is_correct'].mean():.2%}\",",
    "    '',",
    "    '## Régressions / Améliorations',",
    "    f\"P1 Régressions: {len(reg1)} | P1 Améliorations: {len(imp1)}\",",
    "    f\"P2 Régressions: {len(reg2)} | P2 Améliorations: {len(imp2)}\",",
    "]",
    "synthese_path = FIGURES_DIR / 'synthese_compare.txt'",
    "synthese_path.write_text('\\n'.join(lines), encoding='utf-8')",
    "print('\\n'.join(lines))",
    "print(f'\\n>>> Synthèse sauvegardée: {synthese_path}')",
]))

# --- 17. Export ---
cells.append(md(["## 13. Export des résultats détaillés"]))

cells.append(code([
    "export = df_bl_1[['index','type','section_id']].copy()",
    "export['P1_bl_correct'] = df_bl_1['is_correct'].values",
    "export['P1_ft_correct'] = df_ft_1['is_correct'].values",
    "export['P2_bl_correct'] = df_bl_2['is_correct'].values",
    "export['P2_ft_correct'] = df_ft_2['is_correct'].values",
    "export['prompt_len_p1'] = df_bl_1['prompt_len'].values",
    "export['prompt_len_p2'] = df_bl_2['prompt_len'].values",
    "out_csv = FIGURES_DIR / 'error_analysis_compare.csv'",
    "export.to_csv(out_csv, index=False)",
    "print(f'Résultats détaillés sauvegardés: {out_csv}')",
    "print(f'Figures: {FIGURES_DIR}')",
]))

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(__file__).resolve().parent / "compare_error.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print("Notebook écrit:", out_path)
