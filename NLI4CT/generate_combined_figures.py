#!/usr/bin/env python3
"""Génère les 3 figures macro F1 (par type, section, longueur) et les copie dans le rapport."""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score

NLI4CT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR_1 = NLI4CT_ROOT / "results" / "Prompt 1"
RESULTS_DIR_2 = NLI4CT_ROOT / "results" / "Prompt 2"
FIGURES_DIR = NLI4CT_ROOT / "results" / "compare_figures"
REPORT_DIR = NLI4CT_ROOT.parent / "CORIA_TALN_2026___Finetuning_NLI"
GOLD_TEST_JSON = NLI4CT_ROOT / "Gold_test.json"
GOLD_JSONL_1 = RESULTS_DIR_1 / "Gold_test_formatted_old_prompt.jsonl" if (RESULTS_DIR_1 / "Gold_test_formatted_old_prompt.jsonl").exists() else RESULTS_DIR_1 / "Gold_test_formatted.jsonl"
GOLD_JSONL_2 = RESULTS_DIR_2 / "Gold_test_formatted.jsonl"

def extract_premise_hypothesis(text):
    if "HYPOTHESIS:" not in text:
        return text[:800] if len(text) > 800 else text, text[:500] if len(text) > 500 else text
    before, after = text.split("HYPOTHESIS:", 1)
    premise = before.replace("PREMISE:", "").strip()
    for sep in (".\n\nAnswer", "?\n\nAnswer", "? Answer", "\n\nAnswer"):
        if sep in after:
            after = after.split(sep)[0]
    return premise, after.strip()

def load_one_prompt(csv_bl, csv_ft, jsonl_path):
    bl = pd.read_csv(csv_bl)
    ft = pd.read_csv(csv_ft)
    for df in (bl, ft):
        if df["is_correct"].dtype == object:
            df["is_correct"] = df["is_correct"].apply(lambda x: str(x).strip().lower() == "true")
    meta_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if len(msgs) < 2:
                continue
            user_content = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
            prem, stmt = extract_premise_hypothesis(user_content)
            meta_list.append({"index": idx, "premise_jsonl": prem, "statement": stmt})
    df_meta = pd.DataFrame(meta_list)
    bl = bl.merge(df_meta, on="index", how="left")
    ft = ft.merge(df_meta, on="index", how="left")
    for df in (bl, ft):
        df["premise"] = df["premise_jsonl"].astype(str)
        df["hypothesis"] = df["statement"].astype(str)
        df["prompt_len"] = df["premise"].str.len() + df["hypothesis"].str.len()

    def safe_bin(s):
        q = s.quantile([0, 1 / 3, 2 / 3, 1])
        e = np.sort(q.unique())
        n = max(1, len(e) - 1)
        if n == 1:
            return pd.Series("Moyen", index=s.index)
        return pd.cut(s, bins=e, labels=["Court", "Moyen", "Long"][:n], include_lowest=True)

    for df in (bl, ft):
        df["length_bin"] = safe_bin(df["prompt_len"])
    return bl, ft

def df_for_f1(df):
    pred = df["predicted_label"].astype(str).str.strip().str.upper()
    return df[pred != "UNKNOWN"].copy()

def macro_f1_by(df, col):
    df = df_for_f1(df)
    def f1_group(g):
        if len(g) < 2:
            return np.nan
        return f1_score(g["true_label"], g["predicted_label"], average="macro", zero_division=0)

    res = df.groupby(col, observed=True).apply(f1_group, include_groups=False)
    n = df.groupby(col, observed=True).size()
    return pd.DataFrame({"n": n, "macro_f1": res}).reset_index().set_index(col)

def main():
    csv_bl_1 = list(RESULTS_DIR_1.glob("pred_bl_*.csv"))[0]
    csv_ft_1 = list(RESULTS_DIR_1.glob("pred_ft_*.csv"))[0]
    csv_bl_2 = list(RESULTS_DIR_2.glob("pred_bl_*.csv"))[0]
    csv_ft_2 = list(RESULTS_DIR_2.glob("pred_ft_*.csv"))[0]

    df_bl_1, df_ft_1 = load_one_prompt(csv_bl_1, csv_ft_1, GOLD_JSONL_1)
    df_bl_2, df_ft_2 = load_one_prompt(csv_bl_2, csv_ft_2, GOLD_JSONL_2)

    if GOLD_TEST_JSON.exists():
        with open(GOLD_TEST_JSON, "r", encoding="utf-8") as f:
            gold = json.load(f)
        keys = list(gold.keys())
        if len(keys) == len(df_bl_1):
            type_sec = pd.DataFrame(
                [
                    {
                        "index": i,
                        "type": gold[keys[i]].get("Type", "N/A"),
                        "section_id": gold[keys[i]].get("Section_id", "N/A"),
                    }
                    for i in range(len(keys))
                ]
            )
            for df in (df_bl_1, df_ft_1, df_bl_2, df_ft_2):
                df["type"] = type_sec.set_index("index").reindex(df["index"])["type"].values
                df["section_id"] = type_sec.set_index("index").reindex(df["index"])["section_id"].values

    # --- Figure 02: type ---
    t1_bl = macro_f1_by(df_bl_1, "type")
    t1_ft = macro_f1_by(df_ft_1, "type")
    t2_bl = macro_f1_by(df_bl_2, "type")
    t2_ft = macro_f1_by(df_ft_2, "type")
    tab_type = pd.DataFrame(
        {
            "P1 Baseline": t1_bl["macro_f1"],
            "P1 Finetuné": t1_ft["macro_f1"],
            "P2 Baseline": t2_bl["macro_f1"],
            "P2 Finetuné": t2_ft["macro_f1"],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = list(tab_type.columns)
    n_series = len(cols)
    w = 0.8 / max(n_series, 1)
    x = np.arange(len(tab_type.index))
    for i, c in enumerate(cols):
        x_pos = x + (i - n_series / 2 + 0.5) * w
        ax.bar(x_pos, tab_type[c], w, label=c)
        for j, v in enumerate(tab_type[c]):
            if pd.notna(v):
                ax.text(x_pos[j], v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tab_type.index)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.set_title("Macro F1 par type — les deux finetunings")
    plt.tight_layout()
    for d in (FIGURES_DIR, REPORT_DIR):
        fig.savefig(d / "02_accuracy_par_type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 03: section ---
    s1_bl = macro_f1_by(df_bl_1, "section_id")
    s1_ft = macro_f1_by(df_ft_1, "section_id")
    s2_bl = macro_f1_by(df_bl_2, "section_id")
    s2_ft = macro_f1_by(df_ft_2, "section_id")
    tab_sec = pd.DataFrame(
        {
            "P1 Baseline": s1_bl["macro_f1"],
            "P1 Finetuné": s1_ft["macro_f1"],
            "P2 Baseline": s2_bl["macro_f1"],
            "P2 Finetuné": s2_ft["macro_f1"],
        }
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    cols = list(tab_sec.columns)
    n_series = len(cols)
    w = 0.8 / max(n_series, 1)
    x = np.arange(len(tab_sec.index))
    for i, c in enumerate(cols):
        x_pos = x + (i - n_series / 2 + 0.5) * w
        ax.bar(x_pos, tab_sec[c], w, label=c)
        for j, v in enumerate(tab_sec[c]):
            if pd.notna(v):
                ax.text(x_pos[j], v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tab_sec.index, rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.set_title("Macro F1 par section — les deux finetunings")
    plt.tight_layout()
    for d in (FIGURES_DIR, REPORT_DIR):
        fig.savefig(d / "03_accuracy_par_section.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 04: length ---
    l1_bl = macro_f1_by(df_bl_1, "length_bin")
    l1_ft = macro_f1_by(df_ft_1, "length_bin")
    l2_bl = macro_f1_by(df_bl_2, "length_bin")
    l2_ft = macro_f1_by(df_ft_2, "length_bin")
    idx_len = l1_bl.index.union(l2_bl.index).union(l1_ft.index).union(l2_ft.index).unique()
    tab_len = pd.DataFrame(index=idx_len, columns=["P1 Baseline", "P1 Finetuné", "P2 Baseline", "P2 Finetuné"])
    for col, ser in [
        ("P1 Baseline", l1_bl["macro_f1"]),
        ("P1 Finetuné", l1_ft["macro_f1"]),
        ("P2 Baseline", l2_bl["macro_f1"]),
        ("P2 Finetuné", l2_ft["macro_f1"]),
    ]:
        tab_len[col] = ser
    tab_len = tab_len.fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = [c for c in tab_len.columns if c in tab_len]
    n_series = len(cols)
    w = 0.8 / max(n_series, 1)
    x = np.arange(len(tab_len))
    for i, c in enumerate(cols):
        x_pos = x + (i - n_series / 2 + 0.5) * w
        ax.bar(x_pos, tab_len[c], w, label=c)
        for j, v in enumerate(tab_len[c]):
            if pd.notna(v) and v != 0:
                ax.text(x_pos[j], v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tab_len.index)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.set_title("Macro F1 par longueur — les deux finetunings")
    plt.tight_layout()
    for d in (FIGURES_DIR, REPORT_DIR):
        fig.savefig(d / "04_accuracy_par_longueur.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Figures générées et copiées dans", FIGURES_DIR, "et", REPORT_DIR)

if __name__ == "__main__":
    main()
