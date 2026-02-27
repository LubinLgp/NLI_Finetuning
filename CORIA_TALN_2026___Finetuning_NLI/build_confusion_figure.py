#!/usr/bin/env python3
"""
Génère la figure des 4 matrices de confusion (P1 BL, P1 FT, P2 BL, P2 FT)
à partir des fichiers resultats_* du dossier NLI4CT/results/.
Sortie : fig_nli4ct_confusion.png dans le répertoire courant (CORIA).
"""
import re
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Répertoire du script = CORIA_TALN_2026___Finetuning_NLI
SCRIPT_DIR = Path(__file__).resolve().parent
NLI4CT_ROOT = SCRIPT_DIR.parent / "NLI4CT"
RESULTS_P1 = NLI4CT_ROOT / "results" / "Prompt 1"
RESULTS_P2 = NLI4CT_ROOT / "results" / "Prompt 2"


def find_file(base_name, directory):
    """Trouve le fichier avec ou sans extension .out"""
    p = directory / base_name
    if p.exists():
        return p
    p_out = directory / (base_name + ".out")
    if p_out.exists():
        return p_out
    return None


def parse_confusion_matrix(path):
    """
    Parse le bloc "--- Matrice de Confusion ---" dans un fichier resultats_*.
    Retourne une matrice 2x2 : rows = [Entailment, Contradiction], cols = [Entailment, Contradiction].
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    # Lignes type: "Vrai Entail.  :               107 |                139"
    #             "Vrai Contrad. :                25 |                222"
    mat = np.zeros((2, 2), dtype=int)
    for line in text.splitlines():
        line = line.strip()
        if "Vrai Entail" in line or "Vrai Entail." in line:
            m = re.findall(r"(\d+)\s*\|\s*(\d+)", line)
            if m:
                mat[0, 0], mat[0, 1] = int(m[0][0]), int(m[0][1])
        if "Vrai Contrad" in line or "Vrai Contrad." in line:
            m = re.findall(r"(\d+)\s*\|\s*(\d+)", line)
            if m:
                mat[1, 0], mat[1, 1] = int(m[0][0]), int(m[0][1])
    return mat


def main():
    files = [
        (find_file("resultats_bl_qwen7b_NLI4CT_prompt1", RESULTS_P1), "P1 Baseline"),
        (find_file("resultats_ft_qwen7b_NLI4CT_prompt1", RESULTS_P1), "P1 Fine-tuné"),
        (find_file("resultats_bl_qwen7b_NLI4CT_prompt2", RESULTS_P2), "P2 Baseline"),
        (find_file("resultats_ft_qwen7b_NLI4CT_prompt2", RESULTS_P2), "P2 Fine-tuné"),
    ]
    for path, _ in files:
        if path is None:
            raise FileNotFoundError(f"Fichier introuvable pour {_}")

    matrices = []
    for path, title in files:
        m = parse_confusion_matrix(path)
        matrices.append((m, title))

    # Figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    labels = ["Entailment", "Contradiction"]
    for ax, (mat, title) in zip(axes, matrices):
        # mat: row = true, col = predicted
        im = ax.imshow(mat, cmap="Reds", aspect="auto", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Vrai label")
        ax.set_xlabel("Prédit")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black", fontsize=12)
        ax.set_title(title)
    plt.tight_layout()
    out = SCRIPT_DIR / "fig_nli4ct_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {out}")


if __name__ == "__main__":
    main()
