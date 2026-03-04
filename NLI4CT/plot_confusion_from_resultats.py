#!/usr/bin/env python3
"""
Génère la figure des 4 matrices de confusion (P1 Baseline, P1 Fine-tuné, P2 Baseline, P2 Fine-tuné)
à partir des fichiers resultats_* des dossiers Prompt 1 et Prompt 2.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NLI4CT_ROOT = Path(__file__).resolve().parent
PROMPT1 = NLI4CT_ROOT / "results" / "Prompt 1"
PROMPT2 = NLI4CT_ROOT / "results" / "Prompt 2"
REPORT_DIR = NLI4CT_ROOT.parent / "CORIA_TALN_2026___Finetuning_NLI"

FILES = [
    (PROMPT1 / "resultats_bl_qwen7b_NLI4CT_prompt1", "P1 Baseline"),
    (PROMPT1 / "resultats_ft_qwen7b_NLI4CT_prompt1", "P1 Fine-tuné"),
    (PROMPT2 / "resultats_bl_qwen7b_NLI4CT_prompt2.out", "P2 Baseline"),
    (PROMPT2 / "resultats_ft_qwen7b_NLI4CT_prompt2.out", "P2 Fine-tuné"),
]


def parse_confusion_matrix(path: Path) -> np.ndarray:
    """Extrait la matrice 2x2 du bloc '--- Matrice de Confusion ---'."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # "Vrai Entail.  :               107 |                139"
    # "Vrai Contrad. :                25 |                222"
    pattern = re.compile(
        r"Vrai (?:Entail\.|Contrad\.)\s*:\s*(\d+)\s*\|\s*(\d+)"
    )
    matches = pattern.findall(text)
    if len(matches) != 2:
        raise ValueError(f"Expected 2 matrix lines in {path}, got {len(matches)}")
    row_entail = [int(matches[0][0]), int(matches[0][1])]   # Prédit Entail, Prédit Contrad
    row_contrad = [int(matches[1][0]), int(matches[1][1])]
    return np.array([row_entail, row_contrad])


def main():
    labels = ["Entailment", "Contradiction"]
    matrices = []
    for filepath, _ in FILES:
        if filepath.exists():
            matrices.append(parse_confusion_matrix(filepath))
    vmax = max(m.max() for m in matrices) if matrices else 300

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes_flat = axes.ravel()
    im_ref = None

    for ax, (filepath, title) in zip(axes_flat, FILES):
        if not filepath.exists():
            ax.text(0.5, 0.5, f"Fichier absent:\n{filepath.name}", ha="center", va="center")
            ax.set_title(title)
            continue
        cm = parse_confusion_matrix(filepath)
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        if im_ref is None:
            im_ref = im
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")
        ax.set_title(title)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im_ref, cax=cbar_ax, label="Effectif")
    out = REPORT_DIR / "fig_nli4ct_confusion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure enregistrée : {out}")


if __name__ == "__main__":
    main()
