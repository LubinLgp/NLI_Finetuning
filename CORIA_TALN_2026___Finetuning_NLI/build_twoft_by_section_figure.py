#!/usr/bin/env python3
"""
Génère une figure combinant les graphiques :
- 09_two_wrong_by_section.png  (cas où les 2 finetunings se trompent, par section)
- 10_two_ok_by_section.png     (cas où les 2 finetunings ont raison, par section)

Sortie : fig_twoft_by_section.png dans le répertoire courant (CORIA).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SCRIPT_DIR = Path(__file__).resolve().parent
NLI4CT_ROOT = SCRIPT_DIR.parent / "NLI4CT"
FIG_DIR = NLI4CT_ROOT / "results" / "compare_figures"


def main() -> None:
    wrong_path = FIG_DIR / "09_two_wrong_by_section.png"
    ok_path = FIG_DIR / "10_two_ok_by_section.png"

    if not wrong_path.exists() or not ok_path.exists():
        raise FileNotFoundError(f"Fichiers introuvables : {wrong_path} ou {ok_path}")

    img_wrong = mpimg.imread(wrong_path)
    img_ok = mpimg.imread(ok_path)

    # Figure verticale 2 lignes x 1 colonne
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    axes[0].imshow(img_wrong)
    axes[0].axis("off")

    axes[1].imshow(img_ok)
    axes[1].axis("off")

    plt.tight_layout()
    out = SCRIPT_DIR / "fig_twoft_by_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {out}")


if __name__ == "__main__":
    main()

