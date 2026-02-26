# Dossier analyse — Rapport des résultats NLI4CT

Ce dossier contient le rapport LaTeX synthétisant les résultats des notebooks d’analyse d’erreurs.

## Fichiers

- **report.tex** : source du rapport.
- **report.pdf** : rapport compilé (généré par `pdflatex report.tex`).

## Compilation

Depuis ce dossier (`results/analyse/`) :

```bash
pdflatex report.tex
pdflatex report.tex
```

Les figures sont chargées depuis les dossiers parents : `../compare_figures/`, `../Prompt 1/figures/`, `../Prompt 2/figures/`, `../Fewshot/figures/`. Ne pas déplacer le dossier `analyse` sans ces dossiers.

## Contenu du rapport

1. Accuracy globale (tous les modèles)
2. Résultats par type (Single / Comparison) et par section
3. Effet de la longueur du prompt
4. Régressions et améliorations (baseline vs finetuné, few-shot vs finetuné)
5. Patterns d’accord entre P1 Finetuné, P2 Finetuné et Few-shot
6. Comparaisons deux-à-deux
7. Cas « les 3 en erreur » / « les 3 corrects »
8. Matrices de confusion et erreurs par section
9. Synthèses par approche et conclusions
