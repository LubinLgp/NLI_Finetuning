Figures NLI4PR pour le rapport LaTeX CORIA-TALN 2026
====================================================

Ces figures sont générées par les notebooks du projet NLI4PR. Pour les obtenir :

1. Exécuter les notebooks suivants (depuis NLI_Finetuning/NLI4PR/) :

   - results/compare_all_baseline.ipynb
   - results/compare_all_finetuning.ipynb
   - results/Prompt_NLI/compare_nli_baseline_finetune.ipynb
   - results/Prompt_clinical_matching/compare_clinical_matching_baseline_finetune.ipynb
   - results/Prompt_cot/compare_cot_baseline_finetune.ipynb
   - results/Prompt_cot/analyse_cot_finetune.ipynb  (pour la matrice de confusion)
   - results/analyse_finetuning.ipynb  (analyse d'erreurs 123 candidats -> NLI4PR/graphes/)

2. Lancer le script de copie (depuis ce dossier ou la racine CORIA) :
   - Windows : copy_figures_nli4pr.bat
   - Ou copier à la main les PNG listés ci-dessous.

Fichiers attendus dans ce dossier (figures_nli4pr/) :
----------------------------------------------------
- compare_all_baseline_accuracy.png       <- results/graphes/
- compare_all_finetuning_accuracy.png     <- results/graphes/
- nli_compare_baseline_finetune_accuracy.png           <- results/Prompt_NLI/graphes/
- clinical_matching_compare_baseline_finetune_accuracy.png <- results/Prompt_clinical_matching/graphes/
- cot_compare_baseline_finetune_accuracy.png           <- results/Prompt_cot/graphes/
- cot_finetune_confusion_combined.png     <- results/Prompt_cot/graphes/ ou results/graphes/
- comparison_candidats_vs_autres_pol.png  <- NLI4PR/graphes/
- comparison_candidats_vs_autres_medical.png
- comparison_detailed_pol.png
- comparison_detailed_medical.png
- boxplots_pol.png
- boxplots_medical.png

Après copie, recompiler le PDF (pdflatex x2 ou compile_pdf.sh).
