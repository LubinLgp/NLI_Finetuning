@echo off
setlocal

rem Dossier courant = CORIA_TALN_2026___Finetuning_NLI
set SRC=..\NLI4PR
set DST=%~dp0figures_nli4pr

if not exist "%DST%" mkdir "%DST%"

echo Copie des figures NLI4PR dans %DST%

rem 1) Comparaison globale des 3 prompts (baseline / finetune)
copy "%SRC%\results\graphes\compare_all_baseline_accuracy.png" "%DST%\compare_all_baseline_accuracy.png" >nul
copy "%SRC%\results\graphes\compare_all_finetuning_accuracy.png" "%DST%\compare_all_finetuning_accuracy.png" >nul

rem 2) Comparaisons baseline vs finetune par prompt (macro F1)
copy "%SRC%\results\Prompt_NLI\graphes\prompt1_compare_baseline_finetune_macro_f1.png" "%DST%\prompt1_compare_baseline_finetune_macro_f1.png" >nul
copy "%SRC%\results\Prompt_clinical_matching\graphes\prompt2_compare_baseline_finetune_macro_f1.png" "%DST%\prompt2_compare_baseline_finetune_macro_f1.png" >nul
copy "%SRC%\results\Prompt_cot\graphes\prompt3_compare_baseline_finetune_macro_f1.png" "%DST%\prompt3_compare_baseline_finetune_macro_f1.png" >nul

rem 3) Matrice de confusion CoT finetune (NLI4PR)
copy "%SRC%\results\Prompt_cot\graphes\cot_finetune_confusion_combined.png" "%DST%\cot_finetune_confusion_combined.png" >nul

rem 4) Figures d'analyse linguistique NLI4PR (analyse_finetuning.ipynb)
copy "%SRC%\graphes\comparison_candidats_vs_autres_pol.png" "%DST%\comparison_candidats_vs_autres_pol.png" >nul
copy "%SRC%\graphes\comparison_candidats_vs_autres_medical.png" "%DST%\comparison_candidats_vs_autres_medical.png" >nul
copy "%SRC%\graphes\comparison_detailed_pol.png" "%DST%\comparison_detailed_pol.png" >nul
copy "%SRC%\graphes\comparison_detailed_medical.png" "%DST%\comparison_detailed_medical.png" >nul
copy "%SRC%\graphes\boxplots_pol.png" "%DST%\boxplots_pol.png" >nul
copy "%SRC%\graphes\boxplots_medical.png" "%DST%\boxplots_medical.png" >nul

echo Terminé.
endlocal