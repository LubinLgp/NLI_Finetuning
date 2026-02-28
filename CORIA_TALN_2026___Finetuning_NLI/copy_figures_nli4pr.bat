@echo off
REM Copie les figures NLI4PR vers figures_nli4pr/ pour le rapport LaTeX.
REM À lancer après avoir exécuté les notebooks NLI4PR (compare_all_*, compare_*_baseline_finetune, analyse_cot_finetune, analyse_finetuning).
REM Depuis le dossier CORIA_TALN_2026___Finetuning_NLI, NLI4PR est au même niveau (NLI_Finetuning/)
set SCRIPT_DIR=%~dp0
set NLI4PR_RESULTS=%SCRIPT_DIR%..\NLI4PR\results
set NLI4PR_GRAPHES=%SCRIPT_DIR%..\NLI4PR\graphes
set DEST=%SCRIPT_DIR%figures_nli4pr
if not exist "%DEST%" mkdir "%DEST%"

echo Copie des figures NLI4PR...
if exist "%NLI4PR_RESULTS%\graphes\compare_all_baseline_accuracy.png" (
  copy /Y "%NLI4PR_RESULTS%\graphes\compare_all_baseline_accuracy.png" "%DEST%\"
) else echo Manquant: compare_all_baseline_accuracy.png
if exist "%NLI4PR_RESULTS%\graphes\compare_all_finetuning_accuracy.png" (
  copy /Y "%NLI4PR_RESULTS%\graphes\compare_all_finetuning_accuracy.png" "%DEST%\"
) else echo Manquant: compare_all_finetuning_accuracy.png

if exist "%NLI4PR_RESULTS%\Prompt_NLI\graphes\nli_compare_baseline_finetune_accuracy.png" (
  copy /Y "%NLI4PR_RESULTS%\Prompt_NLI\graphes\nli_compare_baseline_finetune_accuracy.png" "%DEST%\"
) else echo Manquant: nli_compare_baseline_finetune_accuracy.png

if exist "%NLI4PR_RESULTS%\Prompt_clinical_matching\graphes\clinical_matching_compare_baseline_finetune_accuracy.png" (
  copy /Y "%NLI4PR_RESULTS%\Prompt_clinical_matching\graphes\clinical_matching_compare_baseline_finetune_accuracy.png" "%DEST%\"
) else echo Manquant: clinical_matching_compare_baseline_finetune_accuracy.png

if exist "%NLI4PR_RESULTS%\Prompt_cot\graphes\cot_compare_baseline_finetune_accuracy.png" (
  copy /Y "%NLI4PR_RESULTS%\Prompt_cot\graphes\cot_compare_baseline_finetune_accuracy.png" "%DEST%\"
) else echo Manquant: cot_compare_baseline_finetune_accuracy.png

if exist "%NLI4PR_RESULTS%\Prompt_cot\graphes\cot_finetune_confusion_combined.png" (
  copy /Y "%NLI4PR_RESULTS%\Prompt_cot\graphes\cot_finetune_confusion_combined.png" "%DEST%\"
) else if exist "%NLI4PR_RESULTS%\graphes\cot_finetune_confusion_combined.png" (
  copy /Y "%NLI4PR_RESULTS%\graphes\cot_finetune_confusion_combined.png" "%DEST%\"
) else echo Manquant: cot_finetune_confusion_combined.png

REM Figures analyse d'erreurs (123 candidats) — analyse_finetuning.ipynb -> NLI4PR/graphes/
echo Copie des figures analyse d'erreurs...
if exist "%NLI4PR_GRAPHES%\comparison_candidats_vs_autres_pol.png" (
  copy /Y "%NLI4PR_GRAPHES%\comparison_candidats_vs_autres_pol.png" "%DEST%\"
) else echo Manquant: comparison_candidats_vs_autres_pol.png
if exist "%NLI4PR_GRAPHES%\comparison_candidats_vs_autres_medical.png" (
  copy /Y "%NLI4PR_GRAPHES%\comparison_candidats_vs_autres_medical.png" "%DEST%\"
) else echo Manquant: comparison_candidats_vs_autres_medical.png
if exist "%NLI4PR_GRAPHES%\comparison_detailed_pol.png" (
  copy /Y "%NLI4PR_GRAPHES%\comparison_detailed_pol.png" "%DEST%\"
) else echo Manquant: comparison_detailed_pol.png
if exist "%NLI4PR_GRAPHES%\comparison_detailed_medical.png" (
  copy /Y "%NLI4PR_GRAPHES%\comparison_detailed_medical.png" "%DEST%\"
) else echo Manquant: comparison_detailed_medical.png
if exist "%NLI4PR_GRAPHES%\boxplots_pol.png" (
  copy /Y "%NLI4PR_GRAPHES%\boxplots_pol.png" "%DEST%\"
) else echo Manquant: boxplots_pol.png
if exist "%NLI4PR_GRAPHES%\boxplots_medical.png" (
  copy /Y "%NLI4PR_GRAPHES%\boxplots_medical.png" "%DEST%\"
) else echo Manquant: boxplots_medical.png

echo Termine. Verifiez le contenu de %DEST%\
