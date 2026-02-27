#!/bin/bash
# Compile le PDF avec plusieurs passes pour que les \ref (figures, tableaux) 
# s'affichent correctement (évite les « ?? »).
# Usage: ./compile_pdf.sh   ou   bash compile_pdf.sh
cd "$(dirname "$0")"
TEX=coria-taln2026-example.tex
echo "=== 1re passe pdflatex ==="
pdflatex -interaction=nonstopmode "$TEX"
echo ""
echo "=== 2e passe pdflatex (résolution des \\ref) ==="
pdflatex -interaction=nonstopmode "$TEX"
echo ""
echo "=== 3e passe pdflatex (stabilisation liens / TOC) ==="
pdflatex -interaction=nonstopmode "$TEX"
echo "Terminé. Ouvrir: ${TEX%.tex}.pdf"
