import pandas as pd
import sys

# =====================================================================
# CONFIGURATION DES CHEMINS (À MODIFIER SELON TES BESOINS)
# =====================================================================
POL_CSV_PATH = "Fewshot_kate/eval_fewshot_pol.csv"
MEDICAL_CSV_PATH = "Fewshot_kate/eval_fewshot_medical.csv"
OUTPUT_CSV_PATH = "Fewshot_kate/eval_fewshot_combined.csv"
# =====================================================================

def main():
    print("🚀 Démarrage de la fusion des résultats...")
    
    # 1. Chargement du fichier POL
    print(f"📂 Chargement du fichier POL : {POL_CSV_PATH}")
    try:
        df_pol = pd.read_csv(POL_CSV_PATH)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {POL_CSV_PATH} : {e}")
        sys.exit(1)

    # 2. Chargement du fichier MEDICAL
    print(f"📂 Chargement du fichier MEDICAL : {MEDICAL_CSV_PATH}")
    try:
        df_medical = pd.read_csv(MEDICAL_CSV_PATH)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {MEDICAL_CSV_PATH} : {e}")
        sys.exit(1)

    # 3. Vérification stricte du format (mêmes colonnes)
    if list(df_pol.columns) != list(df_medical.columns):
        print("\n❌ ERREUR FATALE : Les deux fichiers CSV n'ont pas les mêmes colonnes !")
        print(f"➡️ Colonnes POL     : {list(df_pol.columns)}")
        print(f"➡️ Colonnes MEDICAL : {list(df_medical.columns)}")
        print("Veuillez vérifier vos fichiers avant de relancer.")
        sys.exit(1)
    
    print("✅ Format validé : Les deux fichiers ont des colonnes identiques.")

    # 4. Ajout intelligent de la colonne "statement_type"
    # On l'insère juste après la colonne 'index' si elle existe, sinon à la fin
    if 'index' in df_pol.columns:
        idx_pos = df_pol.columns.get_loc('index') + 1
        df_pol.insert(idx_pos, 'statement_type', 'pol')
        df_medical.insert(idx_pos, 'statement_type', 'medical')
    else:
        df_pol['statement_type'] = 'pol'
        df_medical['statement_type'] = 'medical'

    # 5. Concaténation des deux DataFrames
    print("🔄 Fusion des données en cours...")
    df_combined = pd.concat([df_pol, df_medical], ignore_index=True)

    # 6. Sauvegarde du fichier final
    try:
        df_combined.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n🎉 SUCCÈS ! Fichier final sauvegardé sous : {OUTPUT_CSV_PATH}")
        print(f"📊 Lignes POL     : {len(df_pol)}")
        print(f"📊 Lignes MEDICAL : {len(df_medical)}")
        print(f"📈 Total combiné  : {len(df_combined)} lignes")
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde de {OUTPUT_CSV_PATH} : {e}")

if __name__ == "__main__":
    main()