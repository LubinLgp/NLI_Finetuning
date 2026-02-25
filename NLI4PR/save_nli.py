import pandas as pd
import json
import os

# 1. Définition du prompt système (Baseline NLI)
SYSTEM_PROMPT = "Classify the relationship between the premise and the hypothesis. Respond with only one word: 'Entailment' or 'Contradiction'."

# 2. Chemins vers les fichiers d'entrée Parquet 
# (Vérifie que le dossier "data/" correspond bien à ton arborescence, sinon enlève "data/")
PARQUET_FILES = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet'
}

def create_jsonl(df, output_path, hypothesis_col):
    """
    Fonction qui convertit un DataFrame en fichier JSONL avec la structure OpenAI.
    :param df: Le DataFrame contenant les données Parquet
    :param output_path: Le nom du fichier de sortie (ex: baseline_nli_train_pol.jsonl)
    :param hypothesis_col: La colonne à utiliser pour l'hypothèse ('statement_pol' ou 'statement_medical')
    """
    jsonl_data = []
    
    for _, row in df.iterrows():
        # Extraction des données en gérant les valeurs nulles
        premise = str(row['premise']) if pd.notna(row['premise']) else ""
        hypothesis = str(row[hypothesis_col]) if pd.notna(row[hypothesis_col]) else ""
        label = str(row['label']) if pd.notna(row['label']) else ""
        
        # Construction du message utilisateur
        user_content = f"Premise: {premise}\n\nHypothesis: {hypothesis}"
        
        # Création de la structure attendue par OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label}
        ]
        
        jsonl_data.append({"messages": messages})
    
    # Écriture dans le fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"  ✅ Sauvegardé : {output_path} ({len(jsonl_data)} lignes basées sur '{hypothesis_col}')")


def main():
    print("🚀 Lancement de la génération des 6 fichiers Baseline NLI...")
    print(f"📝 Prompt utilisé : '{SYSTEM_PROMPT}'\n")
    
    # Parcours des 3 jeux de données (train, validation, test)
    for split_name, file_path in PARQUET_FILES.items():
        if os.path.exists(file_path):
            print(f"📂 Traitement de {split_name} ({file_path})...")
            df = pd.read_parquet(file_path)
            
            # 1er fichier : Version POL (vulgarisée)
            output_pol = f"nli_{split_name}_pol.jsonl"
            create_jsonl(df, output_pol, 'statement_pol')
            
            # 2ème fichier : Version MEDICALE (experte)
            output_medical = f"nli_{split_name}_medical.jsonl"
            create_jsonl(df, output_medical, 'statement_medical')
            print("-" * 50)
            
        else:
            print(f"❌ Attention : Le fichier {file_path} est introuvable. Vérifiez le chemin.")
            print("-" * 50)
            
    print("🎉 Terminé ! Vos 6 fichiers d'entraînement sont prêts.")

if __name__ == "__main__":
    main()