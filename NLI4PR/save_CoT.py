import pandas as pd
import json
import os

# Chemins vers les fichiers d'entrée Parquet 
PARQUET_FILES = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet'
}

def create_jsonl_cot(df, output_path, hypothesis_col):
    """
    Convertit le DataFrame en JSONL avec le prompt Chain of Thought (CoT).
    """
    jsonl_data = []
    
    for _, row in df.iterrows():
        # Extraction des données
        premise = str(row['premise']) if pd.notna(row['premise']) else ""
        hypothesis = str(row[hypothesis_col]) if pd.notna(row[hypothesis_col]) else ""
        label = str(row['label']) if pd.notna(row['label']) else ""
        
        # Construction du message utilisateur (avec le prompt CoT et "criteria")
        user_content = (
            f'Does the patient with the statement "{hypothesis}" satisfy the following clinical trial admission criteria ?\n'
            f'"{premise}"\n\n'
            f"First, explain your reasoning step-by-step by comparing the patient's characteristics to the inclusion and exclusion criteria.\n"
            f"Then, conclude on a new line with only one word: 'Entailment' or 'Contradiction'."
        )
        
        # Création de la structure attendue par OpenAI
        messages = [
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
    print("🚀 Lancement de la génération des 6 fichiers Chain of Thought (CoT)...")
    print("📝 Format : Prompt de raisonnement étape par étape.\n")
    
    # Parcours des 3 jeux de données
    for split_name, file_path in PARQUET_FILES.items():
        if os.path.exists(file_path):
            print(f"📂 Traitement de {split_name} ({file_path})...")
            df = pd.read_parquet(file_path)
            
            # 1er fichier : Version POL
            output_pol = f"cot_{split_name}_pol.jsonl"
            create_jsonl_cot(df, output_pol, 'statement_pol')
            
            # 2ème fichier : Version MEDICALE
            output_medical = f"cot_{split_name}_medical.jsonl"
            create_jsonl_cot(df, output_medical, 'statement_medical')
            print("-" * 50)
            
        else:
            print(f"❌ Attention : Le fichier {file_path} est introuvable.")
            print("-" * 50)
            
    print("🎉 Terminé ! Vos 6 fichiers 'Chain of Thought' sont prêts.")

if __name__ == "__main__":
    main()