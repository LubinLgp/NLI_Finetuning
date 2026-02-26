# Interprétation des résultats — Analyse linguistique des 4 cas

## Données utilisées

- **Les 2 corrects** : 299 exemples  
- **Les 2 en erreur** : 71 exemples  
- **P1✓ KATE✗** (finetuning seul a raison) : 72 exemples  
- **P1✗ KATE✓** (KATE seul a raison) : 58 exemples  

---

## 1. Résultats significatifs (tests statistiques)

### Différence entre les 4 groupes (Kruskal-Wallis)

| Feature            | p-value | Significatif ? |
|--------------------|--------|----------------|
| lexical_jaccard    | 0.99   | Non            |
| lexical_coverage   | 0.12   | Non            |
| **numeric_total**  | **0.017** | **Oui**     |
| neg_total          | 0.10   | Non (proche)   |
| **words_premise**  | **0.002** | **Oui**     |
| words_hypothesis   | 0.05   | Limite         |

**Conclusion globale** : Les 4 cas se distinguent surtout par la **densité numérique** et la **longueur de la prémisse**. Le chevauchement lexical (Jaccard, coverage) ne varie pas de façon significative entre les groupes.

---

## 2. Interprétations ciblées

### 2.1 Les 2 en erreur vs Les 2 corrects

- **Négations (neg_total)** : **significatif** (p = 0,015).  
  Les cas où **les deux modèles se trompent** ont en moyenne **plus de négations** (2,31 vs 0,82).  
  **Interprétation** : Les deux approches (finetuning et few-shot) échouent plus souvent quand la contradiction passe par des **négations explicites** (ou un langage plus négatif). Les cas “faciles” (les 2 corrects) en ont moins.

- **Coverage lexical** : tendance (p = 0,066).  
  Quand les 2 se trompent, la part des mots de l’hypothèse présents dans la prémisse est légèrement plus élevée (0,38 vs 0,34).  
  **Interprétation** : Cohérent avec un piège classique NLI : **fort chevauchement lexical** alors que la réponse est Contradiction (contradiction subtile). La tendance n’est pas significative au seuil 5 % mais va dans ce sens.

- **Longueur prémisse** : tendance (p = 0,09).  
  Les erreurs partagées ont des **prémisses plus longues** (169 vs 113 mots).  
  **Interprétation** : Les cas où les deux se trompent sont un peu plus **complexes** (plus de texte à intégrer).

### 2.2 Complémentarité : P1✓ KATE✗ vs P1✗ KATE✓

- **Densité numérique (numeric_total)** : **significatif** (p = 0,014).  
  Les cas où **KATE a raison et le finetuning se trompe** ont une **densité numérique nettement plus élevée** (61,2 vs 42,9 : chiffres, %, unités).  
  **Interprétation** : Contrairement à l’hypothèse “few-shot mauvais en numérique”, ici **KATE réussit mieux** sur les exemples **très chargés en chiffres/seuils/unités** quand P1 échoue. Le finetuning (P1) semble plus souvent en échec sur ces cas numériques ; le few-shot avec exemples en contexte aide peut-être à réutiliser des patrons (dosages, critères d’âge, etc.).

- **Longueur de la prémisse (words_premise)** : **significatif** (p = 0,049).  
  Quand **KATE a raison et P1 se trompe**, les **prémisses sont plus longues** (221 vs 124 mots).  
  **Interprétation** : KATE (few-shot) tire peut-être parti du **contexte long** (plus d’infos dans la prémisse) grâce aux exemples en contexte. Le modèle finetuné pourrait être plus sensible à la longueur ou au bruit dans les longs textes.

### 2.3 Ce qui ne distingue pas les groupes

- **Chevauchement lexical (Jaccard)** : aucune différence significative entre les 4 cas (p ≈ 0,99).  
  **Interprétation** : Le piège “fort overlap → Entailment à tort” n’apparaît pas ici comme **facteur discriminant** entre les 4 sous-ensembles ; il peut quand même jouer au niveau d’exemples individuels.

- **Négations** : pas de différence significative entre **P1✓ KATE✗** et **P1✗ KATE✓** (p ≈ 0,92).  
  **Interprétation** : La complémentarité entre les deux modèles ne s’explique pas par une différence nette de **charge en négations** dans ces deux sous-ensembles.

---

## 3. Synthèse opérationnelle

| Question | Réponse |
|----------|--------|
| **Pourquoi les 2 se trompent parfois ?** | Davantage de **négations** (et tendance : prémisses plus longues, coverage un peu plus élevé). |
| **Où le finetuning (P1) est-il seul à réussir ?** | Sur des exemples **moins numériques** et **prémisses plus courtes** (profil “P1✓ KATE✗”). |
| **Où KATE est-il seul à réussir ?** | Sur des exemples **très numériques** (seuils, unités) et **prémisses plus longues** (profil “P1✗ KATE✓”). |
| **Le chevauchement lexical est-il discriminant ?** | Non, pas entre les 4 groupes. |
| **Les négations ?** | Oui pour **distinguer “2 en erreur” vs “2 corrects”** (plus de négations quand les 2 se trompent). Pas pour distinguer les deux cas de complémentarité. |

---

## 4. Recommandations

1. **Enseignement des seuils numériques** : Le finetuning pourrait être renforcé sur des exemples **riches en chiffres, pourcentages et unités** (critères d’éligibilité, dosages, âges) pour mieux rivaliser avec KATE sur ces cas.
2. **Gestion des négations** : Les deux modèles échouent plus quand il y a beaucoup de négations ; des données d’entraînement ou des exemples few-shot ciblant les **contradictions par négation** pourraient aider.
3. **Longueur de la prémisse** : KATE semble mieux exploiter les **longues prémisses** ; pour le finetuning, vérifier (architecture, fenêtre, pooling) qu’il utilise bien tout le contexte long.
4. **Ensemble / fallback** : Sur les cas **numériques** et **longue prémisse**, un mécanisme de type “préférer KATE” ou combiner les deux prédictions pourrait être testé.
