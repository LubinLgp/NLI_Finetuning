# Comparaison : analyse univariée Eligibility (NLI4CT) vs analyse linguistique (NLI4PR)

## 1. Ce qu’on compare

| | NLI4CT (fin du notebook `analyse_linguistique_4_cas.ipynb`) | NLI4PR (`analyse_finetuning.ipynb`) |
|---|---|---|
| **Population** | Section Eligibility uniquement (132 ex.). Comparaison : modèle a raison vs modèle se trompe. | Tout le jeu de test (POL + MEDICAL). Comparaison : **candidats** (123 indices où le CoT finetuné a tort sur POL et MEDICAL) vs **autres** (reste du test). |
| **Question** | Quand KATE / P1 se trompent sur Eligibility, à quoi ressemblent les exemples (longueur, numérique, négations) ? | À quoi ressemblent les exemples où le modèle se trompe sur les deux variantes (candidats) par rapport au reste ? |

---

## 2. Tendances observées

### NLI4CT (Eligibility, univariée)

- **Quand KATE se trompe** (vs quand KATE a raison) : plus de négations (3,23 vs 1,79), prémisse plus longue (195,6 vs 159 mots), densité numérique un peu plus élevée (17,8 vs 15,3).
- **Quand P1 se trompe** (vs quand P1 a raison) : prémisse beaucoup plus longue (320,4 vs 105,2 mots), densité numérique plus forte (25,9 vs 11,8), beaucoup plus de négations (4,44 vs 1,34).

En résumé sur NLI4CT Eligibility : **les erreurs sont associées à des prémisses plus longues, une densité numérique plus élevée et plus de négations.**

### NLI4PR (candidats vs autres)

- **Candidats** (erreurs communes POL+MEDICAL) : prémisses **plus courtes** (≈198 vs 252 mots), densité numérique **plus faible** (≈30,9–32,8 vs 37,3–40,1), couverture lexicale plus basse.
- En MEDICAL : plus de négations dans le statement pour les candidats (1,72 vs 1,42).

En résumé sur NLI4PR : **les erreurs (candidats) sont associées à des prémisses plus courtes et une densité numérique plus faible** ; négations plus élevées seulement en MEDICAL (dans le statement).

---

## 3. Même tendance ou pas ?

**Non.** Les tendances sont **inversées** pour la longueur et la densité numérique :

- **Longueur de la prémisse** : sur NLI4CT (Eligibility), les erreurs vont avec des **prémisses plus longues** ; sur NLI4PR, les erreurs (candidats) vont avec des **prémisses plus courtes**.
- **Densité numérique** : sur NLI4CT, les erreurs vont avec une **densité plus élevée** ; sur NLI4PR, avec une **densité plus faible**.
- **Négations** : les deux analyses associent davantage de négations aux cas difficiles (NLI4CT : neg_total ; NLI4PR : surtout neg_statement en MEDICAL), mais les définitions des métriques ne sont pas identiques (voir ci‑dessous).

---

## 4. Pourquoi des différences ?

### 4.1 Contexte et définition des “erreurs”

- **NLI4CT** : une seule section (Eligibility), une seule tâche NLI (prémisse = extrait de protocole, hypothèse = phrase à évaluer). On compare “KATE/P1 a raison” vs “KATE/P1 se trompe” sur ces 132 ex.
- **NLI4PR** : tout le test, tâche CoT (prémisse = critères d’essai, statement = énoncé patient POL ou MEDICAL). Les “candidats” sont les indices où le modèle a tort **à la fois** en POL et en MEDICAL. La “prémisse” dans NLI4PR = critères (souvent longs) ; ce n’est pas la même chose que la prémisse NLI4CT.

Donc : **population, tâche et rôle de la “prémisse”** ne sont pas les mêmes, ce qui peut à lui seul expliquer des profils opposés (ex. sur NLI4PR, les critères courts ou peu numériques pourraient être plus ambigus ou plus difficiles).

### 4.2 Métriques pas calculées de la même façon

| Métrique | NLI4CT (`analyse_linguistique_4_cas.ipynb`) | NLI4PR (`analyse_finetuning.ipynb`) |
|----------|--------------------------------------------|-------------------------------------|
| **numeric_total** | `chiffres + pct + unités + **nombres en toutes lettres**` (e.g. one, two, hundred, thousand). `numeric_density` retourne **4** valeurs ; la 4ᵉ est `words_num`. | `chiffres + pct + unités` uniquement. `numeric_density` retourne **3** valeurs. Pas de comptage des nombres en toutes lettres. |
| **neg_total / neg_presence** | Tokenisation (NLTK si dispo, sinon regex). Liste de mots anglais (not, no, never, none, …) + tokens se terminant par `n't`. | Regex uniquement : liste français + anglais (non, aucun, sans, ni, jamais, pas, n'est, not, no, never, cannot, can't, etc.). Pas de tokenisation, pas de règle spécifique pour `n't`. |
| **Jaccard / coverage** | Même idée (intersection/union, coverage = part des mots de l’hypothèse dans la prémisse). Tokenisation : `[a-zA-Z0-9À-ÿ]+` en minuscules. | Même idée, même type de tokenisation. |
| **words_premise** | Nombre de tokens alphanumériques (prémisse = extrait de protocole). | Idem (prémisse = critères d’essai). |

Conséquences :

- **Densité numérique** : les valeurs ne sont **pas directement comparables** entre NLI4CT et NLI4PR (NLI4CT inclut les nombres en lettres, NLI4PR non). Les ordres de grandeur et les écarts “erreurs vs corrects” peuvent diverger.
- **Négations** : même nom de métrique mais **définition différente** (anglais + n't vs français+anglais regex, avec ou sans tokenisation). Les niveaux et les écarts ne sont pas strictement comparables.

---

## 5. Conclusion

- **Tendances** : on **ne** observe **pas** les mêmes tendances entre les deux analyses : sur NLI4CT (Eligibility), les erreurs vont avec prémisses plus longues et densité numérique plus forte ; sur NLI4PR, les erreurs (candidats) vont avec prémisses plus courtes et densité numérique plus faible. Les négations sont liées aux difficultés dans les deux cas, mais avec des métriques différentes.
- **Causes possibles** : (1) différence de tâche et de population (Eligibility seule vs tout NLI4PR ; NLI vs CoT ; “prémisse” = protocole vs critères) ; (2) **métriques non identiques** (numeric_total avec ou sans nombres en lettres ; négations avec des listes et des implémentations différentes).
- **Recommandation** : pour comparer de façon propre NLI4CT et NLI4PR, il faudrait soit (a) recalculer les métriques NLI4PR avec les mêmes formules que NLI4CT (numeric_total avec nombres en lettres, negation_presence avec NLTK + liste anglaise + n't), soit (b) documenter explicitement les différences de calcul dans le rapport et interpréter les comparaisons avec prudence (profils de difficulté différents selon le jeu et la tâche, plutôt qu’une même “règle” linguistique).
