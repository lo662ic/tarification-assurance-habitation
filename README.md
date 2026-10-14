# Tarification en assurance habitation

Projet académique, Master 2 Statistique et Économie du Risque (ex-ISEFAR), parcours Data Science et Actuariat, Université Paris Nanterre. Modélisation d'un portefeuille d'assurance habitation en Python : tarification (OLS, IV/2SLS, GLM Gamma), modèles censurés (Tobit, Heckman, Double-Hurdle), modèles de comptage (Poisson, Binomiale Négative, ZINB) et analyse de survie (Kaplan-Meier, Cox).

Tout le travail, avec ses résultats et ses interprétations, se trouve dans le notebook [`tarification_habitation.ipynb`](tarification_habitation.ipynb).

## Contexte

À partir d'un portefeuille de 5 352 ménages assurés en multirisque habitation, le projet répond à cinq questions :

1. Que révèle l'analyse descriptive, et quels problèmes statistiques conditionnent la modélisation ?
2. Comment modéliser le coût de sinistre, en tenant compte de l'hétéroscédasticité et d'une éventuelle endogénéité des variables de police ?
3. Comment modéliser un sinistre comportant une très forte proportion de zéros (modèles censurés) ?
4. Comment modéliser le nombre de sinistres ?
5. Quels facteurs expliquent la durée de souscription ?

## Données

Le jeu de données est fictif (cours) et inclus dans le dépôt avec l'accord de l'enseignant : `dat_clean.csv`, 5 352 ménages, 27 variables, sans valeur manquante.

| Variable | Description |
|---|---|
| `Sinistre0` | Montant de dommage cumulé sur la période (variable de coût totale) |
| `Sinistre1`, `Sinistre2`, `Sinistre3` | Montants de dommage par type de sinistre (types 1, 2 et 3, non précisés davantage) |
| `NSin` | Nombre de sinistres du ménage |
| `Police1`, `Police2`, `Police3` | Cotisations au titre des compléments de police de type 1, 2 et 3 |
| `Duree` | Durée de la police 1 (durée de souscription) |
| `censure` | Indicateur d'événement pour l'analyse de survie (1 = événement observé, 0 = contrat encore actif en fin d'observation), selon la convention du cours |
| `RUC` | Revenu par unité de consommation (les unités de consommation dépendent de l'âge des membres du ménage) |
| `reves`, `crevpp`, `cs` | Revenu du foyer (estimé par déclaration, codé par tranche), quartile de revenu par personne, catégorie sociale de revenu (aisée, moyenne supérieure, moyenne inférieure, modeste) |
| `pcs` | Catégorie socio-professionnelle (8 modalités, des agriculteurs exploitants aux retraités et autres inactifs) |
| `region` | Région, codée de 1 à 9 (1 = Île-de-France) |
| `Ahabi` | Taille d'agglomération (libellé « habitat » dans le dictionnaire de variables) : communes rurales, unités urbaines de 2 000 à 9 999, de 10 000 à 99 999 et de 100 000 habitants et plus, Paris et agglomération |
| `Atyph` | Statut d'occupation (propriétaire, locataire, non déclaré) |
| `agecat` | Classe d'âge du chef de ménage (21-40, 41-50, 51-60, 61-96) |
| `Acompm`, `nbpers`, `enfants`, `Anat`, `Bauto` | Composition du ménage, nombre de personnes, présence d'enfants, nationalité du ménage, possession d'un véhicule |

Les variables `habi`, `Nbadulte`, `durPolice1` et `censure` ne figurent pas dans le dictionnaire de variables fourni avec le jeu de données. L'unité monétaire des montants n'est pas confirmée (le dictionnaire évoque des milliers d'euros, avec un point d'interrogation) : les commentaires portent sur des ordres de grandeur relatifs entre profils.

## Démarche et résultats

### Partie 1 : analyse descriptive

- `NSin` est fortement surdispersé (variance/moyenne = 3,42) : le Poisson est inadapté a priori.
- `Sinistre0` n'est pas normal (Shapiro-Wilk) et sa distribution est bimodale.
- `Sinistre1`, `Sinistre2` et `Sinistre3` comportent beaucoup de zéros (76,33 %, 89,63 % et 33,26 %).

### Partie 2 : coût de sinistre (`Sinistre0`)

- OLS sur log(Sinistre0), sans puis avec les variables de police. Hétéroscédasticité détectée (Breusch-Pagan) et corrigée par des écarts-types robustes (HC1).
- Endogénéité des polices testée par IV/2SLS (instruments : durée de souscription et catégorie de revenu par personne) : non détectée (Wu-Hausman, p = 0,18 ; Wooldridge robuste, p = 0,17) et instruments cohérents (Sargan, p = 0,86).
- Les polices ne sont pas significatives, individuellement ni conjointement (test F robuste, p = 0,29 ; rapport de vraisemblance du GLM, p = 0,25) : le modèle a priori est retenu.
- **Modèle final : GLM Gamma à lien log** (AIC = 24 385,66 ; RMSE = 2,15 ; MAE = 1,72, en échantillon). Seuls RUC et la composition du ménage sont significatifs, y compris sous écarts-types robustes : un couple avec enfants a un coût attendu inférieur de 44,5 % à la référence.
- Grille de relativités (exp des coefficients) et coût attendu sur cinq profils types.

### Partie 3 : sinistres censurés (`Sinistre2`)

`Sinistre2` est la variable la plus censurée (89,63 % de zéros ; 555 montants positifs).

- **Tobit** de type I codé à la main (maximum de vraisemblance, gradient analytique, BFGS) : log-vraisemblance = −2 664,82 ; AIC = 5 389,65 ; BIC = 5 587,21.
- **Heckman en deux étapes** : sans variable d'exclusion, le ratio de Mills est colinéaire avec les régresseurs (R² auxiliaire = 1,000). Le test de sélection est non concluant (p = 0,74).
- **Double-Hurdle** (probit pour la survenance, log-normale pour le montant) : log-vraisemblance = −2 332,20 ; AIC = 4 782,41 ; BIC = 5 170,94. Les deux modèles ne sont pas emboîtés : la comparaison repose sur l'AIC et le BIC, après correction du jacobien du logarithme. Le Double-Hurdle est nettement préféré. La composition du ménage joue sur la survenance mais pas sur le montant, ce que le Tobit ne peut pas représenter.

### Partie 4 : nombre de sinistres (`NSin`)

Les polices et la durée ne sont pas utilisées comme variables explicatives.

- Poisson rejeté (dispersion de Pearson = 2,52).
- Binomiale Négative (NB2) : α = 0,53 ; AIC = 26 128,04. Elle sous-estime la proportion de zéros (14,4 % prédits pour 26,0 % observés).
- **ZINB codé à la main** (vraisemblance et gradient analytique), retenu : AIC = 23 952,17 ; BIC = 24 156,31 ; proportion de zéros prédite = 25,3 %. Les modèles sont comparés par AIC/BIC, le test du rapport de vraisemblance standard ne s'appliquant pas (π = 0 est en bordure du domaine), et par un tableau de fréquences observées et attendues. BFGS signale une convergence formelle non atteinte, mais la norme du gradient est de 4,9·10⁻⁵ et la hessienne est définie positive.

### Partie 5 : durée de souscription

- Kaplan-Meier : durée médiane de 187 ; tests du log-rank significatifs pour la taille d'agglomération (p = 5,4·10⁻³), le véhicule (p = 5,1·10⁻²²) et l'âge (p = 4,3·10⁻⁶¹), avec les médianes et leurs intervalles de confiance.
- **Modèle de Cox** (indice de concordance = 0,726 ; test du rapport de vraisemblance = 1 945,47 sur 28 ddl). La composition du ménage est le facteur dominant : HR = 0,18 pour les personnes seules (risque de résiliation environ 5,5 fois plus faible), 0,47 pour les couples sans enfant et 1,64 pour les couples avec enfants, par rapport à la référence.
- L'effet de l'âge, marqué en univarié, disparaît une fois les autres variables contrôlées : les modèles de Cox emboîtés montrent qu'il s'atténue surtout à l'entrée de la composition du ménage (confusion).
- Le test de proportionnalité des risques rejette l'hypothèse pour 11 coefficients sur 28 (notamment RUC et la composition du ménage) : les rapports de risques sont à lire comme des effets moyens.

## Limites

- Le détail des types de sinistre (`Sinistre1` à `Sinistre3`), l'unité de `Duree` et l'unité monétaire ne sont pas documentés.
- `NSin` présente des valeurs absentes (aucun ménage avec `NSin` = 1) que le ZINB ne reproduit pas (304 ménages attendus avec `NSin` = 1) : le modèle ajuste la proportion de zéros, pas toute la forme de la distribution.
- Heckman : pas de variable d'exclusion, ratio de Mills colinéaire.
- Double-Hurdle : mêmes variables dans les deux équations (59 paramètres pour 555 observations positives), avec un risque de sur-ajustement.
- IV : instruments faibles, en particulier pour Police3 (statistique de Wald de première étape = 12,48 pour 4 instruments, soit F ≈ 3,1 ; R² de Shea = 0,0008) : les tests d'endogénéité sont peu puissants.
- Cox : hypothèse de risques proportionnels rejetée pour 11 coefficients sur 28.
- Les mesures d'erreur (RMSE, MAE) et la concordance sont calculées sur l'échantillon d'estimation, sans jeu de test.

## Reproduire

1. Installer les dépendances : `pip install -r requirements.txt`
2. Ouvrir `tarification_habitation.ipynb` (le fichier `dat_clean.csv` est à la racine du dépôt) et exécuter les cellules dans l'ordre.

## Outils

Python, pandas, numpy, scipy, statsmodels, patsy, linearmodels, lifelines, numdifftools, matplotlib.
