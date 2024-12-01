## 1. **Introduction**

### **1.1. Pourquoi XGBoost pour les séries temporelles ?**

XGBoost (Extreme Gradient Boosting) présente plusieurs avantages pour l’analyse et la modélisation des séries temporelles, en particulier lorsque les données présentent des relations complexes et non linéaires. Comparé à d'autres modèles de machine learning, il se distingue sur plusieurs points :

1. Gestion des relations non linéaires

XGBoost excelle dans la capture de relations non linéaires complexes entre les variables. Dans les séries temporelles, les relations entre les observations passées (features) et les valeurs futures peuvent être hautement non linéaires, et XGBoost peut les modéliser efficacement sans nécessiter une transformation explicite des données.

2. Flexibilité dans la création des features

Contrairement aux modèles traditionnels comme les modèles ARIMA, qui nécessitent des hypothèses strictes (stationnarité, linéarité), XGBoost permet de construire des features adaptées, comme :

- **Lags temporels** : valeurs retardées.
- **Moyennes mobiles** : pour lisser les tendances.
- **Composantes saisonnières** : capturant les variations périodiques.
- **Variables exogènes** : intègrant des informations externes ou explicatives (exemple : données économiques, météo, etc.).

Cette flexibilité permet d'enrichir le modèle avec des informations pertinentes.

3. Gestion des relations complexes entre features

XGBoost peut détecter des interactions complexes entre les variables (par exemple, entre un lag particulier et une composante saisonnière). Ces interactions peuvent être difficiles à capturer avec des modèles linéaires ou même des réseaux neuronaux qui nécessitent souvent une ingénierie des features très élaborée.

4. Robustesse face aux données bruitées ou manquantes

XGBoost utilise des arbres de décision comme base et est donc relativement robuste face :

- **Au bruit** : Il ne suppose pas que les données suivent une forme spécifique, ce qui en fait un choix approprié lorsque les séries temporelles présentent des anomalies ou des irrégularités.
- **Aux données manquantes** : Les arbres de décision gèrent naturellement les valeurs manquantes en trouvant les meilleures subdivisions, et XGBoost implémente ce mécanisme de manière efficace.

5. Adaptabilité aux séries temporelles multivariées

XGBoost peut intégrer plusieurs séries temporelles ou variables exogènes sans modification significative de l'architecture. Cela le rend utile pour des tâches comme :

- **Prédiction multivariée** : Utilisation d’autres séries ou données comme prédicteurs.
- **Apprentissage sur des données croisées** : Prédire une série à partir d'autres.

6. Rapidité et performance

Grâce à ses optimisations (parallélisation, gestion mémoire efficace, régularisation), XGBoost est souvent plus rapide que d’autres approches comme les réseaux neuronaux, tout en offrant des performances comparables sur des séries temporelles de taille moyenne.

7. Evitement de l'overfitting

La régularisation intégrée (L1 et L2) permet de contrôler la complexité des arbres, réduisant ainsi le risque de surapprentissage, ce qui est particulièrement utile pour les séries temporelles avec des tendances complexes et des cycles variés.

8. Expliquabilité

Comparé à des modèles complexes comme les réseaux neuronaux, XGBoost offre une meilleure **expliquabilité**, notamment grâce à des outils comme SHAP (SHapley Additive exPlanations) ou des analyses d’importance des variables, qui permettent de comprendre l’impact des features sur les prédictions.

9. Adaptabilité à des horizons de prévision variables

XGBoost peut être utilisé aussi bien pour des prévisions à court terme (1 pas de temps) que pour des prévisions à long terme (multi-step). L'approche multi-step peut être réalisée via :

- **Modèles séparés** : Un modèle par horizon de prédiction.
- **Approches récursives** : Utilisation des prédictions précédentes comme inputs.

**Limites**

Cependant, XGBoost n’est pas toujours la solution idéale :

- **Pas natif pour la séquence temporelle** : Il n'intègre pas automatiquement les dépendances temporelles comme les modèles récurrents (RNN, LSTM) ou Transformer.
- **Nécessite de l’ingénierie des features** : Les lags et les autres caractéristiques doivent être explicitement définis, ce qui peut être coûteux en termes de temps et d’expertise.
### **1.2. Objectif et contexte du projet**

Dans mon cas c'est parfait, et les limites n'en sont pas vraiment, un travail préalable d'exploration a été fait. L'explicabilité est la plus importante dans les conditions dans lesquelles on est d'un point de vue métier et parce qu'on veut expliquer une variable plutôt macro.

## 2. **Comprendre XGBoost**


### **2.1. Intuition de base**

XGBoost (Extreme Gradient Boosting) repose sur une amélioration du concept de boosting, une méthode d'ensemble qui combine plusieurs modèles faibles pour créer un modèle fort, capable de réaliser des prédictions précises. Contrairement au boosting traditionnel, XGBoost optimise chaque étape de ce processus en intégrant des améliorations pour augmenter la vitesse, la précision et la robustesse, tout en réduisant le risque de surapprentissage.

#### **Boosting vs Bagging**

Avant d'expliquer XGBoost, il est utile de distinguer le boosting et le bagging (utilisé par exemple dans les forêts aléatoires), deux approches d'ensemble courantes dans le machine learning.

- Bagging
	- Dans le bagging, plusieurs mod-les sont entraînées **en parallèle**, chacun sur un échantillon différent des données (échantillonnage avec remise).
	- Les prédictions des modèles sont ensuite combinées (par exemple, via une moyenne ou un vote majoritaire) pour réduire la variance et améliorer la robustesse.
	- L'objectif principal du bagging est de **réduire le surapprentissage** en généralisant mieux.
- Boosting
	- Contrairement au bagging, le boosting entraîne les modèles **séquentiellement**. Chaque nouveau modèle corrige les erreurs des modèles précédents en accordant plus de poids aux observations mal prédites.
	- Ce processus met davantage l'accent sur **la réduction du biais** et conduit souvent à des modèles plus performants sur des données complexes et difficiles
	- Le boosting peut cependant être plus sujet au surapprentissage si les modèles deviennent trop complexes.

XGBoost est un algo de boosting assez optimisé et complet, adaptable à un tas de cas d'usages

#### **Boosting**

Le boosting, dans son essence, suit un processus itératif :

1. **Modèles faibles successifs** : Chaque modèle est simple (comme un arbre de décision peu profond) et légèrement meilleur qu'un choix aléatoire.
2. **Correction des erreurs** : Les prédictions des nouveaux modèles visent à corriger les erreurs des modèles précédents.
3. **Importance des erreurs** : Les exemples mal prédits reçoivent plus de poids, augmentant leur influence dans les étapes suivantes.

Cependant, les algorithmes traditionnels comme AdaBoost se concentrent principalement sur la pondération des exemples mal classés et ne gèrent pas efficacement des données volumineuses ou complexes.

#### **Les améliorations de XGBoost**

XGBoost optimise le boosting traditionnel grâce à plusieurs innovations, regroupées autour de trois axes principaux : **précision**, **vitesse** et **généralisation**.

1. **Utilisation d'un objectif différentiable (Gradient Boosting)**

Au lieu de simplement pondérer les exemples mal prédits, XGBoost minimise explicitement une fonction de coût différentiable, comme :

- L'erreur quadratique pour les problèmes de régression.
- L'entropie croisée pour les problèmes de classification.

Chaque nouvel arbre est construit pour minimiser les résidus (erreurs des prédictions précédentes).
Cela se fait via :

- **Le gradient (dérivée première)** : Indique dans quelle direction ajuster le modèle.
- **La Hessienne (dérivée seconde)** : Fournit des informations sur la courbure pour accélérer la convergence.

Grâce à cette approche, XGBoost apprend à corriger non seulement la direction des erreurs, mais aussi leur intensité, permettant une optimisation plus fine.

2. **Régularisation pour éviter le surapprentissage**

XGBoost introduit des mécanismes explicites pour contrôler la complexité des modèles :
- **Pénalisation L1 (lasso)** : Encourage la réduction des coefficients inutiles, ce qui simplifie le modèle.
- **Pénalisation L2 (ridge)** : Limite la taille des coefficients pour éviter des prédictions extrêmes.
- **Taille minimale des feuilles** : Imposée pour limiter la profondeur des arbres et prévenir le surapprentissage.

3. **Gestion efficace des données**

XGBoost est conçu pour traiter efficacement des ensembles de données volumineux :

- **Blocage des splits** : Au lieu d'explorer toutes les possibilités de division dans un arbre, XGBoost regroupe les calculs pour optimiser la mémoire et accélérer le traitement.
- **Parallélisation** : Les calculs sont répartis sur plusieurs cœurs pour accélérer l'entraînement.
- **Gestion native des valeurs manquantes** : XGBoost attribue automatiquement les observations manquantes à la branche la plus adaptée lors de la construction des arbres.

4. **Flexibilité dans les données et les tâches**

XGBoost est très flexible et s’adapte facilement à des types de données variés ou à des tâches complexes, comme :

- **Intégration de variables exogènes** : XGBoost peut gérer facilement des prédicteurs externes (météo, indicateurs économiques, etc.).
- **Prévision multivariée** : Idéal pour des tâches combinant plusieurs séries temporelles ou variables d'entrée.

#### **Comparaison avec les approches traditionnelles**

| **Aspect**                         | **Boosting traditionnel**            | **XGBoost**                                                                             |
| ---------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------- |
| **Gestion des erreurs**            | Pondération des exemples mal prédits | Minimisation explicite d’une fonction de coût différentiable via gradient et Hessienne. |
| **Régularisation**                 | Limitée                              | Régularisation L1 et L2 pour éviter le surapprentissage.                                |
| **Construction des arbres**        | Basique                              | Optimisée via parallélisation et gestion mémoire efficace.                              |
| **Gestion des données manquantes** | Pas native                           | Gère les données manquantes de manière automatique.                                     |
| **Performance computationnelle**   | Lente sur grands ensembles           | Très rapide grâce à la parallélisation et au traitement distribué.                      |

Pour résumer, XGBoost combine les forces du boosting en les optimisant pour les rendre plus robustes, rapides et précises. Il surpasse le boosting traditionnel en intégrant :

- Une optimisation guidée par le gradient et la Hessienne.
- Des mécanismes avancés de régularisation.
- Une gestion efficace des données manquantes et bruitées.
- Une construction d’arbres rapide et parallèle.

Ces caractéristiques font de XGBoost un choix privilégié pour de nombreuses tâches, particulièrement lorsqu’il s'agit de modéliser des relations complexes et non linéaires. Comparé au bagging, XGBoost se distingue par son focus sur la correction itérative des biais, tandis que le bagging excelle dans la réduction de la variance.

### **2.2. Spécificités de XGBoost par rapport aux autres algos de gradient boosting**

XGBoost introduit plusieurs innovations majeures qui le distinguent des autres implémentations de **Gradient Boosting**, telles que celles proposées par Scikit-learn ou des bibliothèques similaires. Ces innovations couvrent des aspects liés à la **performance computationnelle**, la **généralisation**, et la **flexibilité du modèle**.

1. Gestion optimisée de la construction des arbres

XGBoost améliore significativement l'algorithme de construction des arbres en intégrant :

- **Split finding algorithm basé sur des historigrammes** : Au lieu de tester chaque valeur possible pour une division, les données sont regroupées en historigrammes , ce qui accélère grandement le processus de recherche.
- **Blocage pour la parallélisation** : XGBoost optimise les calculs pour tirer parti de la parallélisation lors de la recherche des meilleures divisions.

Ces techniques rendent XGBoost plus rapide et adapté à de très grands ensembles de données.

2. Régularisation intégrée (L1 et L2)

XGBoost intègre directement des mécanismes de régularisation pour réduire le surapprentissage, ce qui n'est pas toujours natif dans d'autres implémentations :
- Régularisation L1 : Encourage la sélection de variables pertinentes en annulant les contributions inutiles.
- Régularisation L2 : Empêche les coefficients des feuilles de devenir trop grands, limitant ainsi la complexité du modèle.

Cette régularisation explicite rend XGBoost plus robuste, surtout lorsque les données sont bruitées ou contiennent des caractéristiques inutiles.

3. Gestion native des valeurs manquantes

XGBoost gère directement les données manquantes en les exploitant comme une caractéristique :
- Lors de la construction d'un arbre, il apprend automatiquement la direction (branche) où les observations avec des valeurs manquantes devraient aller, en fonction de la réduction de la réduction de l'erreur.
- Cela évite d'avoir à imputer ou traiter les valeurs manquantes manuellement.

4. Optimisation guidée par la seconde dérivées (Hessienne)

Contrairement aux implémentations classiques de Gradient Boosting qui utilisent uniquement la première dérivée pour ajuster les modèles (gradient), XGBoost utilise également :
- **La Hessienne** (seconde dérivée de la fonction de perte), qui fournit des informations sur la courbure. Cela permet une convergence plus rapide et un ajustement plus précis.

5. Pruning intelligent des arbres

XGBoost introduit un mécanisme de **pruning tardif** des arbres:
- Au lieu de stopper l'ajout de branches dès que le gain devient nul, il permet de continuer à construire l'arbre complet, puis d'élaguer les branches inutiles a posteriori.

6. Gestion de la mémoire et efficacité pour les grands ensembles de données

XGBoost est conçu pour fonctionner efficacement sur des jeux de données volumineux grâce à :
- **Traitement par blocs** : Réduction des besoins en mémoire grâce à une gestion optimisée des calculs.
- **Parallélisation des tâches**: Exploite plusieurs cœurs pour diviser les calculs, notamment lors de la recherche des meilleurs splits. 

7. Support natif pour l'apprentissage distribué

XGBoost permet un entraînement sur plusieurs machines ou coeurs, avec une communication optimisée entre les nœuds, ce qui n'est pas natif

8. Fonctionnalités avancées pour les utilisateurs

XGBoost inclut des fonctionnalités qui facilitent le développement et l'interprétation :
- **Importance des features**: Offre des métriques avancées pour évaluer la contribution de chaque variable.
- **Prise en charge de différentes fonctions de perte**: Outre les classiques (erreur quadratiqu, entropie croisée), XGBoost permet de personnaliser des fonctions de coût pour répondre à des besoins spécifiques.

**Résumé comparatif**

| **Aspect**                         | Gradient Boosting classique       | XGBoost                                                      |
| ---------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| **Optimisation des arbres**        | Recherche exhaustive des splits   | Recherche optimisée par historigrammes et parallélisation    |
| **Régularisation**                 | Limitée                           | Régularisation explicite (L1 et L2)                          |
| **Gestion des données manquantes** | Nécessite un prétraitement manuel | Gestion native des valeurs manquantes                        |
| **Utilisation de la Hessienne**    | Non                               | Oui, pour une convergence plus rapide et plus précise        |
| **Pruning des arbres**             | Pruning précoce                   | Pruning tardif pour une meilleure généralisation             |
| **Apprentissage distribué**        | Non natif                         | Support natif pour des environnements distribués et Big Data |

### **2.3. Comparaison avec d'autres algorithmes (RF, LGBM, etc.)**

La modélisation des séries temporelles avec des algorithmes comme XGBoost, LightGBM, ou Random Forests (RF) repose sur leur capacité à capturer des relations complexes entre les variables. Toutefois, chacun de ces algorithmes présente des différences significatives dans la manière de traiter les données et d’optimiser le modèle.

Voici une comparaison dans le contexte des séries temporelles :

#### XGBoost vs LightGBM

**Traitement des données**

- XGBoost
	- Utilise une méthode d’optimisation par **histogrammes traditionnels**, où chaque split est déterminé en explorant des regroupements fixes de données.
	- Gère les valeurs manquantes nativement, ce qui est utile dans les séries temporelles avec des trous dans les données.
	- Est souvent plus robuste aux petits ensembles de données, ce qui est courant en séries temporelles.
- LightGBM
	- Utilise un algorithme de **Gradient-Based One-Side Sampling (GOSS)** pour réduire le coût des calculs en se concentrant sur les exemples ayant un fort gradient (observations difficiles).
	- Traite plus efficacement les ensembles de données volumineux grâce à une technique de **Leaf-wise Tree Growth**, qui explore les branches les plus prometteuses avant de continuer ailleurs.
	- Peut être plus sensible aux données déséquilibrées ou bruitées, ce qui nécessite un soin particulier dans la préparation des séries.

**Performance computationnelle**

- **XGBoost** est généralement plus lent pour des ensembles volumineux en raison de son approche itérative et exhaustive pour le calcul des splits.
- **LightGBM** est plus rapide dans les cas de séries temporelles complexes ou à haute fréquence grâce à son algorithme d'échantillonnage, mais cela peut conduire à des résultats légèrement moins stables si les hyperparamètres ne sont pas bien réglés.

**Utilisation dans les séries temporelles**

- XGBoost
	- Excelle dans les tâches où des relations complexes doivent être capturées, comme la modélisation de **décalages temporels spécifiques** ou l'intégration de **variables exogènes**.
	- Plus adapté pour les séries de petite à moyenne taille et les environnements nécessitant un contrôle rigoureux de régularisation (L1/L2).
- LightGBM
	- Idéal pour les séries volumineuses et multidimensionnelles où des performances élevées sont requises.
	- Peut nécessiter une attention accrue au surapprentissage dans les séries à forte saisonnalité ou tendance.
#### XGBoost vs LightGBM

**Approche de l'ensemble**

- **XGBoost (Boosting)** :
    - Construit les arbres **séquentiellement**, chaque nouvel arbre corrigeant les erreurs des précédents.
    - Est plus adapté à la modélisation de relations complexes, ce qui est courant dans les séries temporelles avec dépendances non linéaires.
- **Random Forests (Bagging)** :
    - Construit les arbres **en parallèle** sur des échantillons différents des données et combine leurs résultats (moyenne ou vote majoritaire).
    - Se concentre sur la réduction de la variance et peut être moins performant sur des séries temporelles où des biais importants existent.

**Traitement temporel**

- **XGBoost** :
    - Permet de capturer des décalages temporels et des dépendances non linéaires grâce à sa flexibilité dans l’ingestion de nouvelles features comme des lags ou des moyennes mobiles.
    - Convient aux séries temporelles où les relations dépendent fortement de contextes spécifiques (saisons, anomalies, variables exogènes).
- **Random Forests** :
    - Plus limité dans la capture de séquences temporelles, car il n’y a pas de relation directe entre les arbres construits.
    - Fonctionne bien si les dépendances temporelles sont déjà explicites (e.g., via des lags ou transformations pré-calculées)

**Robustesse**

- **XGBoost** :
    - Offre une régularisation explicite (L1/L2), ce qui le rend plus robuste aux données bruitées, mais nécessite un réglage fin des hyperparamètres pour éviter le surapprentissage.
- **Random Forests** :
    - Plus résilient au bruit intrinsèque des séries temporelles grâce à la moyenne des arbres, mais peut avoir du mal à modéliser des comportements non linéaires ou saisonniers complexes.

#### Résumé comparatif

| **Critères**              | **XGBoost**                                  | **LightGBM**                                   | **Random Forests**                                  |
| ------------------------- | -------------------------------------------- | ---------------------------------------------- | --------------------------------------------------- |
| **Traitement des séries** | Capture des relations complexes via boosting | Efficacité et vitesse sur des grands ensembles | Dépendances explicites nécessaires (pré-traitement) |
| **Performance**           | Plus lent, mais précis                       | Très rapide sur de grands volumes              | Stable, mais moins flexible                         |
| **Gestion du bruit**      | Régularisation L1/L2                         | Sensible au bruit si mal configuré             | Résilient au bruit                                  |
| **Applications typiques** | Séries de taille moyenne, décalages précis   | Séries volumineuses, multivariées              | Séries à faible complexité ou pré-traitées          |
#### Conclusion

- **XGBoost** est un excellent choix pour les séries temporelles où il faut modéliser des relations complexes avec une taille de données raisonnable.
- **LightGBM** est plus adapté aux séries volumineuses ou multidimensionnelles nécessitant une rapidité d'exécution.
- **Random Forests** sont une option robuste mais souvent moins performante pour les séries temporelles complexes, sauf si le pré-traitement permet de capturer explicitement les dépendances temporelles.

## 3. **Configurer XGBoost pour des séries temporelles**

### **3.1. Préparer les données pour XGBoost**

Pour utiliser **XGBoost** avec des séries temporelles, il est crucial de transformer les données pour capturer les dépendances temporelles, car XGBoost ne traite pas directement les séquences comme des modèles récursifs (e.g., RNN ou LSTM). Voici une démarche structurée pour préparer vos séries temporelles en entrée :

Avant de commencer, analysez la série pour identifier :

- **Saisonnalités** : Variations périodiques sur des échelles temporelles fixes.
- **Tendances** : Évolution à long terme.
- **Anomalies** : Pics ou creux inhabituels.
- **Stationnarité** : La série est-elle stationnaire ou nécessite-t-elle une transformation ?

#### **Créer des caractéristiques basées sur les décalages (lags)**

XGBoost fonctionne avec des variables indépendantes, il est donc nécessaire de **convertir les dépendances temporelles implicites en caractéristiques explicites**.

- **Décalages simples (lags)** : Créez des colonnes représentant les valeurs précédentes à différents instants.  
	Exemple : Pour une série $y_t$, ajoutez $y_{t-1}, y_{t-2}, \dots, y_{t-k}$​, où $k$ est la fenêtre choisie.
	```python
df['lag_1'] = df['y'].shift(1)
df['lag_2'] = df['y'].shift(2)

```

- **Fenêtres glissantes** : Calculez des moyennes ou médianes sur une période pour capturer des tendances locales.
	Exemple : Moyenne mobile sur 3 jours $\text{mean}(y_{t-3}, y_{t-2}, y_{t-1})$
	```python
df['rolling_mean_3'] = df['y'].rolling(window=3).mean()
```
#### Ajouter des variables temporelles explicites

Les séries temporelles ont souvent des patrons qui dépendent du temps. Ajoutez des **caractéristiques calendaires** :

- **Date et heure** : Jour de la semaine, mois, saison, heure.
```python
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
```
- **Variables périodiques** : Si les données suivent un cycle, encodez le temps de façon cyclique (e.g., jours, heures).
	$\text{sin}(\text{valeur temps}), \text{cos}(\text{valeur temps})$
```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

#### Intégration des tendances et de saisonnalité

1. Ajout des tendances comme une caractéristique

- Les tendances correspondent à l'évolution à long terme de la série. Vous pouvez les intégrer explicitement en tant que **features** :
	- Si la tendance est déjà identifiée (via une régression linéaire, une moyenne mobile à long terme ou un lissage exponentiel), créez une colonne représentant cette tendance.
	- Exemple :
	```python
df['trend'] = np.arange(len(df))  # Index temporel pour modéliser une tendance linéaire.
```
	- Ajoutez cette colonne comme une variable indépendant dans $X$
	
2. Intégration des indicateurs de saisonnalité

- Si la série présente des **saisonnalités** (journalières, hebdomadaires, annuelles), incluez-les directement dans le modèle. Ces indicateurs peuvent être :
	- **Variables binaires** : Par exemple, 1 si c'est un week-end, 0 sinon.
	```python
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
```
- **Variables périodiques cycliques** : Si vous avez utilisé des transformations comme $sin$ et $cos$ pour encoder la périodicité, ajoutez-les comme colonnes dans $X$.
```python
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

3. Variables issues de décompositions

- Si vous avez décomposé la série temporelle (par exemple, avec une méthode STL ou une décomposition classique en tendance, saisonnalité et résidu), incluez les composantes directement comme variables.
	- Exemple avec STL (statsmodels) :
	```python
from statsmodels.tsa.seasonal import STL
stl = STL(df['y'], period=12)
result = stl.fit()
df['stl_trend'] = result.trend
df['stl_seasonal'] = result.seasonal
```
	- Ces colonnes `stl_trend`, `stl_seasonal` sont ajoutés comme features dans $X$.
	
1. Importance des hyperparamètres pour intégrer ces caractéristiques

- Les indicateurs de tendance et de saisonnalité peuvent avoir une échelle différente des autres variables (e.g., lags, variables exogènes). Cela peut influencer leur importance dans le modèle.
	- Si nécessaire, appliquez une **normalisation ou standardisation** uniquement sur les colonnes d'indicateurs :
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['trend_scaled'] = scaler.fit_transform(df[['trend']])
```
#### Incorporer des variables exogènes

Si vous avez des données supplémentaires influençant la série principale (e.g., météo, jours fériés, autres séries), incluez-les comme caractéristiques :

- Température, prix du marché, volume des ventes, etc.
- Utilisez les mêmes techniques de lagging ou transformations si nécessaire.
#### Vérifier et gérer les valeurs manquantes

Les décalages et transformations glissantes introduisent souvent des **valeurs manquantes**. Gérez-les :

- Supprimez les premières lignes affectées.
- Remplissez-les avec une méthode d’imputation (e.g., interpolation linéaire).
#### Transformer pour la stationnarité

Si la série n'est pas stationnaire (e.g., présence de tendances), appliquez des transformations :
- **Différenciation** : Soustrayez les valeurs précédentes ($y_t - y_{t-1}$).
- **Logarithmes** : Réduisent l'impact des variations exponentielles.
#### Diviser en ensembles d'entraînement et de test

Lorsque vous modélisez des séries temporelles, il est crucial de respecter leur **ordre temporel**. Ne pas utiliser de division aléatoire comme dans d'autres problèmes de ML.

- Utilisez une **validation temporelle** (train sur une période passée, test sur une future)
- Exemple de découpe :
```python
train = df[df['date'] < '2023-01-01']
test = df[df['date'] >= '2023-01-01']
```

#### **Normalisation ou standardisation (facultatif)**

Certaines caractéristiques (e.g., lags ou variables exogènes) peuvent bénéficier d’une mise à l’échelle :

- **Min-max scaling** : $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$
- **Standardisation** : $x' = \frac{x - \mu}{\sigma}$
	- Fenêtre glissante et création des features
    - Décalage temporel (lags)
    - Insertion des tendances et des indicateurs de saisonnalité

#### Construction du dataset final

Une fois toutes les caractéristiques préparées, structurez le dataset pour XGBoost avec les colonnes d'entrée (features) et la cible :
- Variables explicatives : $X$ (lags, variables temporelles, exogènes)
- Cible : $y$ (valeurs futures à prédire)

```python
X = df[['lag_1', 'lag_2', 'rolling_mean_3', 'day_of_week']]
y = df['y']
```
#### Validation croisée personnalisée

La validation croisée classique (random split) est inadaptée pour les séries temporelles. Utilisez une **TimeSeriesSplit** pour simuler des prédictions sur des périodes futures :

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

#### Résumé des étapes

1. Créez des décalages et transformations pour capturer les dépendances temporelles.
2. Ajoutez des caractéristiques temporelles explicites et exogènes.
3. Gérez les valeurs manquantes et stationnarisez si nécessaire.
4. Respectez l’ordre temporel dans la division des données.
5. Validez le modèle avec des techniques adaptées au contexte temporel.
### **3.2. Construction du modèle**

La construction d'un modèle XGBoost pour les séries temporelles nécessite une attention particulière aux spécificités des données temporelles. Voici un guide détaillé pour configurer et entraîner un modèle XGBoost adapté à ces données.

#### **Configuration initiale du modèle**

Pour commencer, il est essentiel de configurer correctement le modèle XGBoost en tenant compte des caractéristiques des séries temporelles.

- **Importation des bibliothèques nécessaires :**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
```

- **Préparation des données :**

Assurez-vous que vos données sont prêtes, avec les lags et les caractéristiques temporelles déjà intégrées.

```python
# Supposons que df soit votre DataFrame avec les caractéristiques préparées
X = df.drop(columns=['target'])
y = df['target']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

- **Configuration des paramètres essentiels :**

Les paramètres suivants sont cruciaux pour ajuster le modèle XGBoost :

```python
params = {
    'objective': 'reg:squarederror',  # Pour les séries temporelles, souvent une tâche de régression
    'max_depth': 5,                   # Profondeur maximale des arbres
    'learning_rate': 0.1,             # Taux d'apprentissage
    'n_estimators': 100,              # Nombre d'arbres
    'subsample': 0.8,                 # Fraction des échantillons à utiliser pour chaque arbre
    'colsample_bytree': 0.8,          # Fraction des features à utiliser pour chaque arbre
    'gamma': 0,                       # Réduction minimale de la perte requise pour faire une partition
    'alpha': 0,                       # Régularisation L1
    'lambda': 1                       # Régularisation L2
}
```

#### **Spécificités pour les séries temporelles**

- **Gestion des interactions non linéaires :**

XGBoost est capable de capturer des interactions non linéaires grâce à sa structure en arbre. Assurez-vous que vos données incluent des caractéristiques qui peuvent interagir de manière non linéaire, comme des lags ou des transformations cycliques.

- **Configuration pour capturer les dépendances temporelles :**

Utilisez des lags et des caractéristiques temporelles explicites pour aider le modèle à comprendre les dépendances temporelles.

```python
# Exemple de création de lags
df['lag_1'] = df['target'].shift(1)
df['lag_2'] = df['target'].shift(2)
```

- **Bonnes pratiques pour éviter le surapprentissage :**

  - Utilisez la régularisation (paramètres `alpha` et `lambda`) pour contrôler la complexité du modèle.
  - Limitez la profondeur des arbres (`max_depth`) pour éviter des modèles trop complexes.
  - Utilisez la validation croisée pour évaluer la performance du modèle.

#### **Approches possibles**

- **Prédiction directe vs récursive :**
  - **Prédiction directe :** Entraînez un modèle pour chaque horizon de prédiction.
  - **Prédiction récursive :** Utilisez les prédictions précédentes comme entrées pour les prédictions futures.

```python
# Exemple de prédiction récursive
for t in range(1, forecast_horizon + 1):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Utilisez y_pred pour ajuster X_test pour la prochaine prédiction
```

- **Modèle unique vs multiple :**
  - **Modèle unique :** Un seul modèle pour toutes les prédictions.
  - **Modèle multiple :** Un modèle par horizon de prédiction ou par série.

- **Gestion des horizons de prédiction :**
  - Choisissez entre un modèle unique pour tous les horizons ou des modèles séparés pour chaque horizon, en fonction de la complexité des données et des besoins en précision.

En suivant ces étapes et en ajustant les paramètres en fonction de vos données spécifiques, vous pouvez construire un modèle XGBoost efficace pour les séries temporelles. Assurez-vous de tester différentes configurations et de valider les résultats pour obtenir les meilleures performances possibles.

## 4. **Hyperparamètres et tuning**

### **4.1. Hyperparamètres clés**

Les hyperparamètres de XGBoost peuvent être regroupés en plusieurs catégories, chacune ayant un impact spécifique sur le comportement du modèle. Pour les séries temporelles, certains paramètres nécessitent une attention particulière en raison de la nature séquentielle des données.

#### **Paramètres de contrôle de la complexité**

Ces paramètres déterminent la structure et la complexité des arbres individuels.

1. **max_depth** (profondeur maximale)
   - **Rôle** : Limite la profondeur maximale de chaque arbre.
   - **Impact** : Contrôle directement la complexité du modèle et sa capacité à capturer des interactions.
   - **Valeurs typiques** : 3-10 (séries temporelles : 4-6)
   - **Recommandation** : Pour les séries temporelles, commencer avec une valeur modérée (5-6) pour permettre la capture des dépendances temporelles sans surapprentissage.

```python
params = {
    'max_depth': 5,  # Valeur conservative pour séries temporelles
    'min_child_weight': 1,
    'gamma': 0.1
}
```

2. **min_child_weight**
   - **Rôle** : Définit le poids minimal nécessaire pour créer un nouveau nœud.
   - **Impact** : Aide à contrôler le surapprentissage en évitant les subdivisions trop spécifiques.
   - **Valeurs typiques** : 1-10
   - **Recommandation** : Augmenter pour les séries bruitées ou avec peu d'observations.

3. **gamma** (gamma)
   - **Rôle** : Seuil minimal de réduction de perte pour créer une nouvelle partition.
   - **Impact** : Contrôle la création de nouveaux nœuds, agit comme régularisateur.
   - **Valeurs typiques** : 0-0.5
   - **Recommandation** : Augmenter pour les séries avec forte saisonnalité pour éviter la surmodélisation des motifs saisonniers.

#### **Paramètres de régularisation**

Ces paramètres contrôlent l'apprentissage et la régularisation du modèle.

1. **learning_rate** (eta)
   - **Rôle** : Contrôle la contribution de chaque nouvel arbre.
   - **Impact** : Affecte la vitesse et la stabilité de l'apprentissage.
   - **Valeurs typiques** : 0.01-0.3
   - **Recommandation** : Pour les séries temporelles, privilégier des valeurs plus faibles (0.01-0.1) pour une meilleure stabilité.

```python
params = {
    'learning_rate': 0.05,  # Valeur conservative
    'n_estimators': 1000,   # Augmenté pour compenser le learning rate faible
    'lambda': 1.0,
    'alpha': 0.0
}
```

2. **n_estimators**
   - **Rôle** : Nombre total d'arbres à construire.
   - **Impact** : Détermine la capacité d'apprentissage totale du modèle.
   - **Valeurs typiques** : 100-1000+
   - **Recommandation** : Augmenter avec la diminution du learning_rate. Pour les séries temporelles, prévoir plus d'arbres pour capturer les motifs complexes.

3. **lambda** (L2) et **alpha** (L1)
   - **Rôle** : Contrôlent respectivement la régularisation L2 et L1.
   - **Impact** : Réduisent l'overfitting et la complexité du modèle.
   - **Valeurs typiques** : 
     - lambda : 0-3
     - alpha : 0-3
   - **Recommandation** : Augmenter lambda pour les séries avec forte autocorrélation.

#### **Paramètres de sous-échantillonnage**

Ces paramètres contrôlent l'utilisation des données et des features lors de la construction des arbres.

1. **subsample**
   - **Rôle** : Fraction des observations utilisées pour chaque arbre.
   - **Impact** : Réduit la variance et aide à éviter l'overfitting.
   - **Valeurs typiques** : 0.5-1.0
   - **Recommandation** : Pour les séries temporelles, maintenir une valeur élevée (0.8-1.0) pour préserver la structure temporelle.

```python
params = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 1.0,
    'colsample_bynode': 1.0
}
```

2. **colsample_bytree**, **colsample_bylevel**, **colsample_bynode**
   - **Rôle** : Contrôlent l'échantillonnage des features à différents niveaux.
   - **Impact** : Réduisent l'overfitting et augmentent la robustesse.
   - **Valeurs typiques** : 0.5-1.0
   - **Recommandation** : Pour les séries temporelles avec beaucoup de features dérivées (lags, moyennes mobiles), utiliser colsample_bytree pour réduire la dépendance aux features individuelles.

#### **D. Ajustements spécifiques aux séries temporelles**

Les paramètres doivent être ajustés en fonction des caractéristiques spécifiques de vos séries temporelles :

1. **Fréquence des données**
```python
# Configuration pour données haute fréquence
params_high_freq = {
    'max_depth': 4,          # Réduire pour éviter l'overfitting sur le bruit
    'learning_rate': 0.01,   # Plus faible pour plus de stabilité
    'subsample': 0.9,        # Maintenir structure temporelle
    'n_estimators': 1500     # Plus d'arbres pour compenser
}

# Configuration pour données basse fréquence
params_low_freq = {
    'max_depth': 6,          # Permettre plus de complexité
    'learning_rate': 0.1,    # Plus élevé car moins de bruit
    'subsample': 0.8,        # Plus de flexibilité
    'n_estimators': 500      # Moins d'arbres nécessaires
}
```

2. **Présence de saisonnalité**
```python
# Configuration pour séries avec forte saisonnalité
params_seasonal = {
    'max_depth': 8,          # Capturer motifs complexes
    'gamma': 0.2,            # Contrôler splits sur motifs saisonniers
    'colsample_bytree': 0.7  # Réduire dépendance aux features individuelles
}
```

3. **Longueur de l'historique**
```python
# Configuration pour historique court
params_short = {
    'max_depth': 4,
    'min_child_weight': 2,   # Plus conservateur
    'subsample': 0.9         # Préserver données limitées
}

# Configuration pour historique long
params_long = {
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.7         # Plus de flexibilité
}
```

#### **Pièges à éviter**

1. **Surapprentissage temporel**
   - Ne pas utiliser de max_depth trop élevé qui pourrait capturer du bruit temporel.
   - Maintenir subsample élevé pour préserver la structure temporelle.

2. **Sous-apprentissage des motifs**
   - Ne pas trop limiter la profondeur des arbres si présence de saisonnalité complexe.
   - Ajuster n_estimators en fonction de la complexité des patterns.

3. **Mauvaise gestion des features temporelles**
   - Attention à colsample_* avec beaucoup de lags : risque de perdre des dépendances importantes.
   - Adapter min_child_weight à la fréquence des données.

#### **Impact sur les performances**

| Paramètre | ↑ Temps d'entraînement | ↑ Mémoire | ↑ Risque d'overfitting |
|-----------|------------------------|------------|------------------------|
| max_depth | ↑↑↑ | ↑↑ | ↑↑↑ |
| n_estimators | ↑↑↑ | ↑↑↑ | ↑ |
| learning_rate | ↓ | - | ↓↓ |
| subsample | ↓ | - | ↓ |

### **4.2. Méthodes d'optimisation**

L'optimisation des hyperparamètres est cruciale pour obtenir les meilleures performances de XGBoost sur les séries temporelles. Plusieurs approches sont possibles, chacune avec ses avantages et contraintes.

#### **A. Grid Search**

La recherche par grille (Grid Search) est une approche systématique qui teste toutes les combinaisons possibles d'hyperparamètres dans un espace de recherche prédéfini.

1. **Principe et mise en œuvre**

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Définition de l'espace de recherche
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.8, 0.9, 1.0]
}

# Configuration de la validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=5, test_size=24)  # Pour données mensuelles

# Initialisation du modèle et de la recherche
model = xgb.XGBRegressor()
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # Utilisation de tous les cœurs disponibles
    verbose=2
)

# Exécution de la recherche
grid_search.fit(X_train, y_train)
```

2. **Bonnes pratiques pour séries temporelles**

- Utiliser `TimeSeriesSplit` pour respecter l'ordre temporel
- Adapter la taille des splits à la fréquence des données
- Commencer avec une grille grossière puis affiner

```python
# Exemple d'approche en deux étapes
# 1. Grille grossière
coarse_param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1]
}

# 2. Grille fine autour des meilleurs paramètres
fine_param_grid = {
    'max_depth': [5, 6, 7],  # Centré autour du meilleur résultat
    'learning_rate': [0.05, 0.075, 0.1]
}
```

3. **Analyse des résultats**

```python
# Extraction et visualisation des résultats
import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame(grid_search.cv_results_)
best_params = grid_search.best_params_

# Visualisation de l'impact des paramètres
plt.figure(figsize=(10, 6))
for lr in param_grid['learning_rate']:
    mask = results['param_learning_rate'] == lr
    plt.plot(
        results[mask]['param_max_depth'],
        results[mask]['mean_test_score'],
        label=f'lr={lr}'
    )
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.legend()
```

#### **B. Random Search**

La recherche aléatoire échantillonne aléatoirement l'espace des hyperparamètres, offrant souvent un meilleur rapport efficacité/temps de calcul.

1. **Configuration et implémentation**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Définition de l'espace de recherche
param_distributions = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.19),  # Distribution uniforme entre 0.01 et 0.2
    'n_estimators': randint(100, 1500),
    'subsample': uniform(0.7, 0.3),  # Entre 0.7 et 1.0
    'colsample_bytree': uniform(0.7, 0.3)
}

# Configuration de la recherche
random_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(),
    param_distributions=param_distributions,
    n_iter=100,  # Nombre d'itérations
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
```

2. **Stratégies d'optimisation**

```python
# Fonction d'évaluation personnalisée pour séries temporelles
def custom_scorer(y_true, y_pred):
    # Pondération plus forte pour les erreurs récentes
    weights = np.linspace(0.5, 1.0, len(y_true))
    return -np.average((y_true - y_pred)**2, weights=weights)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    scoring=make_scorer(custom_scorer),
    n_iter=100,
    cv=tscv
)
```

#### **C. Optimisation Bayésienne**

L'optimisation bayésienne utilise un modèle probabiliste pour guider la recherche des hyperparamètres de manière plus efficace.

1. **Mise en œuvre avec Optuna**

```python
import optuna

def objective(trial):
    # Définition de l'espace de recherche
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
    }
    
    # Validation croisée temporelle
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        scores.append(model.best_score)
    
    return np.mean(scores)

# Création et exécution de l'étude
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

2. **Analyse des résultats**

```python
# Visualisation de l'importance des paramètres
optuna.visualization.plot_param_importances(study)

# Visualisation des courbes de convergence
optuna.visualization.plot_optimization_history(study)

# Extraction des meilleurs paramètres
best_params = study.best_params
```

#### **Comparaison des méthodes**

| Méthode | Avantages | Inconvénients | Recommandé pour |
|---------|-----------|---------------|-----------------|
| Grid Search | Exhaustif, reproductible | Coût computationnel élevé | Petits espaces de recherche |
| Random Search | Bon rapport efficacité/coût | Peut manquer l'optimal | Grands espaces de recherche |
| Optuna | Efficace, adaptatif | Plus complexe à configurer | Optimisation fine |

#### **Recommandations pratiques**

1. **Choix de la méthode selon le contexte**
   - Petit dataset : Grid Search
   - Dataset moyen : Random Search
   - Grand dataset : Optimisation Bayésienne

2. **Optimisation des ressources**
   - Utiliser la parallélisation quand possible
   - Implémenter early stopping
   - Commencer avec un sous-ensemble des données

3. **Validation et monitoring**
   - Surveiller l'overfitting pendant l'optimisation
   - Valider sur plusieurs métriques
   - Garder une période de test intouchée

```python
# Exemple de configuration avec early stopping
early_stopping = {
    'early_stopping_rounds': 50,
    'eval_metric': 'rmse',
    'verbose': False
}

# Validation sur multiple métriques
scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring=scoring,
    refit='rmse'  # Optimise sur RMSE mais calcule aussi MAE
)
```

## 5. **Évaluation du modèle**

### **5.1. Métriques adaptées aux séries temporelles**
    - RMSE, MAPE, métriques basées sur le lag
### **5.2. Validation croisée pour séries temporelles**
    - Explication du TimeSeriesSplit
### **5.3. Interprétabilité des résultats**
    - Importance des features
    - SHAP values pour XGBoost

## 6. **Résultats et analyse**

### **6.1. Résultats obtenus (relation entre les séries)**
### **6.2. Analyse des biais et des erreurs**
### **6.3. Améliorations possibles**

### 7. **Conclusion et perspectives**

- Synthèse des découvertes















---

## Prompts pour approfondir chaque section

### 1. **Introduction**

- _"Quels sont les avantages d'utiliser XGBoost pour les séries temporelles par rapport à d'autres modèles de machine learning ?"_

### 2. **Comprendre XGBoost**

- _"Explique-moi l'intuition de base derrière XGBoost et comment il optimise le boosting traditionnel."_
- _"Quelles sont les principales innovations de XGBoost par rapport aux autres implémentations de Gradient Boosting ?"_
- _"Quelle est la différence entre XGBoost et LightGBM dans le contexte de la modélisation de séries temporelles ?"_

### 3. **Configurer XGBoost pour des séries temporelles**

- _"Comment préparer des séries temporelles pour être utilisées comme input dans XGBoost ?"_
- _"Quelles techniques permettent d'intégrer des indicateurs de saisonnalité dans XGBoost ?"_
- _"Comment choisir les lags temporels les plus pertinents pour une modélisation XGBoost ?"_

### 4. **Hyperparamètres et tuning**

- _"Quels sont les hyperparamètres les plus importants de XGBoost et comment influencent-ils le modèle ?"_
- _"Quelle est la meilleure méthode pour optimiser les hyperparamètres de XGBoost sur des séries temporelles ?"_

### 5. **Évaluation du modèle**

- _"Quels types de validation croisée sont adaptés aux séries temporelles et pourquoi ?"_
- _"Comment interpréter l'importance des features dans XGBoost ?"_
- _"Comment les valeurs SHAP peuvent-elles aider à interpréter les prédictions de XGBoost ?"_

### 6. **Résultats et analyse**

- _"Comment analyser la qualité des prédictions d'un modèle XGBoost appliqué aux séries temporelles ?"_

### 7. **Conclusion et perspectives**

- _"Quels sont les défis restants dans l'utilisation de XGBoost pour des séries temporelles ?"_