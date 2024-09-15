
Nom du papier : **Train faster, generalize better : Stability of stochastic gradient descent**

Auteurs : Moritz Hardt∗ Benjamin Recht† Yoram Singer‡

## En gros

Ils montrent que les modèles paramétriques entraînés par la descente de gradient stochastique ont une erreur de généralisation qui tend vers zéro avec quelques itérations

## La contribution de l'article

Ils veulent créer une borne à la généralisation pour les modèles entraînés par la SGD, en établissant une forte connexion entre l'erreur de généralisation et les propriétés de stabilité du modèle.

Ils montrent que le gradient stochastique est uniformément stable, ce qui s'applique également aux cas non-convexes.

## Explication des termes

### Risque et risque empirique

Supposons que nous ayons un modèle supervisé. On a une distribution $\mathcal{D}$ sur des occurrences d'un espace $Z$. On a un sample $S = (z_1, z_2, \dots, z_n)$ de $n$ points tirés indépendamment selon la distribution $\mathcal{D}$.

Le but c'est d'entraîner le modèle de paramètres $w$ afin d'obtenir un faible risque de population. On note $f$ notre fonction de perte, qui mesure l'erreur du modèle de paramètres w sur le point de donnée z.

$$\mathbb{R}\left[w\right]\overset{def}{=}\mathbb{E}_{z\sim\mathcal{D}}\mathbb{E}\left[f(w;z)\right]$$
En pratique, on ne peut calculer que le risque empirique, donc la moyenne de l'espérance effectivement mesuré en chaque point.
$$\mathbb{R}_S\left[w\right]=\frac{1}{m}\sum_{z\in S}^{i=1}\mathbb{E}\left[f(w;z_i)\right]$$
### Generalization error

L'erreur de généralisation, c'est la différence entre le risque et le risque empirique
$$R\left[w\right] - R_S\left[w\right]$$
Cette tautologie permet de bien retenir ce qu'est l'erreur de généralisation
$$R = R_S+(R-R_S)$$
En gros, l'idée c'est que si on modifie les valeurs de l'échantillon $S$, tout en restant conforme à la distribution $\mathcal{D}$  (définie sur l'espace $Z$), alors on veut pas que la différence $R\left[w\right] - R_S\left[w\right]$ augmente de manière significative. Cela signifierait que les paramètres $w$ permettent au modèle de généraliser correctement sur le problème que l'on cherche à résoudre.
### Stabilité

On peut dire qu'en espérance, la généralisation est égale à la stabilité

Écrivons cette affirmation formellement
$$\mathbb{E}\left[R - R_S\right] = \Delta$$
Concrètement, qu'est-ce qu'on veut dire par là ? Cela signifie que l'on s'attend à ce que notre modèle soit insensible aux légères perturbations de données. Le problème mathématiques devient un problème algorithmique, le modèle est stable du moment que l'algorithme utilisé est suffisamment robuste.

Voici un développement pour illustrer mon propos

Pour la démonstration, on ne parlera plus des poids $w$ du modèle mais de l'application d'un algorithme $A$ qui mappe un échantillon (=sample) $S$ à un modèle $A(S)$. Cet algorithme doit être robuste, il doit être symétrique.

Reprenons $S = (z_1, z_2, \dots, z_n)$ où chaque $z_j$ est tiré indépendamment selon la distribution $\mathcal{D}$ et introduisons $S' = (z_1', z_2', \dots, z_n')$ où chaque $z_j'$ est tiré indépendamment selon la distribution $\mathcal{D}$ , on a bien
$$\mathbb{E}\left[R_S\right]=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n} f(A(S);z_i)\right]$$
$$\mathbb{E}\left[R_S\right]=\frac{1}{n}\sum_{i=1}^{n} \mathbb{E}\left[f(A(S);z_i)\right]$$
$$\mathbb{E}\left[R\right]=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n} f(A(S');z_i')\right]$$
$$\mathbb{E}\left[R-R_S\right]=\frac{1}{n}\sum_{i=1}^{n}\mathbb{E}\left[f(A(S');z_i')-f(A(S);z_i)\right]$$
Là on introduit un sample $S^i$, dans lequel seul un élément est tiré indépendamment de $\mathcal{D}$ quand les autres sont tirés indépendamment selon la distribution $\mathcal{D}$ : $S^i=(z_1, z_2, \dots, z_{i-1}, z_{i}', z_{i+1}, \dots, z_n)$ avec $i=1, \dots, n$

En appliquant cette opération, on peut dériver l'égalité suivante
$$\mathbb{E}\left[f(A(S);z_i)\right]=\mathbb{E}\left[f(A(S^i);z_i')\right]=\mathbb{E}\left[f(A(S');z_i')\right] - \delta_i$$

Essayons de comprendre en quoi cette formule est vraie en reprenant tout depuis le début.

Nous considérons les espérances sur les paires $(A(S),z_i)$, où l'espérance est prise sur toutes les réalisations possibles des échantillons et des points de données. Rappelons que les variables aléatoires sont **indépendantes**, tirées selon la **même distribution $\mathcal{D}$**.

Si on réécrit les espérances pour $S$ et $S^i$, ça donne ça

$$\mathbb{E}_{S,z_i}\left[f(A(S);z_i)\right]=E_{z_1,\dots,z_n,z_i}\left[f(A(z_1,\dots,z_n);z_i)\right]$$
$$\mathbb{E}_{S^i,z_i}\left[f(A(S^i);z_i')\right]=E_{z_1,\dots,z_i-1,z_i',z_i+1,\dots,z_n}\left[f(A(z_1,\dots,z_i-1,z_i',z_i+1,\dots,z_n);z_i')\right]$$
L'espérance est linéaire c'est pour ça, on reviendra pas là-dessus c'est trivial.
On a dit que les données $z_i$ et $z_{i}'$ sont tirées indépendamment de la même distribution $\mathcal{D}$.

Rappelons désormais que l'algorithme $A$ est supposé symétrique par rapport à ses arguments : c'est-à-dire que permuter les éléments de l'échantillon $S$ n'affecte pas la distribution du modèle $A(S)$ produit par l'algorithme.
En clair, il traite de la même façon $A(z_1,z_2)$ et $A(z_2,z_1)$, et de la même façon $A(S)$ et $A(S^i)$ (ce qui est intéressant).

Ainsi :
- La distribution de $A(S)$ est la même que celle de $A(S^i)$ lorsqu'on échange $z_i$ et $z_i'$
- Les espérances des fonctions de perte correspondantes sont égales.

Ça nous donne la première égalité
$$\mathbb{E}_{S,z_i}\left[f(A(S);z_i)\right]=\mathbb{E}_{S^i,z_i}\left[f(A(S^i);z_i')\right]$$
Considérons maintenant
$$\mathbb{E}_{S',z_i'}\left[f(A(S');z_i')\right]$$
On introduit $\delta_i$, le terme qui représente la différence induite par le remplacement de tous les points de données.
$$\delta_i=\mathbb{E}_{S',z_i'}\left[f(A(S');z_i')\right]-E_{S^i,z_i}\left[f(A(S^i);z_i')\right]$$
$\delta_i$ mesure l'impact sur l'espérance de la perte lorsque l'on remplace tous les points de données
$$\mathbb{E}_{S',z_i'}\left[f(A(S');z_i')\right]=E_{S^i,z_i}\left[f(A(S^i);z_i')\right]+\delta_i$$
On en conclue bien que 
$$\mathbb{E}\left[f(A(S);z_i)\right]=\mathbb{E}\left[f(A(S^i);z_i')\right]=\mathbb{E}\left[f(A(S');z_i')\right] - \delta_i$$
On peut enfin conclure en additionnant les $\delta_i$ :
$$\mathbb{E}\left[R - R_S\right] = \Delta=\frac{1}{n}\sum_{i=1}^{n}\delta_i$$
Maintenant que l'on y voit plus clair, on va prendre un exemple concret pour comprendre ce que ça signifie dans un problème concret.

Les samples $S$, $S^i$ et $S'$ peuvent être interprétés comme suit :
- $S$ (échantillon initial) : C'est notre ensemble d'entraînement initial, composé de $n$ exemples $z_1,z_2,\dots,z_n$. Chaque $z_i$ pourrait représenter, par exemple, une paire entrée-sortie comme une image et son étiquette correspondante.
- $S'$ (nouvel échantillon): Il s'agit d'un nouvel ensemble de données, entièrement indépendant de $S$, où chaque élément $z_i'$ est tiré de la même distribution $\mathcal{D}$. Dans la pratique, cela correspondrait à un nouvel ensemble de test ou de validation utilisé pour évaluer la performance réelle de notre modèle sur des données qu'il n'a pas vues auparavant.
- **$Si$ (échantillon modifié) :** C'est l'ensemble $S$ où un seul exemple $z_i$ a été remplacé par un nouvel exemple $z_i'$​. Cela simule une petite perturbation dans vos données d'entraînement, reflétant la variabilité naturelle qui peut se produire lors de la collecte des données.

Dans ce contexte, si notre algorithme est stable, le modèle devrait donner des prédictions similaires même si on met à jour ou qu'on modifie légèrement notre ensemble de données d'entraînement.

Quand on parlera de régularisation, on parlera de stabilité, stay tuned le sang plomb 95
### Stabilité uniforme

La stabilité uniforme est une notion plus rigoureuse que la simple stabilité, car elle impose que la différence de performance soit bornée de manière uniforme pour tous les ensembles de données et tous les points d'échantillons possibles, pas seulement en moyenne. C'est une condition très forte qui assure que l'algorithme est stable dans un sens global.

$$|\Delta|\leq\underset{S,S'}{sup}\,\underset{z}{sup}|f(A(S),z)-f(A(S'),z)|$$
$|\Delta|$ est la **différence maximale** entre les performances de l'algorithme $A$ appliqué aux deux ensembles d'entraînement $S$ et $S'$, mesurée sur tous les points $z$.

Autrement dit, la stabilité uniforme garantit que, quelle que soit la petite perturbation apportée à l'ensemble d'entraînement, la différence entre les performances reste bornée **de manière uniforme**, pour tous les points $z$ et tous les ensembles $S$, $S'$.

L'utilisation du supremum est suffisamment parlante, voici une implémentation possible de cela :

```python
# Calculer la différence maximale entre les prédictions des deux modèles (S et S') 
def calculate_max_difference(model1, model2, X_test):
	preds1 = predict(model1, X_test)
	preds2 = predict(model2, X_test)
	differences = np.abs(preds1 - preds2)
	return np.max(differences)
```
