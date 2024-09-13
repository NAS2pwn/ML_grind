La descente de gradient stochastique est utilisée, évidemment, pour minimiser une fonction de coût.

En revanche, là où pour une descente de gradient classique on calcule la dérivée de la fonction de coût par rapport à tous les exemples du dataset, une descente de gradient stochastique va la calculer sur un exemple aléatoire ou un petit lot d'exemples.

## Rappels sur la descente de gradient classique

Pour rappel, le gradient est la dérivée de la fonction de perte en fonction de ses paramètres d'entrée.

La dérivée partielle de la fonction de perte par le réglage d'un paramètre d'entrée nous dit comment et à quel point il faut ajuster ce paramètre afin de réduire la fonction de perte.

voici la formule de la descente de gradient

$$\nabla J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} L(h_\theta(x_i), y_i)$$
- $J(\theta)$ est la fonction de coût que l'on cherche à minimiser
- $\nabla J(\theta)$ est le gradient de la fonction de coût, en gros le vecteur avec la dérivée partielle de la fonction de coût pour chaque paramètre : $\nabla J(\theta) = \left[ \frac{\partial J(\theta)}{\theta_0},  \frac{\partial J(\theta)}{\theta_1},  \frac{\partial J(\theta)}{\theta_2}, \dots \right]$ avec le vecteur de paramètres $\theta = \left[\theta_0, \theta_1, \theta_2, \dots \right]$
- m le nombre total d'exemples dans le dataset
- $L(h_\theta(x_i),y_i)$ et la perte pour l'exemple $i$
- $h_\theta(x_i)$ est la prédiction du modèle
- $\nabla_\theta$ représente le gradient pour un exemple spécifique, et donc on fait la moyenne de tous ces gradients locaux pour avoir le gradient global

Pour rappel, on met à jour les poids de cette façon (où $\alpha$ est le taux d'apprentissage)

$$\theta = \theta-\alpha\nabla J(0)$$

Il y a plusieurs problèmes avec cela :
- **Le temps de calcul** : Parcourir tout le dataset à chaque mise à jour des poids est coûteux, surtout pour un gros dataset.
- **La régularité de la convergence** : La trajectoire de la descente classique est trop régulière et risque de se coincer dans des minima locaux
- **Le surapprentissage** : La descente de gradient classique est plus susceptible de surapprendre, car elle prend en compte toutes les petites variations de données.

>**Note :** La descente de gradient dans un réseau de neurones est efficace grâce à la rétropropagation, qui calcule les dérivées de manière optimisée en réutilisant les résultats intermédiaires. Cela permet de réduire le coût de calcul, car on n'a pas besoin de recalculer les dérivées pour chaque paramètre individuellement. C'est une des raisons du succès de ces derniers !
## Introduction à la descente de gradient stochastique

Pour rendre l'optimisation plus efficace, on introduit la descente de gradient stochastique. Au lieu de calculer le gradient sur l'ensemble du jeu de données, SGD calcule le gradient sur un seul exemple $(x_i,y_i)$ à chaque itération.

La fonction de coût devient alors approximée pour chaque exemple :

$$J(\theta) \approx L(h_\theta(x_i),y_i)$$

Et à chaque itération on met à jour $\theta$ en utilisant le gradient d'un seul exemple au lieu de la somme de tous les exemples :
$$\theta = \theta - \alpha\nabla_\theta L(h_\theta (x_i),y_i)$$

En introduisant de la variabilité, on ajoute de l'irrégularité qui permet de vesqui des optimum locaux et on vesqui aussi le surapprentissage
## Mini-batch SGD

Plutôt qu'une seule valeur, on peut tout aussi bien prendre un petit ensemble aléatoire de données. on appelle ça le **mini-batch SGD**.

Supposons que l'ensemble de données contienne $m$ exemples, que nous appelons $(x_1,y_1),(x_2,y_2),(x_3,y_3),\dots,(x_m,y_m)$. Au lieu de calculer le gradient sur l'ensemble des $m$ exemples comme dans la descente de gradient classique, ou sur un seul exemple comme dans le SGD pur, on choisit un **mini-batch** de taille $B$, avec $B$ exemples pris au hasard.

la fonction de coût pour un mini-batch est alors :

$$J_{mini\_batch}(\theta)=\frac{1}{B}\sum_{i\in batch} L(h_\theta(x_i),y_i)$$

Ainsi, à chaque itération, on met à jour les paramètres $\theta$ selon la règle suivante :
$$\theta = \theta - \alpha\nabla J_{mini\_batch}(\theta)$$
Le mini-batch SGD permet de réduire la variabilité du SGD pur, sans forcément être dans l'excès comme la descente de gradient classique