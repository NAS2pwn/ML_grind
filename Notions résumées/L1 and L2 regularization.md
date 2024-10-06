L2 et L1 regularization sont des outils essentiels en machine learning pour éviter le surapprentissage, surtout quand on a beaucoup de paramètres par rapport à la quantité de données disponibles.

Ces méthodes de régularisation ajoutent une pénalité aux poids du modèle afin d'améliorer sa capacité à généraliser sur des données nouvelles.

Cette approche devient cruciale lorsque le nombre de caractéristiques dépasse largement le nombre de points de données, ce qui rend le modèle susceptible de s'ajuster parfaitement aux données d'entraînement mais de mal performer sur des données inconnues.

### L2 Regularization (Ridge Regression)

L2 regularization, ou Ridge, pénalise les grandes valeurs des poids en ajoutant une somme des carrés des poids dans la fonction de coût. Cela force les poids à se rapprocher de zéro, sans toutefois les annuler complètement. Cette technique est particulièrement utile dans des situations où chaque paramètre apporte une contribution pertinente, mais que l'on veut éviter qu'ils deviennent trop dominants, ce qui risquerait de provoquer un surapprentissage.

Imaginons que nous avons un modèle de régression avec beaucoup de caractéristiques, mais seulement quelques échantillons. Sans régularisation, le modèle peut s'ajuster de manière très précise aux données d'entraînement, ce qui conduit à des prédictions inefficaces sur de nouveaux échantillons. L2 regularization résout ce problème en réduisant la complexité du modèle, permettant ainsi d'obtenir des prédictions plus stables, même quand le nombre d’échantillons est inférieur au nombre de paramètres. Cela se manifeste souvent dans des problèmes comme la régression linéaire où l'on cherche à minimiser une erreur quadratique, tout en évitant des solutions où les poids seraient exagérément grands.

Mathématiquement, la régularisation L2 s’écrit comme suit :
$$J_{ridge}​(\theta)=Erreur(\theta)+\lambda \sum_{i = 1}^{n}\theta_i^2$$​Le paramètre $\lambda$ contrôle la force de la régularisation. Plus $\lambda$ est grand, plus les poids sont contraints à être proches de zéro.
### L1 Regularization (Lasso Regression)

L1 regularization, aussi appelée Lasso, suit une approche différente en ajoutant une pénalité basée sur la somme des valeurs absolues des poids. Cela a pour effet de pousser certains poids à devenir exactement nuls. L'intuition derrière L1 est d'encourager la parcimonie (sparse solutions), c'est-à-dire un modèle où seules les caractéristiques les plus pertinentes sont sélectionnées. 
En fait, cette méthode est souvent utilisée pour la sélection automatique des caractéristiques, car elle élimine complètement les variables non importantes.

Si vous avez un dataset avec un grand nombre de caractéristiques, dont certaines sont peu ou pas informatives, L1 regularization va automatiquement mettre leurs coefficients à zéro, gardant seulement celles qui sont vraiment utiles.

Par exemple, dans des modèles de haute dimension comme ceux utilisés dans la reconnaissance d'images ou l'analyse génomique, L1 peut simplifier la tâche de sélection des variables sans avoir à appliquer des techniques de réduction de dimension séparées.

Mathématiquement, L1 regularization est définie comme suit :

$$J_{lasso}​(\theta)=Erreur(\theta)+\lambda \sum_{i = 1}^{n}|\theta_i|$$
Le paramètre $\lambda$ contrôle la force de la régularisation. Plus $\lambda$ est grand, plus les poids sont contraints à être proches de zéro.

### Définir $\lambda$

Pour déterminer la valeur optimale de $\lambda$ (le paramètre de régularisation), on utilise souvent la technique de la validation croisée (cross-validation). L'idée est de tester plusieurs valeurs de $\lambda$ en divisant les données d'entraînement en sous-ensembles, appelés _folds_. On entraîne alors le modèle sur une partie des folds et on évalue sa performance sur les folds restants. Ce processus est répété plusieurs fois avec différentes partitions des données, ce qui permet d'obtenir une estimation robuste de la performance pour chaque valeur de $\lambda$. Le $\lambda$qui minimise l'erreur de validation (ou maximise la performance, selon la métrique choisie) est alors sélectionné comme la valeur optimale. Cela permet d'éviter le choix arbitraire du paramètre et garantit que le modèle généralise mieux aux nouvelles données.
### Différence entre L1 et L2

La différence majeure entre L1 et L2 est leur effet sur les poids. L2 regularization réduit uniformément la taille de tous les poids, les rendant plus petits sans pour autant les annuler. Cela est utile lorsqu'on suppose que toutes les variables du modèle ont un effet non négligeable. D'un autre côté, L1 regularization pousse directement certains poids à zéro, en sélectionnant ainsi les caractéristiques importantes et en excluant les autres. Cela rend L1 particulièrement adapté aux situations où l'on suspecte que seules quelques variables ont un réel impact.

Si on applique ces techniques à des problèmes réels, dans une régression linéaire où le nombre de caractéristiques est grand mais seulement quelques-unes sont importantes, L1 va sélectionner les plus pertinentes et ignorer les autres. En revanche, si toutes les caractéristiques sont pertinentes mais qu'on veut éviter qu'une caractéristique domine les autres, L2 sera plus approprié.
### Utilisation combinée

Il est aussi possible de combiner L1 et L2 regularization dans une technique appelée **Elastic Net**. Cela permet d'avoir à la fois les avantages de la sélection de caractéristiques de L1 et la régularisation en douceur de L2. Elastic Net est souvent utilisé dans des contextes où il y a beaucoup de corrélation entre les caractéristiques, et où Lasso seul aurait du mal à choisir entre elles.

En résumé, L1 et L2 regularization sont des outils puissants pour contrôler la complexité d'un modèle, que ce soit pour améliorer la prédiction quand les données sont limitées, ou pour simplifier un modèle en sélectionnant automatiquement les caractéristiques les plus pertinentes.