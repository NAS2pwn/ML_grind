[Lien vers le cours entier](Polyseriestemp.pdf)

On peut voir une série temporelle comme une suite d'observations répétées d'un même phénomène à des dates différentes

Les dates sont souvent équidistantes, mais pas nécessairement.

On représente habituellement une série temporelle $(x_t)_{1\leq t\leq T}$ avec $t$ le numéro de l'observation

**Problèmes**

- Prévision : peut-on partir des valeurs $x_1,...,x_T$ pour avoir une idée des valeurs futures $x_{T+1},x_{T+2},...$ ? Evidemment on peut peut pas connaître la dynamique exacte de la série. On pourra toujours tenir compte des valeurs passées ou d'une information auxiliaire mais il existera toujours une composante aléatoire dont il faudra tenir compte, d'autant plus que cette composante est en général autocorrélée. Ainsi, les valeurs  $x_1,...,x_T$ seront supposées être des réalisations de variables aléatoires  $X_1,...,X_T$, dont il faudra spécifier la dynamique. La prévision pourra alors se faire en estimant la projection de $X_{T+1}$ sur un ensemble de fonctionnelles de  $X_1,...,X_T$ (projection linéaire, espérance conditionnelle...). La modélisation doit aussi permettre d'obtenir des intervalles de confiance pour ce type de prévision.
- Résumer la dynamique : des valeurs considérées en enlevant les détails du court terme ou les fluctuations saisonnières. On est intéressés ici par l'estimation d'un tendance. Les fluctuations saisonnières captent un comportement qui se répète avec une certaine périodicité.
- Interpréter le lien entre plusieurs variables : ou interpréter l'influence des valeurs passées d'une variable sur sa valeur présente demande de retrancher les composantes tendancielles et saisonnières (sinon on trouverait des corrélations importantes alors qu'un caractère explicatif est peu pertinent).
- Les séries temporelles multivariées (qui correspondent à l'observation simultanée de plusieurs séries temporelles) : mettent en évidence les effets de corrélation et de causalité entre différentes variables. Il peut alors être intéressant de savoir si les valeurs prises par une variable $x^{(1)}$ sont la conséquence des valeurs prises par la variable $x^{(2)}$ ou le contraire et de regarder les phénomènes d'anticipation entre ces deux variables.
- D'autres problèmes plus spécifiques peuvent aussi se poser : détection de rupture de tendance qui permettent de repérer des changements profonds en macroéconomie, prévision des valeurs extrêmes en climatologie ou en finance, comprendre si les prévisions faites par les entreprises sont en accord avec la conjoncture...

1. **Prévision des valeurs futures**
    - **Contexte** : Utilisation des valeurs passées pour estimer les valeurs futures avec incertitude (composante aléatoire).
    - **Méthodes associées** :
        - **Lissage exponentiel** : Simple, double, ou méthode de Holt-Winters pour les séries avec ou sans saisonnalité.
        - **Modèles ARMA/ARIMA** : Processus autorégressifs et de moyennes mobiles pour modéliser les dépendances stochastiques.
        - **Box et Jenkins** : Approche structurée pour choisir et diagnostiquer les modèles ARIMA.
        - **Prévision probabiliste** : Inclut des intervalles de confiance basés sur la dynamique estimée.
        
1. **Détection des tendances et des fluctuations saisonnières**
    - **Contexte** : Identifier une tendance sous-jacente ou des fluctuations récurrentes dans les séries.
    - **Méthodes associées** :
        - **Décomposition additive ou multiplicative** : Séparer les composantes (tendance, saisonnalité, bruit).
        - **Moyennes mobiles (arithmétiques ou de Henderson)** : Estimation robuste des tendances en éliminant les fluctuations saisonnières.
        - **Désaisonnalisation** : Via régression linéaire ou moyennes mobiles, comme illustré avec les données SNCF.
        
1. **Analyse multivariée et causalité**
    - **Contexte** : Étudier les relations entre différentes séries temporelles (corrélation, causalité, anticipation).
    - **Méthodes associées** :
        - **Modèles VAR (Vecteur Autorégressif)** : Étude des interactions multivariées.
        - **Tests de causalité de Granger** : Identifier des relations causales.

1. **Détection de ruptures et valeurs extrêmes**
    - **Contexte** : Repérer des changements structurels (e.g., macroéconomie) ou analyser des valeurs extrêmes (finance, climat).
    - **Méthodes associées** :
        - **Tests statistiques (Dickey-Fuller, ADF)** : Détection de non-stationnarité ou de ruptures.
        - **Approches bayésiennes** : Détection de ruptures probabilistes.
        - **Analyse des queues** : Modèles extrêmes (e.g., EVT - Extreme Value Theory).

**Méthodes de base et avancées**

1. **Décomposition additive/multiplicative**
    - **Additive** : $X_t = m_t + s_t + U_t$.
    - **Multiplicative** : $X_t = m_t \cdot s_t \cdot U_t$​.  
        Utilisée selon la stabilité ou la proportionnalité des effets saisonniers.
        
2. **Modèles ARIMA et SARIMA**
    - Intègrent les aspects saisonniers et les différences nécessaires pour stationnariser la série.
    - Méthode itérative Box-Jenkins pour ajuster $(p, d, q)$ et leurs composantes saisonnières.
    
3. **Lissage exponentiel**
    - **Simple** : Prévision des séries stationnaires.
    - **Double** : Pour capturer les tendances linéaires.
    - **Holt-Winters** : Additif/multiplicatif pour les séries avec tendance et saisonnalité.
    
4. **Traitement des séries multivariées**
    - Modèles VAR et VARMA pour capturer la dynamique conjointe.
    - Réduction de dimensions avec PCA si de nombreuses séries.

**Exemple de workflow**

1. **Analyse exploratoire** : Graphiques, tests d’autocorrélation, identification de composantes stationnaires/non-stationnaires.
2. **Prétraitement** : Désaisonnalisation, différenciation, ou transformations log.
3. **Modélisation** : ARIMA/SARIMA, lissage, ou modèles VAR selon les objectifs (prévision, causalité).
4. **Validation** : Résidus, tests de portemanteau, intervalles de confiance.

## Modélisations de base pour les séries temporelles

**La décomposition additive** consiste à diviser une série $X_t$ en trois composantes principales :

$$X_t = m_t + s_t + U_t,\quad 1\leq t \leq T$$
où
- $(m_t)_t$ est une composante tendancielle déterministe qui donne le comportement de la variable observée sur le long terme (croissance ou décroissance linéaire, quadratique...). Cette composante peut aussi avoir une expression différente pour différentes périodes (affine par morceaux par exemple). Si on prend par exemple une time serie sur le prix de l'immobilier au m2 dans un quartier, et que ce prix là augmente à la même vitesse chaque année alors on sera sur une tendance affine. Plus généralement, on peut voir cette composante comme une fonction lisse du temps $t$
- $(s_t)_t$ est une suite périodique qui correspond à une composante saisonnière (par exemple de période 12 pour les séries du trafic voyageur puisque c'est annuel, période 4 pour des séries trimestrielles, 24 pour des séries horaires...). Une somme de plusieurs suites de ce type peuvent être pertinentes par exemple une série de températures horaires observées sur plusieurs années nécessite la prise en compte d’une périodicité quotidienne et annuelle)
- $(U_t)_t$ représente une composante irrégulière et aléatoire, le plus souvent de faible amplitude par rapport à la composante saisonnière mais importante en pratique puisque ce terme d'erreur sera le plus souvent autocorrélé (c'est-à-dire que la covariance entre $U_t$ et $U_{t+h}$ sera non nulle). On verra son utilité et les types de modèles qui peuvent être utilisés our l'étudier plus tard.

**La décomposition multiplicative** consiste à représenter une série $X_t$​ cette fois comme le produit de ces trois composantes

....

