http://www.laurentoudre.fr/ast.html

# Introduction

## Qu'est-ce qu'une série temporelle ?

Une série temporelle est une série de data points indexés dans l'ordre temporel

En pratique, des arrays de réels de taille D x N avec D le nombre de dimensions et N le nombre d'échantillons

Le même objet a différents noms selon la discipline scientifique :
- Séries temporelles : maths, stats, économie, finance...
- Signaux : théorie des signaux, physique, ingénierie, simulation
- Séquences : informatique, bioinformatique, datamining

On utilisera ces termes de façon interchangeable

Typical definition : real-valued (or at least ordered) sequential data
(c'est plus compréhensible en anglais)

Une série temporelle peut être :
- Univariée : si elle contient une seule variable observée à des instants successifs (ex : température, ventes quotidienne d'un produit unique)
- Multivariée : si elle contient plusieurs variables mesurées simultanément à chaque instant (ex : données météo complètes avec humidité, vitesse du vent... ou la vente de plusieurs produits pris distinctement, ou le mouvement d'acteurs, bref)

Les séries temporelles multivariées sont évidemment plus difficiles à analyser que les univariées, et courantes en cela qu'on a besoin de données complètes représentant une dynamique pragmatique pour vraiment développer des produits intéressants. Il sera nécessaire d'étudier les corrélations et causalités entre les variables, et de penser au dimensionnement, pour travailler avec de telles séries.

Les séries temporelles sont potentiellement massives (par exemple le son, samplé à une fréquence de 44.1 kHz)

Potentiellement multivariées, multimodales et hétérogènes

Souvent pleines de bruits, de données manquantes (surtout quand il est question de captations physiques), de tendances qui peuvent sembler contradictoires, et de syncrétisme des sources (par exemple si on a une entreprise avec un SI complexe, qu'on veut traiter les données de l'entreprise, et qu'on bricole pour avoir des données exploitables en les aggrégant, genre jsp si t'as 10 usines, et que tu récupères les mêmes données des 10 usines, peut y avoir un biais pour chaque usine ou un format différent qui est chiant à traiter)

Souvent liées à un champ d'application : l'ingénieur ou le data scientist qui travaille sur les données n'est pas forcément formé pour les comprendre pleinement, il a besoin d'un expert du métier avec qui échanger

Contrairement à du traitement d'image basique, annoter des time series nécessite une expertise

Typiquement :
- Données sales, bruitées
- Annotations floues (qui prête à confusion et est très spécialisée, ce qui rend difficile la création de class labels)
- Un expert avec plusieurs années dans le métier, mais incapable de traduire ça dans une annotation ML-compatible

**Comment utiliser le ML dans ce contexte**

La plupart des algos de ML s'en fichent du temps

Comment alors utiliser l'information de temps pour extraire des features et des patterns pertinents qui peuvent être utilisés dans le cadre du ML

Deux visions s'affrontent :
- Physique : la notion de temps a été utilisée et modélisée en physique depuis le 18è siècle et avant (eg. transformation de Fourier)
  Vision : une série temporelle $x[1: N]$ est le résultat de la numérisation d'un phénomène $x(t)$. Les propriétés physiques peuvent être retrouvées et analysées via l'étude de $x[1:N]$ (et vice versa)
- L'aléatoire joue aussi un rôle pour modéliser une classe plus large de signaux
  Vision : une série temporelle $x[1:N]$ est une réalisation d'un processus stochastique $X[1:N]$. Les propriétés statistiques de ce phénomène peuvent être retrouvées et analysées via l'étude de $x[1:N]$ (et vice versa)

Dans la plupart des cas, les deux approches sont combinées.

Le deep learning atteint des résultats à l'état de l'art pour plusieurs tâches MAIS
- De bonnes performances ne signifient pas une bonne compréhension de la donnée
- Le DL est une black box qui ne peut pas apporter satisfaction aux utilisateurs d'un domaine puisqu'ils ne peuvent pas interpréter les résultats
- Bien que certains réseaux de neurones peuvent gérer le temps (e.g. LSTM), ils ne peuvent gérer que max quelques centaines d'échantillons temporels
- Le DL est inefficace dans le contexte de données et d'annotations limitées (scarce je veux dire)

[Are Transformers Effective for Time Series Forecasting ?](https://arxiv.org/pdf/2205.13504)

Les auteurs remettent en question l'efficacité des Transformers pour la prévision à long terme des séries temporelles (LTSF). Bien qu'efficaces pour capturer les corrélations dans des séquences longues, les Transformers perdent des informations temporelles cruciales en raison de leur mécanisme d'attention invariant aux permutations. Ils proposent **LTSF-Linear**, un modèle linéaire simple, qui dépasse largement les performances des Transformers sur neuf jeux de données réels. Ces résultats suggèrent que des modèles simples peuvent être plus adaptés, appelant à reconsidérer l'utilisation du deep learning pour les séries temporelles, y compris d'autres tâches comme la détection d'anomalies.

[Learning representations for time series clustering. Advances in neural information processing systems](https://cseweb.ucsd.edu//~gary/pubs/NeurIPS_2019.pdf#cite.xie2016unsupervised)

Table 1, les algos de DL DEC et IDEC semblent pas super efficaces, en revanche DTCR, qui est du DL, est la meilleure solution (on parle de non supervisé)
Stp prend le temps de lire ça suffisamment pour comprendre l'essentiel stp je comprends pas ces méthodes

In this paper, we propose a novel model called Deep Temporal Clustering Representation (DTCR) to
generate cluster-specific representations. The general structure of DTCR is illustrated in Figure 1.
The encoder maps original time series into a latent space of representations. Then the representations are used to reconstruct the input data with the decoder. At the same time, a K-means objective is integrated into the model to guide the representation learning. Furthermore, we propose a fake-sample generation strategy and auxiliary classification task to enhance the ability of encoder.

[Time Series Machine Learning Results](https://www.timeseriesclassification.com/results.php)

Là je comprends même pas, je trouve pas les résultats affichés slide 18 donc à voir, mais grosso-modo ils performent pas super, faut voir plus précisément ce qu'il en est.

[Anomaly Detection in Time Series: A Comprehensive Evaluation (2022)](https://www.vldb.org/pvldb/vol15/p1779-wenig.pdf)

Là ils ont testé plein de méthodes de détection d'anomalies dans des séries temporelles, et y a une discussion intéressante à la fin

**In line with related work [67], we found that deep learning approaches are not (yet) competitive despite their higher processing effort on training data. We could also confirm that “simple methods yield performance almost as good as more sophisticated methods” [56].**

Still, no single algorithm clearly performs best. We highlighted several algorithms with specific strengths, but the overall performance results call for further research in the following three areas:

- **Flexibility**: No algorithm (or algorithm family) clearly dominates all other approaches and solves all anomaly detection setups.
  To advance the field of anomaly detection, we suggest further research on holistic and hybrid anomaly detection systems that combine existing strengths for the detection of more diverse anomalies in time series with arbitrary characteristics.
  
- **Reliability:** Despite our best efforts, only very few algorithms could process all time series without errors and within common time and memory limits. We therefore emphasize the importance of further research on the robustness and scalability of time series anomaly detection algorithms.
  
- **Simplicity:** Most anomaly detection algorithms of this study were remarkably sensitive to their parameter settings and required on average seven settings.
  What makes this problem worse is that most practical use cases do not have training data for algorithm configuration.
  For this reason, further research on auto-configuring and self-tuning algorithms is very much needed.

[How (not) to use Machine Learning for time series forecasting: Avoiding the pitfalls](https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424)

On peut penser avoir un bon modèle, et en fait se tromper totalement. Se baser sur les métriques d'erreur classiques style mean percentage error ou R2 score peut induire en erreur si ce n'est pas fait avec précaution.

On va traiter l'exemple de l'article, en gros il a voulu prédire l'évolution d'un index boursier avec LSTM, on dirait qu'il a atteint une super accuracy comme on le voit sur le graphe figure 1 et le r2 score figure 2

![[fig1.webp]]

Figure 1 : Prédiction vs test set

![[fig2.webp]]
Figure 2 : R2 score

En fait, il est impossible que la prédiction soit basée sur des motifs qui ont du sens pour la simple et bonne raison qu'il ne s'agit pas d'un vrai indice boursier mais d'un random walk process, un processus complètement stochastique. Il n'y a pas de logique, que du hasard

Comment alors le modèle a pu prévoir la suite de la série temporelle si précisément ?

En fait la raison est assez simple, la série est autocorrelée, la valeur à t+1 est liée à la valeur t

Pour résumer : le modèle ne faisait que répéter le motif précédent, arrivant donc à une prédiction accurate selon tout indicateur mais incapable de raisonner autrement.

blablabla


Bref en gros pour détecter ça, une bonne façon de faire c'est de voir si on peut la rendre stationnaire. Pour rappel, une série temporelle est dite stationnaire si ses propriétés statistiques restent constantes, typiquement : moyenne, variance, structure d'autocorrelation

Exemples de signaux stationnaires ou non

**White Noise (Bruit blanc) : Stationnaire**

- Le bruit blanc est stationnaire car chaque événement a une probabilité égale de se produire, indépendamment des autres événements et du temps.

**Coloured Noise (Bruit coloré) : Stationnaire**

- C'est un bruit blanc filtré (par ex. le bruit rose) où les probabilités entre événements voisins ne sont pas égales.
- Cependant, si les probabilités ne changent pas au cours du temps, le processus reste stationnaire.

**Chirp (Signal montant/descendant) : Non stationnaire**

- Les probabilités entre événements changent dans le temps, car la fréquence du signal augmente ou diminue. Cela modifie la dynamique entre les échantillons.

**Sinusoïde : Stationnaire**

- Une sinusoïde simple a des probabilités constantes entre événements, donc elle est stationnaire.

**Somme de sinusoïdes : Stationnaire**

- Si les périodes et amplitudes des composantes ne changent pas dans le temps, le processus reste stationnaire, car les contraintes entre échantillons sont fixes.

**ECG/EEG (Signaux biologiques) : Non stationnaires**

- Ces signaux reflètent des processus biologiques modulés par des facteurs externes (par exemple, l'activité cérébrale ou la variabilité du rythme cardiaque), donc ils changent dans le temps.
- Exception : sur des fenêtres d'observation courtes (par ex. 30 secondes sous des conditions fixes), ils peuvent être considérés comme approximativement stationnaires.

Bref, pour voir ça on peut utiliser le time differencing, genre `diff()` en gros la différence entre les valeurs successives de la série

TODO : Lire le sequel en se concentrant

https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd-1f/

## Data science pour les séries temporelles

La data science n'est pas lancer des packages de DL jusqu'à obtenir les meilleurs résultats

La datascience vise aussi à comprendre la data, interagir avec des experts, apporter de l'intelligence humaine et de l'expertise et améliorer un savoir

L'intelligence artificielle ne peut pas être intelligente si le data scientist ne l'est pas

Appliquer des méthodes de DL complexes n'exempte pas de la phase de recherche préliminaire


Bref, un datascientist passe plus de temps à collecter de la data et surtout la nettoyer, l'organiser, qu'à faire tourner des algos de ML ou autre

Grosso modo :
- Comprendre la data pour extraire ce qui est pertinent
- Comprendre ce qu'on fait, pourquoi on le fait, comment on le fait : interprétabilité

La représentation est importante, on peut chercher à voir :
- La fréquence
- La saisonnalité
- ... (TODO : remplis cette liste ça te fera gamberger)

Les tâches principales de ML principales pour les séries temporelles sont :
- Prédiction : Prédire les valeurs futures de la série temporelle
- Complétion/interpolation : Combler les trous dans une série temporelle
- Classification : de signaux complets ou de subsequence
- Clustering
- Requête par contexte/indexation : Etant donnée une série temporelle d'entrée, retrouver les séries temporelles les plus proches dans une DB
- Segmentation/detection de points de changement
- Détection d'anomalies
- Extraction de patterns

Les tâches cachées sont :
- Comprendre la donnée : comprendre d'où elles viennent, comment elles sont acquises, quelles sont ses caractéristiques, interagir avec des experts du domaine et comprendre leurs problèmes
- Améliorer la donnée : trouver des espaces de représentation pertinents dans lequels les évènement intéressants peuvent être vus, consolider la donnée (débruiter, detrend, détecter et enlever les valeurs aberrantes)
- Modeler la donnée : modèles physiques/statistiques ou basés sur les experts, simples, adaptifs et interprétables
- Extraire de l'information de la donnée : trouver les patterns répetitifs, les features intéressantes, les change-points, les anomalies...