Un LSTM (Long Short-Term Memory) est un type de réseau de neurones récurrents (RNN) conçu pour mieux traiter et mémoriser des séquences longues. Il surmonte les limites des RNN classiques, notamment l'oubli à long terme.

### 1. Pourquoi les LSTM ?

Les RNN classiques propagent l'information d'une étape à l'autre à travers des cellules, mais sur de longues séquences, ils perdent des informations importantes. C’est ce qu’on appelle le **problème du gradient qui disparaît**, où les informations plus anciennes deviennent trop faibles pour influencer le résultat final. Les LSTM sont conçus pour gérer ces séquences en **mémorisant les informations pertinentes** sur de longues périodes, tout en oubliant celles qui ne sont plus utiles.

### 2. Structure d'un LSTM

Un LSTM se compose de **cellules de mémoire** et de **portes**. Les portes contrôlent le flux d’informations, permettant au réseau de décider quoi mémoriser, quoi oublier, et quoi utiliser pour produire une sortie. Il y a trois portes principales :

- **Porte d’oubli** : Elle décide quelles informations de l’état précédent doivent être oubliées.
- **Porte d’entrée** : Elle sélectionne quelles nouvelles informations doivent être ajoutées à la mémoire.
- **Porte de sortie** : Elle détermine quelles parties de la mémoire vont influencer la sortie à cette étape.

### 3. Fonctionnement étape par étape

#### Étape 1 : Déterminer ce qu’on oublie

La première étape dans une cellule LSTM consiste à appliquer la porte d’oubli. Elle regarde la sortie précédente ($h_t-1$) et l’entrée actuelle ($x_t$​), et décide quels éléments de la cellule précédente doivent être oubliés. Mathématiquement, cela se fait par une **sigmoïde**, qui produit une valeur entre 0 (oublier complètement) et 1 (garder totalement).
$$f_t = \sigma (W_f\,\cdot[h_{t-1},x_t]+b_f)$$
#### Étape 2 : Mise à jour de la mémoire

Ensuite, la porte d’entrée choisit quelles informations nouvelles seront ajoutées à la cellule. Deux choses se passent ici :

- On applique une **sigmoïde** pour savoir quelles informations mettre à jour ($i_t$).
- On crée un vecteur de nouvelles informations potentielles via une fonction **tanh** ($\tilde{C}_t$).

$$i_t = \sigma(W_i\,\cdot[h_{t-1},x_t]+b_i)$$
$$\tilde{C}_t = f_t\,*C_{t-1}+i_t*\tilde{C}_t$$
#### Étape 4 : Calcul de la sortie

Enfin, la porte de sortie décide quelles informations extraites de la cellule actuelle vont être utilisées pour la sortie de l'étape ttt. Une **sigmoïde** décide quelles parties de la cellule seront utilisées, suivie d’une fonction **tanh** pour normaliser.

$$o_t=\sigma(W_o\,\cdot[h_{t-1},x_t]+b_o)$$
$$h_t=o_t*tanh(C_t)$$
### 4. Intérêt concret

L’intérêt des LSTM réside dans leur capacité à capturer **les dépendances à long terme** dans des données séquentielles. Cela est essentiel dans des tâches comme la traduction automatique, la reconnaissance vocale ou l’analyse de séries temporelles, où les informations anciennes influencent les résultats actuels. Contrairement aux RNN classiques, qui oublient rapidement, les LSTM décident intelligemment quelles informations mémoriser et lesquelles oublier, permettant une meilleure performance dans ces applications.

Ainsi, les LSTM représentent une solution élégante au défi d'apprentissage séquentiel dans le temps long.