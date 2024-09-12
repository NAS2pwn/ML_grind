## Softmax

https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/

La fonction d'activation Softmax est souvent placée sur l'output layer des réseaux de neurones.

Elle est souvent utilisée dans les problèmes d'apprentissage multi-classes à un ensemble de features peuvent être liées à l'une de K classes. Elle permet de représenter la confiance du modèle dans ses prédictions, sous une forme si on veut de probabilité.

En gros c'est l'exponentiel de la sortie sur la somme des exponentielles de toutes les sorties.

#### Pourquoi on utilise softmax plutôt qu'une normalisation standard ?

Le Softmax a un avantage intéressant par rapport à la normalisation standard.

Il réagit à une faible stimulation (imagine une image floue) de ton réseau de neurones avec une distribution plutôt uniforme, et à une forte stimulation (par exemple, des nombres élevés, imagine une image nette) avec des probabilités proches de 0 et 1.

Alors que la normalisation standard ne se soucie pas de la taille des valeurs tant que les proportions restent les mêmes.

Regarde ce qui se passe quand le Softmax reçoit des entrées 10 fois plus grandes, c'est-à-dire quand ton réseau de neurones a reçu une image nette et que de nombreux neurones se sont activés.

softmax([1,2]) -> [0.27, 0.73]  Mmmmmmmh est-ce un chat roux, est-ce un renard ?
softmax([10,20]) -> [0.000045, 0.99] CHAAAT, cheef cheeeeef c un chat

std_norm([1,2]) -> [0.33, 0.66] Mmmmmmmh est-ce un chat roux, est-ce un renard ?
std_norm([10,20]) -> [0.33, 0.66] Mmmmmmmh est-ce un chat roux, est-ce un renard ?

## Negative Log-Likelihood (NLL)

En pratique le softmax est utilisé en tandem avec le NLL. C'est fonction de perte est très intéressante si on l'interprète en relation avec le comportement du softmax.

L(y) = -log(y)

Bref en gros plus l'output du softmax est elevé, plus la loss est faible, regarde sur l'article la courbe voilà t'as vu

## Calcul de la dérivée

C'est très important de pouvoir calculer la dérivée de la loss par rapport au vecteur de classes, pour la backpropagation par exemple. Ecoute tout est sur l'article, y a quelques trucs à comprendre mais c'est basique, et finalement ça donne (pk-1) voilà


https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/