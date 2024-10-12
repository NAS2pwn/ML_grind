Le Dropout est une technique permettant de réduire l'overfitting lors de l'entraînement d'un modèle.

Il s'agit de supprimer (de drop out) des neurones, ainsi que les connexions entrantes et sortantes, pendant l'entraînement.

Le choix des neurones à désactiver se fait aléatoirement, avec une probabilité $p$ qui est la même pour chaque neurone.

A chaque epoch, on applique le drop out. Cela signifie qu'à chaque passe (forward propagation), le modèle apprendra avec une configuration de neurones différente, les neurones n'étant pas les mêmes à se désactiver à chaque fois.

