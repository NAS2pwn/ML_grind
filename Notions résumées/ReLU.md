ReLU est l'une des fonctions d'activation les plus utilisés pour la construction de réseaux de neurones. 

Elle est très simple c'est $ReLU(x) = max(0, x)$

Son principal intérêt est d'introduire de la non-linéarité dans le modèle, ce qui permet au réseau de mieux apprendre des relations complexes dans les données.

Sans cette non-linéarité, un réseau ne serait capable de résoudre que des problèmes linéaires, limitant ainsi sa capacité à généraliser à des données plus variées.

En plus, elle atténue le problème du [[vanishing gradien]] : ReLU, en ne saturant pas pour les valeurs positives, permet aux gradients de rester plus grands et donc d'améliorer la vitesse d'apprentissage.

Cependant, elle peut aussi poser certains problèmes, notamment l'inactivation de neurones quand une sortie est toujours zéro, mais en pratique, ses avantages surpassent généralement ces inconvénients.