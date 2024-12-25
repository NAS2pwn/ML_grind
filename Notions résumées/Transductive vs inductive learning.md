Le transductive learning et le inductive learning sont deux paradigmes d'apprentissage supervisé qui diffèrent par leur approche face aux données et leur objectif final. Ces concepts sont particulièrement utiles pour comprendre comment un modèle exploite les données disponibles pour faire des prédictions.

- **Inductive learning** : L'objectif est de généraliser à partir des données d'entraînement pour pouvoir prédire sur des données futures (non observées pendant l'entraînement). Autrement dit, on construit un modèle qui apprend une règle générale, applicable à des données nouvelles.
    
- **Transductive learning** : Ici, l'objectif est de prédire uniquement pour des points spécifiques (généralement des données test ou inconnues mais disponibles à l'avance) sans chercher à généraliser à des données hors de cet ensemble. L'accent est mis sur une correspondance directe entre les données connues (entraînement) et les données à prédire.

## Exemple

Considérons un problème de classification de courriels en **spam** ou **non-spam**.

- **Inductive learning** :  
    Imaginons que vous avez un ensemble d'entraînement avec 1 000 courriels étiquetés comme spam ou non-spam. Un modèle inductif va apprendre une règle générale (par exemple, une fonction basée sur des mots-clés ou une probabilité bayésienne). Cette règle est ensuite utilisée pour classer tout nouveau courriel, même s'il provient d'un autre utilisateur ou d'un autre contexte.
    
- **Transductive learning** :  
    Prenons le même ensemble d'entraînement. Supposons que vous connaissez à l'avance 500 courriels non étiquetés sur lesquels vous devez faire une prédiction (les courriels à classer). Un modèle transductif va s'appuyer non seulement sur les 1 000 courriels étiquetés, mais aussi sur les caractéristiques spécifiques des 500 courriels non étiquetés pour affiner ses prédictions. Ici, il ne s'agit pas d'apprendre une règle générale, mais d'optimiser uniquement pour cet ensemble de courriels précis.

## Formalisation

Soient :

- $X_{train}$​ : les données d'entraînement avec leurs étiquettes associées YtrainY_{train}Ytrain​,
- $X_{test}$ : les données sur lesquelles on souhaite faire des prédictions,
- $f$ : une fonction que l'on apprend pour prédire les étiquettes.

1. **Inductive learning** :  
    Le modèle apprend une fonction $f$ telle que :
$$f(X_{train}) \approx Y_{train}$$
    Puis $f$ est appliqué à $X_{test}$, indépendamment des spécificités de $X_{test}​$. Le modèle cherche à généraliser, et l'objectif est souvent d'optimiser la performance moyenne sur des données inconnues.
    
2. **Transductive learning** :  
    Le modèle prend en compte les données $X_{test}$ dès le départ. Il apprend une fonction $f$ qui minimise l'erreur sur :
   $$ f(X_{test}) \approx Y_{test}$$
    tout en exploitant les relations entre $X_{train}$​, $Y_{train}​$, et $X_{test}​$. Le modèle ne prétend pas prédire pour d'autres données hors de cet ensemble spécifique.

