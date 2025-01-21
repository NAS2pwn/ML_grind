Le **Wilcoxon rank-sum test**, aussi appelé le **Mann-Whitney U test**, est un test statistique non paramétrique utilisé pour comparer deux échantillons indépendants. Il évalue si les deux groupes proviennent d'une même distribution ou si un groupe a tendance à avoir des valeurs plus grandes que l'autre.

### Principales caractéristiques :

- **Non paramétrique** : Il ne suppose pas que les données suivent une distribution normale.
- **Comparaison de deux groupes** : Utilisé lorsque vous voulez comparer les distributions de deux échantillons indépendants.
- **Données ordinales ou continues** : Fonctionne sur des données qui peuvent être ordonnées, mais ne nécessite pas des échelles d'intervalle ou de ratio.

---

### Comment fonctionne le test :

1. **Rangement des données** : Les observations des deux échantillons sont combinées, puis rangées par ordre croissant.
2. **Attribution des rangs** : Chaque observation se voit attribuer un rang. En cas d'ex-aequo, on attribue la moyenne des rangs concernés.
3. **Somme des rangs** : Pour chaque groupe, on calcule la somme des rangs.
4. **Statistique de test** : Une statistique UUU ou WWW est calculée à partir des rangs pour mesurer la différence entre les groupes.
5. **Signification statistique** : On compare UUU à une distribution critique (ou on utilise une ppp-valeur) pour déterminer si la différence est significative.

---

### Hypothèses du test :

- **Hypothèse nulle (H0H_0H0​)** : Les deux échantillons proviennent de populations identiques (les distributions sont similaires).
- **Hypothèse alternative (H1H_1H1​)** : Les deux échantillons proviennent de populations différentes (les distributions diffèrent).

---

### Utilisations typiques :

1. **Comparaison de groupes indépendants** : Ex. : Comparer les scores de performance de deux groupes d'étudiants (un groupe utilisant une méthode pédagogique différente).
2. **Données asymétriques ou non normales** : Lorsque la normalité des données ne peut pas être supposée.
3. **Petits échantillons** : Approprié pour des échantillons de petite taille où les tests paramétriques, comme le ttt-test, ne sont pas fiables.

---

### Exemple simple :

Imaginons deux groupes :

- Groupe A : [7, 8, 5, 6]
- Groupe B : [9, 10, 6, 8]

Les données combinées : [5, 6, 6, 7, 8, 8, 9, 10].  
Rangs : [1, 2.5, 2.5, 4, 5.5, 5.5, 7, 8].  
Somme des rangs pour :

- Groupe A : 4+5.5+1+2.5=134 + 5.5 + 1 + 2.5 = 134+5.5+1+2.5=13,
- Groupe B : 7+8+2.5+5.5=237 + 8 + 2.5 + 5.5 = 237+8+2.5+5.5=23.

Le test détermine si cette différence dans les sommes des rangs est significative.

---

### Limites :

- Ne donne pas d'information sur l'ampleur de l'effet (seulement une différence statistique).
- Ne fait pas de distinction sur les formes exactes des distributions (il compare les tendances centrales, pas les variances ou autres aspects).

Cela en fait un outil précieux lorsque les hypothèses des tests paramétriques ne sont pas respectées.