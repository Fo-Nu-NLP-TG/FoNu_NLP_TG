# Mini-Cours : Comprendre les Scores d'Attention dans les Transformers

## Introduction

Les scores d'attention sont au cœur du mécanisme d'attention, l'innovation clé qui a rendu les modèles Transformer si puissants. Ce mini-cours explique ce que sont les scores d'attention, comment ils sont calculés et pourquoi ils sont si importants dans le traitement du langage naturel.

## Table des matières

1. [Qu'est-ce que l'attention ?](#quest-ce-que-lattention-)
2. [Calcul des scores d'attention](#calcul-des-scores-dattention)
3. [Visualisation des scores d'attention](#visualisation-des-scores-dattention)
4. [Multi-Head Attention](#multi-head-attention)
5. [Impact des scores d'attention](#impact-des-scores-dattention)
6. [Exercices pratiques](#exercices-pratiques)

## Qu'est-ce que l'attention ?

L'attention est un mécanisme qui permet à un modèle de se concentrer sur certaines parties d'une séquence d'entrée lors du traitement d'un élément spécifique. C'est comme lorsque vous lisez une phrase complexe et que vous vous concentrez sur certains mots clés pour comprendre le sens global.

Dans les modèles Transformer, l'attention permet de :
- Capturer les dépendances à longue distance entre les mots
- Traiter les séquences en parallèle (contrairement aux RNN)
- Créer des représentations contextuelles riches pour chaque mot

## Calcul des scores d'attention

Les scores d'attention sont calculés en trois étapes principales :

### 1. Projection des vecteurs Query, Key et Value

Chaque mot de la séquence est d'abord transformé en trois vecteurs différents :
- **Query (Q)** : Ce que nous cherchons
- **Key (K)** : Ce avec quoi nous comparons
- **Value (V)** : Ce que nous récupérons

Ces vecteurs sont obtenus par projection linéaire du vecteur d'embedding initial :

```
Q = W_Q * X
K = W_K * X
V = W_V * X
```

Où X est le vecteur d'embedding et W_Q, W_K, W_V sont des matrices de poids apprenables.

### 2. Calcul des scores de compatibilité

Les scores de compatibilité sont calculés en prenant le produit scalaire entre chaque Query et toutes les Keys :

```
Score = Q * K^T
```

Ce score indique à quel point chaque mot (représenté par sa Key) est pertinent pour le mot actuel (représenté par sa Query).

### 3. Normalisation et pondération

Les scores sont ensuite normalisés par la racine carrée de la dimension des vecteurs de Key pour stabiliser l'apprentissage :

```
Score_normalisé = Score / √d_k
```

Puis, une fonction softmax est appliquée pour obtenir une distribution de probabilité :

```
Attention_weights = softmax(Score_normalisé)
```

Enfin, ces poids sont utilisés pour calculer une somme pondérée des vecteurs Value :

```
Output = Attention_weights * V
```

La formule complète du mécanisme d'attention est donc :

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

## Visualisation des scores d'attention

Les scores d'attention peuvent être visualisés sous forme de cartes de chaleur (heatmaps) pour comprendre comment le modèle se concentre sur différentes parties de la séquence.

Exemple de visualisation pour la phrase "Le chat dort sur le tapis" :

```
   | Le | chat | dort | sur | le | tapis |
---+----+------+------+-----+----+-------|
Le | 0.7|  0.1 |  0.05| 0.05| 0.1|  0.0  |
chat| 0.1|  0.6 |  0.2 | 0.05| 0.0|  0.05 |
dort| 0.0|  0.3 |  0.5 | 0.1 | 0.0|  0.1  |
sur | 0.0|  0.1 |  0.1 | 0.6 | 0.1|  0.1  |
le  | 0.1|  0.0 |  0.0 | 0.1 | 0.3|  0.5  |
tapis| 0.0|  0.05|  0.05| 0.1 | 0.2|  0.6  |
```

Dans cette visualisation, chaque cellule représente le poids d'attention entre le mot de la ligne et le mot de la colonne. Plus la valeur est élevée (et la couleur foncée), plus l'attention est forte.

## Multi-Head Attention

Dans la pratique, les Transformers utilisent l'attention multi-têtes (Multi-Head Attention), qui consiste à exécuter plusieurs mécanismes d'attention en parallèle, chacun avec ses propres matrices de projection W_Q, W_K et W_V.

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) * W_O

où head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
```

Avantages de l'attention multi-têtes :
- Chaque tête peut se concentrer sur différents aspects de la séquence
- Certaines têtes peuvent capturer la syntaxe, d'autres la sémantique
- Améliore la capacité du modèle à modéliser des relations complexes

## Impact des scores d'attention

Les scores d'attention ont un impact majeur sur les performances des modèles Transformer :

1. **Contextualisation** : Ils permettent de créer des représentations contextuelles des mots, tenant compte de leur environnement
2. **Désambiguïsation** : Ils aident à résoudre les ambiguïtés en se concentrant sur les mots pertinents
3. **Traduction** : Dans la traduction automatique, ils alignent implicitement les mots entre les langues source et cible
4. **Interprétabilité** : Ils offrent un aperçu de la façon dont le modèle "raisonne" sur le texte

## Exercices pratiques

### Exercice 1 : Calcul manuel des scores d'attention

Considérons une séquence de 3 mots avec des vecteurs d'embedding simplifiés de dimension 2 :
- Mot 1 : [1, 0]
- Mot 2 : [0, 1]
- Mot 3 : [1, 1]

Avec des matrices de projection simplifiées :
- W_Q = [[1, 0], [0, 1]]
- W_K = [[1, 0], [0, 1]]
- W_V = [[1, 0], [0, 1]]

Calculez les scores d'attention pour cette séquence.

### Exercice 2 : Analyse d'une carte d'attention

Observez la carte d'attention suivante pour la traduction de "The cat sits on the mat" vers "Le chat est assis sur le tapis" :

```
            | The | cat | sits | on  | the | mat |
------------+-----+-----+------+-----+-----+-----|
Le          | 0.8 | 0.1 | 0.0  | 0.0 | 0.1 | 0.0 |
chat        | 0.1 | 0.8 | 0.1  | 0.0 | 0.0 | 0.0 |
est         | 0.0 | 0.1 | 0.4  | 0.3 | 0.1 | 0.1 |
assis       | 0.0 | 0.1 | 0.7  | 0.1 | 0.0 | 0.1 |
sur         | 0.0 | 0.0 | 0.1  | 0.8 | 0.1 | 0.0 |
le          | 0.1 | 0.0 | 0.0  | 0.1 | 0.7 | 0.1 |
tapis       | 0.0 | 0.0 | 0.0  | 0.1 | 0.1 | 0.8 |
```

Questions :
1. Quels mots sont fortement liés dans la traduction ?
2. Y a-t-il des mots qui prêtent attention à plusieurs mots source ?
3. Comment l'attention aide-t-elle à la traduction correcte de "sits" en "est assis" (deux mots en français) ?

## Conclusion

Les scores d'attention sont un mécanisme fondamental qui a transformé le domaine du NLP. En permettant aux modèles de se concentrer dynamiquement sur différentes parties d'une séquence, ils ont ouvert la voie à des architectures plus performantes et plus interprétables.

Dans les modèles Transformer modernes comme BERT, GPT et T5, les mécanismes d'attention sont devenus encore plus sophistiqués, avec des variantes comme l'attention sparse, l'attention locale, ou l'attention avec des contraintes structurelles.

Comprendre les scores d'attention est essentiel pour quiconque souhaite maîtriser le fonctionnement interne des modèles de langage modernes et développer des applications NLP avancées.

---

## Ressources complémentaires

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) par Jay Alammar
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Article original sur les Transformers
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Implémentation annotée du Transformer
- [Visualizing Attention in Transformer-Based Language Models](https://towardsdatascience.com/visualizing-attention-in-transformer-based-language-models-9a1d0c2c4c10) - Article sur la visualisation de l'attention
