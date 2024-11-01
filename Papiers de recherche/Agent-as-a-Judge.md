Agent-as-a-Judge: Evaluate Agents with Agents Mingchen Zhuge

Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi, Vikas Chandra, Jürgen Schmidhuber
Meta AI, KAUST

## Résumé

Les méthodes d'évaluations actuelles ne sont pas adaptées aux systèmes agentiques ; soit ça ne prend pas en compte l'aspect pas à pas de ces systèmes, soit ça demande trop de travail à la mano.

Dans ce papier ils proposent une solution, le framework Agent-as-a-Judge, qui consiste simplement en des agents qui évaluent d'autres agents.

Dans ce papier ils expliquent la technique et montrent que ça marche mieux qu'un framework précédent : LLM-as-a-Judge

## Powerpoint

Voici un lien menant vers une présentation générale du sujet [[Evaluation des agents IA.pdf]]

## Fonctionnement technique

Je vais maintenant creuser le sujet d'un point de vue technique en vue d'une implémentation triviale