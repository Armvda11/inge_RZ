# Analyse Topologique Simplifiée

Ce document explique comment utiliser l'outil simplifié d'analyse topologique du réseau de satellites.

## Objectif

L'outil permet d'analyser la topologie du réseau de satellites en se concentrant sur 4 métriques clés :

1. **Degré moyen (⟨k⟩)** : Nombre moyen de connexions par satellite
2. **Taille de la composante géante (|Gₘₐₓ|/N)** : Fraction des nœuds qui restent connectés
3. **Longueur moyenne des plus courts chemins (ℓ̄)** : Proxy pour la latence minimale
4. **Coefficient de clustering (C)** : Mesure la redondance locale (triangles)

## Fonctionnement

L'analyse évalue l'impact de deux types de pannes sur le réseau :

- **Pannes prévisibles** : Les satellites avec les centralisés les plus élevées tombent en panne à un instant T prédéfini
- **Pannes aléatoires** : À chaque instant, chaque satellite a une probabilité p de tomber en panne

## Utilisation

1. Exécuter l'analyse :

```bash
./analyse_topologique.sh
```

2. Les résultats seront sauvegardés dans le dossier `data_logs/topologie/` :
   - `metriques_topologiques.png` : Graphique des 4 métriques pour les 3 scénarios
   - `metriques_topologiques.csv` : Données brutes pour analyse ultérieure

## Configuration

Les paramètres de simulation peuvent être modifiés dans le fichier `code/config.py` :

- `T_PRED` : Instant où les satellites prédictibles tombent en panne
- `N_PRED` : Nombre de satellites prédictibles qui tombent en panne
- `P_FAIL` : Probabilité de panne aléatoire à chaque pas de temps
- `MAXTEMPS` : Durée totale de la simulation
- Portées de connexion : `MIN_RANGE`, `MID_RANGE`, `MAX_RANGE`

## Interprétation des résultats

Le graphique généré montre l'évolution des métriques en fonction du temps pour les trois scénarios (sans panne, pannes prévisibles, pannes aléatoires).

- Une chute brutale de la taille de la composante géante indique une fragmentation du réseau
- Une augmentation de la longueur moyenne des chemins indique une dégradation des performances
- Une diminution du coefficient de clustering indique une perte de redondance locale
