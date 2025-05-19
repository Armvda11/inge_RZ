# Analyse Topologique d'un Réseau de Nanosatellites

Ce projet simplifié analyse la topologie d'un réseau de nanosatellites face à différents scénarios de panne.

## Métriques Clés

L'analyse se concentre sur 4 métriques fondamentales :

1. **Degré moyen (⟨k⟩)** : Nombre de liens en moyenne par satellite
2. **Taille de la composante géante (|Gₘₐₓ|/N)** : Fraction de nœuds toujours connectés
3. **Longueur moyenne des plus courts chemins (ℓ̄)** : Proxy pour la latence minimale
4. **Coefficient de clustering (C)** : Mesure la redondance locale (triangles)

## Installation

1. Créer un environnement virtuel (recommandé) :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'analyse topologique :
```bash
./analyse_topologique.sh
```

Les résultats seront sauvegardés dans le dossier `data_logs/topologie/`.

## Scénarios de Panne

L'analyse examine deux types de pannes :

- **Panne prévisible** : Un ensemble déterminé de nœuds tombe en panne à l'instant T_PRED
- **Panne aléatoire** : À chaque instant, chaque nœud a une probabilité P_FAIL de tomber en panne

## Configuration

Les paramètres sont configurables dans `code/config.py` :

- `T_PRED` : Instant où les nœuds prédictibles tombent en panne
- `N_PRED` : Nombre de nœuds qui tombent en panne
- `P_FAIL` : Probabilité de panne aléatoire par pas de temps
- Portées de connexion : `MIN_RANGE`, `MID_RANGE`, `MAX_RANGE`

## Structure du Projet Simplifié

```
analyse_topologique.sh     # Script de lancement
ANALYSE_TOPOLOGIE.md       # Documentation détaillée
code/
  ├── analyze_topologie.py # Script principal d'analyse
  ├── config.py            # Configuration du projet
  ├── data/                # Chargement des données
  ├── models/              # Modèles de simulation (Node, Swarm)
  └── simulation/          # Outils de simulation simplifiés
data_logs/
  └── topologie/           # Résultats de l'analyse
```
