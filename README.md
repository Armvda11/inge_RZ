# Étude de la robustesse d'un essaim de nano satellites

Ce projet analyse la robustesse d'un réseau de nanosatellites face à différents scénarios de panne et compare l'efficacité de plusieurs protocoles DTN (Delay Tolerant Network).

## Configuration

Le programme comporte plusieurs paramètres configurables dans `config.py` :

- `PATH` : Emplacement et nom du fichier source des traces
- `MAXTEMPS` : Limite de temps pour la simulation
- `MIN_RANGE`, `MID_RANGE`, `MAX_RANGE` : Portées des satellites (aide à la décision entre différents types d'antennes)
- `WINDOW_SIZE` : Taille de la fenêtre pour l'analyse temporelle (nouveau)
- `NUM_RUNS` : Nombre d'exécutions pour les statistiques (nouveau)

La librairie swarm_sim a été modifiée pour disposer d'éléments nécessaires à la réalisation du projet, comme par exemple des fonctions de visualisation (plot_nodes, plot_edges).

## Installation et exécution

1. Créer un environnement virtuel (recommandé) :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancer la simulation principale :
```bash
cd code
python main.py
```

4. Analyser les résultats en détail :
```bash
cd code
python analyze_results.py
```

## Architecture

Le programme est organisé en plusieurs phases :

1. Chargement et mise en forme des données
2. Analyse topologique du réseau et calcul de métriques
3. Simulation DTN avec plusieurs protocoles (Epidemic, Spray-and-Wait, PRoPHET)
4. Création de scénarios avec pannes de nœuds
5. Analyse comparative des performances des protocoles face aux pannes
6. Génération de visualisations et rapports détaillés
7. Analyse statistique des données de paquets (nouveau)

## Nouveautés

### Journaux de paquets détaillés

La simulation génère désormais deux fichiers CSV :

- `packet_logs.csv` : Enregistre chaque paquet avec protocole, scénario, ID, timestamps d'émission/réception, et nombre de sauts
- `per_packet_metrics.csv` : Calcule des métriques par paquet (délai, etc.)

### Analyse approfondie

Un nouveau script `analyze_results.py` permet :

- Le calcul de statistiques descriptives (moyennes, écarts-types)
- Des tests ANOVA pour évaluer l'effet du protocole et du scénario sur les performances
- La génération de graphiques avancés :
  - Histogrammes des délais par protocole et scénario
  - Boxplots comparatifs du délai et du nombre de sauts
  - Courbes temporelles du delivery ratio par fenêtre

Tous les résultats sont enregistrés dans le dossier `figures/analysis/`.

## Guide d'utilisation

### Exemples d'utilisation du code

#### Accès aux données spatiales
```python
# Afficher les positions de tous les points au temps 70
print(positions[70])

# Afficher la position du nœud 1 au temps 70
print(positions[70][1])

# Afficher la coordonnée x du nœud 1 au temps 70
print(positions[70][1].pos[0])

# Afficher la coordonnée y du nœud 5 au temps 75
print(positions[75][5].pos[1])
```

#### Analyse des logs de paquets
```python
# Charger les logs de paquets
import pandas as pd
packet_logs = pd.read_csv("../figures/packet_logs.csv")

# Filtrer par protocole
epidemic_logs = packet_logs[packet_logs['protocol'] == 'Epidemic']

# Calculer le délai moyen par scénario
avg_delay = packet_logs.groupby(['protocol', 'scenario'])['delay'].mean()
print(avg_delay)
```

#### Exécution des analyses
```bash
# Lancer l'analyse complète
python analyze_results.py

# Visualiser les résultats
open ../figures/analysis/
```
print(Matrixes[70])
## Afficher le cout entre le sat 2 et 3 au temps 70
print(Matrixes[70][2][3])
## Afficher le swarm a l'instant 70
print(Swarms[70])
## Afficher les métriques pour chacuns des graphs
print(AnalyzeGraph(Positions, Swarms, Matrixes))
## Afficher les métriques pour le temps 75
print(AnalyzeGraph(Positions, Swarms, Matrixes)[75])
## Afficher le scenario disposant de la meilleure efficacite
print(GetBestCase(Stats))
## Afficher les 6 noeuds disposant de la plus grande "importance" au temps 78
print(GetTopImportanceNoeud(Matrixes[78], 6))
## Supprimer 6 noeuds du meilleur et pire scénario en utilisant la strategie centralité
print(StrategieCentralite(6))
