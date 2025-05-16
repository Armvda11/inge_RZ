    # config.py
import os

# Configuration centralisée pour tout le projet
CONFIG = {
    'path': '../Traces.csv',  # Chemin relatif depuis le dossier code
    'max_temps': 100,
    'ranges': {
        'max': 60000,
        'mid': 40000,
        'min': 20000
    },
    'outdir': '../data_logs',
    'failure': {
        'predictable_time': 50,  # Instant T où les nœuds prédictibles tombent en panne
        'predictable_nodes': 5,  # Nombre de nœuds prédictibles qui tombent en panne
        'categories': {
            'legere': {
                'random_prob': 0.005,  # 0.5% de probabilité par pas de temps
                'predictable_nodes': 3  # Nombre faible de nœuds prédictibles
            },
            'moyenne': {
                'random_prob': 0.05,   # 1% de probabilité par pas de temps
                'predictable_nodes': 30  # Nombre moyen de nœuds prédictibles
            },
            'lourde': {
                'random_prob': 0.10,   # 2% de probabilité par pas de temps
                'predictable_nodes': 50  # Nombre élevé de nœuds prédictibles
            }
        },
        'active_category': 'moyenne'    # Catégorie par défaut
    },
    'analysis': {
        'window_size': 5,       # Taille de la fenêtre pour l'analyse temporelle
        'num_runs': 1          # Nombre d'exécutions pour les statistiques
    }
}

# Paramètres de pannes de nœuds (pour accès facile)
T_PRED = CONFIG['failure']['predictable_time']

# Utiliser la catégorie active pour déterminer les paramètres de panne
ACTIVE_CATEGORY = CONFIG['failure']['active_category']
N_PRED = CONFIG['failure']['categories'][ACTIVE_CATEGORY]['predictable_nodes']
P_FAIL = CONFIG['failure']['categories'][ACTIVE_CATEGORY]['random_prob']

# Paramètres d'analyse (pour accès facile)
WINDOW_SIZE = CONFIG['analysis']['window_size']
NUM_RUNS = CONFIG['analysis']['num_runs']

# Chemins et constantes
PATH      = CONFIG['path']
MAXTEMPS  = CONFIG['max_temps']
MAX_RANGE = CONFIG['ranges']['max']
MID_RANGE = CONFIG['ranges']['mid']
MIN_RANGE = CONFIG['ranges']['min']
OUTDIR    = CONFIG['outdir']

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTDIR, exist_ok=True)