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
    'outdir': '../figures',
    'failure': {
        'predictable_time': 50,  # Instant T où les nœuds prédictibles tombent en panne
        'predictable_nodes': 5,  # Nombre de nœuds prédictibles qui tombent en panne
        'random_prob': 0.01     # Probabilité qu'un nœud tombe en panne à chaque pas de temps
    },
    'analysis': {
        'window_size': 5,       # Taille de la fenêtre pour l'analyse temporelle
        'num_runs': 1          # Nombre d'exécutions pour les statistiques
    }
}

# Paramètres de pannes de nœuds (pour accès facile)
T_PRED = CONFIG['failure']['predictable_time']
N_PRED = CONFIG['failure']['predictable_nodes']
P_FAIL = CONFIG['failure']['random_prob']

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