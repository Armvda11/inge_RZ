# data/loader.py
import pandas as pd
from models.node import Node
from models.swarm import Swarm
from simulation.metrics import get_weighted_matrix

def load_data(path, max_temps, ranges):
    """
    Charge les données de positionnement des satellites et génère les structures nécessaires.
    
    Args:
        path: Chemin du fichier CSV contenant les données
        max_temps: Temps maximal de la simulation
        ranges: Dictionnaire des portées (min, mid, max)
    
    Returns:
        tuple: (positions, swarms, matrixes, adjacency) pour tous les instants t
    """
    print("### Importation des données ###")
    df = pd.read_csv(path, header=0)

    print("### Reformatage : index temps ###")
    names = [str(i) for i in range(1, len(df.columns)+1)]
    save = df.columns.copy()
    df.columns = names
    df.loc[-1] = save
    df.index += 1
    df.sort_index(inplace=True)

    print("### Reformatage : index satellites ###")
    num_sats = len(df.index) // 3
    satnames = [f"sat{c}{i}" for i in range(1, num_sats+1) for c in ('x','y','z')]
    df['coords'] = satnames
    df.set_index('coords', inplace=True, drop=True)

    dft = df.transpose()
    
    # Extraction des portées
    min_range = ranges['min']
    mid_range = ranges['mid']
    max_range = ranges['max']

    # Fonctions auxiliaires
    def get_nodes(t):
        """Crée les nœuds pour un instant t donné"""
        nodes = {}
        for i in range(1, num_sats+1):
            x = float(dft.loc[str(t)][f"satx{i}"])
            y = float(dft.loc[str(t)][f"saty{i}"])
            z = float(dft.loc[str(t)][f"satz{i}"])
            nodes[i-1] = Node(i-1, x, y, z)
        return nodes

    print("### Création positions & swarms ###")
    positions = {t: get_nodes(t+1) for t in range(max_temps)}
    swarms = {t: Swarm(max_range, list(positions[t].values()))
              for t in range(max_temps)}

    print("### Génération matrices pondérées & adjacence ###")
    matrixes = {t: get_weighted_matrix(swarms[t], min_range, mid_range, max_range) 
               for t in range(max_temps)}
    
    adjacency = {
        t: {
            i: {j for j, w in enumerate(matrixes[t][i]) if w > 0}
            for i in range(num_sats)
        }
        for t in range(max_temps)
    }

    return positions, swarms, matrixes, adjacency, num_sats