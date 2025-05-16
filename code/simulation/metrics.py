# simulation/metrics.py
import numpy as np
import networkx as nx
import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class Metric:
    """Métriques pour l'analyse topologique d'un graphe.
    
    Attributes:
        MeanDegree: Degré moyen du graphe
        MeanClusterCoef: Coefficient de clustering moyen
        Connexity: 1.0 si le graphe est connexe, 0.0 sinon
        Efficiency: Mesure de l'efficience du graphe pondéré
    """
    MeanDegree: float
    MeanClusterCoef: float
    Connexity: float
    Efficiency: float

def swarm_to_graph(swarm, weighted_matrix=None):
    """Convertit un Swarm en graphe NetworkX.
    
    Args:
        swarm: Instance de Swarm (ou None si weighted_matrix est fourni)
        weighted_matrix: Matrice de poids optionnelle
    
    Returns:
        Un graphe NetworkX
    """
    if weighted_matrix is not None:
        G = nx.Graph()
        n = len(weighted_matrix)
        for i in range(n):
            for j in range(i):
                w = weighted_matrix[i][j]
                if w != 0:
                    G.add_edge(i, j, weight=w)
        return G
    return swarm.swarm_to_nxgraph()

def get_mean_degree(G):
    """Calcule le degré moyen d'un graphe.
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        float: le degré moyen
    """
    return sum(d for _, d in G.degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

def get_mean_cluster_coef(G):
    """Calcule le coefficient de clustering moyen d'un graphe.
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        float: coefficient de clustering moyen
    """
    return nx.average_clustering(G) if G.number_of_nodes() > 0 else 0

def get_connexity(G):
    """Vérifie si un graphe est connexe.
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        float: 1.0 si connexe, 0.0 sinon
    """
    return 1.0 if nx.number_connected_components(G) == 1 else 0.0

def get_efficiency(G):
    """Calcule l'efficience globale d'un graphe pondéré.
    
    La fonction retourne 0.0 si le graphe n'est pas connexe.
    Sinon, elle calcule l'inverse de la moyenne harmonique des distances.
    
    Args:
        G: Graphe NetworkX
    
    Returns:
        float: Efficience du graphe entre 0.0 et 1.0
    """
    if get_connexity(G) == 0.0:
        return 0.0
    
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        n = G.number_of_nodes()
        total = sum(1/lengths[i][j] for i in lengths for j in lengths[i] if i != j and lengths[i][j] > 0)
        max_efficiency = n * (n - 1)
        
        return total / max_efficiency if max_efficiency > 0 else 0.0
    except nx.NetworkXNoPath:
        return 0.0

def analyze_single_graph(swarm, matrix):
    """Calcule les métriques pour un seul graphe.
    
    Args:
        swarm: Instance de Swarm
        matrix: Matrice de poids
        
    Returns:
        Metric: Objet contenant les métriques calculées
    """
    G_un = swarm_to_graph(swarm)
    G_wt = swarm_to_graph(None, weighted_matrix=matrix)
    
    return Metric(
        get_mean_degree(G_un),
        get_mean_cluster_coef(G_un),
        get_connexity(G_un),
        get_efficiency(G_wt)
    )

def get_weighted_matrix(swarm, min_range, mid_range, max_range):
    """Génère une matrice pondérée à partir d'un essaim.
    
    Args:
        swarm: Instance de Swarm
        min_range: Portée minimale
        mid_range: Portée moyenne
        max_range: Portée maximale
        
    Returns:
        list: Matrice pondérée
    """
    m20 = swarm.neighbor_matrix(min_range)
    m40 = swarm.neighbor_matrix(mid_range)
    m60 = swarm.neighbor_matrix(max_range)
    
    n = len(m20)
    for i in range(n):
        for j in range(n):
            if m20[i][j] == 0 and m40[i][j] == 1:
                m20[i][j] = 2
            elif m20[i][j] == 0 and m60[i][j] == 1:
                m20[i][j] = 3
    
    return m20

def get_importance(matrix, k):
    """Identifie les k nœuds les plus importants selon la centralité d'intermédiarité.
    
    Cette implémentation se base sur les chemins les plus courts pondérés.
    
    Args:
        matrix: Matrice de poids
        k: Nombre de nœuds à sélectionner
        
    Returns:
        list: Liste des k indices de nœuds les plus importants
    """
    G = swarm_to_graph(None, weighted_matrix=matrix)
    cnt = {i: 0 for i in range(len(matrix))}
    
    for i in cnt:
        for j in cnt:
            if j > i:
                try:
                    path = nx.dijkstra_path(G, i, j, weight='weight')
                    for n in path[1:-1]:
                        cnt[n] += 1
                except nx.NetworkXNoPath:
                    pass
    
    return sorted(cnt, key=cnt.get, reverse=True)[:k]

def get_centrality(swarm, k):
    """Identifie les k nœuds les plus centraux selon la centralité de degré.
    
    Args:
        swarm: Instance de Swarm
        k: Nombre de nœuds à sélectionner
        
    Returns:
        list: Liste des k indices de nœuds les plus centraux
    """
    cen = nx.degree_centrality(swarm.swarm_to_nxgraph())
    return sorted(cen, key=cen.get, reverse=True)[:k]

def compute_per_packet_metrics(packet_logs_path, output_path=None):
    """
    Analyse les logs de paquets et calcule des métriques par paquet.
    
    Args:
        packet_logs_path: Chemin vers le fichier CSV des logs de paquets
        output_path: Chemin où enregistrer les métriques calculées (optional)
        
    Returns:
        DataFrame: DataFrame contenant les métriques par paquet
    """
    # Charger les logs de paquets
    try:
        df = pd.read_csv(packet_logs_path)
        print(f"  - {len(df)} logs de paquets chargés depuis {packet_logs_path}")
    except Exception as e:
        print(f"Erreur lors du chargement des logs de paquets: {e}")
        return None
    
    # Vérifier que toutes les colonnes nécessaires sont présentes
    required_columns = ['protocol', 'scenario', 'packet_id', 't_emit', 't_recv', 'num_hops']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Colonnes manquantes dans le fichier de logs: {missing_columns}")
        return None
    
    # Calculer le délai pour chaque paquet
    df['delay'] = df['t_recv'] - df['t_emit']
    
    # Filtrer les paquets avec délai négatif (erreur)
    invalid_delays = df[df['delay'] < 0]
    if len(invalid_delays) > 0:
        print(f"Attention: {len(invalid_delays)} paquets ont un délai négatif et ont été filtrés")
        df = df[df['delay'] >= 0]
    
    # Exporter les résultats si un chemin est spécifié
    if output_path:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Exporter en CSV
        df.to_csv(output_path, index=False)
        print(f"  - Métriques par paquet exportées vers {output_path}")
    
    return df