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
    # Vérification que l'essaim n'est pas vide
    if not swarm or not swarm.nodes:
        print("ERREUR: Essaim vide dans get_weighted_matrix")
        return [[]]
    
    # Création des matrices de connectivité pour chaque portée
    m20 = swarm.neighbor_matrix(min_range)
    m40 = swarm.neighbor_matrix(mid_range)
    m60 = swarm.neighbor_matrix(max_range)
    
    # Vérifier que les matrices ont la même taille
    n = len(m20)
    if len(m40) != n or len(m60) != n:
        print(f"ERREUR: Les matrices de connectivité ont des tailles différentes: {len(m20)}, {len(m40)}, {len(m60)}")
        return [[0 for _ in range(n)] for _ in range(n)]
    
    # Création de la matrice pondérée
    result = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:  # Pas de connexion d'un nœud avec lui-même
                result[i][j] = 0
            elif m20[i][j] == 1:  # Connexion forte (MIN_RANGE)
                result[i][j] = 1
            elif m40[i][j] == 1:  # Connexion moyenne (MID_RANGE)
                result[i][j] = 2
            elif m60[i][j] == 1:  # Connexion faible (MAX_RANGE)
                result[i][j] = 3
    
    return result

def get_importance(matrix, k):
    """Identifie les k nœuds les plus importants selon la centralité d'intermédiarité.
    
    Cette implémentation utilise directement la fonction de NetworkX pour calculer
    la centralité d'intermédiarité sur un graphe pondéré, ce qui est plus précis
    que notre implémentation précédente.
    
    Args:
        matrix: Matrice de poids
        k: Nombre de nœuds à sélectionner
        
    Returns:
        list: Liste des k indices de nœuds les plus importants
    """
    G = swarm_to_graph(None, weighted_matrix=matrix)
    
    # Si le graphe n'a pas assez de nœuds ou est vide
    if G.number_of_nodes() <= 1:
        return list(range(min(k, len(matrix))))
    
    # Si le graphe n'est pas connexe, le traiter composante par composante
    if nx.number_connected_components(G) > 1:
        # Calculer la centralité sur chaque composante connexe
        combined_bc = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            bc = nx.betweenness_centrality(subgraph, weight='weight', normalized=True)
            combined_bc.update(bc)
        
        # Trier par valeur de centralité
        sorted_nodes = sorted(combined_bc.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:k]]
    else:
        # Calculer la centralité d'intermédiarité normalisée
        bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
        sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:k]]

def get_centrality(swarm, k):
    """Identifie les k nœuds les plus centraux selon la centralité de degré.
    
    Args:
        swarm: Instance de Swarm
        k: Nombre de nœuds à sélectionner
        
    Returns:
        list: Liste des k indices de nœuds les plus centraux
    """
    G = swarm.swarm_to_nxgraph()
    
    # Si le graphe n'a pas assez de nœuds
    if G.number_of_nodes() <= 1:
        return [node.id for node in swarm.nodes[:min(k, len(swarm.nodes))]]
    
    # Calculer la centralité de degré
    degree_cen = nx.degree_centrality(G)
    
    # Trier les nœuds par centralité décroissante
    sorted_nodes = sorted(degree_cen.items(), key=lambda x: x[1], reverse=True)
    
    # Renvoyer les k premiers nœuds
    return [node for node, _ in sorted_nodes[:k]]

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