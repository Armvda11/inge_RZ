# simulation/failure.py
import random
import numpy as np
from config import T_PRED, N_PRED, P_FAIL
from simulation.metrics import get_importance, get_centrality

class NodeFailureManager:
    """Gestionnaire de pannes de nœuds.
    
    Cette classe gère les différentes stratégies de pannes :
    - Pannes prévisibles (batterie faible) : planifie qu'un nœud tombe en panne à t = T_pred
    - Pannes aléatoires : à chaque pas de temps, chaque nœud a une petite probabilité p_fail de s'éteindre
    """
    def __init__(self, positions: dict):
        self.positions = positions
        self.failed_nodes = set()
        self.predictable_failures = set()
        self.failure_times = {}  # Pour tracer l'historique des pannes
        
    def setup_predictable_failures(self, t_pred: int, num_nodes: int, strategy='importance', matrix=None, swarm=None):
        """Configure les pannes prévisibles.
        
        Args:
            t_pred: Instant t où les pannes se produiront
            num_nodes: Nombre de nœuds qui tomberont en panne
            strategy: Stratégie de sélection ('importance', 'centralite', 'aléatoire')
            matrix: Matrice pondérée (nécessaire pour 'importance')
            swarm: Swarm (nécessaire pour 'centralite')
        """
        # Sélection des nœuds selon la stratégie
        if strategy == 'importance' and matrix:
            self.predictable_failures = set(get_importance(matrix, num_nodes))
        elif strategy == 'centralite' and swarm:
            self.predictable_failures = set(get_centrality(swarm, num_nodes))
        else:  # Par défaut, sélection aléatoire
            all_node_ids = list(range(len(self.positions[0])))
            self.predictable_failures = set(random.sample(all_node_ids, min(num_nodes, len(all_node_ids))))
        
        # Enregistrer le moment prévu pour chaque panne
        for node_id in self.predictable_failures:
            self.failure_times[node_id] = t_pred
    
    def apply_failures(self, t: int, positions: dict):
        """Applique les pannes à l'instant t.
        
        Args:
            t: Instant actuel
            positions: Dictionnaire des nœuds actifs
        """
        # Appliquer pannes prévisibles à t = T_pred
        if t == T_PRED:
            for node_id in self.predictable_failures:
                if node_id in positions and positions[node_id].active:
                    positions[node_id].active = False
                    self.failed_nodes.add(node_id)
        
        # Appliquer pannes aléatoires avec probabilité p_fail
        for node_id, node in positions.items():
            if node.active and node_id not in self.predictable_failures and random.random() < P_FAIL:
                node.active = False
                self.failed_nodes.add(node_id)
                self.failure_times[node_id] = t
    
    def get_active_nodes(self, positions: dict):
        """Retourne uniquement les nœuds actifs.
        
        Args:
            positions: Dictionnaire complet des nœuds
            
        Returns:
            dict: Dictionnaire filtré ne contenant que les nœuds actifs
        """
        return {node_id: node for node_id, node in positions.items() if node.active}

def simulate_with_failures(positions, swarms, matrixes, adjacency, num_sats, dest, failure_type='none', best_t=0, plot=True):
    """Simule le réseau avec différents types de pannes.
    
    Args:
        positions: Dictionnaire des positions des nœuds par temps
        swarms: Dictionnaire des essaims par temps
        matrixes: Dictionnaire des matrices pondérées par temps
        adjacency: Dictionnaire des matrices d'adjacence par temps
        num_sats: Nombre total de satellites
        dest: ID du nœud de destination
        failure_type: Type de panne ('none', 'predictable', 'random')
        best_t: Instant avec la meilleure efficacité (pour les pannes prévisibles)
        plot: Générer des graphiques
        
    Returns:
        Tuple: (Métriques topologiques, Delivery Ratio, Delivery Delay) par protocole
    """
    from models.node import Node
    from models.swarm import Swarm
    from simulation.metrics import analyze_single_graph, get_weighted_matrix, Metric
    from protocols.spray_and_wait import SprayAndWait
    from protocols.epidemic import Epidemic
    from protocols.prophet import Prophet
    from config import MAX_RANGE, MID_RANGE, MIN_RANGE, MAXTEMPS
    import matplotlib.pyplot as plt
    from config import OUTDIR
    
    # Copier les positions pour ne pas modifier les originales
    local_positions = {t: {id: Node(id, n.pos[0], n.pos[1], n.pos[2]) 
                          for id, n in positions[t].items()} 
                      for t in range(MAXTEMPS)}
    
    # Initialiser le gestionnaire de pannes
    failure_mgr = NodeFailureManager(local_positions)
    
    # Configurer les pannes selon le type
    if failure_type == 'predictable':
        # Pannes prévisibles à T_PRED
        failure_mgr.setup_predictable_failures(T_PRED, N_PRED, 'importance', 
                                             matrixes[best_t], swarms[best_t])
    
    # Structures pour stocker les résultats
    local_swarms = {}
    local_matrixes = {}
    local_adjacency = {}
    active_nodes_by_time = {}
    
    # Appliquer les pannes à chaque pas de temps et reconstruire les structures
    for t in range(MAXTEMPS):
        # Appliquer les pannes à l'instant t
        if failure_type != 'none':
            failure_mgr.apply_failures(t, local_positions[t])
        
        # Filtrer les nœuds actifs
        active_positions = failure_mgr.get_active_nodes(local_positions[t])
        active_nodes_by_time[t] = list(active_positions.keys())
        
        # Reconstruire les structures
        local_swarms[t] = Swarm(MAX_RANGE, list(active_positions.values()))
        local_matrixes[t] = get_weighted_matrix(local_swarms[t], MIN_RANGE, MID_RANGE, MAX_RANGE)
        
        # Reconstruire la matrice d'adjacence
        local_adjacency[t] = {}
        for node in active_positions.values():
            neighbors = set()
            for other in active_positions.values():
                if node.id != other.id and node.is_neighbor(other, MAX_RANGE):
                    neighbors.add(other.id)
            local_adjacency[t][node.id] = neighbors
    
    # Calculer les métriques topologiques
    local_stats = {t: analyze_single_graph(local_swarms[t], local_matrixes[t]) 
                  for t in range(MAXTEMPS)}
    
    # Métriques moyennes
    avg_metric = Metric(
        np.mean([m.MeanDegree for m in local_stats.values()]),
        np.mean([m.MeanClusterCoef for m in local_stats.values()]),
        np.mean([m.Connexity for m in local_stats.values()]),
        np.mean([m.Efficiency for m in local_stats.values() if m.Connexity == 1.0])
    )
    
    # Simuler les protocoles DTN
    results = {}
    protocols = {
        'Spray-and-Wait': lambda L: SprayAndWait(num_sats, L, dest),
        'Epidemic': lambda _: Epidemic(num_sats, 0, dest),
        'Prophet': lambda p_init: Prophet(num_sats, p_init, 0, dest)
    }
    
    # Tester différentes valeurs de paramètres pour chaque protocole
    for proto_name, proto_factory in protocols.items():
        if proto_name == 'Spray-and-Wait':
            param_values = [2, 5, 10, 15, 20]
            param_name = 'L'
        elif proto_name == 'Prophet':
            param_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            param_name = 'p_init'
        else:  # Epidemic
            param_values = [0]  # Pas de paramètre à varier pour Epidemic
            param_name = 'none'
        
        results[proto_name] = {'param_name': param_name, 'values': []}
        
        for param_value in param_values:
            protocol = proto_factory(param_value)
            
            # Simuler le protocole
            for t in range(MAXTEMPS):
                # Ne traiter que les nœuds actifs pour éviter KeyError
                protocol.step(t, local_adjacency[t])
            
            # Enregistrer les résultats
            results[proto_name]['values'].append({
                'param_value': param_value,
                'delivery_ratio': protocol.delivery_ratio(),
                'delivery_delay': protocol.delivery_delay(),
                'protocol_instance': protocol  # Stocker l'instance du protocole pour accéder aux logs de paquets
            })
    
    # Visualiser les résultats si demandé
    if plot:
        # Graphique d'évolution des métriques avec pannes
        plt.figure(figsize=(10, 6))
        ts = sorted(local_stats.keys())
        plt.plot(ts, [local_stats[t].MeanDegree for t in ts], label='Degré moyen')
        plt.plot(ts, [local_stats[t].MeanClusterCoef for t in ts], label='Clustering')
        plt.plot(ts, [local_stats[t].Connexity for t in ts], label='Connexité')
        plt.plot(ts, [local_stats[t].Efficiency for t in ts], label='Efficience')
        
        # Marquer les instants de panne
        if failure_type == 'predictable':
            plt.axvline(x=T_PRED, color='r', linestyle='--', 
                       label=f'Panne prévisible (t={T_PRED})')
        
        plt.legend()
        plt.grid(True)
        plt.title(f'Évolution des métriques avec pannes ({failure_type})')
        plt.savefig(f"{OUTDIR}/metrics_{failure_type}.png")
        plt.close()
        
        # Graphiques pour les performances des protocoles DTN
        for proto_name, proto_data in results.items():
            if proto_name != 'Epidemic':  # Epidemic n'a pas de paramètre à varier
                param_name = proto_data['param_name']
                param_values = [res['param_value'] for res in proto_data['values']]
                delivery_ratios = [res['delivery_ratio'] for res in proto_data['values']]
                delivery_delays = [res['delivery_delay'] for res in proto_data['values']]
                
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(param_values, delivery_ratios, 'o-')
                plt.xlabel(param_name)
                plt.ylabel('Delivery Ratio')
                plt.title(f'{proto_name}: Delivery Ratio vs {param_name}')
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(param_values, delivery_delays, 'o-')
                plt.xlabel(param_name)
                plt.ylabel('Delivery Delay')
                plt.title(f'{proto_name}: Delivery Delay vs {param_name}')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{OUTDIR}/{proto_name}_{failure_type}.png")
                plt.close()
    
    return avg_metric, results