# simulation/failure.py
import random
import numpy as np
from config import T_PRED, N_PRED, P_FAIL, OUTDIR
import networkx as nx
import matplotlib.pyplot as plt
from simulation.metrics import get_importance, get_centrality, swarm_to_graph, analyze_single_graph, Metric

def plot_degree_distribution(degree_distribution, title="Distribution des degrés", filename=None):
    """Trace la distribution des degrés d'un graphe.
    
    Args:
        degree_distribution: Un dictionnaire où les clés sont les degrés et les valeurs sont les fréquences
        title: Le titre du graphique
        filename: Nom du fichier pour sauvegarder le graphique (sans extension)
    """
    if not degree_distribution:
        print("Pas de distribution de degrés à tracer")
        return
    
    plt.figure(figsize=(10, 6))
    degrees = list(degree_distribution.keys())
    frequencies = list(degree_distribution.values())
    
    plt.bar(degrees, frequencies, color='royalblue', alpha=0.7)
    plt.xlabel('Degré')
    plt.ylabel('Fréquence')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if filename:
        plt.savefig(f"{OUTDIR}/topologie/{filename}.png", bbox_inches='tight')
    plt.close()

def compare_degree_distributions(dist1, dist2, event_name="événement", filename=None):
    """Compare deux distributions de degrés (avant/après un événement).
    
    Args:
        dist1: Distribution des degrés avant l'événement
        dist2: Distribution des degrés après l'événement
        event_name: Nom de l'événement pour le titre
        filename: Nom du fichier pour sauvegarder le graphique (sans extension)
    """
    if not dist1 or not dist2:
        print("Distributions de degrés incomplètes, impossible de comparer")
        return
    
    # Combine keys from both distributions
    all_degrees = sorted(set(list(dist1.keys()) + list(dist2.keys())))
    
    # Create arrays for plotting, filling in zeros for missing degrees
    freq_before = [dist1.get(d, 0) for d in all_degrees]
    freq_after = [dist2.get(d, 0) for d in all_degrees]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(all_degrees))
    width = 0.35
    
    plt.bar(x - width/2, freq_before, width, color='royalblue', alpha=0.7, label='Avant')
    plt.bar(x + width/2, freq_after, width, color='firebrick', alpha=0.7, label='Après')
    
    plt.xlabel('Degré')
    plt.ylabel('Fréquence')
    plt.title(f'Distribution des degrés avant et après {event_name}')
    plt.xticks(x, all_degrees)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if filename:
        plt.savefig(f"{OUTDIR}/topologie/{filename}.png", bbox_inches='tight')
    plt.close()

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
        # Vérifier que les paramètres sont cohérents
        if strategy == 'importance' and matrix is None:
            print("ERREUR: La matrice est requise pour la stratégie 'importance'. Utilisation de la stratégie 'aléatoire'.")
            strategy = 'aléatoire'
        
        if strategy == 'centralite' and swarm is None:
            print("ERREUR: Le swarm est requis pour la stratégie 'centralite'. Utilisation de la stratégie 'aléatoire'.")
            strategy = 'aléatoire'
            
        # S'assurer que num_nodes ne dépasse pas le nombre total de nœuds
        total_nodes = len(self.positions[0])
        if num_nodes > total_nodes // 2:
            print(f"ATTENTION: Le nombre de nœuds à désactiver ({num_nodes}) est supérieur à 50% du total ({total_nodes}).")
            print(f"Limitation à {total_nodes // 3} nœuds pour éviter une fragmentation excessive.")
            num_nodes = max(1, total_nodes // 3)
        
        # Sélection des nœuds selon la stratégie
        selected_nodes = []
        
        if strategy == 'importance' and matrix:
            selected_nodes = get_importance(matrix, num_nodes)
            #print(f"  - Nœuds sélectionnés par importance: {selected_nodes}")
            
        elif strategy == 'centralite' and swarm:
            selected_nodes = get_centrality(swarm, num_nodes)
            #print(f"  - Nœuds sélectionnés par centralité: {selected_nodes}")
            
        else:  # Stratégie aléatoire
            all_node_ids = list(range(total_nodes))
            selected_nodes = random.sample(all_node_ids, min(num_nodes, total_nodes))
           # print(f"  - Nœuds sélectionnés aléatoirement: {selected_nodes}")
        
        # Stocker les nœuds sélectionnés
        self.predictable_failures = set(selected_nodes)
        
        # Enregistrer le moment prévu pour chaque panne
        for node_id in self.predictable_failures:
            self.failure_times[node_id] = t_pred
    
    def setup_random_failures(self, prob_failure):
        """Configure les pannes aléatoires.
        
        Args:
            prob_failure: Probabilité qu'un nœud tombe en panne à chaque pas de temps
        """
        self.random_failure_prob = prob_failure
        
        # Pré-calculer les pannes aléatoires pour chaque temps
        # afin d'avoir un comportement déterministe et cohérent
        max_time = max(self.positions.keys())
        all_nodes = set(self.positions[0].keys())
        self.random_failure_times = {}
        
        # Pour chaque nœud et chaque temps, déterminer s'il tombe en panne
        for node_id in all_nodes:
            for t in range(max_time + 1):
                # Tirer un nombre aléatoire et vérifier s'il est inférieur à la probabilité de panne
                if random.random() < prob_failure:
                    if node_id not in self.random_failure_times:
                        self.random_failure_times[node_id] = t
                        self.failure_times[node_id] = t
    
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
                    self.failure_times[node_id] = t
                    #print(f"  - Nœud {node_id} désactivé (panne prévisible)")
         # Appliquer pannes aléatoires
        # Identifier les nœuds actifs non prévus pour une panne prévisible
        active_nodes = [node_id for node_id, node in positions.items() 
                      if node.active and node_id not in self.predictable_failures]
        
        # D'abord vérifier si nous avons des pannes pré-calculées
        if hasattr(self, 'random_failure_times'):
            # Utiliser des pannes pré-calculées pour un comportement cohérent
            for node_id, failure_time in self.random_failure_times.items():
                if failure_time == t and node_id in active_nodes and positions[node_id].active:
                    positions[node_id].active = False
                    self.failed_nodes.add(node_id)
                    #print(f"  - Nœud {node_id} désactivé (panne aléatoire pré-calculée)")
        else:
            # Comportement legacy: appliquer des pannes aléatoires à la volée
            # Limiter le nombre total de pannes par pas de temps
            # pour éviter une cascade de pannes qui fragmenterait trop le réseau
            max_failures_per_step = max(1, len(active_nodes) // 20)  # Au plus 5% des nœuds actifs
            
            # Tirage aléatoire pour déterminer quels nœuds tombent en panne
            random_failures = []
            for node_id in active_nodes:
                if random.random() < P_FAIL and len(random_failures) < max_failures_per_step:
                    random_failures.append(node_id)
            
            # Appliquer les pannes aléatoires
            for node_id in random_failures:
                positions[node_id].active = False
                self.failed_nodes.add(node_id)
                self.failure_times[node_id] = t
                #print(f"  - Nœud {node_id} désactivé (panne aléatoire à t={t})")
        
        # Utiliser P_FAIL comme probabilité mais limiter le nombre total de pannes par pas de temps
        # pour éviter une cascade de pannes qui fragmenterait trop le réseau
        max_failures_per_step = max(1, len(active_nodes) // 20)  # Au plus 5% des nœuds actifs
        
        # Tirage aléatoire pour déterminer quels nœuds tombent en panne
        random_failures = []
        for node_id in active_nodes:
            if random.random() < P_FAIL and len(random_failures) < max_failures_per_step:
                random_failures.append(node_id)
        
        # Appliquer les pannes aléatoires
        for node_id in random_failures:
            positions[node_id].active = False
            self.failed_nodes.add(node_id)
            self.failure_times[node_id] = t
            #print(f"  - Nœud {node_id} désactivé (panne aléatoire à t={t})")
            
        # Afficher un résumé des pannes
        if t % 10 == 0 or t == T_PRED:  # Tous les 10 pas de temps ou à T_PRED
            active_count = sum(1 for node in positions.values() if node.active)
            total_count = len(positions)
           # print(f"  - t={t}: {active_count}/{total_count} nœuds actifs ({100*active_count/total_count:.1f}%)")
    
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
        print(f"  - Configuration des pannes prévisibles (T={T_PRED}, N={N_PRED})")
        # Utiliser la meilleure stratégie en alternant entre importance et centralité
        # selon les caractéristiques du réseau à cet instant
        best_graph = swarm_to_graph(swarms[best_t], matrixes[best_t])
        
        # Si le graphe est peu connexe, privilégier la centralité de degré
        if nx.density(best_graph) < 0.3:
            print("  - Utilisation de la stratégie 'centralité' (faible densité du réseau)")
            failure_mgr.setup_predictable_failures(T_PRED, N_PRED, 'centralite', 
                                               matrixes[best_t], swarms[best_t])
        else:
            print("  - Utilisation de la stratégie 'importance' (forte densité du réseau)")
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
    
    # Si c'est un scénario avec des pannes prévisibles à T_PRED, visualiser l'impact
    if failure_mgr.predictable_failures and T_PRED is not None and 0 <= T_PRED < MAXTEMPS:
        # Analyser les métriques juste avant et juste après T_PRED
        if T_PRED > 0:
            before_t = T_PRED - 1
        else:
            before_t = 0
        after_t = T_PRED
        
        print(f"  - Analyse de l'impact des pannes prévisibles à T_PRED={T_PRED}...")
        print(f"    Degré moyen: {local_stats[before_t].MeanDegree:.3f} -> {local_stats[after_t].MeanDegree:.3f}")
        print(f"    Connexité: {local_stats[before_t].Connexity:.3f} -> {local_stats[after_t].Connexity:.3f}")
        
        # Visualiser la distribution des degrés avant/après les pannes prévisibles
        compare_degree_distributions(
            local_stats[before_t].DegreeDistribution,
            local_stats[after_t].DegreeDistribution,
            event_name=f"pannes prévisibles à T_PRED={T_PRED}",
            filename=f"degree_dist_before_after_T_PRED_{failure_type}"
        )
        
        # Visualiser l'évolution des mesures de robustesse spécifiquement autour de T_PRED
        plt.figure(figsize=(10, 6))
        t_range = range(max(0, T_PRED-5), min(MAXTEMPS, T_PRED+6))  # Visualisation autour de T_PRED
        
        # Tracer l'évolution du degré moyen et de la connexité autour de T_PRED
        plt.plot([t for t in t_range], 
                [local_stats[t].MeanDegree for t in t_range], 
                'b-', label='Degré moyen <k>', linewidth=2)
        plt.plot([t for t in t_range], 
                [local_stats[t].Connexity for t in t_range], 
                'r-', label='Connexité (Gmax/N)', linewidth=2)
        
        # Marquer T_PRED
        plt.axvline(x=T_PRED, color='g', linestyle='--', label=f'T_PRED={T_PRED}')
        
        plt.legend(loc='best')
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        plt.title(f'Impact des pannes prévisibles ({failure_type}) sur les métriques de robustesse')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/impact_T_PRED_{failure_type}.png")
        plt.close()
    
    # Métriques moyennes
    # Utiliser la distribution des degrés de l'état final (ou un instant représentatif)
    representative_t = MAXTEMPS - 1  # Utiliser le dernier instant comme représentatif
    
    avg_metric = Metric(
        np.mean([m.MeanDegree for m in local_stats.values()]),
        np.mean([m.MeanClusterCoef for m in local_stats.values()]),
        np.mean([m.Connexity for m in local_stats.values()]),
        np.mean([m.Efficiency for m in local_stats.values() if m.Connexity > 0.5]),  # Use Connexity > 0.5 instead of == 1.0
        local_stats[representative_t].DegreeDistribution  # Utiliser la distribution du dernier instant
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
    
    # Journaliser un résumé des résultats
    print(f"\n=== Résumé du scénario: {failure_type} ===")
    print(f"Nombre de nœuds actifs en fin de simulation: {len(active_nodes_by_time[MAXTEMPS-1])}/{num_sats}")
    
    # Recherche de fragmentation du réseau
    disconnected_times = [t for t in local_stats.keys() if local_stats[t].Connexity < 1.0]
    if disconnected_times:
        print(f"ATTENTION: Réseau fragmenté à {len(disconnected_times)} instants: {disconnected_times[:5]}...")
        avg_components = sum(nx.number_connected_components(swarm_to_graph(local_swarms[t])) 
                            for t in disconnected_times) / len(disconnected_times)
        print(f"Nombre moyen de composantes: {avg_components:.2f}")
    else:
        print("Réseau connecté sur toute la durée de la simulation.")
    
    # Statistiques par protocole
    for proto_name, proto_data in results.items():
        best_param_idx = 0
        best_dr = 0
        
        if proto_name != 'Epidemic':  # Epidemic n'a pas de paramètre à varier
            # Trouver le meilleur paramètre
            param_name = proto_data['param_name']
            for i, res in enumerate(proto_data['values']):
                if res['delivery_ratio'] > best_dr:
                    best_dr = res['delivery_ratio']
                    best_param_idx = i
            
            best_param = proto_data['values'][best_param_idx]['param_value']
            print(f"\n{proto_name} - Meilleur paramètre {param_name}={best_param}:")
        else:
            print(f"\n{proto_name}:")
        
        # Afficher les résultats du meilleur paramètre
        best_result = proto_data['values'][best_param_idx]
        print(f"  Delivery Ratio: {best_result['delivery_ratio']:.3f}")
        print(f"  Delivery Delay: {best_result['delivery_delay']:.1f}")
        
        # Vérifier les paquets non livrés
        protocol = best_result['protocol_instance']
        delivered = sum(1 for p in protocol.packet_logs if 't_recv' in p)
        total = len(protocol.packet_logs)
        print(f"  Paquets livrés: {delivered}/{total} ({100*delivered/total:.1f}%)")
    
    return avg_metric, results