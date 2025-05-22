#!/usr/bin/env python3
# test_spray_and_wait_dynamic_failures.py
"""
Script de test pour évaluer la résilience du protocole Spray-and-Wait face à des
pannes dynamiques et continues qui se produisent tout au long de la simulation.

Ce script améliore les tests de résilience en:
1. Implémentant des pannes CONTINUES qui se produisent régulièrement pendant la simulation
2. Proposant plusieurs MODES de panne dynamique:
   - Mode "continuous": pannes aléatoires à chaque pas de temps
   - Mode "cascade": effet domino où les voisins d'un nœud en panne ont plus de risque de tomber en panne
   - Mode "targeted_dynamic": recalcul périodique des nœuds critiques à mettre en panne
3. Permettant une VISUALISATION claire de l'impact des pannes sur la diffusion des messages
4. Testant l'influence de différentes valeurs de L (nombre de copies initial) face à ces pannes
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTDIR
from protocols.spray_and_wait import SprayAndWait
from simulation.failure import NodeFailureManager

class DynamicFailureManager:
    """
    Gestionnaire de pannes dynamiques et continues.
    
    Cette classe étend les fonctionnalités du NodeFailureManager standard
    pour permettre des pannes qui évoluent tout au long de la simulation.
    """
    
    def __init__(self, num_nodes, source, destination):
        """
        Initialise le gestionnaire de pannes dynamiques.
        
        Args:
            num_nodes: Nombre total de nœuds dans le réseau
            source: ID du nœud source (à préserver)
            destination: ID du nœud destination (à préserver)
        """
        self.num_nodes = num_nodes
        self.source = source
        self.destination = destination
        self.failed_nodes = set()
        self.failure_history = []  # Pour suivre l'historique des pannes
        self.cascade_risk = {}  # Pour le mode cascade
        
        # Initialiser les infos de risque pour le mode cascade
        for node_id in range(num_nodes):
            self.cascade_risk[node_id] = 0.0
    
    def apply_failures(self, t, adjacency, mode, params):
        """
        Applique des pannes dynamiques selon le mode spécifié.
        
        Args:
            t: Instant actuel
            adjacency: Dictionnaire d'adjacence du réseau
            mode: Mode de panne ('continuous', 'cascade', 'targeted_dynamic')
            params: Paramètres spécifiques au mode de panne
        
        Returns:
            dict: Nouveau dictionnaire d'adjacence avec les nœuds en panne isolés
        """
        # Garder une trace du nombre de nœuds actifs avant d'appliquer les pannes
        active_nodes_before = len(adjacency)
        
        # Créer une copie modifiable du dictionnaire d'adjacence
        new_adjacency = {node: neighbors.copy() for node, neighbors in adjacency.items()}
        
        # Ne jamais mettre en panne la source ou la destination
        protected_nodes = {self.source, self.destination}
        
        # Identifier les nœuds qui peuvent potentiellement tomber en panne
        eligible_nodes = [node for node in new_adjacency.keys() 
                         if node not in protected_nodes and node not in self.failed_nodes]
        
        # Appliquer les pannes selon le mode
        new_failures = set()
        
        if mode == 'continuous':
            # Mode continu: chaque nœud a une probabilité de tomber en panne à chaque pas de temps
            failure_prob = params.get('failure_prob', 0.02)  # 2% de chance par défaut
            
            # Réduire progressivement la probabilité pour éviter trop de pannes en fin de simulation
            decay_factor = max(0.5, 1.0 - t * 0.01)  # Diminue avec le temps
            adjusted_prob = failure_prob * decay_factor
            
            # Limiter le nombre maximal de pannes par pas de temps
            max_failures = max(1, len(eligible_nodes) // 10)  # Max 10% des nœuds éligibles
            
            # Appliquer des pannes aléatoires
            for node in eligible_nodes:
                if random.random() < adjusted_prob and len(new_failures) < max_failures:
                    new_failures.add(node)
        
        elif mode == 'cascade':
            # Mode cascade: les nœuds voisins de nœuds en panne ont un risque accru de panne
            # Paramètres spécifiques au mode cascade
            base_prob = params.get('base_prob', 0.01)  # Probabilité de base
            cascade_factor = params.get('cascade_factor', 3.0)  # Multiplicateur pour les voisins des nœuds en panne
            max_failures = max(1, len(eligible_nodes) // 15)  # Limiter à environ 6-7% des nœuds éligibles
            
            # Mettre à jour les risques de cascade pour tous les nœuds
            for node in eligible_nodes:
                # Risque de base
                self.cascade_risk[node] = base_prob
                
                # Augmenter le risque pour les voisins de nœuds en panne
                for neighbor in adjacency.get(node, set()):
                    if neighbor in self.failed_nodes:
                        self.cascade_risk[node] += base_prob * cascade_factor
                
                # Plafonner le risque à 80% pour éviter une cascade trop rapide
                self.cascade_risk[node] = min(0.8, self.cascade_risk[node])
            
            # Appliquer les pannes en cascade
            for node in eligible_nodes:
                if random.random() < self.cascade_risk[node] and len(new_failures) < max_failures:
                    new_failures.add(node)
        
        elif mode == 'targeted_dynamic':
            # Mode ciblé dynamique: viser périodiquement les nœuds les plus centraux
            recalculation_interval = params.get('recalculation_interval', 5)  # Recalculer tous les 5 pas de temps
            failure_percentage = params.get('failure_percentage', 0.05)  # 5% des nœuds les plus centraux
            
            # Ne recalculer que périodiquement pour éviter des calculs à chaque pas de temps
            if t % recalculation_interval == 0 and eligible_nodes:
                # Construire le graphe pour calculer la centralité
                G = nx.Graph()
                for node, neighbors in adjacency.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
                
                # Calculer la centralité de betweenness pour tous les nœuds
                try:
                    centrality = nx.betweenness_centrality(G)
                    # Trier les nœuds éligibles par centralité décroissante
                    sorted_nodes = sorted(
                        [(node, centrality.get(node, 0)) for node in eligible_nodes],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Sélectionner un pourcentage des nœuds les plus centraux
                    num_to_fail = max(1, int(len(eligible_nodes) * failure_percentage))
                    most_central = [node for node, _ in sorted_nodes[:num_to_fail]]
                    
                    # Ajouter ces nœuds à la liste des nouvelles pannes
                    new_failures = set(most_central)
                except:
                    # Fallback en cas d'erreur (graphe déconnecté, etc.)
                    num_to_fail = max(1, int(len(eligible_nodes) * failure_percentage))
                    new_failures = set(random.sample(eligible_nodes, min(num_to_fail, len(eligible_nodes))))
        
        # Appliquer les nouvelles pannes au réseau
        for node in new_failures:
            # Retirer toutes les connexions du nœud en panne
            if node in new_adjacency:
                new_adjacency[node] = set()
            
            # Retirer ce nœud des listes d'adjacence des autres nœuds
            for other, neighbors in new_adjacency.items():
                if node in neighbors:
                    neighbors.remove(node)
            
            # Marquer le nœud comme étant en panne
            self.failed_nodes.add(node)
        
        # Enregistrer l'historique des pannes pour ce pas de temps
        self.failure_history.append({
            't': t,
            'new_failures': list(new_failures),
            'total_failures': len(self.failed_nodes),
            'active_nodes': len(adjacency) - len(new_failures)
        })
        
        # Log pour le debugging
        if new_failures:
            print(f"t={t}: {len(new_failures)} nouvelles pannes ({len(self.failed_nodes)}/{self.num_nodes} total) - mode {mode}")
        
        return new_adjacency

def create_multihop_network(t: int, num_nodes: int = 30, dilution_factor: float = 1.0) -> dict[int, set[int]]:
    """
    Crée un réseau test qui simule une topologie en "grappes" mobiles avec scénario multi-sauts.
    
    Args:
        t (int): L'instant de temps (pour simuler le mouvement)
        num_nodes (int): Nombre total de nœuds dans le réseau
        dilution_factor (float): Facteur pour réduire la densité des liens
        
    Returns:
        dict[int, set[int]]: Dictionnaire d'adjacence
    """
    random.seed(42 + t)  # Seed différente pour chaque pas de temps mais reproductible
    
    # Initialiser le réseau vide
    adjacency = {i: set() for i in range(num_nodes)}
    
    # Nombre de clusters - au moins 4 pour un véritable scenario multi-sauts
    cluster_size = 5  # Nombre approximatif de nœuds par cluster
    num_clusters = max(4, num_nodes // cluster_size)
    
    # Attribuer des nœuds à des clusters
    clusters = [set() for _ in range(num_clusters)]
    
    # Réserver la source et la destination
    source, destination = 0, num_nodes - 1
    reserved_nodes = {source, destination}
    regular_nodes = [i for i in range(num_nodes) if i not in reserved_nodes]
    random.shuffle(regular_nodes)
    
    # Calculer la distribution des nœuds par cluster
    nodes_per_cluster = (num_nodes - len(reserved_nodes)) // num_clusters
    remainder = (num_nodes - len(reserved_nodes)) % num_clusters
    
    # Placer la source et la destination
    clusters[0].add(source)  # Source dans le premier cluster
    clusters[-1].add(destination)  # Destination dans le dernier cluster
    
    # Distribuer les autres nœuds
    node_index = 0
    for c in range(num_clusters):
        nodes_for_this_cluster = nodes_per_cluster + (1 if c < remainder else 0)
        for _ in range(nodes_for_this_cluster):
            if node_index < len(regular_nodes):
                clusters[c].add(regular_nodes[node_index])
                node_index += 1
    
    # Établir des connexions à l'intérieur des clusters
    for cluster_idx, cluster in enumerate(clusters):
        cluster_list = list(cluster)
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                # Connectivité variable selon le cluster et le temps
                base_density = 0.7 * dilution_factor
                
                # Les clusters du milieu sont moins denses
                if 0 < cluster_idx < num_clusters - 1:
                    cluster_factor = 0.8
                else:
                    cluster_factor = 1.0
                
                # Variation temporelle mineure
                time_factor = 0.95 + 0.1 * ((t % 5) / 5.0)  # Entre 95% et 105%
                
                connect_prob = base_density * cluster_factor * time_factor
                
                if random.random() < connect_prob:
                    adjacency[cluster_list[i]].add(cluster_list[j])
                    adjacency[cluster_list[j]].add(cluster_list[i])
    
    # Établir des connexions UNIQUEMENT entre clusters adjacents
    for c1 in range(num_clusters - 1):
        c2 = c1 + 1  # Cluster adjacent suivant
        
        # Sélectionner des nœuds passerelles
        num_gateways = 1 if t % 5 != 0 else 2
        
        gateways1 = random.sample(list(clusters[c1]), min(num_gateways, len(clusters[c1])))
        gateways2 = random.sample(list(clusters[c2]), min(num_gateways, len(clusters[c2])))
        
        # Établir des connexions entre passerelles
        for g1 in gateways1:
            for g2 in gateways2:
                base_prob = 0.3 * dilution_factor
                time_factor = 0.2 * ((t % 10) / 10.0)
                
                connect_prob = base_prob + time_factor
                
                if random.random() < connect_prob:
                    adjacency[g1].add(g2)
                    adjacency[g2].add(g1)
    
    # S'assurer qu'il n'y a pas de connexion directe entre source et destination
    if destination in adjacency[source]:
        adjacency[source].remove(destination)
    if source in adjacency[destination]:
        adjacency[destination].remove(source)
    
    return adjacency

def visualize_network(adjacency: dict[int, set[int]], t: int, copies: dict[int, int], 
                      delivered_at: dict, output_dir: str, 
                      failed_nodes=None, 
                      failure_manager=None,
                      filename_prefix="network"):
    """
    Visualise le réseau à un instant donné, montrant la distribution des copies
    et les nœuds en panne.
    
    Args:
        adjacency: Dictionnaire d'adjacence représentant le réseau
        t: L'instant actuel
        copies: Nombre de copies par nœud
        delivered_at: Dictionnaire des nœuds ayant reçu le message
        output_dir: Dossier de sortie pour les visualisations
        failed_nodes: Ensemble des nœuds en panne
        failure_manager: Gestionnaire de pannes dynamiques pour des info supplémentaires
        filename_prefix: Préfixe pour le nom de fichier
    """
    G = nx.Graph()
    failed_nodes = failed_nodes or set()
    
    # Ajouter les nœuds et arêtes
    for node, neighbors in adjacency.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Ajouter les nœuds en panne (même s'ils n'ont pas de connections)
    for node in failed_nodes:
        if node not in G:
            G.add_node(node)
    
    # Préparer la visualisation
    plt.figure(figsize=(14, 10))
    
    # Utiliser un layout qui montre mieux la structure en clusters
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Calculer le nombre maximal de copies pour la normalisation
    max_copies = max(copies.values()) if copies else 1
    
    # Définir les tailles de nœuds en fonction du nombre de copies
    base_size = 800  # Taille de base
    size_factor = 400  # Facteur multiplicatif
    node_sizes = [base_size + (size_factor * (copies.get(node, 0))) for node in G.nodes()]
    
    # Définir les couleurs et formes des nœuds
    node_colors = []
    labels = {}
    edge_colors = []
    edge_widths = []
    node_shapes = []  # Pour distinguer les nœuds en panne
    
    source = 0
    destination = max(adjacency.keys())
    
    # Collecter les info sur les nœuds pour une meilleure visualisation
    for node in G.nodes():
        num_copies = copies.get(node, 0)
        is_failed = node in failed_nodes
        
        # Déterminer la couleur, taille et label pour chaque nœud
        if is_failed:
            # Nœuds en panne: noir avec une croix
            node_colors.append('black')
            labels[node] = f"X{node}"
            node_shapes.append('x')
        elif node == source:
            # Source: vert, avec nombre de copies
            node_colors.append('green')
            labels[node] = f"S:{num_copies}"
            node_shapes.append('o')
        elif node == destination:
            # Destination: doré si livré, sinon rouge
            if node in delivered_at:
                node_colors.append('gold')
                labels[node] = f"D✓:{num_copies} (t={delivered_at[node]})"
                node_shapes.append('o')
            else:
                node_colors.append('red')
                labels[node] = f"D:{num_copies}"
                node_shapes.append('o')
        elif num_copies > 0:
            # Nœuds avec copies: gradient de bleu selon le nombre de copies
            intensity = min(1.0, 0.3 + 0.7 * (num_copies / max_copies))
            node_colors.append((0, 0, intensity))
            labels[node] = f"{node}:{num_copies}"
            node_shapes.append('o')
        else:
            # Nœuds sans copies
            node_colors.append('lightgray')
            labels[node] = str(node)
            node_shapes.append('o')
    
    # Déterminer les couleurs et épaisseurs des arêtes
    for u, v in G.edges():
        if u in failed_nodes or v in failed_nodes:
            edge_colors.append('gray')
            edge_widths.append(0.2)
        elif (copies.get(u, 0) > 0 and copies.get(v, 0) == 0) or (copies.get(u, 0) == 0 and copies.get(v, 0) > 0):
            edge_colors.append('blue')
            edge_widths.append(2.0)
        elif copies.get(u, 0) > 0 and copies.get(v, 0) > 0:
            edge_colors.append('darkblue')
            edge_widths.append(2.5)
        elif u == source or v == source or u == destination or v == destination:
            edge_colors.append('green' if (u == source or v == source) else 'red')
            edge_widths.append(3.0)
        else:
            edge_colors.append('lightgray')
            edge_widths.append(0.5)
    
    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color=edge_colors, width=edge_widths)
    
    # Trier les nœuds selon leurs formes
    nodes_list = list(G.nodes())
    shapes = {'o': [], 'x': []}
    colors = {'o': [], 'x': []}
    sizes = {'o': [], 'x': []}
    
    for i, node in enumerate(nodes_list):
        if i < len(node_shapes):
            shape = node_shapes[i]
            shapes[shape].append(node)
            colors[shape].append(node_colors[i])
            sizes[shape].append(node_sizes[i])
    
    # Dessiner les nœuds selon leurs formes
    if shapes['o']:
        nx.draw_networkx_nodes(G, pos, nodelist=shapes['o'], 
                             node_size=sizes['o'], 
                             node_color=colors['o'], 
                             alpha=0.8, 
                             edgecolors='black', 
                             linewidths=1)
    
    if shapes['x']:
        nx.draw_networkx_nodes(G, pos, nodelist=shapes['x'], 
                             node_size=sizes['x'], 
                             node_color=colors['x'], 
                             alpha=0.8, 
                             edgecolors='black', 
                             linewidths=1, 
                             node_shape='x')
    
    # Dessiner les labels
    for i, node in enumerate(G.nodes()):
        if i < len(node_colors):
            color = node_colors[i]
            is_dark = isinstance(color, tuple) or color in ['green', 'blue', 'darkblue', 'black']
            
            if node in labels:
                nx.draw_networkx_labels(G, pos, {node: labels[node]}, 
                                     font_size=10, 
                                     font_color="white" if is_dark else "black")
    
    plt.title(f"État du réseau à t={t}", fontsize=16)
    plt.axis('off')
    
    # Ajouter des informations sur le statut actuel
    active_nodes = sum(1 for _, c in copies.items() if c > 0)
    total_copies = sum(copies.values())
    delivered = destination in delivered_at
    delivery_info = f"Livré à t={delivered_at[destination]}" if delivered else "Non livré"
    
    status_text = (f"Nœuds actifs: {active_nodes}/{len(G.nodes())-len(failed_nodes)}\n"
                  f"Copies totales: {total_copies}\n"
                  f"Nœuds en panne: {len(failed_nodes)}/{len(G.nodes())}\n"
                  f"Message: {delivery_info}")
    
    plt.figtext(0.02, 0.02, status_text, fontsize=12, 
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Légende
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Source', markersize=15, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='o', color='w', label='Destination', markersize=15, markerfacecolor='red'),
        plt.Line2D([0], [0], marker='o', color='w', label='Destination livrée', markersize=15, markerfacecolor='gold'),
        plt.Line2D([0], [0], marker='o', color='w', label='Relais avec copies', markersize=15, markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='Nœud sans copie', markersize=15, markerfacecolor='lightgray'),
        plt.Line2D([0], [0], marker='x', color='black', label='Nœud en panne', markersize=15, markerfacecolor='black'),
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=3, fontsize=10, frameon=True, facecolor='white')
    
    # Sauvegarder l'image
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename_prefix}_t{t:02d}.png", dpi=300, bbox_inches='tight')
    plt.close()

def test_spray_and_wait_with_dynamic_failures():
    """
    Test du protocole Spray-and-Wait avec des pannes dynamiques et continues
    qui se produisent tout au long de la simulation, pas seulement à un moment précis.
    """
    print("=== Test du protocole Spray-and-Wait avec pannes dynamiques et continues ===")
    
    # Paramètres de simulation
    num_nodes = 30  # Nombre de nœuds dans le réseau
    max_steps = 60  # Nombre de pas de temps de la simulation
    L_values = [4, 8, 16, 32]  # Ajouter ou modifier les valeurs à tester
    ttl_value = 30  # TTL des messages
    distribution_rate = 0.5  # Taux de distribution ralenti pour voir l'impact des pannes
    network_dilution = 0.8  # Dilution du réseau (réduit la densité)
    
    # Modes de test
    test_modes = {
        'continuous': {
            'desc': 'Pannes aléatoires continues tout au long de la simulation',
            'params': {'failure_prob': 0.03}  # 3% de chance qu'un nœud tombe en panne à chaque pas
        },
        'cascade': {
            'desc': 'Pannes en cascade (effet domino)',
            'params': {'base_prob': 0.01, 'cascade_factor': 5.0}
        },
        'targeted_dynamic': {
            'desc': 'Pannes ciblant dynamiquement les nœuds les plus critiques',
            'params': {'recalculation_interval': 5, 'failure_percentage': 0.07}
        }
    }
    
    # Dossier de sortie principal
    main_output_dir = f"{OUTDIR}/protocols/spray_and_wait_dynamic_failures_test"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Demander le mode à exécuter
    print("\nModes de test disponibles:")
    for i, (mode, details) in enumerate(test_modes.items(), 1):
        print(f"{i}. {mode}: {details['desc']}")
    
    try:
        mode_index = int(input("\nChoisissez un mode (1-3): ").strip()) - 1
        failure_mode = list(test_modes.keys())[mode_index]
    except (ValueError, IndexError):
        print("Choix invalide, utilisation du mode 'continuous' par défaut")
        failure_mode = 'continuous'
    
    print(f"\nMode sélectionné: {failure_mode} - {test_modes[failure_mode]['desc']}")
    
    # Configurer la sortie spécifique au mode
    output_dir = f"{main_output_dir}/{failure_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Résultats
    all_results = []
    
    # Pour chaque valeur de L
    for L in L_values:
        print(f"\n{'='*70}")
        print(f"=== TEST AVEC L={L}, MODE={failure_mode.upper()} ===")
        print(f"{'='*70}")
        
        # Initialiser le protocole
        source = 0
        destination = num_nodes - 1
        protocol = SprayAndWait(num_nodes, L, destination, source, binary=True,
                              ttl=ttl_value, distribution_rate=distribution_rate)
        
        # Initialiser le gestionnaire de pannes dynamiques
        failure_manager = DynamicFailureManager(num_nodes, source, destination)
        
        # Variables pour suivre l'état de la simulation
        copies_history = []
        failed_nodes_by_time = []
        active_copies_over_time = []
        delivery_occurred = False
        delivery_time = float('inf')
        
        # Visualiser l'état initial
        print(f"\nConfiguration initiale du réseau (dilution={network_dilution})")
        print(f"Source: {source}, Destination: {destination}")
        initial_adjacency = create_multihop_network(0, num_nodes, dilution_factor=network_dilution)
        visualize_network(initial_adjacency, 0, protocol.copies, protocol.delivered_at, output_dir, 
                        failure_manager.failed_nodes, failure_manager, filename_prefix=f"network_L{L}")
        
        # Affichage en format tableau pour suivre l'évolution
        print(f"\n{'t':>3} | {'Copies actives':^15} | {'Nœuds actifs':^15} | {'Nœuds en panne':^15} | {'Livré':<7}")
        print("-" * 70)
        
        # Boucle principale de simulation
        for t in range(max_steps):
            # Générer le réseau pour l'instant t avec le facteur de dilution
            adjacency = create_multihop_network(t, num_nodes, dilution_factor=network_dilution)
            
            # Appliquer les pannes dynamiques selon le mode choisi
            adjacency = failure_manager.apply_failures(
                t, adjacency, failure_mode, test_modes[failure_mode]['params']
            )
            
            # Enregistrer l'état actuel
            copies_history.append(protocol.copies.copy())
            failed_nodes_by_time.append(len(failure_manager.failed_nodes))
            
            # Nombre de copies actives
            active_copies = sum(1 for _, c in protocol.copies.items() if c > 0)
            active_copies_over_time.append(active_copies)
            
            # Visualiser périodiquement et aux moments clés
            should_visualize = (t == 0 or t % 10 == 0 or t == max_steps - 1)
            if should_visualize or (protocol.dest in protocol.delivered_at and not delivery_occurred):
                visualize_network(adjacency, t, protocol.copies, protocol.delivered_at, output_dir, 
                                failure_manager.failed_nodes, failure_manager, filename_prefix=f"network_L{L}")
            
            # Exécuter le pas de simulation
            protocol.step(t, adjacency)
            
            # Vérifier si la livraison vient de se produire
            if protocol.dest in protocol.delivered_at and not delivery_occurred:
                delivery_occurred = True
                delivery_time = protocol.delivered_at[protocol.dest]
                print(f"\n>>> MESSAGE LIVRÉ À t={delivery_time} <<<")
            
            # Afficher l'état actuel
            nodes_with_copies = [n for n, c in protocol.copies.items() if c > 0]
            total_active_copies = sum(protocol.copies.values())
            delivered = protocol.dest in protocol.delivered_at
            
            delivered_str = f"Oui (t={protocol.delivered_at.get(protocol.dest, 'N/A')})" if delivered else "Non"
            
            print(f"{t:3d} | {total_active_copies:^15d} | {len(adjacency):^15d} | "
                 f"{len(failure_manager.failed_nodes):^15d} | {delivered_str:<7}")
            
            # Arrêter si toutes les copies ont expiré et si le message n'est pas livrable
            if total_active_copies == 0:
                if not delivered:
                    print(f"\n>>> ÉCHEC: TOUTES LES COPIES EXPIRÉES SANS LIVRAISON À t={t} <<<")
                else:
                    print(f"\n>>> SUCCÈS: Livraison terminée, copies épuisées à t={t} <<<")
                break
        
        # Calculer les métriques finales
        delivery_ratio = protocol.delivery_ratio()
        delivery_delay = protocol.delivery_delay()
        overhead_ratio = protocol.overhead_ratio()
        
        # Obtenir des statistiques sur les sauts
        hop_stats = protocol.get_hop_stats()
        dest_hops = hop_stats.get('destination', float('inf'))
        
        # Enregistrer les résultats
        result = {
            'L': L,
            'TTL': ttl_value,
            'failure_mode': failure_mode,
            'network_dilution': network_dilution,
            'distribution_rate': distribution_rate,
            'failed_nodes_final': len(failure_manager.failed_nodes),
            'delivered': bool(delivery_ratio),
            'delivery_delay': delivery_delay if delivery_ratio > 0 else float('inf'),
            'overhead_ratio': overhead_ratio if delivery_ratio > 0 else float('inf'),
            'total_copies': protocol.total_copies_created,
            'hop_count': dest_hops if dest_hops != float('inf') else None
        }
        
        all_results.append(result)
        
        # Rapport sur les résultats
        print(f"\n--- Résultats pour L={L}, Mode={failure_mode} ---")
        if delivery_ratio > 0:
            print(f"✓ Message livré avec succès en {delivery_delay} unités de temps")
            print(f"✓ {protocol.total_copies_created} copies créées au total")
            print(f"✓ Surcharge: {overhead_ratio:.2f} copies par message livré")
            print(f"✓ Sauts pour atteindre la destination: {dest_hops}")
        else:
            print(f"✗ Échec de livraison (0% de taux de livraison)")
            print(f"✗ {protocol.total_copies_created} copies créées, toutes perdues ou expirées")
            print(f"✗ Message bloqué par {len(failure_manager.failed_nodes)} nœuds en panne sur {num_nodes}")
        
        # Graphiques d'analyse
        # 1. Évolution des pannes au fil du temps
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(failed_nodes_by_time)), failed_nodes_by_time, 'r-', linewidth=2, 
               label="Nœuds en panne")
        plt.plot(range(len(active_copies_over_time)), active_copies_over_time, 'b-', linewidth=2, 
               label="Nœuds avec copies")
        
        plt.xlabel("Temps (t)")
        plt.ylabel("Nombre de nœuds")
        plt.title(f"Impact des pannes dynamiques - L={L}, Mode={failure_mode}")
        
        # Marquer l'instant de livraison s'il y a eu livraison
        if delivery_occurred:
            plt.axvline(x=delivery_time, color='green', linestyle='--', 
                      label=f"Livraison (t={delivery_time})")
            
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.savefig(f"{output_dir}/dynamic_failures_L{L}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap de propagation avec overlay des pannes
        if copies_history:
            # Créer un DataFrame pour le heatmap des copies
            df_copies = pd.DataFrame(copies_history).fillna(0)
            plt.figure(figsize=(12, 8))
            plt.title(f"Propagation des copies avec pannes dynamiques - L={L}, Mode={failure_mode}")
            
            # Créer une colormap personnalisée
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'lightblue', 'blue', 'darkblue'])
            plt.imshow(df_copies.values.T, aspect='auto', cmap=cmap, interpolation='nearest')
            
            # Superposer les pannes sur le heatmap
            for t, failure_data in enumerate(failure_manager.failure_history):
                if t >= len(df_copies):  # Éviter les erreurs d'indexation
                    break
                
                for node_id in failure_data['new_failures']:
                    if node_id < df_copies.values.T.shape[0]:  # Vérifier que l'indice est valide
                        plt.scatter(t, node_id, color='red', marker='x', s=50)
            
            plt.xlabel("Temps (t)")
            plt.ylabel("Identifiant du nœud")
            plt.colorbar(label="Nombre de copies")
            
            # Marquer événement important
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='green', linestyle='--', 
                          label=f"Livraison (t={delivery_time})")
            
            plt.legend(loc='best')
            plt.savefig(f"{output_dir}/heatmap_with_failures_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Créer un tableau comparatif
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n=== Tableau comparatif des résultats ===")
        print(df_results.to_string(index=False))
        
        # Sauvegarder les résultats
        df_results.to_csv(f"{output_dir}/resultats_pannes_dynamiques.csv", index=False)
        
        # Créer des graphiques comparatifs
        plt.figure(figsize=(10, 8))
        
        # Taux de livraison vs L
        plt.subplot(2, 1, 1)
        plt.plot(df_results['L'], df_results['delivered'].astype(int), 'o-', linewidth=2, color='green')
        plt.title(f'Taux de livraison avec pannes dynamiques - Mode: {failure_mode}')
        plt.xlabel('L (nombre initial de copies)')
        plt.ylabel('Livraison réussie')
        plt.xticks(df_results['L'])
        plt.yticks([0, 1], ['Échec', 'Succès'])
        plt.grid(True, alpha=0.3)
        
        # Délai de livraison vs L (seulement pour les messages livrés)
        plt.subplot(2, 1, 2)
        delivered_df = df_results[df_results['delivered']]
        
        if not delivered_df.empty:
            plt.plot(delivered_df['L'], delivered_df['delivery_delay'], 'o-', linewidth=2, color='blue')
            plt.title(f'Délai de livraison avec pannes dynamiques - Mode: {failure_mode}')
            plt.xlabel('L (nombre initial de copies)')
            plt.ylabel('Délai de livraison')
            plt.xticks(df_results['L'])
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Aucun message livré", ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparative.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n=== Test terminé ===")
    print(f"Résultats sauvegardés dans: {output_dir}")

if __name__ == "__main__":
    test_spray_and_wait_with_dynamic_failures()
    print("\nTest de résilience aux pannes dynamiques et continues terminé!")
