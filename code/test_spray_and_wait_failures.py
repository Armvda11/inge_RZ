#!/usr/bin/env python3
# test_spray_and_wait_failures.py
"""
Script de test spécifique pour évaluer la résilience du protocole Spray-and-Wait
face à différents scénarios de panne.

Ce script améliore le test des pannes en:
1. Déclenchant les pannes PLUS TÔT (à t=0 ou t=1)
2. Augmentant le nombre de nœuds en panne
3. Permettant de tester avec une topologie plus étendue/diluée
4. Recalculant la centralité à chaque pas de temps
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

def create_multihop_network(t: int, num_nodes: int = 30, cluster_size: int = 5, dilution_factor: float = 1.0) -> dict[int, set[int]]:
    """
    Crée un réseau test dilué qui simule une topologie en "grappes" mobiles avec scénario multi-sauts.
    
    Args:
        t (int): L'instant de temps (pour simuler le mouvement)
        num_nodes (int): Nombre total de nœuds dans le réseau (augmenté pour plus de réalisme)
        cluster_size (int): Nombre approximatif de nœuds par cluster
        dilution_factor (float): Facteur pour réduire la densité des liens (1.0=normal, <1.0=plus dilué)
        
    Returns:
        dict[int, set[int]]: Dictionnaire d'adjacence
    """
    random.seed(42 + t)  # Seed pour reproductibilité, mais différente à chaque instant
    
    # Initialiser le réseau vide
    adjacency = {i: set() for i in range(num_nodes)}
    
    # Nombre de clusters - au moins 4 pour un véritable scenario multi-sauts
    num_clusters = max(4, num_nodes // cluster_size)
    
    # Attribuer des nœuds à des clusters de manière stricte pour former une chaîne
    clusters = [set() for _ in range(num_clusters)]
    
    # Répartir les nœuds équitablement entre les clusters
    # Réserver la source pour le cluster 0 et la destination pour le dernier cluster
    reserved_nodes = {0, num_nodes - 1}  # Source et destination
    regular_nodes = [i for i in range(num_nodes) if i not in reserved_nodes]
    random.shuffle(regular_nodes)  # Mélanger les nœuds pour une attribution aléatoire
    
    # Calculer combien de nœuds par cluster (distribution approximativement égale)
    nodes_per_cluster = (num_nodes - len(reserved_nodes)) // num_clusters
    remainder = (num_nodes - len(reserved_nodes)) % num_clusters
    
    # Attribuer la source au premier cluster et la destination au dernier
    clusters[0].add(0)  # Source dans le premier cluster
    clusters[-1].add(num_nodes - 1)  # Destination dans le dernier cluster
    
    # Distribuer les nœuds restants aux clusters
    node_index = 0
    for c in range(num_clusters):
        # Combien de nœuds attribuer à ce cluster
        nodes_for_this_cluster = nodes_per_cluster + (1 if c < remainder else 0)
        
        # Attribuer les nœuds, en sautant ceux déjà réservés
        for _ in range(nodes_for_this_cluster):
            if node_index < len(regular_nodes):
                clusters[c].add(regular_nodes[node_index])
                node_index += 1
    
    # Établir des connexions à l'intérieur des clusters (connexions intra-cluster)
    # Mais avec une probabilité réduite par le facteur de dilution
    for cluster_idx, cluster in enumerate(clusters):
        cluster_list = list(cluster)
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                # Connectivité intra-cluster variable selon le cluster et le temps
                # Les clusters du milieu ont une connectivité plus faible
                base_density = 0.7 * dilution_factor  # Densité de base réduite par la dilution
                
                # Les clusters du milieu sont moins denses
                if 0 < cluster_idx < num_clusters - 1:
                    # Réduire la densité pour les clusters intermédiaires
                    cluster_factor = 0.8  # Réduction de 20%
                else:
                    # Maintenir une bonne connectivité pour les clusters source et destination
                    cluster_factor = 1.0
                
                # Ajouter une variation temporelle mineure
                time_factor = 0.95 + 0.1 * ((t % 5) / 5.0)  # Entre 95% et 105%
                
                connect_prob = base_density * cluster_factor * time_factor
                
                if random.random() < connect_prob:
                    adjacency[cluster_list[i]].add(cluster_list[j])
                    adjacency[cluster_list[j]].add(cluster_list[i])
    
    # Établir des connexions UNIQUEMENT entre clusters adjacents avec une connectivité très limitée
    # avec probabilité réduite par le facteur de dilution
    for c1 in range(num_clusters - 1):
        c2 = c1 + 1  # Cluster adjacent suivant
        
        # Sélectionner des nœuds de passerelle
        num_gateways = 1 if t % 5 != 0 else 2  # Généralement 1 passerelle, parfois 2
        
        gateways1 = random.sample(list(clusters[c1]), min(num_gateways, len(clusters[c1])))
        gateways2 = random.sample(list(clusters[c2]), min(num_gateways, len(clusters[c2])))
        
        # Établir des connexions entre passerelles avec une probabilité plus faible
        for g1 in gateways1:
            for g2 in gateways2:
                # Connectivité inter-cluster réduite et variable dans le temps
                base_prob = 0.3 * dilution_factor  # Probabilité de base réduite par la dilution
                time_factor = 0.2 * ((t % 10) / 10.0)  # Variation temporelle réduite
                
                connect_prob = base_prob + time_factor
                
                if random.random() < connect_prob:
                    adjacency[g1].add(g2)
                    adjacency[g2].add(g1)
    
    # Absolument s'assurer qu'il n'y a pas de connexion directe entre source et destination
    if num_nodes - 1 in adjacency[0]:
        adjacency[0].remove(num_nodes - 1)
    if 0 in adjacency[num_nodes - 1]:
        adjacency[num_nodes - 1].remove(0)
    
    return adjacency

def get_top_centrality_nodes(adjacency: dict[int, set[int]], num_to_select: int, source: int = 0, dest: int = None):
    """
    Identifie les nœuds avec la plus haute centralité (betweenness) dans le réseau.
    Exclut la source et la destination des candidats.
    
    Args:
        adjacency: Dictionnaire d'adjacence représentant le réseau
        num_to_select: Nombre de nœuds à sélectionner
        source: ID du nœud source (à exclure)
        dest: ID du nœud destination (à exclure)
        
    Returns:
        list: Liste des IDs des nœuds à plus haute centralité
    """
    # Créer un graphe NetworkX à partir du dictionnaire d'adjacence
    G = nx.Graph()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Calculer la centralité de tous les nœuds
    centrality = nx.betweenness_centrality(G)
    
    # Trier par centralité décroissante et filtrer la source et destination
    nodes_to_exclude = {source}
    if dest is not None:
        nodes_to_exclude.add(dest)
    
    sorted_centrality = sorted(
        [(node, score) for node, score in centrality.items() if node not in nodes_to_exclude],
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Sélectionner les top N nœuds
    return [node for node, _ in sorted_centrality[:num_to_select]]

def visualize_network(adjacency: dict[int, set[int]], t: int, copies: dict[int, int], 
                      delivered_at: dict, output_dir: str, 
                      failed_nodes=None, 
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
    
    # Définir les couleurs des nœuds avec un gradient dépendant du nombre de copies
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
    
    status_text = (f"Nœuds actifs: {active_nodes}/{len(G.nodes())}\n"
                  f"Copies totales: {total_copies}\n"
                  f"Nœuds en panne: {len(failed_nodes)}\n"
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

def test_spray_and_wait_with_early_failures():
    """
    Test amélioré du protocole Spray-and-Wait avec des pannes déclenchées
    très tôt dans la simulation, pour vraiment évaluer la résilience du protocole.
    """
    print("=== Test du protocole Spray-and-Wait avec pannes précoces ===")
    print("Ce test évalue spécifiquement la résilience face aux pannes activées TRÈS TÔT")
    
    # Paramètres de simulation ajustables
    num_nodes = 30  # Plus grand réseau pour mieux voir l'impact des pannes
    max_steps = 60  # Plus de temps pour observer la propagation après les pannes
    L_values = [2, 4, 8, 16]  # Tester avec plus de valeurs de L
    ttl_value = 30  # Augmenter le TTL pour permettre plus de tentatives après panne
    distribution_rate = 0.3  # Ralentir la distribution pour voir l'effet des pannes
    
    # Paramètres de panne critiques
    failure_time = 1  # Panne TRÈS précoce (peut être 0 ou 1)
    network_dilution = 0.7  # Diluer le réseau pour le rendre plus vulnérable aux pannes
    
    # Options de test
    test_modes = {
        'early_random': {
            'desc': 'Pannes aléatoires précoces (25% des nœuds)',
            'node_percent': 0.25  # 25% des nœuds tombent en panne
        },
        'early_targeted': {
            'desc': 'Pannes ciblées précoces (nœuds les plus centraux)',
            'node_count': 3  # Les 3 nœuds à plus haute centralité
        },
        'early_critical_targeted': {
            'desc': 'Pannes ciblées précoces sévères (plus de nœuds centraux)',
            'node_count': 5  # Les 5 nœuds à plus haute centralité
        },
        'none': {
            'desc': 'Sans panne (référence)'
        }
    }
    
    # Dossier de sortie
    main_output_dir = f"{OUTDIR}/protocols/spray_and_wait_early_failures_test"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Demander le mode à exécuter
    print("\nModes de test disponibles:")
    for i, (mode, details) in enumerate(test_modes.items(), 1):
        print(f"{i}. {mode}: {details['desc']}")
    
    try:
        mode_index = int(input("\nChoisissez un mode (1-4): ").strip()) - 1
        failure_mode = list(test_modes.keys())[mode_index]
    except (ValueError, IndexError):
        print("Choix invalide, utilisation du mode 'early_targeted' par défaut")
        failure_mode = 'early_targeted'
    
    print(f"\nMode sélectionné: {failure_mode} - {test_modes[failure_mode]['desc']}")
    
    # Configurer la sortie spécifique au mode
    output_dir = f"{main_output_dir}/{failure_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Résultats
    all_results = []
    
    # Pour chaque valeur de L
    for L in L_values:
        print(f"\n{'='*70}")
        print(f"=== TEST AVEC L={L}, MODE={failure_mode.upper()}, PANNE À T={failure_time} ===")
        print(f"{'='*70}")
        
        # Initialiser le protocole
        source = 0
        destination = num_nodes - 1
        protocol = SprayAndWait(num_nodes, L, destination, source, binary=True,
                               ttl=ttl_value, distribution_rate=distribution_rate)
        
        # Variables pour suivre l'état de la simulation
        copies_history = []
        failed_nodes_over_time = []
        active_copies_over_time = []
        delivery_occurred = False
        delivery_time = float('inf')
        failed_nodes = set()
        
        # Générer le réseau initial pour l'identification des nœuds centraux
        initial_adjacency = create_multihop_network(0, num_nodes, dilution_factor=network_dilution)
        
        # Visualiser l'état initial
        print(f"\nConfiguration initiale du réseau (dilution={network_dilution})")
        print(f"Source: {source}, Destination: {destination}")
        visualize_network(initial_adjacency, 0, protocol.copies, protocol.delivered_at, output_dir, 
                         failed_nodes, filename_prefix=f"initial_L{L}")
        
        # Pré-calculer les nœuds à cibler (mais appliquer plus tard à t=failure_time)
        nodes_to_fail = set()
        if failure_mode != 'none':
            if failure_mode == 'early_random':
                # Pannes aléatoires: x% des nœuds
                num_to_fail = int(num_nodes * test_modes[failure_mode]['node_percent'])
                candidates = [n for n in range(1, num_nodes-1) if n != source and n != destination]
                nodes_to_fail = set(random.sample(candidates, min(num_to_fail, len(candidates))))
                print(f"Nœuds sélectionnés pour tomber en panne à t={failure_time} (aléatoire): {sorted(nodes_to_fail)}")
            
            elif 'targeted' in failure_mode:
                # Pannes ciblées: nœuds les plus centraux
                num_to_fail = test_modes[failure_mode]['node_count']
                central_nodes = get_top_centrality_nodes(initial_adjacency, num_to_fail, source, destination)
                nodes_to_fail = set(central_nodes)
                print(f"Nœuds les plus centraux qui tomberont en panne à t={failure_time}: {sorted(nodes_to_fail)}")
        
        # Affichage en format tableau pour suivre l'évolution
        print(f"\n{'t':>3} | {'Copies actives':^15} | {'Nœuds avec copies':^20} | {'Nœuds en panne':^15} | {'Livré':<7}")
        print("-" * 90)
        
        # Boucle principale de simulation
        for t in range(max_steps):
            # Générer le réseau pour l'instant t avec le facteur de dilution
            adjacency = create_multihop_network(t, num_nodes, dilution_factor=network_dilution)
            
            # Appliquer les pannes TRÈS TÔT dans la simulation
            if failure_mode != 'none' and t == failure_time:
                print(f"\n>>> ACTIVATION DES PANNES À t={t} <<<")
                
                # Créer une copie du dictionnaire d'adjacence
                adjacency_with_failures = {}
                for node, neighbors in adjacency.items():
                    adjacency_with_failures[node] = neighbors.copy()
                
                # Supprimer les connexions des nœuds en panne
                for node in nodes_to_fail:
                    failed_nodes.add(node)
                    if node in adjacency_with_failures:
                        # Le nœud ne peut plus communiquer
                        adjacency_with_failures[node] = set()
                        # Supprimer le nœud des listes d'adjacence des autres nœuds
                        for other, neighbors in adjacency_with_failures.items():
                            if node in neighbors:
                                neighbors.remove(node)
                
                # Mettre à jour l'adjacence
                adjacency = adjacency_with_failures
                
                # Mettre à jour les copies (les copies sur les nœuds en panne sont perdues)
                num_copies_lost = 0
                for node in failed_nodes:
                    if protocol.copies.get(node, 0) > 0:
                        num_copies_lost += protocol.copies[node]
                        protocol.copies[node] = 0
                
                print(f"Nœuds en panne: {sorted(failed_nodes)} ({len(failed_nodes)} nœuds)")
                if num_copies_lost > 0:
                    print(f"ATTENTION: {num_copies_lost} copies ont été perdues à cause des pannes!")
            
            # Enregistrer l'état actuel
            copies_history.append(protocol.copies.copy())
            failed_nodes_over_time.append(len(failed_nodes))
            
            # Nombre de copies actives
            active_copies = sum(1 for _, c in protocol.copies.items() if c > 0)
            active_copies_over_time.append(active_copies)
            
            # Visualiser aux instants-clés
            should_visualize = (t == 0 or t == failure_time or 
                               t % 10 == 0 or t == max_steps - 1)
            if should_visualize or (protocol.dest in protocol.delivered_at and not delivery_occurred):
                visualize_network(adjacency, t, protocol.copies, protocol.delivered_at, output_dir, 
                                 failed_nodes, filename_prefix=f"network_L{L}")
            
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
            failed_str = ', '.join(map(str, sorted(failed_nodes))) if len(failed_nodes) <= 5 else f"{len(failed_nodes)} nœuds"
            nodes_str = ', '.join(map(str, nodes_with_copies)) if len(nodes_with_copies) <= 10 else f"{len(nodes_with_copies)} nœuds"
            
            print(f"{t:3d} | {total_active_copies:^15d} | {nodes_str:^20} | {failed_str:^15} | {delivered_str:<7}")
            
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
            'failure_time': failure_time,
            'num_nodes': num_nodes,
            'dilution': network_dilution,
            'failed_nodes_count': len(failed_nodes),
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
            print(f"✗ Message bloqué par les pannes des nœuds: {sorted(failed_nodes)}")
        
        # Visualiser l'évolution des copies
        if copies_history:
            # Heatmap de propagation
            df_copies = pd.DataFrame(copies_history).fillna(0)
            plt.figure(figsize=(12, 8))
            plt.title(f"Propagation des copies - L={L}, Mode={failure_mode}")
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'lightblue', 'blue', 'darkblue'])
            plt.imshow(df_copies.values.T, aspect='auto', cmap=cmap, interpolation='nearest')
            plt.xlabel("Temps (t)")
            plt.ylabel("Identifiant du nœud")
            plt.colorbar(label="Nombre de copies")
            
            # Marquer événements importants
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='green', linestyle='--', 
                          label=f"Livraison (t={delivery_time})")
            
            plt.axvline(x=failure_time, color='red', linestyle='-', 
                      label=f"Pannes (t={failure_time})")
            
            plt.legend(loc='best')
            plt.savefig(f"{output_dir}/heatmap_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Graphique des copies actives
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(active_copies_over_time)), active_copies_over_time, 
                   'b-', linewidth=2, label="Nœuds avec copies")
            
            plt.xlabel("Temps (t)")
            plt.ylabel("Nombre de nœuds avec copies")
            plt.title(f"Impact des pannes - L={L}, Mode={failure_mode}")
            
            # Marquer événements
            plt.axvline(x=failure_time, color='red', linestyle='-', 
                      label=f"Activation des pannes (t={failure_time})")
            
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='green', linestyle='--', 
                          label=f"Livraison (t={delivery_time})")
            
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.savefig(f"{output_dir}/active_nœuds_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Créer un tableau comparatif
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n=== Tableau comparatif des résultats ===")
        print(df_results.to_string(index=False))
        
        # Sauvegarder les résultats
        df_results.to_csv(f"{output_dir}/resultats_pannes_precoces.csv", index=False)
        
        # Créer des graphiques comparatifs
        plt.figure(figsize=(10, 8))
        
        # Taux de livraison vs L
        plt.subplot(2, 1, 1)
        plt.plot(df_results['L'], df_results['delivered'].astype(int), 'o-', linewidth=2, color='green')
        plt.title(f'Taux de livraison avec pannes à t={failure_time} - Mode: {failure_mode}')
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
            plt.title(f'Délai de livraison avec pannes à t={failure_time} - Mode: {failure_mode}')
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
    
    return df_results if all_results else None

if __name__ == "__main__":
    test_spray_and_wait_with_early_failures()
    print("\nTest de résilience aux pannes précoces terminé avec succès!")
