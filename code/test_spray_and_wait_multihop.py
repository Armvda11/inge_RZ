#!/usr/bin/env python3
# test_spray_and_wait_multihop.py
"""
Script de test pour le protocole Spray-and-Wait en scénario multi-sauts.
Ce script simule un environnement où le message doit traverser plusieurs
relais intermédiaires avant d'atteindre sa destination.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
import random
from matplotlib.colors import LinearSegmentedColormap

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTDIR
from protocols.spray_and_wait import SprayAndWait
from simulation.failure import NodeFailureManager

def create_multihop_network(t: int, num_nodes: int = 20, cluster_size: int = 5) -> dict[int, set[int]]:
    """
    Crée un réseau test qui simule une topologie en "grappes" mobiles avec un véritable
    scénario multi-sauts où la source et la destination sont séparées par plusieurs clusters.
    La topologie garantit un véritable scénario multi-sauts en plaçant les clusters
    dans une chaîne linéaire et en ne permettant les connexions qu'entre clusters adjacents.
    
    Args:
        t (int): L'instant de temps (pour simuler le mouvement)
        num_nodes (int): Nombre total de nœuds dans le réseau
        cluster_size (int): Nombre approximatif de nœuds par cluster
        
    Returns:
        dict[int, set[int]]: Dictionnaire d'adjacence
    """
    random.seed(42 + t)  # Seed pour reproductibilité, mais différente à chaque instant
    
    # Initialiser le réseau vide
    adjacency = {i: set() for i in range(num_nodes)}
    
    # Nombre de clusters - au moins 4 pour un véritable scenario multi-sauts, mais aussi assez
    # pour éviter que le réseau ne soit trop dense et ne permette de contourner les sauts
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
    
    # Introduire un peu de mobilité: certains nœuds peuvent changer de cluster au fil du temps
    # Mais maintenir la mobilité contrôlée pour assurer que le message doit traverser chaque cluster
    if t > 0 and t % 3 == 0:  # Tous les 3 pas de temps
        # Trouver des candidats pour le mouvement (pas la source ni la destination)
        mobile_candidates = [n for c_idx, cluster in enumerate(clusters) 
                             for n in cluster 
                             if n != 0 and n != num_nodes - 1]
        
        # Choisir quelques nœuds à déplacer (max 10% des nœuds)
        num_to_move = max(1, len(mobile_candidates) // 10)
        nodes_to_move = random.sample(mobile_candidates, min(num_to_move, len(mobile_candidates)))
        
        for node in nodes_to_move:
            # Trouver le cluster actuel du nœud
            current_cluster = next(i for i, cluster in enumerate(clusters) if node in cluster)
            
            # Déterminer les clusters possibles pour déplacement (clusters adjacents seulement)
            possible_clusters = []
            if current_cluster > 0:  # Peut aller au cluster précédent
                possible_clusters.append(current_cluster - 1)
            if current_cluster < num_clusters - 1:  # Peut aller au cluster suivant
                possible_clusters.append(current_cluster + 1)
            
            if possible_clusters:
                # Choisir un cluster adjacent au hasard
                new_cluster = random.choice(possible_clusters)
                
                # Déplacer le nœud
                clusters[current_cluster].remove(node)
                clusters[new_cluster].add(node)
    
    # Établir des connexions à l'intérieur des clusters (connexions intra-cluster)
    for cluster_idx, cluster in enumerate(clusters):
        cluster_list = list(cluster)
        # Au lieu d'une clique complète, créer une topologie plus réaliste et éparse
        # avec une densité de connexion variable selon le cluster
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                # Connectivité intra-cluster variable selon le cluster et le temps
                # Les clusters au milieu ont une connectivité plus faible
                base_density = 0.7  # Densité de base
                
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
    # pour forcer un routage multi-sauts réaliste et ralentir la propagation
    for c1 in range(num_clusters - 1):
        c2 = c1 + 1  # Cluster adjacent suivant
        
        # Sélectionner des nœuds de passerelle
        # Le nombre de passerelles dépend du temps pour simuler la mobilité
        # Réduire considérablement le nombre de passerelles (de 1 à 2, mais souvent 1)
        num_gateways = 1 if t % 5 != 0 else 2  # Généralement 1 passerelle, parfois 2
        
        # S'assurer qu'il y a toujours au moins une passerelle entre clusters adjacents,
        # mais limiter les possibilités de passage entre clusters
        gateways1 = random.sample(list(clusters[c1]), min(num_gateways, len(clusters[c1])))
        gateways2 = random.sample(list(clusters[c2]), min(num_gateways, len(clusters[c2])))
        
        # Établir des connexions entre passerelles avec une probabilité plus faible
        # pour rendre le passage entre clusters plus difficile
        for g1 in gateways1:
            for g2 in gateways2:
                # Connectivité inter-cluster réduite et variable dans le temps
                # Réduire significativement la probabilité de connexion
                base_prob = 0.3  # Probabilité de base réduite
                time_factor = 0.2 * ((t % 10) / 10.0)  # Variation temporelle réduite
                
                # Pour les clusters qui contiennent la destination,
                # rendre les connexions encore plus difficiles dans les premiers pas de temps
                if c2 == num_clusters - 1 and t < 5:
                    connect_prob = base_prob * 0.5 + time_factor  # Encore plus difficile
                else:
                    connect_prob = base_prob + time_factor  # Entre 30% et 50%
                
                if random.random() < connect_prob:
                    adjacency[g1].add(g2)
                    adjacency[g2].add(g1)
    
    # Des connexions très rares entre clusters non-adjacents pour simuler des opportunités
    # de raccourcis exceptionnels, mais en veillant à garder la structure multi-sauts
    if t % 15 == 0 and t > 5:  # Extrêmement rarement (tous les 15 pas de temps et après t=5)
        # Limiter à un seul raccourci possible par pas de temps
        shortcut_created = False
        
        # Essayer de créer un raccourci entre clusters éloignés avec une probabilité très faible
        for c1 in range(num_clusters - 2):
            if shortcut_created:
                break
                
            for c2 in range(c1 + 2, min(c1 + 4, num_clusters)):  # Limiter la distance du raccourci
                # Plus les clusters sont éloignés, plus la probabilité est faible
                distance = c2 - c1
                shortcut_prob = 0.02 / distance  # Probabilité très faible
                
                # Interdire les raccourcis directs entre source et destination
                if c1 == 0 and c2 == num_clusters - 1:
                    continue  # Sauter cette combinaison
                
                if random.random() < shortcut_prob and not shortcut_created:
                    # Choisir des nœuds au hasard dans les clusters respectifs
                    if clusters[c1] and clusters[c2]:  # Vérifier que les clusters ne sont pas vides
                        node1 = random.choice(list(clusters[c1]))
                        node2 = random.choice(list(clusters[c2]))
                        
                        # Éviter de connecter la source et la destination
                        if (node1 == 0 and node2 == num_nodes - 1) or (node1 == num_nodes - 1 and node2 == 0):
                            continue
                            
                        # Créer un lien de raccourci temporaire
                        adjacency[node1].add(node2)
                        adjacency[node2].add(node1)
                        shortcut_created = True
    
    # Absolument s'assurer qu'il n'y a pas de connexion directe entre source et destination
    # pour garantir un véritable scénario multi-sauts
    if num_nodes - 1 in adjacency[0]:
        adjacency[0].remove(num_nodes - 1)
    if 0 in adjacency[num_nodes - 1]:
        adjacency[num_nodes - 1].remove(0)
    
    return adjacency

def visualize_network(adjacency: dict[int, set[int]], t: int, copies: dict[int, int], delivered_at: dict, output_dir: str, failed_nodes=None):
    """
    Visualise le réseau à un instant donné, montrant la distribution des copies.
    Améliore la visualisation avec une coloration graduelle selon le nombre de copies,
    des indications sur le TTL des messages et un affichage plus clair des connexions.
    
    Args:
        adjacency (dict): Dictionnaire d'adjacence représentant le réseau
        t (int): L'instant actuel
        copies (dict): Nombre de copies par nœud
        delivered_at (dict): Dictionnaire des nœuds ayant reçu le message
        output_dir (str): Dossier de sortie pour les visualisations
        failed_nodes (set): Ensemble des nœuds en panne
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
    # Avec un paramètre k plus grand pour espacer les nœuds
    pos = nx.spring_layout(G, seed=42, k=0.3)  # Position fixe pour une meilleure comparaison, espacée
    
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
    
    # Calculer les clusters pour la visualisation
    # Pour déterminer les clusters, on regroupe les nœuds selon leur position dans le graphe
    num_clusters = max(4, len(G.nodes()) // 5)  # Estimation du nombre de clusters
    
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
            # Plus le nœud a de copies, plus il est foncé
            intensity = min(1.0, 0.3 + 0.7 * (num_copies / max_copies))
            node_colors.append((0, 0, intensity))  # RGB pour bleu
            labels[node] = f"{node}:{num_copies}"
            node_shapes.append('o')
        else:
            # Nœuds sans copies
            node_colors.append('lightgray')
            labels[node] = str(node)
            node_shapes.append('o')
    
    # Déterminer les couleurs et épaisseurs des arêtes pour mieux visualiser les chemins
    # Les arêtes entre clusters sont plus épaisses
    for u, v in G.edges():
        # Les arêtes qui touchent des nœuds en panne sont grises et très fines
        if u in failed_nodes or v in failed_nodes:
            edge_colors.append('gray')
            edge_widths.append(0.2)
        # Vérifier si l'un des nœuds a des copies et l'autre non
        # Cela pourrait indiquer un chemin de diffusion potentiel
        elif (copies.get(u, 0) > 0 and copies.get(v, 0) == 0) or (copies.get(u, 0) == 0 and copies.get(v, 0) > 0):
            edge_colors.append('blue')  # Arête de diffusion potentielle
            edge_widths.append(2.0)
        # Arêtes entre nœuds avec copies
        elif copies.get(u, 0) > 0 and copies.get(v, 0) > 0:
            edge_colors.append('darkblue')  # Arête entre nœuds actifs
            edge_widths.append(2.5)
        # Arêtes impliquant la source ou la destination
        elif u == source or v == source or u == destination or v == destination:
            edge_colors.append('green' if (u == source or v == source) else 'red')
            edge_widths.append(3.0)
        else:
            # Autres arêtes
            edge_colors.append('lightgray')
            edge_widths.append(0.5)
    
    # Dessiner le graphe
    # Dessiner d'abord les arêtes avec les couleurs et épaisseurs calculées
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color=edge_colors, width=edge_widths)
    
    # Dessiner les nœuds avec des formes et couleurs différentes selon leur état
    # Dessiner les nœuds réguliers (pas en panne)
    nodes_list = list(G.nodes())
    shapes = {'o': [], 'x': []}
    colors = {'o': [], 'x': []}
    sizes = {'o': [], 'x': []}
    
    # Trier les nœuds selon leurs formes
    for i, node in enumerate(nodes_list):
        if i < len(node_shapes):  # S'assurer que l'index est valide
            shape = node_shapes[i]
            shapes[shape].append(node)
            colors[shape].append(node_colors[i])
            sizes[shape].append(node_sizes[i])
    
    # Dessiner les nœuds réguliers (cercles)
    if shapes['o']:
        nx.draw_networkx_nodes(G, pos, nodelist=shapes['o'], 
                             node_size=sizes['o'], 
                             node_color=colors['o'], 
                             alpha=0.8, 
                             edgecolors='black', 
                             linewidths=1)
    
    # Dessiner les nœuds en panne (X)
    if shapes['x']:
        nx.draw_networkx_nodes(G, pos, nodelist=shapes['x'], 
                             node_size=sizes['x'], 
                             node_color=colors['x'], 
                             alpha=0.8, 
                             edgecolors='black', 
                             linewidths=1, 
                             node_shape='x')
    
    # Ajuster les labels pour une meilleure visibilité selon la couleur du nœud
    for i, node in enumerate(G.nodes()):
        if i < len(node_colors):  # S'assurer que l'index est valide
            color = node_colors[i]
            # Déterminer si le fond du nœud est foncé
            is_dark = isinstance(color, tuple) or color in ['green', 'blue', 'darkblue', 'black']
            
            # S'assurer que le nœud a un label
            if node in labels:
                # Dessiner chaque label individuellement avec la couleur de texte adaptée
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
    
    # Ajouter une légende plus détaillée
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Source', markersize=15, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='o', color='w', label='Destination', markersize=15, markerfacecolor='red'),
        plt.Line2D([0], [0], marker='o', color='w', label='Destination livrée', markersize=15, markerfacecolor='gold'),
        plt.Line2D([0], [0], marker='o', color='w', label='Relais avec copies', markersize=15, markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='Nœud sans copie', markersize=15, markerfacecolor='lightgray'),
        plt.Line2D([0], [0], marker='x', color='black', label='Nœud en panne', markersize=15, markerfacecolor='black'),
        plt.Line2D([0], [0], color='darkblue', linewidth=2, label='Lien entre nœuds actifs'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Lien de diffusion potentiel')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=3, fontsize=10, frameon=True, facecolor='white')
    
    # Sauvegarder l'image avec une résolution plus élevée
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/network_t{t:02d}.png", dpi=300, bbox_inches='tight')
    plt.close()

def test_spray_and_wait_multihop():
    """
    Test du protocole Spray-and-Wait dans un scénario multi-sauts avec simulation de pannes.
    Ce test évalue l'impact de différents modes de pannes sur les métriques de performance:
    - Taux de livraison (Delivery Ratio)
    - Délai moyen de livraison (Average Delivery Delay)
    - Surcharge de copies (Overhead Ratio)
    - Nombre moyen de sauts (Hop Count)
    
    Le test permet d'exécuter un seul mode ou tous les modes de panne pour comparaison.
    """
    print("=== Test du protocole Spray-and-Wait en scénario multi-sauts ===")
    print("Évaluation des performances avec métriques DTN avancées")
    
    # Paramètres de simulation
    num_nodes = 20
    max_steps = 50  # Plus d'étapes pour observer l'évolution complète même après livraison
    L_values = [2, 4, 8]  # Nombre initial de copies réduit pour mieux voir l'effet des pannes
    ttl_value = 20  # Time-to-Live pour les copies (en pas de temps)
    distribution_rate = 0.2  # Taux de distribution des copies très ralenti pour mieux voir l'effet des pannes
    
    # Paramètres de simulation de pannes
    enable_failures = True
    
    # Option pour exécuter toutes les simulations ou choisir un mode spécifique
    print("Options de simulation des pannes:")
    print("1. Un seul mode de panne")
    print("2. Tous les modes de panne (pour comparaison)")
    choice = input("Entrez votre choix (1/2) [1]: ").strip() or "1"
    
    valid_modes = ["none", "random", "targeted", "region"]
    failure_modes_to_run = []
    
    if choice == "1":
        # Demander le mode de panne avec gestion d'erreur
        while True:
            failure_mode = input(f"Choisissez un mode de panne ({'/'.join(valid_modes)}) [none]: ").strip().lower() or "none"
            if failure_mode in valid_modes:
                failure_modes_to_run = [failure_mode]
                break
            else:
                print(f"Mode invalide. Veuillez choisir parmi: {', '.join(valid_modes)}")
    else:
        # Exécuter tous les modes de panne pour comparaison
        failure_modes_to_run = valid_modes
        print(f"Exécution de tous les modes de panne: {', '.join(valid_modes)}")
    
    # Détermine quand les pannes commencent (très tôt dans la simulation, à t=2)
    # Cela permet aux pannes d'avoir un réel impact sur la livraison des messages
    failure_time = 2
    
    # Pour stocker les résultats de tous les modes de simulation
    all_results = []
    
    # Dossier de sortie principal
    main_output_dir = f"{OUTDIR}/protocols/spray_and_wait_multihop_test"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Pour chaque mode de panne sélectionné
    for failure_mode in failure_modes_to_run:
        print(f"\n{'='*70}")
        print(f"=== EXÉCUTION DU MODE DE PANNE: {failure_mode.upper()}")
        print(f"{'='*70}\n")
        
        # Si le mode est "none", désactiver les pannes
        enable_failures = failure_mode != "none"
        
        # Dossier de sortie spécifique au mode
        output_dir = f"{main_output_dir}"
        if failure_mode != "none":
            output_dir += f"_{failure_mode}_failure"
        os.makedirs(output_dir, exist_ok=True)
        
        # Stocker les résultats pour chaque valeur de L
        results = []
        
        # Initialiser les positions des nœuds pour le gestionnaire de pannes
        positions = {}
        adjacency_initial = create_multihop_network(0, num_nodes)
        G = nx.Graph()
        for node, neighbors in adjacency_initial.items():
            G.add_node(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        pos = nx.spring_layout(G, seed=42)
        positions = {node: (x, y) for node, (x, y) in pos.items()}
        
        # Initialiser le gestionnaire de pannes
        failure_manager = NodeFailureManager(positions)
        
        # Configurer les pannes selon le mode choisi
        if enable_failures:
            # Stocker les nœuds qui vont tomber en panne pour une utilisation dans la boucle principale
            nodes_to_fail = []
            
            if failure_mode == "random":
                # Pannes aléatoires: 20% des nœuds tombent en panne à t=failure_time
                # Exclure source et destination pour ne pas bloquer complètement la propagation
                failure_nodes = random.sample(list(range(1, num_nodes-1)), num_nodes // 5)
                print(f"Nœuds qui tomberont en panne à t={failure_time}: {failure_nodes}")
                nodes_to_fail = failure_nodes
                for node in failure_nodes:
                    failure_manager.failure_times[node] = failure_time
            
            elif failure_mode == "targeted":
                # Pannes ciblées: trouver les nœuds avec le plus haut degré (centraux)
                degrees = {node: len(neighbors) for node, neighbors in adjacency_initial.items()}
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                
                # Cibler les 3 nœuds les plus connectés, excluant source et destination
                targeted_nodes = [node for node, degree in sorted_nodes if node != 0 and node != num_nodes-1][:3]
                print(f"Nœuds centraux qui tomberont en panne à t={failure_time}: {targeted_nodes}")
                nodes_to_fail = targeted_nodes
                for node in targeted_nodes:
                    failure_manager.failure_times[node] = failure_time
            
            elif failure_mode == "region":
                # Pannes régionales: choisir un nœud et ses voisins
                # Trouver un nœud central (pas la source ni la destination)
                potential_central_nodes = [n for n in range(1, num_nodes-1) if n != 0 and n != num_nodes-1]
                if potential_central_nodes:
                    central_node = random.choice(potential_central_nodes)
                    region_nodes = [central_node] + list(adjacency_initial[central_node])
                    # Filtrer pour ne pas inclure la source ou la destination
                    region_nodes = [n for n in region_nodes if n != 0 and n != num_nodes-1]
                    print(f"Région de pannes à t={failure_time}: {region_nodes}")
                    nodes_to_fail = region_nodes
                    for node in region_nodes:
                        failure_manager.failure_times[node] = failure_time
    
    # Tester différentes valeurs de L
    for L in L_values:
        # Initialiser protocole Spray-and-Wait avec TTL et taux de distribution
        source = 0
        destination = num_nodes - 1  # Le dernier nœud est la destination
        protocol = SprayAndWait(num_nodes, L, destination, source, binary=True, 
                               ttl=ttl_value, distribution_rate=distribution_rate)
        
        # Réinitialiser le gestionnaire de pannes pour chaque test avec une valeur L différente
        failure_manager.failed_nodes = set()
        
        # Si c'est la première itération, afficher les informations sur le mode de panne
        if L == L_values[0] and enable_failures:
            print(f"\n--- Mode de panne: {failure_mode.upper()} ---")
            print(f"Les pannes se produiront à t={failure_time}")
            if hasattr(failure_manager, 'failure_times'):
                affected_nodes = sorted(failure_manager.failure_times.keys())
                print(f"Nœuds qui seront affectés: {affected_nodes}")
            print("---")
        
        # Suivi de la distribution des copies
        copies_history = []
        hops_logs = []
        active_copies_over_time = []  # Pour suivre l'évolution du nombre de copies actives
        failed_nodes_over_time = []   # Pour suivre l'évolution des pannes
        
        # Exécuter la simulation
        print(f"\nTest multi-sauts avec L={L}, TTL={ttl_value}, Rate={distribution_rate:.2f}, Mode de panne: {failure_mode}")
        print(f"Source: {source}, Destination: {destination}")
        print(f"{'t':>3} | {'Copies actives':^15} | {'Nœuds avec copies':^20} | {'Nœuds en panne':^15} | {'Livré':<7}")
        print("-" * 90)
        
        # Conserver l'état de livraison pour savoir quand la livraison a eu lieu
        delivery_occurred = False
        delivery_time = float('inf')
        
        for t in range(max_steps):
            # Générer le réseau pour l'instant t
            adjacency = create_multihop_network(t, num_nodes)
            
            # Appliquer les pannes si nécessaire
            if enable_failures and failure_mode != "none":
                # Mettre à jour les nœuds en panne au temps défini
                if t == failure_time:
                    print(f"Activation des pannes à t={t}")
                    for node, fail_time in failure_manager.failure_times.items():
                        if t >= fail_time:
                            # Vérifier que ni la source ni la destination ne tombent en panne
                            if node != source and node != destination:
                                failure_manager.failed_nodes.add(node)
                
                # Appliquer l'effet des pannes sur le réseau à chaque pas de temps
                if failure_manager.failed_nodes:
                    # Créer une copie du dictionnaire d'adjacence
                    adjacency_with_failures = {}
                    for node, neighbors in adjacency.items():
                        adjacency_with_failures[node] = neighbors.copy()
                    
                    # Supprimer les connexions des nœuds en panne
                    for node in failure_manager.failed_nodes:
                        if node in adjacency_with_failures:
                            # Le nœud ne peut plus communiquer
                            adjacency_with_failures[node] = set()
                            # Supprimer le nœud des listes d'adjacence des autres nœuds
                            for other, neighbors in adjacency_with_failures.items():
                                if node in neighbors:
                                    neighbors.remove(node)
                                    
                    # Remplacer le dictionnaire d'adjacence original par celui modifié avec les pannes
                    adjacency = adjacency_with_failures
                    
                    # Si c'est la première fois que les pannes sont activées pour cette valeur de L
                    if t == failure_time:
                        num_copies_lost = 0
                        # Vérifier quels nœuds en panne avaient des copies du message
                        for node in failure_manager.failed_nodes:
                            if protocol.copies.get(node, 0) > 0:
                                num_copies_lost += protocol.copies[node]
                                # Mettre à jour le nombre de copies (les copies dans les nœuds en panne sont perdues)
                                protocol.copies[node] = 0
                        
                        if num_copies_lost > 0:
                            print(f"Attention: {num_copies_lost} copies ont été perdues à cause des pannes!")
            
            # Avant de faire un pas, enregistrer l'état actuel
            copies_history.append(protocol.copies.copy())
            failed_nodes_over_time.append(len(failure_manager.failed_nodes))
            
            # Compter les copies actives avant l'étape
            active_copies = sum(1 for _, c in protocol.copies.items() if c > 0)
            active_copies_over_time.append(active_copies)
            
            # Visualiser le réseau à certains instants-clés ou quand le message est livré
            should_visualize = t % 4 == 0 or t == max_steps - 1
            if should_visualize or (protocol.dest in protocol.delivered_at and not delivery_occurred):
                visualize_network(adjacency, t, protocol.copies, protocol.delivered_at, output_dir, 
                                 failed_nodes=failure_manager.failed_nodes)
            
            # Exécuter un pas de simulation
            protocol.step(t, adjacency)
            
            # Vérifier si la livraison vient de se produire
            if protocol.dest in protocol.delivered_at and not delivery_occurred:
                delivery_occurred = True
                delivery_time = protocol.delivered_at[protocol.dest]
            
            # Compter les nœuds actifs avec des copies
            nodes_with_copies = [n for n, c in protocol.copies.items() if c > 0]
            total_active_copies = sum(protocol.copies.values())
            
            # Utiliser le compteur total de copies créées
            total_copies_created = protocol.total_copies_created
            
            # Afficher l'état actuel
            delivered = protocol.dest in protocol.delivered_at
            delivered_str = f"Oui (t={protocol.delivered_at.get(protocol.dest, 'N/A')})" if delivered else "Non"
            failed_str = ', '.join(map(str, sorted(failure_manager.failed_nodes))) if failure_manager.failed_nodes else "-"
            print(f"{t:3d} | {total_active_copies:^15d} | {', '.join(map(str, nodes_with_copies)):^20} | {failed_str:^15} | {delivered_str:<7}")
            
            # Enregistrer les données de sauts pour analyse
            if protocol.num_hops:
                for node, hops in protocol.num_hops.items():
                    if node != source:  # Ignorer la source (0 sauts)
                        hops_logs.append({
                            't': t,
                            'node': node,
                            'hops': hops,
                            'is_destination': node == destination
                        })
            
            # Arrêter si toutes les copies ont expiré ou si les copies sont épuisées
            if total_active_copies == 0 and t > delivery_time + 5:
                print(f"Toutes les copies ont expiré ou ont été consommées à t={t}")
                break
        
        # Visualisation de la progression des copies sur une carte de chaleur
        if copies_history:
            df_copies = pd.DataFrame(copies_history)
            # Remplir les valeurs manquantes par 0
            df_copies = df_copies.fillna(0)
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Propagation des copies dans le temps (L={L}, TTL={ttl_value}, Rate={distribution_rate:.2f})")
            
            # Créer une carte de chaleur
            # Convertir les données en tableau numpy
            heat_data = df_copies.values
            
            # Définir un gradient de couleurs personnalisé
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'lightblue', 'blue', 'darkblue'])
            
            # Afficher la carte de chaleur
            plt.imshow(heat_data.T, aspect='auto', cmap=cmap, interpolation='nearest')
            
            # Ajouter des étiquettes
            plt.xlabel("Temps (t)")
            plt.ylabel("Identifiant du nœud")
            plt.colorbar(label="Nombre de copies")
            
            # Ajuster les étiquettes des axes
            plt.yticks(range(num_nodes), [str(i) for i in range(num_nodes)])
            plt.xticks(range(0, len(df_copies), 5), [str(i) for i in range(0, len(df_copies), 5)])
            
            # Marquer le moment de la livraison et les pannes
            lines = []
            if delivery_occurred:
                lines.append((delivery_time, 'red', '--', f"Livraison à t={delivery_time}"))
            
            if enable_failures and failure_mode != "none" and failure_time < len(df_copies):
                lines.append((failure_time, 'orange', ':', f"Début des pannes à t={failure_time}"))
            
            for time, color, style, label in lines:
                plt.axvline(x=time, color=color, linestyle=style, label=label)
            
            if lines:
                plt.legend()
            
            # Sauvegarder la figure
            plt.savefig(f"{output_dir}/heatmap_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Visualiser l'évolution du nombre de copies actives au fil du temps et des pannes
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Tracer l'évolution des copies actives
            ax1.plot(range(len(active_copies_over_time)), active_copies_over_time, 'b-', linewidth=2, label="Nœuds avec copies")
            ax1.set_xlabel("Temps (t)")
            ax1.set_ylabel("Nombre de nœuds avec copies", color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Si des pannes sont activées, ajouter un deuxième axe pour les pannes
            if enable_failures and failure_mode != "none":
                ax2 = ax1.twinx()
                ax2.plot(range(len(failed_nodes_over_time)), failed_nodes_over_time, 'r--', linewidth=2, label="Nœuds en panne")
                ax2.set_ylabel("Nombre de nœuds en panne", color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.set_ylim(bottom=0)  # S'assurer que l'axe commence à 0
                
                # Marquer le début des pannes
                if failure_time < len(active_copies_over_time):
                    ax1.axvline(x=failure_time, color='orange', linestyle=':', label=f"Début des pannes à t={failure_time}")
            
            # Marquer le moment de la livraison
            if delivery_occurred and delivery_time < len(active_copies_over_time):
                ax1.axvline(x=delivery_time, color='green', linestyle='--', label=f"Livraison à t={delivery_time}")
            
            # Combiner les légendes des deux axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            if enable_failures and failure_mode != "none":
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            else:
                ax1.legend(loc="best")
            
            plt.title(f"Impact des pannes sur la distribution des copies (L={L}, TTL={ttl_value})")
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f"{output_dir}/active_copies_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Calculer les métriques finales
        delivery_ratio = protocol.delivery_ratio()
        delivery_delay = protocol.delivery_delay()
        hop_stats = protocol.get_hop_stats() if hasattr(protocol, 'get_hop_stats') else {'max': 0, 'destination': None}
        
        # Calculer les statistiques sur les sauts
        if hops_logs:
            df_hops = pd.DataFrame(hops_logs)
            max_hops = df_hops['hops'].max() if not df_hops.empty else 0
            destination_hops = df_hops[df_hops['is_destination']]['hops'].max() if not df_hops[df_hops['is_destination']].empty else None
            
            # Visualiser l'évolution du nombre de sauts
            plt.figure(figsize=(10, 6))
            
            # Tracer le nombre de sauts pour chaque nœud en fonction du temps
            for node in sorted(df_hops['node'].unique()):
                node_data = df_hops[df_hops['node'] == node]
                if not node_data.empty:
                    is_dest = node == destination
                    plt.plot(node_data['t'], node_data['hops'], 
                             marker='o' if is_dest else '.', 
                             linestyle='-' if is_dest else '--',
                             linewidth=2 if is_dest else 1,
                             label=f"Nœud {node}" + (" (Destination)" if is_dest else ""))
            
            plt.title(f"Évolution du nombre de sauts par nœud (L={L}, TTL={ttl_value})")
            plt.xlabel("Temps (t)")
            plt.ylabel("Nombre de sauts")
            plt.grid(True, alpha=0.3)
            
            # Marquer le moment de la livraison et des pannes
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='green', linestyle='--', label=f"Livraison à t={delivery_time}")
            
            if enable_failures and failure_mode != "none":
                plt.axvline(x=failure_time, color='orange', linestyle=':', label=f"Début des pannes à t={failure_time}")
            
            # N'afficher dans la légende que la destination et quelques autres nœuds représentatifs
            handles, labels = plt.gca().get_legend_handles_labels()
            # Filtrer pour ne montrer que les 5 premiers nœuds + la destination si présente
            dest_indices = [i for i, label in enumerate(labels) if "Destination" in label]
            delivery_idx = [i for i, label in enumerate(labels) if "Livraison" in label]
            failure_idx = [i for i, label in enumerate(labels) if "pannes" in label]
            
            selected_idx = []
            if dest_indices:
                selected_idx += dest_indices
            if delivery_idx:
                selected_idx += delivery_idx
            if failure_idx:
                selected_idx += failure_idx
            
            # Ajouter quelques nœuds représentatifs
            other_nodes = [i for i, label in enumerate(labels) 
                         if "Nœud" in label and i not in dest_indices][:3]  # Max 3 autres nœuds
            selected_idx += other_nodes
            
            # S'assurer qu'il n'y a pas de doublons dans les indices
            selected_idx = sorted(set(selected_idx))
            
            if selected_idx:
                plt.legend([handles[i] for i in selected_idx], [labels[i] for i in selected_idx], loc="best")
            else:
                plt.legend(loc="best")
            
            plt.savefig(f"{output_dir}/hops_evolution_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            max_hops = hop_stats['max']
            destination_hops = hop_stats.get('destination', None)
        
        # Enregistrer les résultats
        # Utiliser le nouveau compteur total_copies_created au lieu de la somme des copies actuelles
        total_copies = protocol.total_copies_created
        results.append({
            'L': L,
            'TTL': ttl_value,
            'Distribution_Rate': distribution_rate,
            'Failure_Mode': failure_mode,
            'delivered': destination in protocol.delivered_at,
            'delivery_time': protocol.delivered_at.get(destination, float('inf')),
            'total_copies': total_copies,
            'max_hops': max_hops,
            'destination_hops': destination_hops if destination_hops is not None else 'N/A',
            'overhead': total_copies / (1 if delivery_ratio == 0 else delivery_ratio),
            'num_failed_nodes': len(failure_manager.failed_nodes)
        })
        
        print(f"\nRésultats pour L={L}, TTL={ttl_value}, Rate={distribution_rate:.2f}, Mode de panne: {failure_mode}:")
        if delivery_ratio > 0:
            print(f"  - Taux de livraison (DR): {delivery_ratio:.2f}")
            print(f"  - Délai de livraison: {delivery_delay}")
            print(f"  - Surcharge (OH): {total_copies:.0f} copies / {delivery_ratio:.0f} message = {total_copies/delivery_ratio:.2f}")
            print(f"  - Nombre de sauts pour la destination: {destination_hops if destination_hops is not None else 'N/A'}")
            print(f"  - Détails: Message livré en {delivery_delay} unités de temps avec {total_copies} copies créées")
        else:
            print(f"  - Taux de livraison (DR): 0.00 (échec)")
            print(f"  - Délai de livraison: ∞ (message non livré)")
            print(f"  - Surcharge (OH): ∞ (impossible à calculer)")
            print(f"  - Nombre total de copies créées: {total_copies}")
            print(f"  - Nombre maximum de sauts observés: {max_hops}")
    
        # Créer un tableau comparatif pour ce mode
        if results:
            # Ajouter les résultats au tableau global pour comparaison ultérieure
            for result in results:
                result_copy = result.copy()
                result_copy['failure_mode'] = failure_mode
                all_results.append(result_copy)
            
            df_results = pd.DataFrame(results)
            print("\nTableau comparatif pour le mode de panne", failure_mode.upper())
            print(df_results.to_string(index=False))
            
            # Sauvegarder les résultats en CSV
            csv_filename = f"resultats_multihop_{failure_mode}.csv"
            df_results.to_csv(f"{output_dir}/{csv_filename}", index=False)
            print(f"Résultats sauvegardés dans {output_dir}/{csv_filename}")
            
            # Calculer des métriques avancées
            df_results['delivery_ratio'] = df_results['delivered'].apply(lambda x: 1.0 if x else 0.0)  # Taux de livraison
            df_results['delivery_delay'] = df_results['delivery_time'].apply(lambda x: x if x != float('inf') else 0)  # Délai moyen
            df_results['overhead_ratio'] = df_results['total_copies'] / df_results.apply(
                lambda row: 1 if row['delivered'] else float('inf'), axis=1)  # OH = copies/messages_livrés
            
            # Convertir hop count en numérique
            df_results['hop_count'] = pd.to_numeric(df_results['destination_hops'], errors='coerce')
            
            # Créer des graphiques comparatifs avec des courbes au lieu des barres
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Taux de livraison vs L
            axes[0, 0].plot(df_results['L'], df_results['delivery_ratio'], 'o-', linewidth=2, color='green', markersize=8)
            axes[0, 0].set_title('Taux de livraison (Delivery Ratio)')
            axes[0, 0].set_xlabel('L (nombre initial de copies)')
            axes[0, 0].set_ylabel('DR = messages livrés/émis')
            axes[0, 0].set_ylim([0, 1.1])  # Entre 0 et 1 avec un peu de marge
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Délai de livraison vs L
            axes[0, 1].plot(df_results['L'], df_results['delivery_delay'], 'o-', linewidth=2, color='blue', markersize=8)
            axes[0, 1].set_title('Délai moyen de livraison')
            axes[0, 1].set_xlabel('L (nombre initial de copies)')
            axes[0, 1].set_ylabel('Délai (unités de temps)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Surcharge de copies (Overhead) vs L
            axes[1, 0].plot(df_results['L'], df_results['overhead_ratio'], 'o-', linewidth=2, color='red', markersize=8)
            axes[1, 0].set_title('Surcharge de copies (Overhead Ratio)')
            axes[1, 0].set_xlabel('L (nombre initial de copies)')
            axes[1, 0].set_ylabel('OH = copies créées/messages livrés')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Nombre moyen de sauts vs L
            axes[1, 1].plot(df_results['L'], df_results['hop_count'], 'o-', linewidth=2, color='purple', markersize=8)
            axes[1, 1].set_title('Nombre moyen de sauts (Hop Count)')
            axes[1, 1].set_xlabel('L (nombre initial de copies)')
            axes[1, 1].set_ylabel('Nombre de sauts')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Ajouter les valeurs numériques sur chaque point des courbes
            for ax in axes.flat:
                for line in ax.get_lines():
                    x_data, y_data = line.get_data()
                    for x, y in zip(x_data, y_data):
                        if not np.isnan(y) and not np.isinf(y):  # S'assurer que la valeur n'est pas NaN ou inf
                            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                                       xytext=(0,5), ha='center', fontweight='bold')
            
            plt.tight_layout()
            fig_filename = f"comparaison_multihop_{failure_mode}.png"
            plt.savefig(f"{output_dir}/{fig_filename}", dpi=300)
            print(f"Graphique comparatif sauvegardé dans {output_dir}/{fig_filename}")
    
    # À la fin de toutes les simulations, créer un graphique comparatif global
    if len(all_results) > 0 and len(failure_modes_to_run) > 1:
        print("\n\n" + "="*80)
        print("COMPARAISON DE TOUS LES MODES DE PANNE")
        print("="*80)
        
        # Convertir tous les résultats en DataFrame
        df_all = pd.DataFrame(all_results)
        
        # Calculer des métriques avancées
        df_all['delivery_ratio'] = df_all['delivered'].apply(lambda x: 1.0 if x else 0.0)
        df_all['delivery_delay'] = df_all['delivery_time'].apply(lambda x: x if x != float('inf') else 0)
        df_all['overhead_ratio'] = df_all['total_copies'] / df_all.apply(
            lambda row: 1 if row['delivered'] else float('inf'), axis=1)
        df_all['hop_count'] = pd.to_numeric(df_all['destination_hops'], errors='coerce')
        
        # Sauvegarder les résultats globaux en CSV
        df_all.to_csv(f"{main_output_dir}/resultats_multihop_tous_modes.csv", index=False)
        
        # Créer un graphique comparatif global
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Palette de couleurs pour différencier les modes de panne
        colors = {'none': 'green', 'random': 'red', 'targeted': 'blue', 'region': 'purple'}
        markers = {'none': 'o', 'random': 's', 'targeted': '^', 'region': 'D'}
        
        # Regrouper par mode de panne et L
        for failure_mode in failure_modes_to_run:
            df_mode = df_all[df_all['failure_mode'] == failure_mode]
            
            # Trier par L pour avoir des courbes cohérentes
            df_mode = df_mode.sort_values('L')
            
            # 1. Taux de livraison
            axes[0, 0].plot(df_mode['L'], df_mode['delivery_ratio'], 
                          marker=markers[failure_mode], linestyle='-', linewidth=2, 
                          color=colors[failure_mode], markersize=8, label=failure_mode.capitalize())
            
            # 2. Délai de livraison
            axes[0, 1].plot(df_mode['L'], df_mode['delivery_delay'], 
                          marker=markers[failure_mode], linestyle='-', linewidth=2, 
                          color=colors[failure_mode], markersize=8, label=failure_mode.capitalize())
            
            # 3. Surcharge (Overhead)
            axes[1, 0].plot(df_mode['L'], df_mode['overhead_ratio'], 
                          marker=markers[failure_mode], linestyle='-', linewidth=2, 
                          color=colors[failure_mode], markersize=8, label=failure_mode.capitalize())
            
            # 4. Nombre de sauts
            axes[1, 1].plot(df_mode['L'], df_mode['hop_count'], 
                          marker=markers[failure_mode], linestyle='-', linewidth=2, 
                          color=colors[failure_mode], markersize=8, label=failure_mode.capitalize())
        
        # Titres et labels
        axes[0, 0].set_title('Taux de livraison (Delivery Ratio)', fontsize=14)
        axes[0, 0].set_xlabel('L (nombre initial de copies)', fontsize=12)
        axes[0, 0].set_ylabel('DR = messages livrés/émis', fontsize=12)
        axes[0, 0].set_ylim([0, 1.1])
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=12)
        
        axes[0, 1].set_title('Délai moyen de livraison', fontsize=14)
        axes[0, 1].set_xlabel('L (nombre initial de copies)', fontsize=12)
        axes[0, 1].set_ylabel('Délai (unités de temps)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=12)
        
        axes[1, 0].set_title('Surcharge de copies (Overhead Ratio)', fontsize=14)
        axes[1, 0].set_xlabel('L (nombre initial de copies)', fontsize=12)
        axes[1, 0].set_ylabel('OH = copies créées/messages livrés', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=12)
        
        axes[1, 1].set_title('Nombre moyen de sauts (Hop Count)', fontsize=14)
        axes[1, 1].set_xlabel('L (nombre initial de copies)', fontsize=12)
        axes[1, 1].set_ylabel('Nombre de sauts', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{main_output_dir}/comparaison_tous_modes.png", dpi=300)
        print(f"Graphique comparatif global sauvegardé dans {main_output_dir}/comparaison_tous_modes.png")

if __name__ == "__main__":
    test_spray_and_wait_multihop()
    print("\nTest multi-sauts terminé avec succès!")
