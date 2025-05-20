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
from matplotlib.colors import LinearSegmentedColormap

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTDIR
from protocols.spray_and_wait import SprayAndWait

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
    import random
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

def visualize_network(adjacency: dict[int, set[int]], t: int, copies: dict[int, int], delivered_at: dict, output_dir: str):
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
    """
    G = nx.Graph()
    
    # Ajouter les nœuds et arêtes
    for node, neighbors in adjacency.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
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
    
    source = 0
    destination = max(adjacency.keys())
    
    # Calculer les clusters pour la visualisation
    # Pour déterminer les clusters, on regroupe les nœuds selon leur position dans le graphe
    num_clusters = max(4, len(G.nodes()) // 5)  # Estimation du nombre de clusters
    
    # Collecter les info sur les nœuds pour une meilleure visualisation
    for node in G.nodes():
        num_copies = copies.get(node, 0)
        
        if node == source:
            # Source: vert, avec nombre de copies
            node_colors.append('green')
            labels[node] = f"S:{num_copies}"
        elif node == destination:
            # Destination: doré si livré, sinon rouge
            if node in delivered_at:
                node_colors.append('gold')
                labels[node] = f"D✓:{num_copies} (t={delivered_at[node]})"
            else:
                node_colors.append('red')
                labels[node] = f"D:{num_copies}"
        elif num_copies > 0:
            # Nœuds avec copies: gradient de bleu selon le nombre de copies
            # Plus le nœud a de copies, plus il est foncé
            intensity = min(1.0, 0.3 + 0.7 * (num_copies / max_copies))
            node_colors.append((0, 0, intensity))  # RGB pour bleu
            labels[node] = f"{node}:{num_copies}"
        else:
            # Nœuds sans copies
            node_colors.append('lightgray')
            labels[node] = str(node)
    
    # Déterminer les couleurs et épaisseurs des arêtes pour mieux visualiser les chemins
    # Les arêtes entre clusters sont plus épaisses
    for u, v in G.edges():
        # Vérifier si l'un des nœuds a des copies et l'autre non
        # Cela pourrait indiquer un chemin de diffusion potentiel
        if (copies.get(u, 0) > 0 and copies.get(v, 0) == 0) or (copies.get(u, 0) == 0 and copies.get(v, 0) > 0):
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
    
    # Dessiner les nœuds avec un contour noir pour une meilleure visibilité
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                         alpha=0.8, edgecolors='black', linewidths=1)
    
    # Ajuster les labels pour une meilleure visibilité selon la couleur du nœud
    for node, color in zip(G.nodes(), node_colors):
        # Déterminer si le fond du nœud est foncé
        is_dark = isinstance(color, tuple) or color in ['green', 'blue', 'darkblue']
        
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
    Test du protocole Spray-and-Wait dans un scénario multi-sauts.
    Ce test simule un environnement où le message doit traverser plusieurs
    nœuds intermédiaires avant d'atteindre la destination.
    """
    print("=== Test du protocole Spray-and-Wait en scénario multi-sauts ===")
    
    # Paramètres de simulation
    num_nodes = 20
    max_steps = 50  # Plus d'étapes pour observer l'évolution complète même après livraison
    L_values = [4, 8, 16]  # Nombre initial de copies
    ttl_value = 20  # Time-to-Live pour les copies (en pas de temps)
    distribution_rate = 0.4  # Taux de distribution des copies (ralentir la diffusion significativement)
    
    # Dossier de sortie
    output_dir = f"{OUTDIR}/protocols/spray_and_wait_multihop_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Stocker les résultats pour chaque valeur de L
    results = []
    
    # Tester différentes valeurs de L
    for L in L_values:
        # Initialiser protocole Spray-and-Wait avec TTL et taux de distribution
        source = 0
        destination = num_nodes - 1  # Le dernier nœud est la destination
        protocol = SprayAndWait(num_nodes, L, destination, source, binary=True, 
                               ttl=ttl_value, distribution_rate=distribution_rate)
        
        # Suivi de la distribution des copies
        copies_history = []
        hops_logs = []
        active_copies_over_time = []  # Pour suivre l'évolution du nombre de copies actives
        
        # Exécuter la simulation
        print(f"\nTest multi-sauts avec L={L}, TTL={ttl_value}, Rate={distribution_rate:.2f}:")
        print(f"Source: {source}, Destination: {destination}")
        print(f"{'t':>3} | {'Copies actives':^15} | {'Nœuds avec copies':^20} | {'Livré':<7}")
        print("-" * 70)
        
        # Conserver l'état de livraison pour savoir quand la livraison a eu lieu
        delivery_occurred = False
        delivery_time = float('inf')
        
        for t in range(max_steps):
            # Générer le réseau pour l'instant t
            adjacency = create_multihop_network(t, num_nodes)
            
            # Avant de faire un pas, enregistrer l'état actuel
            copies_history.append(protocol.copies.copy())
            
            # Compter les copies actives avant l'étape
            active_copies = sum(1 for _, c in protocol.copies.items() if c > 0)
            active_copies_over_time.append(active_copies)
            
            # Visualiser le réseau à certains instants-clés ou quand le message est livré
            should_visualize = t % 4 == 0 or t == max_steps - 1
            if should_visualize or (protocol.dest in protocol.delivered_at and not delivery_occurred):
                visualize_network(adjacency, t, protocol.copies, protocol.delivered_at, output_dir)
            
            # Exécuter un pas de simulation
            protocol.step(t, adjacency)
            
            # Vérifier si la livraison vient de se produire
            if protocol.dest in protocol.delivered_at and not delivery_occurred:
                delivery_occurred = True
                delivery_time = protocol.delivered_at[protocol.dest]
            
            # Compter les nœuds actifs avec des copies
            nodes_with_copies = [n for n, c in protocol.copies.items() if c > 0]
            total_active_copies = sum(protocol.copies.values())
            
            # Afficher l'état actuel
            delivered = protocol.dest in protocol.delivered_at
            delivered_str = f"Oui (t={protocol.delivered_at.get(protocol.dest, 'N/A')})" if delivered else "Non"
            print(f"{t:3d} | {total_active_copies:^15d} | {', '.join(map(str, nodes_with_copies)):^20} | {delivered_str:<7}")
            
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
            
            # Marquer le moment de la livraison
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='red', linestyle='--', label=f"Livraison à t={delivery_time}")
                plt.legend()
            
            # Sauvegarder la figure
            plt.savefig(f"{output_dir}/heatmap_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Visualiser l'évolution du nombre de copies actives au fil du temps
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(active_copies_over_time)), active_copies_over_time, 'b-', linewidth=2)
            plt.title(f"Évolution du nombre de nœuds avec des copies (L={L}, TTL={ttl_value})")
            plt.xlabel("Temps (t)")
            plt.ylabel("Nombre de nœuds avec copies")
            plt.grid(True, alpha=0.3)
            
            # Marquer le moment de la livraison
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='red', linestyle='--', label=f"Livraison à t={delivery_time}")
                plt.legend()
            
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
            
            # Marquer le moment de la livraison
            if delivery_occurred:
                plt.axvline(x=delivery_time, color='red', linestyle='--', label=f"Livraison à t={delivery_time}")
            
            # N'afficher dans la légende que la destination et quelques autres nœuds représentatifs
            handles, labels = plt.gca().get_legend_handles_labels()
            # Filtrer pour ne montrer que les 5 premiers nœuds + la destination si présente
            dest_indices = [i for i, label in enumerate(labels) if "Destination" in label]
            delivery_idx = [i for i, label in enumerate(labels) if "Livraison" in label]
            
            if dest_indices and len(handles) > 6:
                # Si la destination est présente dans les labels
                dest_idx = dest_indices[0] 
                selected_idx = list(range(min(5, len(handles)))) + ([dest_idx] if dest_idx >= 5 else [])
                # Ajouter la ligne de livraison si présente
                if delivery_idx:
                    selected_idx += delivery_idx
                # S'assurer qu'il n'y a pas de doublons dans les indices
                selected_idx = sorted(set(selected_idx))
                plt.legend([handles[i] for i in selected_idx], [labels[i] for i in selected_idx])
            else:
                # Sinon juste afficher les 5 premiers ou tous s'il y en a moins de 5
                plt.legend(loc='best')
            
            plt.savefig(f"{output_dir}/hops_evolution_L{L}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            max_hops = hop_stats['max']
            destination_hops = hop_stats.get('destination', None)
        
        # Enregistrer les résultats
        total_copies = sum(protocol.copies.values())
        results.append({
            'L': L,
            'TTL': ttl_value,
            'Distribution_Rate': distribution_rate,
            'delivered': destination in protocol.delivered_at,
            'delivery_time': protocol.delivered_at.get(destination, float('inf')),
            'total_copies': total_copies,
            'max_hops': max_hops,
            'destination_hops': destination_hops if destination_hops is not None else 'N/A',
            'overhead': total_copies / (1 if delivery_ratio == 0 else delivery_ratio)
        })
        
        print(f"\nRésultats pour L={L}, TTL={ttl_value}, Rate={distribution_rate:.2f}:")
        if delivery_ratio > 0:
            print(f"  - Message livré: Oui")
            print(f"  - Délai de livraison: {delivery_delay}")
            print(f"  - Nombre total de copies créées: {total_copies}")
            print(f"  - Nombre maximum de sauts: {max_hops}")
            print(f"  - Nombre de sauts pour la destination: {destination_hops if destination_hops is not None else 'N/A'}")
        else:
            print(f"  - Message non livré dans le délai imparti")
            print(f"  - Nombre total de copies créées: {total_copies}")
            print(f"  - Nombre maximum de sauts observés: {max_hops}")
    
    # Créer un tableau comparatif
    if results:
        df_results = pd.DataFrame(results)
        print("\nTableau comparatif:")
        print(df_results.to_string(index=False))
        
        # Sauvegarder les résultats en CSV
        df_results.to_csv(f"{output_dir}/resultats_multihop.csv", index=False)
        print(f"Résultats sauvegardés dans {output_dir}/resultats_multihop.csv")
        
        # Créer des graphiques comparatifs
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Délai de livraison vs L
        axes[0, 0].bar(df_results['L'].astype(str), df_results['delivery_time'])
        axes[0, 0].set_title('Délai de livraison')
        axes[0, 0].set_xlabel('L (nombre initial de copies)')
        axes[0, 0].set_ylabel('Temps')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Nombre de copies totales vs L
        axes[0, 1].bar(df_results['L'].astype(str), df_results['total_copies'])
        axes[0, 1].set_title('Total des copies créées')
        axes[0, 1].set_xlabel('L (nombre initial de copies)')
        axes[0, 1].set_ylabel('Nombre de copies')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Nombre de sauts maximum vs L
        axes[1, 0].bar(df_results['L'].astype(str), df_results['max_hops'])
        axes[1, 0].set_title('Nombre maximum de sauts')
        axes[1, 0].set_xlabel('L (nombre initial de copies)')
        axes[1, 0].set_ylabel('Sauts')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Nombre de sauts pour atteindre la destination vs L
        # Convertir la colonne en numérique si possible
        df_results['destination_hops_num'] = pd.to_numeric(df_results['destination_hops'], errors='coerce')
        axes[1, 1].bar(df_results['L'].astype(str), df_results['destination_hops_num'])
        axes[1, 1].set_title('Sauts pour atteindre la destination')
        axes[1, 1].set_xlabel('L (nombre initial de copies)')
        axes[1, 1].set_ylabel('Sauts')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparaison_multihop.png", dpi=300)
        print(f"Graphique comparatif sauvegardé dans {output_dir}/comparaison_multihop.png")

if __name__ == "__main__":
    test_spray_and_wait_multihop()
    print("\nTest multi-sauts terminé avec succès!")
