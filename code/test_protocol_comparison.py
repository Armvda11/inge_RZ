#!/usr/bin/env python3
# test_protocol_comparison.py
"""
Script de comparaison des protocoles Spray-and-Wait et PRoPHET face à des pannes dynamiques.

Ce script permet d'exécuter les deux protocoles sur les mêmes conditions de réseau et de pannes,
pour comparer directement leurs performances. Les métriques suivies sont:
- Taux de livraison
- Délai de livraison
- Overhead (surcharge réseau)
- Résilience aux pannes
- Débit moyen

Modes de panne disponibles:
- continuous: pannes aléatoires à chaque pas de temps
- cascade: effet domino où les voisins d'un nœud en panne ont plus de risque de tomber en panne
- targeted_dynamic: recalcul périodique des nœuds critiques à mettre en panne
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from tabulate import tabulate

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTDIR
from protocols.spray_and_wait import SprayAndWait
from protocols.prophet import Prophet
from simulation.failure import NodeFailureManager
from performance_metrics import PerformanceTracker, generate_comparative_table

class DynamicFailureManager:
    """
    Gestionnaire de pannes dynamiques et continues qui peut être appliqué
    de manière cohérente à plusieurs protocoles en parallèle.
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
        
        # S'assurer qu'il y a toujours des nœuds éligibles pour éviter que les pannes ne s'arrêtent
        # en gardant au moins 40% du réseau opérationnel
        max_failures_allowed = int(0.6 * self.num_nodes)
        
        # Éviter d'appliquer trop de pannes si on approche de la limite
        if len(self.failed_nodes) >= max_failures_allowed:
            # Si on a déjà beaucoup de pannes, réduire la probabilité ou appliquer moins de pannes
            if random.random() < 0.3:  # 30% de chance de faire une panne quand même
                eligible_nodes = eligible_nodes[:1]  # Limiter à un seul nœud
            else:
                return new_adjacency
        
        # Appliquer les pannes selon le mode choisi
        if mode == 'continuous':
            # Mode continu: chaque nœud a une probabilité fixe de tomber en panne à chaque pas
            failure_prob = params.get('failure_prob', 0.05)
            
            for node in eligible_nodes:
                if random.random() < failure_prob:
                    self.failed_nodes.add(node)
                    self.failure_history.append((t, node))
        
        elif mode == 'cascade':
            # Mode cascade: les voisins des nœuds en panne ont plus de risque de tomber en panne
            base_prob = params.get('base_prob', 0.02)
            cascade_factor = params.get('cascade_factor', 3.0)
            
            # D'abord, augmenter le risque pour les voisins des nœuds récemment tombés en panne
            for node in eligible_nodes:
                # Vérifier si le nœud a des voisins en panne
                failed_neighbors = [n for n in adjacency.get(node, set()) if n in self.failed_nodes]
                
                # Augmenter le risque proportionnellement au nombre de voisins en panne
                if failed_neighbors:
                    self.cascade_risk[node] += len(failed_neighbors) * 0.1
            
            # Ensuite, appliquer les pannes basées sur le risque
            for node in eligible_nodes:
                failure_probability = base_prob + self.cascade_risk[node] * cascade_factor
                if random.random() < min(failure_probability, 0.5):  # Limiter à 50% max
                    self.failed_nodes.add(node)
                    self.failure_history.append((t, node))
        
        elif mode == 'targeted_dynamic':
            # Mode ciblé dynamique: recalcule périodiquement les nœuds les plus critiques
            recalculation_interval = params.get('recalculation_interval', 5)
            failure_percentage = params.get('failure_percentage', 0.1)
            
            # Ne recalculer les cibles que tous les X pas de temps
            if t % recalculation_interval == 0:
                # Construire un graphe pour l'analyse
                G = nx.Graph()
                for node, neighbors in adjacency.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
                
                # Identifier les nœuds critiques (par centralité)
                if len(G) > 0:
                    # Calculer la centralité de degré comme mesure d'importance
                    centrality = nx.degree_centrality(G)
                    
                    # Trier les nœuds par centralité décroissante (les plus importants d'abord)
                    sorted_nodes = sorted(
                        [(node, centrality.get(node, 0)) for node in eligible_nodes],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Déterminer combien de nœuds mettre en panne
                    num_failures = max(1, int(failure_percentage * len(eligible_nodes)))
                    
                    # Sélectionner les nœuds les plus centraux pour les mettre en panne
                    for i in range(min(num_failures, len(sorted_nodes))):
                        node = sorted_nodes[i][0]
                        self.failed_nodes.add(node)
                        self.failure_history.append((t, node))
        
        # Supprimer les nœuds en panne du dictionnaire d'adjacence
        new_adjacency = {
            node: neighbors for node, neighbors in new_adjacency.items() 
            if node not in self.failed_nodes
        }
        
        # Supprimer également les nœuds en panne des listes de voisins
        for node, neighbors in new_adjacency.items():
            new_adjacency[node] = {n for n in neighbors if n not in self.failed_nodes}
        
        # Garder une trace du nombre de nœuds actifs après avoir appliqué les pannes
        active_nodes_after = len(new_adjacency)
        
        # Journalisation pour débogage
        if active_nodes_before != active_nodes_after:
            print(f"[t={t}] Pannes appliquées: {active_nodes_before} → {active_nodes_after} nœuds actifs")
        
        return new_adjacency

def create_multihop_network(t: int, num_nodes: int = 30, dilution_factor: float = 1.0) -> dict[int, set[int]]:
    """
    Crée un réseau test qui simule une topologie en "grappes" mobiles avec scénario multi-sauts.
    Version améliorée: Plus grande distance entre source et destination, avec plus de sauts intermédiaires.
    
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
    
    # Plus de clusters pour augmenter la distance entre source et destination
    cluster_size = 4  # Nombre réduit de nœuds par cluster pour avoir plus de clusters
    num_clusters = max(6, num_nodes // cluster_size)  # Au moins 6 clusters pour garantir la distance
    
    # Attribuer des nœuds à des clusters
    clusters = [set() for _ in range(num_clusters)]
    
    # Réserver la source et la destination
    source, destination = 0, num_nodes - 1  # Source au début, destination à la fin
    reserved_nodes = {source, destination}
    regular_nodes = [i for i in range(num_nodes) if i not in reserved_nodes]
    random.shuffle(regular_nodes)
    
    # Calculer la distribution des nœuds par cluster
    nodes_per_cluster = (num_nodes - len(reserved_nodes)) // num_clusters
    remainder = (num_nodes - len(reserved_nodes)) % num_clusters
    
    # Placer la source dans le premier cluster et la destination dans le dernier
    # pour maximiser la distance entre eux
    clusters[0].add(source)  # Source dans le premier cluster
    clusters[-1].add(destination)  # Destination dans le dernier cluster
    
    # Distribuer les autres nœuds
    node_index = 0
    for c in range(num_clusters):
        # Ajouter plus de nœuds aux premiers clusters s'il y a un reste
        extra = 1 if remainder > 0 and c < remainder else 0
        
        # Ajouter les nœuds au cluster actuel
        for _ in range(nodes_per_cluster + extra):
            if node_index < len(regular_nodes):
                clusters[c].add(regular_nodes[node_index])
                node_index += 1
    
    # Établir des connexions à l'intérieur des clusters
    for cluster_idx, cluster in enumerate(clusters):
        for i in cluster:
            for j in cluster:
                if i != j and random.random() < 0.8 * dilution_factor:  # 80% de chance de connexion intra-cluster
                    adjacency[i].add(j)
                    adjacency[j].add(i)  # Connexion bidirectionnelle
    
    # Établir des connexions UNIQUEMENT entre clusters adjacents
    # et réduire la probabilité de connexion entre clusters pour augmenter les sauts
    for c1 in range(num_clusters - 1):
        c2 = c1 + 1  # Cluster adjacents uniquement
        prob_inter_cluster = 0.3 * dilution_factor  # Probabilité de connexion réduite entre clusters
        
        # Établir quelques connexions entre les deux clusters
        for i in clusters[c1]:
            for j in clusters[c2]:
                if random.random() < prob_inter_cluster:
                    adjacency[i].add(j)
                    adjacency[j].add(i)  # Connexion bidirectionnelle
    
    # S'assurer qu'il n'y a pas de connexion directe entre source et destination
    # et pas de connexions entre clusters non-adjacents
    if destination in adjacency[source]:
        adjacency[source].remove(destination)
    
    if source in adjacency[destination]:
        adjacency[destination].remove(source)
    
    # Bloquer les connexions entre clusters non adjacents pour forcer les multi-sauts
    for c1 in range(num_clusters):
        for c2 in range(num_clusters):
            # Si les clusters ne sont pas adjacents (différence > 1) et ne sont pas les mêmes
            if abs(c1 - c2) > 1:
                # Supprimer toute connexion entre ces clusters
                for i in clusters[c1]:
                    for j in clusters[c2]:
                        if j in adjacency[i]:
                            adjacency[i].remove(j)
                        if i in adjacency[j]:
                            adjacency[j].remove(i)
    
    return adjacency

def compare_protocols_with_dynamic_failures():
    """
    Comparaison des protocoles Spray-and-Wait et PRoPHET face à des pannes dynamiques.
    
    Cette fonction exécute plusieurs simulations avec les deux protocoles dans les mêmes
    conditions de réseau et de pannes, afin de comparer directement leurs performances.
    """
    print("=== Comparaison de Spray-and-Wait et PRoPHET avec pannes dynamiques ===")
    
    # Paramètres de simulation
    num_nodes = 40  # Nombre de nœuds dans le réseau
    max_steps = 100  # Nombre maximum de pas de simulation
    
    # Paramètres pour les protocoles
    spray_and_wait_params = {
        'L_values': [4, 8, 16],  # Valeurs de L à tester
        'ttl': 40,  # Time-to-Live
        'distribution_rate': 0.4  # Taux de distribution
    }
    
    prophet_params = {
        'p_init_values': [0.3, 0.5],  # Valeurs de P_init à tester
        'beta': 0.25,  # Facteur de transitivité
        'gamma': 0.98,  # Facteur de vieillissement
        'ttl': 40,  # Time-to-Live
        'distribution_rate': 0.4  # Taux de distribution
    }
    
    # Paramètres de simulation
    network_dilution = 0.7  # Dilution du réseau
    
    # Modes de test avec paramètres ajustés
    test_modes = {
        'continuous': {
            'desc': 'Pannes aléatoires continues tout au long de la simulation',
            'params': {'failure_prob': 0.04}
        },
        'cascade': {
            'desc': 'Pannes en cascade (effet domino)',
            'params': {'base_prob': 0.02, 'cascade_factor': 5.0}
        },
        'targeted_dynamic': {
            'desc': 'Pannes ciblant dynamiquement les nœuds les plus critiques',
            'params': {'recalculation_interval': 4, 'failure_percentage': 0.08}
        }
    }
    
    # Dossier de sortie principal
    main_output_dir = f"{OUTDIR}/protocols/comparison_dynamic_failures"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Demander le mode à exécuter
    print("\nModes de test disponibles:")
    for i, (mode, details) in enumerate(test_modes.items(), 1):
        print(f"{i}. {mode}: {details['desc']}")
    
    try:
        choice = int(input("\nChoisissez un mode de test (1-3): "))
        failure_mode = list(test_modes.keys())[choice - 1]
    except (ValueError, IndexError):
        print("Choix invalide. Mode 'continuous' sélectionné par défaut.")
        failure_mode = 'continuous'
    
    print(f"\nMode sélectionné: {failure_mode} - {test_modes[failure_mode]['desc']}")
    
    # Configurer la sortie spécifique au mode
    output_dir = f"{main_output_dir}/{failure_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Source et destination fixes
    source = 0  # Premier nœud
    destination = num_nodes - 1  # Dernier nœud
    
    # Résultats à collecter
    all_results = []
    
    # Créer un gestionnaire de pannes pour chaque combinaison de protocoles et paramètres
    failure_mgr = DynamicFailureManager(num_nodes, source, destination)
    
    # Liste pour stocker les résultats adjacents à chaque pas de temps
    # (pour les réutiliser entre les protocoles)
    saved_adjacency = {}
    
    # ---------- Exécuter les simulations pour Spray-and-Wait ----------
    for L in spray_and_wait_params['L_values']:
        print(f"\n--- Test de Spray-and-Wait avec L = {L} ---")
        
        # Réinitialiser le gestionnaire de pannes pour chaque test
        failure_mgr = DynamicFailureManager(num_nodes, source, destination)
        
        # Initialiser le protocole Spray-and-Wait
        protocol = SprayAndWait(
            num_nodes=num_nodes,
            L=L,
            dest=destination,
            source=source,
            ttl=spray_and_wait_params['ttl'],
            distribution_rate=spray_and_wait_params['distribution_rate']
        )
        
        # Tracker de performances
        perf_tracker = PerformanceTracker(num_nodes, source, destination)
        
        # Simuler le réseau
        for t in range(max_steps):
            # Créer ou récupérer l'état du réseau à ce pas de temps
            if t not in saved_adjacency:
                # Créer le réseau initial
                adjacency = create_multihop_network(t, num_nodes, network_dilution)
                
                # Appliquer les pannes dynamiques
                adjacency = failure_mgr.apply_failures(t, adjacency, failure_mode, test_modes[failure_mode]['params'])
                
                # Sauvegarder pour les autres protocoles
                saved_adjacency[t] = adjacency
            else:
                adjacency = saved_adjacency[t]
            
            # Exécuter un pas de simulation
            protocol.step(t, adjacency)
            
            # Enregistrer les métriques pour ce pas de temps
            active_nodes = len(adjacency)
            failed_nodes = num_nodes - active_nodes
            perf_tracker.record_step(t, protocol, active_nodes, failed_nodes)
            
            # Si le message est livré et qu'on est déjà 20 pas après la livraison, on peut arrêter
            if protocol.message_delivered and t > protocol.delivered_at[destination] + 20:
                print(f"  Message livré à t={protocol.delivered_at[destination]}, simulation arrêtée à t={t}")
                break
        
        # Calculer les statistiques finales
        perf_tracker.calculate_final_stats(protocol, max_steps)
        
        # Stocker les résultats
        results_dict = {
            'protocol': 'Spray-and-Wait',
            'param': L,  # L pour Spray-and-Wait
            'delivered': protocol.message_delivered,
            'delivery_time': protocol.delivered_at.get(destination, float('inf')),
            'hops': protocol.num_hops.get(destination, None),
            'copies_created': protocol.total_copies_created,
            'overhead_ratio': protocol.overhead_ratio(),
            'avg_throughput': perf_tracker.final_stats.get('avg_throughput', 0),
            'resilience': perf_tracker.final_stats.get('resilience_score', 0),
            'failure_rate': perf_tracker.final_stats.get('avg_failure_rate', 0)
        }
        
        all_results.append(results_dict)
        
        # Afficher un résumé
        print(f"  {'Livré' if protocol.message_delivered else 'Non livré'} - "
              f"Délai: {protocol.delivered_at.get(destination, 'N/A')} - "
              f"Copies: {protocol.total_copies_created}")
    
    # ---------- Exécuter les simulations pour PRoPHET ----------
    for p_init in prophet_params['p_init_values']:
        print(f"\n--- Test de PRoPHET avec P_init = {p_init} ---")
        
        # Réinitialiser le gestionnaire de pannes pour chaque test
        failure_mgr = DynamicFailureManager(num_nodes, source, destination)
        
        # Initialiser le protocole PRoPHET
        protocol = Prophet(
            num_nodes=num_nodes,
            p_init=p_init,
            dest=destination,
            source=source,
            gamma=prophet_params['gamma'],
            beta=prophet_params['beta'],
            ttl=prophet_params['ttl'],
            distribution_rate=prophet_params['distribution_rate']
        )
        
        # Tracker de performances
        perf_tracker = PerformanceTracker(num_nodes, source, destination)
        
        # Simuler le réseau
        for t in range(max_steps):
            # Utiliser le même état de réseau que pour Spray-and-Wait pour une comparaison équitable
            if t in saved_adjacency:
                adjacency = saved_adjacency[t]
            else:
                # Au cas où nous aurions besoin de générer des données supplémentaires
                adjacency = create_multihop_network(t, num_nodes, network_dilution)
                adjacency = failure_mgr.apply_failures(t, adjacency, failure_mode, test_modes[failure_mode]['params'])
                saved_adjacency[t] = adjacency
            
            # Exécuter un pas de simulation
            protocol.step(t, adjacency)
            
            # Enregistrer les métriques pour ce pas de temps
            active_nodes = len(adjacency)
            failed_nodes = num_nodes - active_nodes
            perf_tracker.record_step(t, protocol, active_nodes, failed_nodes)
            
            # Si le message est livré et qu'on est déjà 20 pas après la livraison, on peut arrêter
            if protocol.message_delivered and t > protocol.delivered_at[destination] + 20:
                print(f"  Message livré à t={protocol.delivered_at[destination]}, simulation arrêtée à t={t}")
                break
        
        # Calculer les statistiques finales
        perf_tracker.calculate_final_stats(protocol, max_steps)
        
        # Stocker les résultats
        results_dict = {
            'protocol': 'PRoPHET',
            'param': p_init,  # P_init pour PRoPHET
            'delivered': protocol.message_delivered,
            'delivery_time': protocol.delivered_at.get(destination, float('inf')),
            'hops': protocol.num_hops.get(destination, None),
            'copies_created': protocol.total_copies_created,
            'overhead_ratio': protocol.overhead_ratio(),
            'avg_throughput': perf_tracker.final_stats.get('avg_throughput', 0),
            'resilience': perf_tracker.final_stats.get('resilience_score', 0),
            'failure_rate': perf_tracker.final_stats.get('avg_failure_rate', 0)
        }
        
        all_results.append(results_dict)
        
        # Afficher un résumé
        print(f"  {'Livré' if protocol.message_delivered else 'Non livré'} - "
              f"Délai: {protocol.delivered_at.get(destination, 'N/A')} - "
              f"Copies: {protocol.total_copies_created}")
    
    # ---------- Générer des rapports de comparaison ----------
    if all_results:
        # Convertir en DataFrame pour analyse
        results_df = pd.DataFrame(all_results)
        
        # Sauvegarder les résultats bruts
        results_df.to_csv(f"{output_dir}/resultats_comparison.csv", index=False)
        
        # Créer des visualisations comparatives
        create_comparison_plots(results_df, output_dir, failure_mode)
        
        # Générer un rapport HTML interactif
        create_html_report(results_df, output_dir, failure_mode, test_modes[failure_mode]['desc'])
        
        # Afficher un tableau comparatif dans la console
        print("\n=== Tableau comparatif des résultats ===")
        
        # Table pour Spray-and-Wait
        print("\nRésultats pour Spray-and-Wait:")
        spray_results = results_df[results_df['protocol'] == 'Spray-and-Wait']
        print(tabulate(
            spray_results[['param', 'delivered', 'delivery_time', 'hops', 'copies_created', 'overhead_ratio']],
            headers=['L', 'Livré', 'Délai', 'Sauts', 'Copies', 'Overhead'],
            tablefmt='pretty'
        ))
        
        # Table pour PRoPHET
        print("\nRésultats pour PRoPHET:")
        prophet_results = results_df[results_df['protocol'] == 'PRoPHET']
        print(tabulate(
            prophet_results[['param', 'delivered', 'delivery_time', 'hops', 'copies_created', 'overhead_ratio']],
            headers=['P_init', 'Livré', 'Délai', 'Sauts', 'Copies', 'Overhead'],
            tablefmt='pretty'
        ))
    
    # Demander à l'utilisateur s'il souhaite ouvrir le rapport HTML dans le navigateur
    if os.path.exists(f"{output_dir}/rapport_comparatif.html"):
        choice = input("\nOuvrir le rapport détaillé dans un navigateur? (o/n): ").lower()
        if choice.startswith('o'):
            import webbrowser
            webbrowser.open(f"file://{output_dir}/rapport_comparatif.html")
    
    print(f"\n{'='*60}")
    print(f"=== Comparaison terminée avec succès ===")
    print(f"Résultats complets sauvegardés dans: {output_dir}")
    print(f"{'='*60}")

def create_comparison_plots(results_df, output_dir, failure_mode):
    """
    Crée des graphiques comparatifs entre les protocoles.
    
    Args:
        results_df: DataFrame contenant les résultats de toutes les simulations
        output_dir: Dossier de sortie pour les graphiques
        failure_mode: Mode de panne utilisé
    """
    # ---- Graphique comparatif principal ----
    plt.figure(figsize=(12, 10))
    
    # Organiser les données par protocole
    spray_results = results_df[results_df['protocol'] == 'Spray-and-Wait']
    prophet_results = results_df[results_df['protocol'] == 'PRoPHET']
    
    # 1. Délai de livraison
    plt.subplot(2, 2, 1)
    
    # Spray-and-Wait
    plt.bar(
        [f"S&W (L={p})" for p in spray_results['param']],
        [t if t != float('inf') else max_steps for t in spray_results['delivery_time']],
        color='skyblue',
        alpha=0.7
    )
    
    # PRoPHET
    plt.bar(
        [f"PRoPHET (P={p})" for p in prophet_results['param']],
        [t if t != float('inf') else max_steps for t in prophet_results['delivery_time']],
        color='orange',
        alpha=0.7
    )
    
    plt.title('Délai de livraison')
    plt.ylabel('Temps (pas)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Overhead (copies créées)
    plt.subplot(2, 2, 2)
    
    # Spray-and-Wait
    plt.bar(
        [f"S&W (L={p})" for p in spray_results['param']],
        spray_results['copies_created'],
        color='skyblue',
        alpha=0.7
    )
    
    # PRoPHET
    plt.bar(
        [f"PRoPHET (P={p})" for p in prophet_results['param']],
        prophet_results['copies_created'],
        color='orange',
        alpha=0.7
    )
    
    plt.title('Copies créées')
    plt.ylabel('Nombre de copies')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Résilience
    plt.subplot(2, 2, 3)
    
    # Spray-and-Wait
    plt.bar(
        [f"S&W (L={p})" for p in spray_results['param']],
        spray_results['resilience'],
        color='skyblue',
        alpha=0.7
    )
    
    # PRoPHET
    plt.bar(
        [f"PRoPHET (P={p})" for p in prophet_results['param']],
        prophet_results['resilience'],
        color='orange',
        alpha=0.7
    )
    
    plt.title('Score de résilience')
    plt.ylabel('Résilience')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Débit moyen
    plt.subplot(2, 2, 4)
    
    # Spray-and-Wait
    plt.bar(
        [f"S&W (L={p})" for p in spray_results['param']],
        spray_results['avg_throughput'],
        color='skyblue',
        alpha=0.7
    )
    
    # PRoPHET
    plt.bar(
        [f"PRoPHET (P={p})" for p in prophet_results['param']],
        prophet_results['avg_throughput'],
        color='orange',
        alpha=0.7
    )
    
    plt.title('Débit moyen')
    plt.ylabel('Copies/pas')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparaison_protocoles_{failure_mode}.png")
    plt.close()

def create_html_report(results_df, output_dir, failure_mode, failure_desc):
    """
    Crée un rapport HTML interactif pour comparer les protocoles.
    
    Args:
        results_df: DataFrame contenant les résultats
        output_dir: Dossier de sortie
        failure_mode: Mode de panne utilisé
        failure_desc: Description du mode de panne
    """
    # Organiser les données par protocole pour le rapport
    spray_results = results_df[results_df['protocol'] == 'Spray-and-Wait']
    prophet_results = results_df[results_df['protocol'] == 'PRoPHET']
    
    # Préparer les tableaux HTML
    spray_table = tabulate(
        spray_results[['param', 'delivered', 'delivery_time', 'hops', 'copies_created', 'overhead_ratio',
                       'avg_throughput', 'resilience', 'failure_rate']],
        headers=['L', 'Livré', 'Délai', 'Sauts', 'Copies', 'Overhead', 'Débit', 'Résilience', 'Taux panne'],
        tablefmt='html'
    )
    
    prophet_table = tabulate(
        prophet_results[['param', 'delivered', 'delivery_time', 'hops', 'copies_created', 'overhead_ratio',
                         'avg_throughput', 'resilience', 'failure_rate']],
        headers=['P_init', 'Livré', 'Délai', 'Sauts', 'Copies', 'Overhead', 'Débit', 'Résilience', 'Taux panne'],
        tablefmt='html'
    )
    
    # Créer le rapport HTML
    with open(f"{output_dir}/rapport_comparatif.html", "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comparaison Spray-and-Wait vs PRoPHET - {failure_mode}</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f9f9f9;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .chart {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .conclusion {{
                    background-color: #f0f7ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 30px;
                }}
                .true {{
                    color: green;
                    font-weight: bold;
                }}
                .false {{
                    color: red;
                }}
                footer {{
                    margin-top: 30px;
                    text-align: center;
                    font-size: 0.8em;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Comparaison des protocoles DTN face aux pannes dynamiques</h1>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Mode de panne:</strong> {failure_mode} - {failure_desc}</p>
                
                <div class="chart">
                    <h2>Résultats comparatifs</h2>
                    <img src="comparaison_protocoles_{failure_mode}.png" alt="Graphique comparatif" style="max-width:100%;">
                </div>
                
                <h2>Résultats détaillés pour Spray-and-Wait</h2>
                {spray_table}
                
                <h2>Résultats détaillés pour PRoPHET</h2>
                {prophet_table}
                
                <div class="conclusion">
                    <h2>Analyse comparative</h2>
                    <p>Cette comparaison met en évidence les différences de performance entre les protocoles
                    Spray-and-Wait et PRoPHET dans un scénario de pannes dynamiques.</p>
                    
                    <h3>Points clés à observer:</h3>
                    <ul>
                        <li><strong>Délai de livraison:</strong> Mesure du temps nécessaire pour délivrer le message</li>
                        <li><strong>Overhead:</strong> Surcharge réseau générée par chaque protocole</li>
                        <li><strong>Résilience:</strong> Capacité à livrer malgré les pannes</li>
                    </ul>
                    
                    <p>Les deux protocoles présentent des caractéristiques différentes:</p>
                    <ul>
                        <li><strong>Spray-and-Wait</strong> est plus déterministe et économe en ressources, avec un nombre fixe de copies.</li>
                        <li><strong>PRoPHET</strong> est plus adaptatif et exploite l'historique des rencontres, ce qui peut être avantageux dans certains scénarios.</li>
                    </ul>
                </div>
            </div>
            
            <footer>
                Rapport généré automatiquement - Système de simulation DTN
            </footer>
        </body>
        </html>
        """)

if __name__ == "__main__":
    max_steps = 100  # Valeur globale utilisée dans certaines fonctions
    compare_protocols_with_dynamic_failures()
