#!/usr/bin/env python3
# test_prophet_dynamic_failures.py
"""
Script de test pour évaluer la résilience du protocole PRoPHET (Probabilistic Routing Protocol using History
of Encounters and Transitivity) face à des pannes dynamiques et continues.

Ce script est basé sur le test de Spray-and-Wait et utilise les mêmes mécanismes de pannes dynamiques:
1. Implémentation de pannes CONTINUES qui se produisent régulièrement pendant la simulation
2. Plusieurs MODES de panne dynamique:
   - Mode "continuous": pannes aléatoires à chaque pas de temps
   - Mode "cascade": effet domino où les voisins d'un nœud en panne ont plus de risque de tomber en panne
   - Mode "targeted_dynamic": recalcul périodique des nœuds critiques à mettre en panne
3. VISUALISATION claire de l'impact des pannes sur la diffusion des messages
4. Test de l'influence de différentes valeurs de P_init (probabilité initiale) face à ces pannes
5. Analyse continue des métriques de délai et débit tout au long de la simulation
6. Affichage amélioré des résultats sous forme de tableaux détaillés
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
from protocols.prophet import Prophet
from simulation.failure import NodeFailureManager
from performance_metrics import PerformanceTracker, generate_comparative_table

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

def visualize_network(adjacency: dict[int, set[int]], t: int, copies: dict[int, int], 
                      delivered_at: dict, output_dir: str, 
                      failed_nodes=None, 
                      failure_manager=None,
                      filename_prefix="network"):
    """
    Fonction désactivée - les visualisations de réseau ne sont plus requises par l'utilisateur.
    
    Cette fonction était utilisée pour visualiser le réseau à un instant donné,
    montrant la distribution des copies et les nœuds en panne.
    """
    # Fonction désactivée - ne génère plus de visualisations
    pass

def test_prophet_with_dynamic_failures():
    """
    Test du protocole PRoPHET avec des pannes dynamiques et continues
    qui se produisent tout au long de la simulation, pas seulement à un moment précis.
    
    Version adaptée pour PRoPHET: Test de différentes valeurs de P_init, gamma et beta,
    et meilleure observation des effets sur les délais et débits. Calcul continu des 
    métriques et affichage amélioré des tableaux de résultats.
    """
    print("=== Test du protocole PRoPHET avec pannes dynamiques et continues ===")
    
    # Paramètres de simulation
    num_nodes = 40  # Augmenter le nombre de nœuds pour avoir plus de clusters et de sauts
    max_steps = 100  # Augmenter le nombre de pas pour mieux observer les effets à long terme
    p_init_values = [0.1, 0.3, 0.5, 0.7]  # Différentes valeurs de P_init à tester
    beta_value = 0.25  # Facteur de transitivité
    gamma_value = 0.98  # Facteur de vieillissement
    ttl_value = 40  # TTL augmenté pour permettre plus de sauts
    distribution_rate = 0.4  # Taux de distribution ralenti pour mieux observer l'impact des pannes
    network_dilution = 0.7  # Plus de dilution pour simuler un réseau plus épars
    
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
    main_output_dir = f"{OUTDIR}/protocols/prophet_dynamic_failures_test"
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
    
    # Résultats
    all_results = []
    
    # Pour chaque valeur de P_init
    for p_init in p_init_values:
        print(f"\n--- Test avec P_init = {p_init} ---")
        
        # Source et destination fixes
        source = 0  # Premier nœud
        destination = num_nodes - 1  # Dernier nœud
        
        # Créer le gestionnaire de pannes dynamiques
        failure_mgr = DynamicFailureManager(num_nodes, source, destination)
        
        # Initialiser le protocole PRoPHET
        protocol = Prophet(
            num_nodes=num_nodes,
            p_init=p_init,
            dest=destination,
            source=source,
            gamma=gamma_value,
            beta=beta_value,
            ttl=ttl_value,
            distribution_rate=distribution_rate
        )
        
        # Tracker de performances pour suivre les métriques tout au long de la simulation
        perf_tracker = PerformanceTracker(num_nodes, source, destination)
        
        # Créer un DataFrame pour enregistrer les métriques au fil du temps
        metrics_df = pd.DataFrame(columns=['t', 'active_nodes', 'failed_nodes', 'copies', 'delivered'])
        
        # Simuler le réseau
        for t in range(max_steps):
            # Créer le réseau à ce pas de temps
            adjacency = create_multihop_network(t, num_nodes, network_dilution)
            
            # Appliquer les pannes dynamiques
            adjacency = failure_mgr.apply_failures(t, adjacency, failure_mode, test_modes[failure_mode]['params'])
            
            # Exécuter un pas de simulation
            protocol.step(t, adjacency)
            
            # Enregistrer les métriques pour ce pas de temps
            active_nodes = len(adjacency)
            failed_nodes = num_nodes - active_nodes
            perf_tracker.record_step(t, protocol, active_nodes, failed_nodes)
            
            # Ajouter les métriques à notre DataFrame
            metrics_df.loc[len(metrics_df)] = {
                't': t,
                'active_nodes': active_nodes,
                'failed_nodes': failed_nodes,
                'copies': sum(protocol.copies.values()),
                'delivered': destination in protocol.delivered_at
            }
            
            # Visualiser le réseau à certains instants clés (tous les 4 pas)
            if t % 4 == 0:
                visualize_network(
                    adjacency=adjacency,
                    t=t,
                    copies=protocol.copies,
                    delivered_at=protocol.delivered_at,
                    output_dir=output_dir,
                    failed_nodes=failure_mgr.failed_nodes,
                    failure_manager=failure_mgr,
                    filename_prefix=f"prophet_p{p_init}_network"
                )
            
            # Si le message est livré et qu'on est déjà 20 pas après la livraison, on peut arrêter
            if protocol.message_delivered and t > protocol.delivered_at[destination] + 20:
                print(f"  Message livré à t={protocol.delivered_at[destination]}, simulation arrêtée à t={t}")
                break
        
        # Calculer les statistiques finales
        perf_tracker.calculate_final_stats(protocol, max_steps)
        
        # Sauvegarder les métriques
        metrics_df.to_csv(f"{output_dir}/metrics_P{p_init}.csv", index=False)
        
        # Créer des visualisations des métriques
        create_performance_plots(metrics_df, protocol, perf_tracker, p_init, output_dir)
        
        # Afficher les résultats
        results_dict = {
            'P_init': p_init,
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
        
        # Afficher un tableau de résultats
        print("\nRésultats pour P_init =", p_init)
        print("-" * 50)
        print(f"Message livré: {protocol.message_delivered}")
        if protocol.message_delivered:
            print(f"Temps de livraison: {protocol.delivered_at[destination]}")
            print(f"Nombre de sauts: {protocol.num_hops[destination]}")
        print(f"Copies créées: {protocol.total_copies_created}")
        print(f"Overhead ratio: {protocol.overhead_ratio()}")
        print(f"Débit moyen: {perf_tracker.final_stats.get('avg_throughput', 0):.4f} copies/pas")
        print(f"Score de résilience: {perf_tracker.final_stats.get('resilience_score', 0):.4f}")
        print(f"Taux moyen de pannes: {perf_tracker.final_stats.get('avg_failure_rate', 0):.4f}")
    
    # Créer un tableau comparatif amélioré
    if all_results:
        comparison_table = generate_comparative_table(all_results)
        print("\nTableau comparatif des résultats:")
        print(comparison_table)
        
        # Sauvegarder le tableau dans un fichier
        with open(f"{output_dir}/resultats_prophet.txt", "w") as f:
            f.write("Résultats du test de PRoPHET avec pannes dynamiques\n")
            f.write(f"Mode de panne: {failure_mode}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(comparison_table)
        
        # Créer une version HTML plus interactive
        html_table = generate_comparative_table(all_results, table_format='html')
        with open(f"{output_dir}/rapport_detaille.html", "w") as f:
            f.write(f"""
            <html>
            <head>
                <title>Rapport de test PRoPHET - Mode {failure_mode}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .success {{ color: green; }}
                    .failure {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Rapport détaillé: Test PRoPHET avec pannes dynamiques ({failure_mode})</h1>
                <p>Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Mode de panne: <strong>{test_modes[failure_mode]['desc']}</strong></p>
                {html_table}
                <hr>
                <h2>Configuration de test</h2>
                <ul>
                    <li>Nombre de nœuds: {num_nodes}</li>
                    <li>Pas de simulation max: {max_steps}</li>
                    <li>Beta (transitivité): {beta_value}</li>
                    <li>Gamma (vieillissement): {gamma_value}</li>
                    <li>TTL: {ttl_value}</li>
                    <li>Taux de distribution: {distribution_rate}</li>
                </ul>
            </body>
            </html>
            """)
    
    # Demander à l'utilisateur s'il souhaite ouvrir le rapport HTML dans le navigateur
    if all_results and os.path.exists(f"{output_dir}/rapport_detaille.html"):
        choice = input("\nOuvrir le rapport détaillé dans un navigateur? (o/n): ").lower()
        if choice.startswith('o'):
            import webbrowser
            webbrowser.open(f"file://{output_dir}/rapport_detaille.html")
    
    print(f"\n{'='*60}")
    print(f"=== Test terminé avec succès ===")
    print(f"Résultats complets sauvegardés dans: {output_dir}")
    print(f"{'='*60}")

def create_performance_plots(metrics_df, protocol, perf_tracker, p_init, output_dir):
    """
    Crée des graphiques de performance pour analyser le comportement du protocole.
    
    Args:
        metrics_df: DataFrame contenant les métriques au fil du temps
        protocol: Instance du protocole utilisé
        perf_tracker: Tracker de performance
        p_init: Valeur de P_init utilisée
        output_dir: Dossier de sortie pour les graphiques
    """
    # Conversion en DataFrame pandas pour plus de facilité
    perf_df = perf_tracker.get_metrics_dataframe()
    
    # ---- Graphique principal: évolution du nombre de copies et des nœuds actifs/en panne ----
    plt.figure(figsize=(12, 8))
    
    # Sous-graphique 1: Copies actives et nœuds actifs
    plt.subplot(2, 1, 1)
    plt.plot(metrics_df['t'], metrics_df['copies'], 'b-', label='Copies actives')
    plt.plot(metrics_df['t'], metrics_df['active_nodes'], 'g-', label='Nœuds actifs')
    plt.plot(metrics_df['t'], metrics_df['failed_nodes'], 'r-', label='Nœuds en panne')
    
    # Marquer le moment de la livraison si elle a eu lieu
    if protocol.message_delivered:
        delivery_time = protocol.delivered_at[protocol.dest]
        plt.axvline(x=delivery_time, color='purple', linestyle='--', label=f'Livré à t={delivery_time}')
    
    plt.title(f'Évolution des copies et pannes - PRoPHET (P_init={p_init})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Nombre')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Sous-graphique 2: Métriques de performance
    plt.subplot(2, 1, 2)
    if not perf_df.empty and 'throughput' in perf_df.columns:
        plt.plot(perf_df['t'], perf_df['throughput'], 'm-', label='Débit')
    if not perf_df.empty and 'coverage_rate' in perf_df.columns:
        plt.plot(perf_df['t'], perf_df['coverage_rate'], 'c-', label='Taux de couverture')
    
    plt.title(f'Métriques de performance au fil du temps - PRoPHET (P_init={p_init})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Valeur')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dynamic_failures_P{p_init}.png")
    plt.close()
    
    # ---- Graphique détaillé pour analyse approfondie ----
    plt.figure(figsize=(14, 10))
    
    # Sous-graphique 1: Copies actives
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['t'], metrics_df['copies'], 'b-', linewidth=2)
    if protocol.message_delivered:
        plt.axvline(x=protocol.delivered_at[protocol.dest], color='purple', linestyle='--')
    plt.title(f'Copies actives - PRoPHET (P_init={p_init})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Nombre de copies')
    plt.grid(True, alpha=0.3)
    
    # Sous-graphique 2: Nœuds actifs vs en panne
    plt.subplot(2, 2, 2)
    plt.stackplot(metrics_df['t'], metrics_df['active_nodes'], metrics_df['failed_nodes'],
                 labels=['Nœuds actifs', 'Nœuds en panne'],
                 colors=['lightgreen', 'lightcoral'], alpha=0.7)
    plt.title(f'État des nœuds - PRoPHET (P_init={p_init})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Nombre de nœuds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Sous-graphique 3: Débit et taux de couverture
    plt.subplot(2, 2, 3)
    if not perf_df.empty:
        if 'throughput' in perf_df.columns:
            plt.plot(perf_df['t'], perf_df['throughput'], 'm-', linewidth=2, label='Débit')
        if 'coverage_rate' in perf_df.columns:
            plt.plot(perf_df['t'], perf_df['coverage_rate'], 'c-', linewidth=2, label='Taux de couverture')
    plt.title(f'Métriques de performance - PRoPHET (P_init={p_init})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Valeur')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Sous-graphique 4: Taux de panne et délai de livraison
    plt.subplot(2, 2, 4)
    if not perf_df.empty and 'failure_rate' in perf_df.columns:
        plt.plot(perf_df['t'], perf_df['failure_rate'], 'r-', linewidth=2, label='Taux de panne')
    
    # Texte résumant les résultats finaux
    delivery_text = "Livré" if protocol.message_delivered else "Non livré"
    delivery_time = protocol.delivered_at.get(protocol.dest, "N/A")
    hops = protocol.num_hops.get(protocol.dest, "N/A")
    copies = protocol.total_copies_created
    
    results_text = f"""
    Résultat: {delivery_text}
    Temps de livraison: {delivery_time}
    Sauts: {hops}
    Copies créées: {copies}
    """
    
    plt.text(0.1, 0.5, results_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.title(f'Résultats finaux - PRoPHET (P_init={p_init})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Taux de panne')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dynamic_failures_P{p_init}_detailed.png")
    plt.close()

if __name__ == "__main__":
    test_prophet_with_dynamic_failures()
