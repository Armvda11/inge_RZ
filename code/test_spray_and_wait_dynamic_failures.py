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
from tabulate import tabulate  # Pour les tableaux formatés

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTDIR
from protocols.spray_and_wait import SprayAndWait
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
                return new_adjacency  # Pas de nouvelles pannes ce tour-ci
        
        # Appliquer les pannes selon le mode
        new_failures = set()
        
        if mode == 'continuous':
            # Mode continu: chaque nœud a une probabilité de tomber en panne à chaque pas de temps
            failure_prob = params.get('failure_prob', 0.03)  # 3% de chance par défaut
            
            # Ne pas réduire la probabilité autant avec le temps pour maintenir des pannes continues
            # Utiliser une fonction sinusoïdale pour varier la probabilité au cours du temps
            time_factor = 0.8 + 0.4 * abs(math.sin(t * 0.1))  # Varie entre 0.8 et 1.2
            adjusted_prob = failure_prob * time_factor
            
            # Limiter le nombre maximal de pannes par pas de temps mais assurer au moins une panne
            max_failures = max(1, len(eligible_nodes) // 8)  # Max 12.5% des nœuds éligibles
            
            # Appliquer des pannes aléatoires
            for node in eligible_nodes:
                if random.random() < adjusted_prob and len(new_failures) < max_failures:
                    new_failures.add(node)
            
            # S'assurer qu'il y a au moins une nouvelle panne pour maintenir une dynamique continue
            if not new_failures and eligible_nodes:
                new_failures.add(random.choice(eligible_nodes))
        
        elif mode == 'cascade':
            # Mode cascade: les nœuds voisins de nœuds en panne ont un risque accru de panne
            # Paramètres spécifiques au mode cascade
            base_prob = params.get('base_prob', 0.02)  # Probabilité de base augmentée
            cascade_factor = params.get('cascade_factor', 5.0)  # Multiplicateur pour les voisins des nœuds en panne
            max_failures = max(1, len(eligible_nodes) // 10)  # Limiter à environ 10% des nœuds éligibles
            
            # Mettre à jour les risques de cascade pour tous les nœuds
            for node in eligible_nodes:
                # Risque de base avec variation temporelle pour maintenir les pannes
                self.cascade_risk[node] = base_prob * (0.8 + 0.4 * abs(math.sin(t * 0.1)))
                
                # Augmenter le risque pour les voisins de nœuds en panne
                for neighbor in adjacency.get(node, set()):
                    if neighbor in self.failed_nodes:
                        self.cascade_risk[node] += base_prob * cascade_factor
                
                # Plafonner le risque à 85% pour éviter une cascade trop rapide
                self.cascade_risk[node] = min(0.85, self.cascade_risk[node])
            
            # Appliquer les pannes en cascade
            for node in eligible_nodes:
                if random.random() < self.cascade_risk[node] and len(new_failures) < max_failures:
                    new_failures.add(node)
            
            # S'assurer qu'il y a au moins une nouvelle panne pour maintenir une dynamique continue
            if not new_failures and eligible_nodes:
                # Choisir le nœud avec le risque le plus élevé
                highest_risk_node = max(eligible_nodes, key=lambda n: self.cascade_risk[n])
                new_failures.add(highest_risk_node)
        
        elif mode == 'targeted_dynamic':
            # Mode ciblé dynamique: viser périodiquement les nœuds les plus centraux
            recalculation_interval = params.get('recalculation_interval', 4)  # Recalcul plus fréquent
            failure_percentage = params.get('failure_percentage', 0.08)  # Plus de nœuds ciblés
            
            # Recalculer plus souvent pour maintenir des pannes continues
            if t % recalculation_interval == 0 or not self.failure_history[-1]['new_failures'] if self.failure_history else True:
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
                    
                    # Calculer le nombre de nœuds à mettre en panne (au moins 1)
                    num_to_fail = max(1, int(len(eligible_nodes) * failure_percentage))
                    num_to_fail = min(num_to_fail, len(sorted_nodes))
                    
                    # Sélectionner les nœuds les plus centraux
                    most_central = [node for node, _ in sorted_nodes[:num_to_fail]]
                    
                    # Ajouter ces nœuds à la liste des nouvelles pannes
                    new_failures = set(most_central)
                except:
                    # Fallback en cas d'erreur (graphe déconnecté, etc.)
                    num_to_fail = max(1, int(len(eligible_nodes) * failure_percentage))
                    num_to_fail = min(num_to_fail, len(eligible_nodes))
                    new_failures = set(random.sample(eligible_nodes, num_to_fail))
            else:
                # Entre les recalculs, ajouter quand même quelques pannes aléatoires
                failure_prob = 0.02
                for node in eligible_nodes:
                    if random.random() < failure_prob and len(new_failures) < 2:
                        new_failures.add(node)
        
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
            print(f"t={t}: {len(new_failures)} nouvelles pannes ({len(self.failed_nodes)}/{self.num_nodes} total - {len(self.failed_nodes)/self.num_nodes*100:.1f}%) - mode {mode}")
        
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
    source, destination = 0, num_nodes - 1
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
                # Réduire la densité pour augmenter le nombre de sauts
                base_density = 0.6 * dilution_factor  # Densité réduite
                
                # Les clusters du milieu sont moins denses pour forcer les chemins multi-sauts
                if 0 < cluster_idx < num_clusters - 1:
                    cluster_factor = 0.7  # Encore moins dense dans les clusters intermédiaires
                else:
                    cluster_factor = 0.9
                
                # Variation temporelle mineure
                time_factor = 0.95 + 0.1 * ((t % 5) / 5.0)  # Entre 95% et 105%
                
                connect_prob = base_density * cluster_factor * time_factor
                
                if random.random() < connect_prob:
                    adjacency[cluster_list[i]].add(cluster_list[j])
                    adjacency[cluster_list[j]].add(cluster_list[i])
    
    # Établir des connexions UNIQUEMENT entre clusters adjacents
    # et réduire la probabilité de connexion entre clusters pour augmenter les sauts
    for c1 in range(num_clusters - 1):
        c2 = c1 + 1  # Cluster adjacent suivant
        
        # Sélectionner des nœuds passerelles
        num_gateways = 1 if t % 5 != 0 else 2
        
        # Limiter le nombre de passerelles pour augmenter la distance
        gateways1 = random.sample(list(clusters[c1]), min(num_gateways, len(clusters[c1])))
        gateways2 = random.sample(list(clusters[c2]), min(num_gateways, len(clusters[c2])))
        
        # Établir des connexions entre passerelles avec probabilité réduite
        for g1 in gateways1:
            for g2 in gateways2:
                base_prob = 0.25 * dilution_factor  # Moins de chance d'avoir des liens entre clusters
                time_factor = 0.2 * ((t % 10) / 10.0)
                
                connect_prob = base_prob + time_factor
                
                if random.random() < connect_prob:
                    adjacency[g1].add(g2)
                    adjacency[g2].add(g1)
    
    # S'assurer qu'il n'y a pas de connexion directe entre source et destination
    # et pas de connexions entre clusters non-adjacents
    if destination in adjacency[source]:
        adjacency[source].remove(destination)
    if source in adjacency[destination]:
        adjacency[destination].remove(source)
    
    # Bloquer les connexions entre clusters non adjacents pour forcer les multi-sauts
    for c1 in range(num_clusters):
        for c2 in range(num_clusters):
            if abs(c1 - c2) > 1:  # Si les clusters ne sont pas adjacents
                for n1 in clusters[c1]:
                    for n2 in clusters[c2]:
                        if n2 in adjacency[n1]:
                            adjacency[n1].remove(n2)
                        if n1 in adjacency[n2]:
                            adjacency[n2].remove(n1)
    
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

def test_spray_and_wait_with_dynamic_failures():
    """
    Test du protocole Spray-and-Wait avec des pannes dynamiques et continues
    qui se produisent tout au long de la simulation, pas seulement à un moment précis.
    
    Version améliorée: Plus de sauts entre source et destination, pannes continues, 
    et meilleure observation des effets sur les délais et débits. Calcul continu des 
    métriques et affichage amélioré des tableaux de résultats.
    """
    print("=== Test du protocole Spray-and-Wait avec pannes dynamiques et continues ===")
    
    # Paramètres de simulation
    num_nodes = 40  # Augmenter le nombre de nœuds pour avoir plus de clusters et de sauts
    max_steps = 100  # Augmenter le nombre de pas pour mieux observer les effets à long terme
    L_values = [4, 8, 16, 32]  # Ajouter ou modifier les valeurs à tester
    ttl_value = 40  # TTL augmenté pour permettre plus de sauts
    distribution_rate = 0.4  # Taux de distribution ralenti pour mieux observer l'impact des pannes
    network_dilution = 0.7  # Plus de dilution pour simuler un réseau plus épars
    
    # Modes de test avec paramètres ajustés
    test_modes = {
        'continuous': {
            'desc': 'Pannes aléatoires continues tout au long de la simulation',
            'params': {'failure_prob': 0.04}  # 4% de chance qu'un nœud tombe en panne à chaque pas
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
        
        # Initialiser le tracker de performances
        performance_tracker = PerformanceTracker(num_nodes, source, destination)
        
        # Visualiser l'état initial
        print(f"\nConfiguration initiale du réseau (dilution={network_dilution})")
        print(f"Source: {source}, Destination: {destination}")
        initial_adjacency = create_multihop_network(0, num_nodes, dilution_factor=network_dilution)
        
        # Affichage des en-têtes du tableau de progression
        print(f"\n{'='*80}")
        print(f"{'t':^5} | {'Copies':^8} | {'Copies Actives':^14} | {'Nœuds Actifs':^12} | {'En Panne':^8} | {'Taux Panne':^10} | {'Livré':^12}")
        print(f"{'-'*80}")
        
        # Fenêtre mobile pour l'affichage (affiche seulement les 10 derniers pas de temps)
        display_window = 10
        
        # Boucle principale de simulation
        for t in range(max_steps):
            # Générer le réseau pour l'instant t avec le facteur de dilution
            adjacency = create_multihop_network(t, num_nodes, dilution_factor=network_dilution)
            
            # Appliquer les pannes dynamiques selon le mode choisi
            adjacency = failure_manager.apply_failures(
                t, adjacency, failure_mode, test_modes[failure_mode]['params']
            )
            
            # Exécuter le pas de simulation
            protocol.step(t, adjacency)
            
            # Enregistrer les métriques pour ce pas de temps
            performance_tracker.record_step(
                t, 
                protocol, 
                len(adjacency), 
                len(failure_manager.failed_nodes)
            )
            
            # Afficher l'état actuel (utiliser un affichage en ligne pour économiser l'espace)
            # N'afficher que tous les 5 pas ou les événements importants
            total_active_copies = sum(protocol.copies.values())
            nodes_with_copies = sum(1 for n, c in protocol.copies.items() if c > 0)
            delivered = protocol.dest in protocol.delivered_at
            failure_rate = len(failure_manager.failed_nodes) / num_nodes
            
            delivered_str = f"Oui (t={protocol.delivered_at.get(protocol.dest, 'N/A')})" if delivered else "Non"
            
            # Afficher le tableau seulement à certaines conditions pour ne pas surcharger l'écran
            should_display = (
                t % 5 == 0 or  # Tous les 5 pas
                t == 0 or      # Premier pas
                t == max_steps - 1 or  # Dernier pas
                delivered and protocol.dest not in protocol.delivered_at  # Moment de livraison
            )
            
            if should_display or t < 10:  # Toujours afficher les 10 premiers pas
                print(f"{t:5d} | {protocol.total_copies_created:8d} | {total_active_copies:14d} | {len(adjacency):12d} | "
                      f"{len(failure_manager.failed_nodes):8d} | {failure_rate*100:8.1f}% | {delivered_str:12}")
            
            # Notification spéciale en cas de livraison
            if protocol.dest in protocol.delivered_at and not performance_tracker.delivery_occurred:
                print(f"\n{'='*50}")
                print(f"🎉 MESSAGE LIVRÉ À t={protocol.delivered_at[protocol.dest]} 🎉")
                print(f"{'='*50}\n")
                
                # Afficher un tableau détaillé des métriques au moment de la livraison
                performance_tracker.print_progress_table(t)
            
            # Arrêter si toutes les copies ont expiré
            if total_active_copies == 0:
                if not delivered:
                    print(f"\n❌ ÉCHEC: Toutes les copies expirées sans livraison à t={t}")
                else:
                    print(f"\n✅ SUCCÈS: Livraison terminée, copies épuisées à t={t}")
                break
            
            # Tous les 20 pas, afficher un tableau récapitulatif des derniers pas
            if t > 0 and t % 20 == 0:
                performance_tracker.print_progress_table(t, window_size=5)
        
        # Calculer les statistiques finales
        performance_tracker.calculate_final_stats(protocol, t+1)
        
        # Afficher le rapport détaillé
        performance_tracker.print_final_report()
        
        # Enregistrer les résultats pour comparaison
        result = {
            'L': L,
            'TTL': ttl_value,
            'failure_mode': failure_mode,
            'network_dilution': network_dilution,
            'distribution_rate': distribution_rate,
            'failed_nodes_final': len(failure_manager.failed_nodes),
            'delivered': bool(protocol.delivery_ratio()),
            'delivery_delay': performance_tracker.final_stats['delivery_delay'],
            'overhead_ratio': performance_tracker.final_stats['overhead_ratio'],
            'total_copies': protocol.total_copies_created,
            'hop_count': performance_tracker.final_stats['hop_count'],
            'avg_throughput': performance_tracker.final_stats['avg_throughput'],
            'resilience_score': performance_tracker.final_stats['resilience_score'],
            'avg_failure_rate': performance_tracker.final_stats['avg_failure_rate']
        }
        
        all_results.append(result)
        
        # Sauvegarder les métriques détaillées pour cette simulation
        metrics_df = performance_tracker.get_metrics_dataframe()
        metrics_df.to_csv(f"{output_dir}/metrics_L{L}.csv", index=False)
        
        # Graphiques d'analyse
        # 1. Évolution des métriques au fil du temps
        plt.figure(figsize=(14, 8))
        
        # 1.1 Premier graphique: Copies vs Pannes
        plt.subplot(2, 1, 1)
        plt.plot(metrics_df['t'], metrics_df['failed_nodes'], 'r-', linewidth=2, label="Nœuds en panne")
        plt.plot(metrics_df['t'], metrics_df['nodes_with_copies'], 'b-', linewidth=2, label="Nœuds avec copies")
        plt.plot(metrics_df['t'], metrics_df['total_copies'], 'g-', linewidth=2, label="Copies totales")
        
        plt.xlabel("Temps (t)")
        plt.ylabel("Nombre")
        plt.title(f"Impact des pannes sur la distribution des copies - L={L}, Mode={failure_mode}")
        
        # Marquer l'instant de livraison s'il y a eu livraison
        if performance_tracker.delivery_occurred:
            plt.axvline(x=performance_tracker.delivery_time, color='green', linestyle='--', 
                      label=f"Livraison (t={performance_tracker.delivery_time})")
            
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # 1.2 Second graphique: Taux de couverture et de pannes
        plt.subplot(2, 1, 2)
        plt.plot(metrics_df['t'], metrics_df['coverage_rate'] * 100, 'b-', linewidth=2, 
               label="Taux de couverture (%)")
        plt.plot(metrics_df['t'], metrics_df['failure_rate'] * 100, 'r-', linewidth=2, 
               label="Taux de panne (%)")
        plt.plot(metrics_df['t'], metrics_df['throughput'], 'g-', linewidth=2, 
               label="Débit (copies/t)")
        
        plt.xlabel("Temps (t)")
        plt.ylabel("Pourcentage / Débit")
        plt.title(f"Taux de couverture et impact des pannes - L={L}, Mode={failure_mode}")
        
        # Marquer l'instant de livraison s'il y a eu livraison
        if performance_tracker.delivery_occurred:
            plt.axvline(x=performance_tracker.delivery_time, color='green', linestyle='--', 
                      label=f"Livraison (t={performance_tracker.delivery_time})")
            
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dynamic_failures_L{L}_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Créer un tableau comparatif amélioré
    if all_results:
        # Convertir en DataFrame pour sauvegarder les résultats
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{output_dir}/resultats_pannes_dynamiques.csv", index=False)
        
        # Afficher un tableau comparatif amélioré
        print("\n" + "="*80)
        print("TABLEAU COMPARATIF DES RÉSULTATS")
        print("="*80)
        
        # Utiliser la fonction de génération de tableau comparatif
        comparative_table = generate_comparative_table(all_results)
        print(comparative_table)
        
        # Sauvegarder le tableau formaté dans un fichier texte
        with open(f"{output_dir}/tableau_comparatif.txt", 'w') as f:
            f.write(comparative_table)
        
        # Créer des graphiques comparatifs améliorés
        plt.figure(figsize=(12, 10))
        
        # 1. Livraison et Délai vs L
        plt.subplot(2, 2, 1)
        plt.plot(df_results['L'], df_results['delivered'].astype(int), 'o-', linewidth=2, color='green')
        plt.title('Taux de livraison vs L')
        plt.xlabel('L (copies initiales)')
        plt.ylabel('Livraison réussie')
        plt.xticks(df_results['L'])
        plt.yticks([0, 1], ['Échec', 'Succès'])
        plt.grid(True, alpha=0.3)
        
        # 2. Délai vs L
        plt.subplot(2, 2, 2)
        delivered_df = df_results[df_results['delivered']]
        
        if not delivered_df.empty:
            plt.plot(delivered_df['L'], delivered_df['delivery_delay'], 'o-', linewidth=2, color='blue')
            plt.title('Délai de livraison vs L')
            plt.xlabel('L (copies initiales)')
            plt.ylabel('Délai de livraison')
            plt.xticks(df_results['L'])
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Aucun message livré", ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # 3. Débit moyen vs L
        plt.subplot(2, 2, 3)
        plt.plot(df_results['L'], df_results['avg_throughput'], 'o-', linewidth=2, color='purple')
        plt.title('Débit moyen vs L')
        plt.xlabel('L (copies initiales)')
        plt.ylabel('Débit moyen (copies/t)')
        plt.xticks(df_results['L'])
        plt.grid(True, alpha=0.3)
        
        # 4. Score de résilience vs L
        plt.subplot(2, 2, 4)
        plt.plot(df_results['L'], df_results['resilience_score'], 'o-', linewidth=2, color='orange')
        plt.title('Score de résilience vs L')
        plt.xlabel('L (copies initiales)')
        plt.ylabel('Score de résilience')
        plt.xticks(df_results['L'])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparative_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Générer un rapport HTML détaillé avec toutes les métriques
        try:
            # Récupérer les métriques détaillées pour la valeur de L avec la meilleure résilience
            best_L = df_results.iloc[df_results['resilience_score'].argmax()]['L']
            best_metrics_file = f"{output_dir}/metrics_L{int(best_L)}.csv"
            
            if os.path.exists(best_metrics_file):
                detailed_metrics_df = pd.read_csv(best_metrics_file)
                
                # Informations sur le protocole
                protocol_info = {
                    'failure_mode': failure_mode,
                    'num_nodes': num_nodes,
                    'ttl_value': ttl_value,
                    'network_dilution': network_dilution,
                    'distribution_rate': distribution_rate
                }
                
                print("\nRésultats de la simulation :")
                print("---------------------------")
                print(f"Mode de panne : {failure_mode}")
                print(f"Nombre de nœuds : {num_nodes}")
                print(f"TTL : {ttl_value}")
                print(f"Dilution du réseau : {network_dilution}")
                print(f"Taux de distribution : {distribution_rate}")
            else:
                print(f"\n⚠️ Impossible de trouver les métriques détaillées pour L={best_L}")
        except Exception as e:
            print(f"\n⚠️ Erreur lors de la génération du rapport HTML: {str(e)}")
    
    # Demander à l'utilisateur s'il souhaite ouvrir le rapport HTML dans le navigateur
    if all_results and os.path.exists(f"{output_dir}/rapport_detaille.html"):
        try:
            open_html = input("\nSouhaitez-vous ouvrir le rapport HTML dans votre navigateur? (o/n): ").lower().strip()
            if open_html == 'o' or open_html == 'oui' or open_html == 'y' or open_html == 'yes':
                import webbrowser
                print("Ouverture du rapport HTML...")
                webbrowser.open(f"file://{os.path.abspath(output_dir)}/rapport_detaille.html")
        except Exception as e:
            print(f"Impossible d'ouvrir le navigateur: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"=== Test terminé avec succès ===")
    print(f"Résultats complets sauvegardés dans: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_spray_and_wait_with_dynamic_failures()
    print("\nTest de résilience aux pannes dynamiques et continues terminé!")
