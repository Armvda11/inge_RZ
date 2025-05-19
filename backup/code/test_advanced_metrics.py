#!/usr/bin/env python3
# test_advanced_metrics.py
"""
Script de test pour l'analyse avancée de robustesse du réseau.
Crée plusieurs graphes d'exemple et calcule les métriques avancées pour valider l'implémentation.
"""

import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
from simulation.advanced_metrics import compute_advanced_metrics
from config import OUTDIR

# Import optionnel du module d'analyse des composantes
try:
    from simulation.component_analysis import analyze_fragmentation, compare_fragmentation_patterns
    has_component_analysis = True
except ImportError:
    has_component_analysis = False

def test_basic_metrics():
    """Test des métriques sur un graphe avant/après panne."""
    print("\nTest des métriques de base avant/après panne...")
    
    # Créer un graphe d'exemple
    print("Création d'un graphe d'exemple...")
    
    # Graphe avant panne (bien connecté)
    G_before = nx.watts_strogatz_graph(20, 4, 0.3)
    
    # Graphe après panne (supprimer quelques nœuds centraux)
    G_after = G_before.copy()
    
    # Identifier les nœuds les plus centraux
    centrality = nx.betweenness_centrality(G_after)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:5]]  # Supprimer les 5 nœuds les plus centraux
    
    # Supprimer ces nœuds
    for node in top_nodes:
        G_after.remove_node(node)
    
    # Calculer les métriques avancées pour les deux graphes
    print("Calcul des métriques avant panne...")
    metrics_before = compute_advanced_metrics(G_before)
    
    print("Calcul des métriques après panne...")
    metrics_after = compute_advanced_metrics(G_after)
    
    # Afficher les résultats
    print("\nRésultats:")
    print(f"{'Métrique':<25} {'Avant':<10} {'Après':<10} {'Variation (%)':<15}")
    print("-" * 60)
    
    # Degré moyen
    delta_k = ((metrics_after.mean_degree - metrics_before.mean_degree) / metrics_before.mean_degree) * 100
    print(f"{'Degré moyen':<25} {metrics_before.mean_degree:.3f} {metrics_after.mean_degree:.3f} {delta_k:.2f}%")
    
    # Taille de la composante géante
    delta_gcc = ((metrics_after.giant_component_size - metrics_before.giant_component_size) / metrics_before.giant_component_size) * 100
    print(f"{'Composante géante':<25} {metrics_before.giant_component_size:.3f} {metrics_after.giant_component_size:.3f} {delta_gcc:.2f}%")
    
    # Longueur moyenne des chemins
    delta_l = ((metrics_after.avg_path_length - metrics_before.avg_path_length) / metrics_before.avg_path_length) * 100
    print(f"{'Longueur chemins':<25} {metrics_before.avg_path_length:.3f} {metrics_after.avg_path_length:.3f} {delta_l:.2f}%")
    
    # Diamètre
    delta_d = ((metrics_after.diameter - metrics_before.diameter) / metrics_before.diameter) * 100
    print(f"{'Diamètre':<25} {metrics_before.diameter:.1f} {metrics_after.diameter:.1f} {delta_d:.2f}%")
    
    # Coefficient de clustering
    delta_c = ((metrics_after.clustering_coefficient - metrics_before.clustering_coefficient) / metrics_before.clustering_coefficient) * 100
    print(f"{'Clustering':<25} {metrics_before.clustering_coefficient:.3f} {metrics_after.clustering_coefficient:.3f} {delta_c:.2f}%")
    
    # Distribution des composantes
    print("\nDistribution des composantes avant:")
    print(metrics_before.components_distribution)
    
    print("\nDistribution des composantes après:")
    print(metrics_after.components_distribution)
    
    # Si le module d'analyse des composantes est disponible, l'utiliser
    if has_component_analysis:
        print("\nTest de l'analyse des composantes...")
        compare_fragmentation_patterns(G_before, G_after, "test_basic", 
                                     os.path.join(OUTDIR, "advanced"))

def test_specific_structures():
    """Test des métriques sur différentes structures de graphes connues."""
    print("\nTest des métriques sur des structures spécifiques...")
    
    # Liste des graphes à tester
    graphs = {
        "Complet (K10)": nx.complete_graph(10),
        "Chemin (P10)": nx.path_graph(10),
        "Étoile (S10)": nx.star_graph(9),  # 1 centre + 9 branches = 10 nœuds
        "Cycle (C10)": nx.cycle_graph(10),
        "Complet biparti (K5,5)": nx.complete_bipartite_graph(5, 5),
        "Grille 4x4": nx.grid_2d_graph(4, 4),
        "Graphe aléatoire (ER, p=0.2)": nx.erdos_renyi_graph(20, 0.2, seed=42)
    }
    
    # Calculer et afficher les métriques pour chaque graphe
    results = {}
    for name, G in graphs.items():
        metrics = compute_advanced_metrics(G)
        results[name] = metrics
        
        print(f"\n{name} ({G.number_of_nodes()} nœuds, {G.number_of_edges()} liens):")
        print(f"  - Degré moyen (⟨k⟩): {metrics.mean_degree:.3f}")
        print(f"  - Composante géante (|Gₘₐₓ|/N): {metrics.giant_component_size:.3f}")
        print(f"  - Longueur moyenne chemins (ℓ̄): {metrics.avg_path_length:.3f}")
        print(f"  - Diamètre (D): {metrics.diameter}")
        print(f"  - Coefficient clustering (C): {metrics.clustering_coefficient:.3f}")
    
    # Créer une figure pour visualiser les graphes et leurs métriques
    fig = plt.figure(figsize=(18, 12))
    
    # Disposition des sous-graphiques
    ncols = 3
    nrows = (len(graphs) + ncols - 1) // ncols
    
    for i, (name, G) in enumerate(graphs.items()):
        metrics = results[name]
        
        # Tracer le graphe
        ax = fig.add_subplot(nrows, ncols, i + 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, font_weight='bold', ax=ax)
        
        # Ajouter le titre avec les métriques principales
        title = f"{name}\n"
        title += f"⟨k⟩={metrics.mean_degree:.1f}, |Gₘₐₓ|/N={metrics.giant_component_size:.2f}\n"
        title += f"ℓ̄={metrics.avg_path_length:.2f}, D={metrics.diameter}, C={metrics.clustering_coefficient:.2f}"
        ax.set_title(title)
    
    # Créer le répertoire avancé si nécessaire
    advanced_dir = os.path.join(OUTDIR, 'advanced')
    os.makedirs(advanced_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f"{advanced_dir}/test_graphs.png")
    print(f"\nVisualisation des graphes de test enregistrée: {advanced_dir}/test_graphs.png")
    plt.close()

def test_fragmentation():
    """Test de la fragmentation du réseau."""
    if not has_component_analysis:
        print("\nModule d'analyse des composantes non disponible. Test de fragmentation ignoré.")
        return
    
    print("\nTest de fragmentation progressive du réseau...")
    
    # Créer un graphe bien connecté (préférer un graphe plus simple)
    G_full = nx.watts_strogatz_graph(50, 6, 0.2, seed=42)
    
    # Graphes avec différents degrés de fragmentation
    G_stages = []
    
    # Graphe 1: Complet
    G_stages.append(("Original", G_full.copy()))
    
    # Graphe 2: Retirer 10% des nœuds aléatoirement
    G2 = G_full.copy()
    nodes_to_remove = list(G2.nodes())[:5]  # Retirer 5 nœuds sur 50
    G2.remove_nodes_from(nodes_to_remove)
    G_stages.append(("10% nœuds supprimés", G2))
    
    # Graphe 3: Retirer 20% des nœuds centraux
    G3 = G_full.copy()
    centrality = nx.betweenness_centrality(G3)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:10]]  # 10 nœuds sur 50 = 20%
    G3.remove_nodes_from(top_nodes)
    G_stages.append(("20% nœuds centraux supprimés", G3))
    
    # Analyser chaque étape
    all_metrics = []
    for i, (name, G) in enumerate(G_stages):
        metrics = compute_advanced_metrics(G)
        frag_stats = analyze_fragmentation(G, f"stage{i}", i)
        
        print(f"\n{name} ({G.number_of_nodes()} nœuds, {G.number_of_edges()} liens):")
        print(f"  - Degré moyen (⟨k⟩): {metrics.mean_degree:.3f}")
        print(f"  - Composante géante (|Gₘₐₓ|/N): {metrics.giant_component_size:.3f}")
        print(f"  - Nombre composantes: {len(metrics.components_distribution)}")
        print(f"  - Facteur fragmentation: {frag_stats['fragmentation_factor']:.3f}")
    
    # Comparer les étapes successives
    for i in range(1, len(G_stages)):
        prev_name, prev_G = G_stages[i-1]
        curr_name, curr_G = G_stages[i]
        
        print(f"\nComparaison {prev_name} → {curr_name}:")
        compare_fragmentation_patterns(prev_G, curr_G, f"compare_{i-1}_to_{i}", 
                                     os.path.join(OUTDIR, "advanced"))

def main():
    """Point d'entrée principal du script."""
    print("### Test des métriques avancées de robustesse ###")
    
    try:
        # Test des métriques de base
        test_basic_metrics()
        
        # Test des structures spécifiques
        test_specific_structures()
        
        # Test de fragmentation
        test_fragmentation()
        
        print("\nTests terminés avec succès!")
        
    except Exception as e:
        print(f"\nERREUR lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
