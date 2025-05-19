"""
Module d'analyse des composantes connexes du réseau satellite.
Ce module fournit des outils pour analyser en détail les petites composantes
qui se forment après fragmentation du réseau suite à des pannes.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from config import OUTDIR

def analyze_fragmentation(G, scenario_name="unknown", time_point=0, output_dir=None):
    """
    Analyse détaillée de la fragmentation d'un graphe en composantes connexes.
    
    Args:
        G: Graphe NetworkX à analyser
        scenario_name: Nom du scénario (pour les rapports)
        time_point: Point temporel de l'analyse
        output_dir: Répertoire de sortie pour les graphiques
        
    Returns:
        dict: Dictionnaire contenant les statistiques de fragmentation
    """
    if output_dir is None:
        output_dir = os.path.join(OUTDIR, 'advanced')
        os.makedirs(output_dir, exist_ok=True)
    
    # Trouver toutes les composantes connexes
    components = list(nx.connected_components(G))
    num_components = len(components)
    
    # Si le graphe est connecté, retourner des statistiques simples
    if num_components <= 1:
        return {
            'num_components': num_components,
            'is_connected': True,
            'giant_component_size': G.number_of_nodes(),
            'giant_component_fraction': 1.0,
            'fragmentation_factor': 0.0,
            'small_components': [],
            'isolated_nodes': []
        }
    
    # Trier les composantes par taille décroissante
    components_by_size = sorted(components, key=len, reverse=True)
    component_sizes = [len(c) for c in components_by_size]
    
    # Composante géante
    giant_component = components_by_size[0]
    giant_size = len(giant_component)
    giant_fraction = giant_size / G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0
    
    # Facteur de fragmentation (1 - taille relative de la composante géante)
    fragmentation_factor = 1.0 - giant_fraction
    
    # Analyser les petites composantes (taille < 10% de la composante géante)
    small_components = []
    isolated_nodes = []
    
    for i, comp in enumerate(components_by_size[1:]):  # Skip the giant component
        comp_size = len(comp)
        comp_fraction = comp_size / G.number_of_nodes()
        
        # Récupérer les caractéristiques de base de la composante
        subgraph = G.subgraph(comp)
        avg_degree = sum(d for _, d in subgraph.degree()) / comp_size if comp_size > 0 else 0
        
        if comp_size == 1:
            # Nœuds isolés
            node_id = list(comp)[0]
            isolated_nodes.append(node_id)
        else:
            # Petites composantes (non-isolées)
            small_components.append({
                'size': comp_size,
                'fraction': comp_fraction,
                'avg_degree': avg_degree,
                'diameter': nx.diameter(subgraph) if comp_size > 1 else 0,
                'nodes': list(comp)
            })
    
    # Visualiser la distribution des tailles de composantes
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(component_sizes) + 1), component_sizes)
    plt.axhline(y=giant_size * 0.1, color='r', linestyle='-', label="10% de la composante géante")
    plt.title(f"Distribution des tailles de composantes - {scenario_name} à t={time_point}")
    plt.xlabel('Index de composante')
    plt.ylabel('Taille de composante')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/component_sizes_{scenario_name}_t{time_point}.png")
    plt.close()
    
    return {
        'num_components': num_components,
        'is_connected': False,
        'giant_component_size': giant_size,
        'giant_component_fraction': giant_fraction,
        'fragmentation_factor': fragmentation_factor,
        'small_components': small_components,
        'isolated_nodes': isolated_nodes,
        'component_sizes': component_sizes
    }


def compare_fragmentation_patterns(before_G, after_G, scenario_name="unknown", output_dir=None):
    """
    Compare les motifs de fragmentation avant et après un événement (panne).
    
    Args:
        before_G: Graphe NetworkX avant l'événement
        after_G: Graphe NetworkX après l'événement
        scenario_name: Nom du scénario
        output_dir: Répertoire de sortie pour les graphiques
        
    Returns:
        dict: Statistiques de comparaison des fragmentations
    """
    if output_dir is None:
        output_dir = os.path.join(OUTDIR, 'advanced')
        os.makedirs(output_dir, exist_ok=True)
    
    # Analyser les fragmentations avant et après
    before_stats = analyze_fragmentation(before_G, scenario_name, "avant")
    after_stats = analyze_fragmentation(after_G, scenario_name, "après")
    
    # Calculer les différences clés
    node_loss = before_G.number_of_nodes() - after_G.number_of_nodes()
    edge_loss = before_G.number_of_edges() - after_G.number_of_edges()
    component_increase = after_stats['num_components'] - before_stats['num_components']
    
    # Analyser comment la composante géante s'est fragmentée
    before_giant = set(max(nx.connected_components(before_G), key=len))
    after_components = list(nx.connected_components(after_G))
    
    # Pour chaque nœud qui était dans la composante géante avant,
    # voir dans quelle composante il se trouve après
    before_to_after_mapping = {}
    for i, comp in enumerate(after_components):
        comp_nodes = set(comp)
        overlap = before_giant.intersection(comp_nodes)
        before_to_after_mapping[i] = {
            'size': len(comp),
            'overlap': len(overlap),
            'overlap_fraction': len(overlap) / len(before_giant) if before_giant else 0
        }
    
    # Visualiser la fragmentation
    plt.figure(figsize=(12, 8))
    
    # Graphique principal: comparaison des tailles de composantes
    plt.subplot(211)
    plt.bar(range(1, len(before_stats.get('component_sizes', [0])) + 1), 
            before_stats.get('component_sizes', [0]), 
            alpha=0.5, label="Avant")
    plt.bar(range(1, len(after_stats.get('component_sizes', [0])) + 1), 
            after_stats.get('component_sizes', [0]), 
            alpha=0.5, label="Après")
    plt.title(f"Comparaison des tailles de composantes - {scenario_name}")
    plt.xlabel('Index de composante')
    plt.ylabel('Taille')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sous-graphique: distribution des petites composantes
    plt.subplot(212)
    before_sizes = Counter([len(c) for c in nx.connected_components(before_G)])
    after_sizes = Counter([len(c) for c in nx.connected_components(after_G)])
    
    # Fusionner les clés
    all_sizes = sorted(set(list(before_sizes.keys()) + list(after_sizes.keys())))
    
    # Créer les barres
    plt.bar([x - 0.2 for x in all_sizes], 
            [before_sizes.get(size, 0) for size in all_sizes], 
            width=0.4, label="Avant")
    plt.bar([x + 0.2 for x in all_sizes], 
            [after_sizes.get(size, 0) for size in all_sizes], 
            width=0.4, label="Après")
    
    plt.title("Distribution des tailles de composantes")
    plt.xlabel('Taille de composante')
    plt.ylabel('Nombre de composantes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(f"{output_dir}/fragmentation_analysis_{scenario_name}.png")
    plt.close()
    
    return {
        'node_loss': node_loss,
        'edge_loss': edge_loss,
        'component_increase': component_increase,
        'before_stats': before_stats,
        'after_stats': after_stats,
        'fragmentation_mapping': before_to_after_mapping
    }


def analyze_critical_nodes(G, scenario_name="unknown", output_dir=None):
    """
    Analyse des nœuds critiques dont la suppression fragmenterait le réseau.
    
    Args:
        G: Graphe NetworkX à analyser
        scenario_name: Nom du scénario
        output_dir: Répertoire de sortie pour les graphiques
        
    Returns:
        list: Liste des nœuds critiques avec leurs métriques
    """
    if output_dir is None:
        output_dir = os.path.join(OUTDIR, 'advanced')
        os.makedirs(output_dir, exist_ok=True)
    
    # Si le graphe n'est pas connecté, on identifie les nœuds critiques
    # dans chaque composante séparément
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        all_critical_nodes = []
        
        for i, comp in enumerate(components):
            if len(comp) > 2:  # Ignorer les composantes trop petites
                subgraph = G.subgraph(comp)
                critical = analyze_critical_nodes_component(subgraph, f"{scenario_name}_comp{i}")
                all_critical_nodes.extend(critical)
        
        return all_critical_nodes
    
    return analyze_critical_nodes_component(G, scenario_name, output_dir)


def analyze_critical_nodes_component(G, component_name="unknown", output_dir=None):
    """
    Analyse des nœuds critiques dans une composante connexe.
    
    Args:
        G: Graphe NetworkX connexe à analyser
        component_name: Nom de la composante
        output_dir: Répertoire de sortie pour les graphiques
        
    Returns:
        list: Liste des nœuds critiques avec leurs métriques
    """
    if output_dir is None:
        output_dir = os.path.join(OUTDIR, 'advanced')
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculer les mesures de centralité
    betweenness = nx.betweenness_centrality(G)
    articulation_points = list(nx.articulation_points(G))
    
    # Fusionner les deux mesures pour trouver les nœuds critiques
    critical_nodes = []
    
    # 1. Points d'articulation (leur suppression fragmente le graphe)
    for node in articulation_points:
        critical_nodes.append({
            'node_id': node,
            'betweenness': betweenness.get(node, 0),
            'is_articulation': True,
            'criticality_score': betweenness.get(node, 0) + 1.0  # Bonus pour être un point d'articulation
        })
    
    # 2. Nœuds à forte centralité d'intermédiarité (même s'ils ne sont pas des points d'articulation)
    top_betweenness = sorted([(n, b) for n, b in betweenness.items() if n not in articulation_points],
                            key=lambda x: x[1], reverse=True)[:10]  # Top 10
                            
    for node, b_score in top_betweenness:
        if b_score > 0.1:  # Seuil minimal de centralité
            critical_nodes.append({
                'node_id': node,
                'betweenness': b_score,
                'is_articulation': False,
                'criticality_score': b_score
            })
    
    # Trier par score de criticité
    critical_nodes.sort(key=lambda x: x['criticality_score'], reverse=True)
    
    # Visualiser les nœuds critiques (si pas trop grand)
    if len(G) <= 100:
        plt.figure(figsize=(10, 10))
        
        # Position des nœuds (layout force-directed)
        pos = nx.spring_layout(G)
        
        # Dessiner tous les nœuds et arêtes
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.5)
        
        # Mettre en évidence les nœuds critiques
        articulation_nodes = [n['node_id'] for n in critical_nodes if n['is_articulation']]
        high_centrality_nodes = [n['node_id'] for n in critical_nodes if not n['is_articulation']]
        
        if articulation_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=articulation_nodes, 
                                node_color='red', node_size=100)
        
        if high_centrality_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=high_centrality_nodes,
                                node_color='orange', node_size=80)
        
        plt.title(f"Points critiques - {component_name} ({len(articulation_nodes)} articulations)")
        plt.axis('off')
        plt.savefig(f"{output_dir}/critical_nodes_{component_name}.png")
        plt.close()
    
    return critical_nodes


def generate_fragmentation_report(results, output_dir=None):
    """
    Génère un rapport détaillé sur la fragmentation du réseau.
    
    Args:
        results: Dictionnaire des résultats d'analyse de fragmentation
        output_dir: Répertoire de sortie pour le rapport
        
    Returns:
        str: Chemin vers le fichier de rapport généré
    """
    if output_dir is None:
        output_dir = os.path.join(OUTDIR, 'advanced')
        os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "fragmentation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("# Analyse détaillée de la fragmentation du réseau\n\n")
        
        for scenario, data in results.items():
            f.write(f"## Scénario: {scenario}\n\n")
            
            before = data['before_stats']
            after = data['after_stats']
            
            f.write(f"### Statistiques générales\n\n")
            f.write(f"- Perte de nœuds: {data['node_loss']}\n")
            f.write(f"- Perte de liens: {data['edge_loss']}\n")
            f.write(f"- Augmentation du nombre de composantes: {data['component_increase']}\n\n")
            
            f.write(f"### Composante géante\n\n")
            f.write(f"- Avant: {before['giant_component_size']} nœuds ({before['giant_component_fraction']:.2%})\n")
            f.write(f"- Après: {after['giant_component_size']} nœuds ({after['giant_component_fraction']:.2%})\n")
            f.write(f"- Variation: {after['giant_component_fraction'] - before['giant_component_fraction']:.2%}\n\n")
            
            if len(after['isolated_nodes']) > 0:
                f.write(f"### Nœuds isolés après fragmentation: {len(after['isolated_nodes'])}\n\n")
            
            if len(after['small_components']) > 0:
                f.write(f"### Petites composantes après fragmentation: {len(after['small_components'])}\n\n")
                f.write("| Taille | Fraction | Degré moyen | Diamètre |\n")
                f.write("|--------|----------|-------------|----------|\n")
                
                for comp in after['small_components']:
                    f.write(f"| {comp['size']} | {comp['fraction']:.2%} | {comp['avg_degree']:.2f} | ")
                    f.write(f"{comp['diameter']} |\n")
                
                f.write("\n")
    
    return report_path
