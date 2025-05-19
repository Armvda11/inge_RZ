# simulation/advanced_metrics.py
"""
Module contenant des métriques avancées pour l'analyse de la robustesse du réseau satellite.
Ce module implémente les 5 métriques clés demandées pour quantifier précisément l'effet des pannes.
"""

import numpy as np
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from config import OUTDIR, T_PRED

@dataclass
class AdvancedMetrics:
    """
    Métriques avancées pour l'analyse de la robustesse du réseau.
    
    Attributes:
        mean_degree: Densité moyenne (⟨k⟩) - Indicateur de redondance
        giant_component_size: Taille normalisée de la composante géante (|Gₘₐₓ|/N)
        avg_path_length: Longueur moyenne des plus courts chemins (ℓ̄)
        diameter: Diamètre de Gₘₐₓ (D) - Pire cas de distance en sauts
        clustering_coefficient: Coefficient de clustering moyen (C)
        components_distribution: Distribution des tailles de composantes
        node_count: Nombre total de nœuds dans le graphe
        giant_component_nodes: Identifiants des nœuds de la composante géante
        is_connected: Booléen indiquant si le graphe est connexe
    """
    mean_degree: float
    giant_component_size: float
    avg_path_length: float
    diameter: float
    clustering_coefficient: float
    components_distribution: Dict[int, int] = field(default_factory=dict)
    node_count: int = 0
    giant_component_nodes: List[int] = field(default_factory=list)
    is_connected: bool = False


def compute_advanced_metrics(G):
    """
    Calcule les métriques avancées pour un graphe donné.
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        AdvancedMetrics: Objet contenant les métriques avancées
    """
    if G.number_of_nodes() == 0:
        return AdvancedMetrics(0.0, 0.0, 0.0, 0.0, 0.0, {}, 0, [], False)
    
    # Degré moyen (Densité moyenne ⟨k⟩)
    mean_degree = sum(d for _, d in G.degree()) / G.number_of_nodes()
    
    # Coefficient de clustering
    clustering_coefficient = nx.average_clustering(G)
    
    # Distribution des composantes
    components = list(nx.connected_components(G))
    components_sizes = [len(c) for c in components]
    components_distribution = {}
    for size in components_sizes:
        components_distribution[size] = components_distribution.get(size, 0) + 1
    
    # Taille de la composante géante et son diamètre
    if not components:
        giant_component_size = 0.0
        avg_path_length = 0.0
        diameter = 0
        giant_component_nodes = []
        is_connected = False
    else:
        # Trouver la plus grande composante connexe
        giant_component = max(components, key=len)
        # Validation explicite pour calculer la taille fractionnaire (|Gₘₐₓ|/N) avec précision
        initial_node_count = G.number_of_nodes()
        giant_component_size = len(giant_component) / initial_node_count if initial_node_count > 0 else 0.0
        giant_component_nodes = list(giant_component)
        # Vérifier si tous les nœuds sont dans une seule composante
        is_connected = len(giant_component) == initial_node_count  # Connectivité stricte
        
        # Log détaillé pour comprendre la fragmentation du réseau
        if len(components) > 1:
            component_sizes = sorted([len(c) for c in components], reverse=True)
            if len(component_sizes) > 1:
                second_largest = component_sizes[1]
                ratio_to_largest = second_largest / len(giant_component) if len(giant_component) > 0 else 0
                # Si la deuxième composante est significative (>10% de la plus grande), le signaler
                if ratio_to_largest > 0.1:
                    # Le réseau est fragmenté de manière significative
                    fragmentation_factor = 1 - giant_component_size  # Plus proche de 1 = plus fragmenté
                    # print(f"    - FRAGMENTATION: {len(components)} composantes. Plus grande={len(giant_component)} nœuds, "
                    #      f"2ème plus grande={second_largest} nœuds ({ratio_to_largest:.2%} de la plus grande)")
                    # print(f"    - Facteur de fragmentation: {fragmentation_factor:.3f}")
        else:
            # print(f"    - Réseau connecté: une seule composante avec {len(giant_component)} nœuds")
            pass
        
        # Calcul du diamètre (uniquement sur la composante géante)
        # Le diamètre est la plus longue distance en sauts dans la composante géante
        # Sous-graphe de la composante géante
        giant_subgraph = G.subgraph(giant_component)
        
        # Diamètre de la composante géante
        if len(giant_component) > 1:
            try:
                # Diamètre (plus long des plus courts chemins) - toujours sur Gₘₐₓ
                if len(giant_component) < 1000:  # Pour les petits graphes
                    # Utiliser l'algorithme exact pour le diamètre
                    diameter = nx.diameter(giant_subgraph)
                else:  # Pour les grands graphes, calcul plus précis
                    # Utiliser l'algorithme de Floyd-Warshall pour les grands graphes
                    # en sélectionnant un échantillon représentatif
                    sample_size = min(200, len(giant_component))
                    sample_nodes = np.random.choice(list(giant_component), sample_size, replace=False)
                    max_path = 0
                    
                    # Pour chaque nœud de l'échantillon, calculer ses plus courts chemins
                    # vers tous les autres nœuds et garder la plus grande valeur
                    for u in sample_nodes:
                        path_lengths = nx.single_source_shortest_path_length(giant_subgraph, u)
                        if path_lengths:  # S'assurer que path_lengths n'est pas vide
                            current_max = max(path_lengths.values())
                            if current_max > max_path:
                                max_path = current_max
                    diameter = max_path
            except nx.NetworkXError as e:
                # Gérer le cas d'erreur avec logs détaillés
                print(f"    - AVERTISSEMENT: Erreur lors du calcul du diamètre: {e}")
                diameter = float('inf')
        else:
            diameter = 0
        
        # Calcul de la longueur moyenne des chemins (ℓ̄) sur toutes les paires connectées
        # Il faut calculer cette métrique sur chaque composante connexe et pondérer par le nombre de paires
        path_lengths = []
        total_pairs = 0
        total_weighted_length = 0
        
        # Pour chaque composante (y compris les petits sous-graphes)
        for component in components:
            # Si la composante contient plus d'un nœud
            if len(component) > 1:
                subgraph = G.subgraph(component)
                try:
                    # On peut utiliser average_shortest_path_length sur cette composante connexe
                    component_avg_path = nx.average_shortest_path_length(subgraph)
                    # Calculer le nombre de paires dans cette composante
                    num_pairs = len(component) * (len(component) - 1) / 2
                    # Ajouter à nos compteurs
                    total_weighted_length += component_avg_path * num_pairs
                    total_pairs += num_pairs
                    
                    # Conserver l'information pour les logs détaillés
                    path_lengths.append((len(component), component_avg_path, num_pairs))
                except nx.NetworkXError:
                    # Ne devrait pas arriver car on traite une composante connexe
                    print(f"    - AVERTISSEMENT: Impossible de calculer la longueur moyenne des chemins pour une composante de taille {len(component)}")
        
        # Calculer la moyenne pondérée sur toutes les composantes
        if total_pairs > 0:
            avg_path_length = total_weighted_length / total_pairs
            
            # Log détaillé pour comprendre la distribution
            if len(path_lengths) > 1:
                components_info = [f"Composante {idx+1}: {size} nœuds, ℓ̄={avg:.3f}" 
                                   for idx, (size, avg, _) in enumerate(path_lengths)]
                components_str = ", ".join(components_info)
                #print(f"    - Calcul de ℓ̄ sur {len(path_lengths)} composantes: {components_str}")
        else:
            avg_path_length = 0.0
    
    return AdvancedMetrics(
        mean_degree=mean_degree,
        giant_component_size=giant_component_size,
        avg_path_length=avg_path_length,
        diameter=diameter,
        clustering_coefficient=clustering_coefficient,
        components_distribution=components_distribution,
        node_count=G.number_of_nodes(),
        giant_component_nodes=giant_component_nodes,
        is_connected=is_connected
    )


def extract_metrics_by_scenario(metrics_dict):
    """
    Extrait les métriques par scénario (sans panne, prévisible, aléatoire).
    
    Args:
        metrics_dict: Dictionnaire des métriques avancées par temps et scénario
        
    Returns:
        DataFrame: DataFrame contenant les métriques résumées par scénario
    """
    scenarios = list(metrics_dict.keys())
    results = []
    
    for scenario in scenarios:
        # Trouver T_PRED dans les données
        times = sorted(metrics_dict[scenario].keys())
        
        # Utiliser les instants exacts t=49 (avant) et t=50 (après) pour l'analyse
        # Assurer que nous avons bien les snapshots précis avant et après T_PRED
        if 49 in times and 50 in times:
            before_t = 49  # Instant juste avant T_PRED
            after_t = 50   # Instant juste après T_PRED (T_PRED lui-même)
            print(f"  - Scénario {scenario}: Utilisation des snapshots t={before_t} et t={after_t}")
        elif T_PRED is not None and T_PRED in times:
            t_pred_idx = times.index(T_PRED)
            before_t = times[max(0, t_pred_idx - 1)]  # Instant juste avant T_PRED
            after_t = T_PRED  # T_PRED lui-même
            print(f"  - Scénario {scenario}: Fallback sur les snapshots t={before_t} et t={after_t}")
        else:
            # Si T_PRED n'existe pas, utiliser le milieu de la simulation
            mid_idx = len(times) // 2
            before_t = times[max(0, mid_idx - 1)]
            after_t = times[min(len(times) - 1, mid_idx)]
            print(f"  - Scénario {scenario}: Utilisation des snapshots au milieu: t={before_t} et t={after_t}")
        
        # Calculer les moyennes avant et après T_PRED
        before_metrics = metrics_dict[scenario][before_t]
        after_metrics = metrics_dict[scenario][after_t]
        
        # Calculer les variations relatives
        delta_k = calc_percent_change(after_metrics.mean_degree, before_metrics.mean_degree)
        delta_gcc = calc_percent_change(after_metrics.giant_component_size, before_metrics.giant_component_size)
        delta_l = calc_percent_change(after_metrics.avg_path_length, before_metrics.avg_path_length)
        delta_d = calc_percent_change(after_metrics.diameter, before_metrics.diameter)
        delta_c = calc_percent_change(after_metrics.clustering_coefficient, before_metrics.clustering_coefficient)
        
        # Ajouter les résultats au tableau
        results.append({
            'scenario': scenario,
            'before_mean_degree': before_metrics.mean_degree,
            'after_mean_degree': after_metrics.mean_degree,
            'delta_mean_degree': delta_k,
            
            'before_giant_component': before_metrics.giant_component_size,
            'after_giant_component': after_metrics.giant_component_size,
            'delta_giant_component': delta_gcc,
            
            'before_path_length': before_metrics.avg_path_length,
            'after_path_length': after_metrics.avg_path_length,
            'delta_path_length': delta_l,
            
            'before_diameter': before_metrics.diameter,
            'after_diameter': after_metrics.diameter,
            'delta_diameter': delta_d,
            
            'before_clustering': before_metrics.clustering_coefficient,
            'after_clustering': after_metrics.clustering_coefficient,
            'delta_clustering': delta_c,
            
            'before_time': before_t,
            'after_time': after_t,
        })
    
    return pd.DataFrame(results)


def calc_percent_change(new_val, old_val):
    """
    Calcule le pourcentage de variation entre deux valeurs.
    
    Args:
        new_val: Nouvelle valeur
        old_val: Ancienne valeur (référence)
        
    Returns:
        float: Pourcentage de variation
    """
    if old_val == 0:
        return float('inf') if new_val > 0 else 0
    return ((new_val - old_val) / old_val) * 100


def plot_time_series_metrics(metrics_dict, output_dir=OUTDIR):
    """
    Trace l'évolution temporelle des métriques clés pour chaque scénario.
    
    Args:
        metrics_dict: Dictionnaire des métriques avancées par temps et scénario
        output_dir: Répertoire de sortie pour les graphiques
    """
    scenarios = list(metrics_dict.keys())
    colors = {'none': 'blue', 'predictable': 'red', 'random': 'green'}
    labels = {'none': 'Sans panne', 'predictable': 'Pannes prévisibles', 'random': 'Pannes aléatoires'}
    
    # Créer le répertoire avancé si nécessaire
    advanced_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(advanced_dir, exist_ok=True)
    
    # 1. Tracer <k> et |Gmax|/N
    plt.figure(figsize=(12, 5))
    
    # Premier graphique: Degré moyen <k>
    plt.subplot(1, 2, 1)
    for scenario in scenarios:
        times = sorted(metrics_dict[scenario].keys())
        values = [metrics_dict[scenario][t].mean_degree for t in times]
        plt.plot(times, values, '-', color=colors.get(scenario, 'black'), label=f"{labels.get(scenario, scenario)}")
    
    # Ajouter une ligne verticale à T_PRED
    if T_PRED is not None:
        plt.axvline(x=T_PRED, color='purple', linestyle='--', label=f'T_PRED={T_PRED}')
    
    plt.title('Évolution du degré moyen ⟨k⟩')
    plt.xlabel('Temps')
    plt.ylabel('Degré moyen')
    plt.legend()
    plt.grid(True)
    
    # Deuxième graphique: Taille de la composante géante |Gmax|/N
    plt.subplot(1, 2, 2)
    for scenario in scenarios:
        times = sorted(metrics_dict[scenario].keys())
        values = [metrics_dict[scenario][t].giant_component_size for t in times]
        plt.plot(times, values, '-', color=colors.get(scenario, 'black'), label=f"{labels.get(scenario, scenario)}")
    
    # Ajouter une ligne verticale à T_PRED
    if T_PRED is not None:
        plt.axvline(x=T_PRED, color='purple', linestyle='--', label=f'T_PRED={T_PRED}')
    
    # Ajouter une ligne horizontale à 0.5 (seuil critique)
    plt.axhline(y=0.5, color='red', linestyle=':', label='Seuil critique (0.5)')
    
    plt.title('Évolution de la taille de la composante géante |Gₘₐₓ|/N')
    plt.xlabel('Temps')
    plt.ylabel('|Gₘₐₓ|/N')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{advanced_dir}/advanced_connectivity_metrics.png")
    plt.close()
    
    # 2. Tracer ℓ̄ avec D en surimpression
    plt.figure(figsize=(10, 6))
    
    # Première courbe: Longueur moyenne des plus courts chemins ℓ̄
    for scenario in scenarios:
        times = sorted(metrics_dict[scenario].keys())
        path_values = [metrics_dict[scenario][t].avg_path_length for t in times]
        diameter_values = [metrics_dict[scenario][t].diameter for t in times]
        
        # Longueur moyenne des chemins (ligne pleine)
        plt.plot(times, path_values, '-', color=colors.get(scenario, 'black'), 
                 label=f"{labels.get(scenario, scenario)} - ℓ̄")
        
        # Diamètre (ligne pointillée)
        plt.plot(times, diameter_values, '--', color=colors.get(scenario, 'black'), 
                 label=f"{labels.get(scenario, scenario)} - D")
    
    # Ajouter une ligne verticale à T_PRED
    if T_PRED is not None:
        plt.axvline(x=T_PRED, color='purple', linestyle='--', label=f'T_PRED={T_PRED}')
    
    plt.title('Évolution de la longueur moyenne des chemins (ℓ̄) et du diamètre (D)')
    plt.xlabel('Temps')
    plt.ylabel('Nombre de sauts')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{advanced_dir}/advanced_path_metrics.png")
    plt.close()
    
    # 3. Tracer coefficient de clustering C
    plt.figure(figsize=(10, 6))
    
    for scenario in scenarios:
        times = sorted(metrics_dict[scenario].keys())
        values = [metrics_dict[scenario][t].clustering_coefficient for t in times]
        plt.plot(times, values, '-', color=colors.get(scenario, 'black'), 
                 label=f"{labels.get(scenario, scenario)}")
    
    # Ajouter une ligne verticale à T_PRED
    if T_PRED is not None:
        plt.axvline(x=T_PRED, color='purple', linestyle='--', label=f'T_PRED={T_PRED}')
    
    plt.title('Évolution du coefficient de clustering (C)')
    plt.xlabel('Temps')
    plt.ylabel('Coefficient de clustering moyen')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{advanced_dir}/advanced_clustering_metrics.png")
    plt.close()


def plot_component_size_distribution(metrics_dict, output_dir=OUTDIR):
    """
    Trace la distribution des tailles de composantes avant et après T_PRED.
    
    Args:
        metrics_dict: Dictionnaire des métriques avancées par temps et scénario
        output_dir: Répertoire de sortie pour les graphiques
    """
    scenarios = list(metrics_dict.keys())
    
    # Créer le répertoire avancé si nécessaire
    advanced_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(advanced_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5 * len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        times = sorted(metrics_dict[scenario].keys())
        
        # Trouver l'index correspondant à T_PRED
        if T_PRED is not None and T_PRED in times:
            t_pred_idx = times.index(T_PRED)
            before_t = times[max(0, t_pred_idx - 1)]
            after_t = T_PRED
        else:
            # Si T_PRED n'existe pas, utiliser le milieu de la simulation
            mid_idx = len(times) // 2
            before_t = times[max(0, mid_idx - 1)]
            after_t = times[min(len(times) - 1, mid_idx)]
        
        # Récupérer les distributions avant et après
        before_distribution = metrics_dict[scenario][before_t].components_distribution
        after_distribution = metrics_dict[scenario][after_t].components_distribution
        
        # Préparer les données pour le graphique
        before_sizes = list(before_distribution.keys())
        before_counts = list(before_distribution.values())
        after_sizes = list(after_distribution.keys())
        after_counts = list(after_distribution.values())
        
        # Tracer la distribution avant T_PRED
        plt.subplot(len(scenarios), 2, 2*i+1)
        plt.bar(before_sizes, before_counts, alpha=0.7)
        plt.title(f"{scenario} - Distribution avant T={before_t}")
        plt.xlabel('Taille de la composante')
        plt.ylabel('Nombre de composantes')
        plt.grid(True)
        
        # Tracer la distribution après T_PRED
        plt.subplot(len(scenarios), 2, 2*i+2)
        plt.bar(after_sizes, after_counts, color='orange', alpha=0.7)
        plt.title(f"{scenario} - Distribution après T={after_t}")
        plt.xlabel('Taille de la composante')
        plt.ylabel('Nombre de composantes')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{advanced_dir}/component_size_distribution.png")
    plt.close()


def plot_degree_distribution(metrics_dict, output_dir=OUTDIR):
    """
    Trace la distribution des degrés avant et après T_PRED.
    
    Args:
        metrics_dict: Dictionnaire des métriques avancées par temps et scénario
        output_dir: Répertoire de sortie pour les graphiques
    """
    scenarios = list(metrics_dict.keys())
    
    # Créer le répertoire avancé si nécessaire
    advanced_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(advanced_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5 * len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        times = sorted(metrics_dict[scenario].keys())
        
        # Trouver l'index correspondant à T_PRED
        if T_PRED is not None and T_PRED in times:
            t_pred_idx = times.index(T_PRED)
            before_t = times[max(0, t_pred_idx - 1)]
            after_t = T_PRED
        else:
            # Si T_PRED n'existe pas, utiliser le milieu de la simulation
            mid_idx = len(times) // 2
            before_t = times[max(0, mid_idx - 1)]
            after_t = times[min(len(times) - 1, mid_idx)]
        
        # Récupérer les graphes avant et après
        before_degree_avg = metrics_dict[scenario][before_t].mean_degree
        after_degree_avg = metrics_dict[scenario][after_t].mean_degree
        
        # Récupérer les distributions de degré avant et après
        # (Nous devrons modifier la simulation pour stocker ces distributions)
        
        plt.subplot(len(scenarios), 1, i+1)
        plt.title(f"{scenario} - Degré moyen: {before_degree_avg:.2f} → {after_degree_avg:.2f}")
        plt.xlabel('Temps')
        plt.ylabel('Degré moyen')
        plt.grid(True)
        
        # Tracer l'évolution du degré moyen avec surbrillance avant/après T_PRED
        t_values = times
        degree_values = [metrics_dict[scenario][t].mean_degree for t in t_values]
        
        # Marquer les points avant et après
        plt.plot(t_values, degree_values, '-', color='blue', alpha=0.7)
        plt.scatter([before_t], [before_degree_avg], color='green', s=100, label=f'Avant (t={before_t})')
        plt.scatter([after_t], [after_degree_avg], color='red', s=100, label=f'Après (t={after_t})')
        
        # Ajouter une ligne verticale à T_PRED
        if T_PRED is not None:
            plt.axvline(x=T_PRED, color='purple', linestyle='--', label=f'T_PRED={T_PRED}')
        
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{advanced_dir}/degree_evolution.png")
    plt.close()


def export_metrics_table(metrics_df, output_dir=OUTDIR):
    """
    Exporte les métriques au format CSV et génère un tableau formatté pour le rapport.
    
    Args:
        metrics_df: DataFrame contenant les métriques résumées par scénario
        output_dir: Répertoire de sortie pour le fichier CSV
    """
    # Créer le répertoire avancé si nécessaire
    advanced_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(advanced_dir, exist_ok=True)
    
    # Exporter en CSV
    csv_path = os.path.join(advanced_dir, "advanced_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # Créer un joli tableau formatté pour le rapport
    report_path = os.path.join(advanced_dir, "advanced_metrics_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("# Résumé des métriques de robustesse du réseau\n\n")
        
        # Entête du tableau
        f.write("| Scénario | Mesure | Avant | Après | Variation (%) |\n")
        f.write("|----------|--------|-------|-------|---------------|\n")
        
        # Lignes du tableau pour chaque scénario
        for _, row in metrics_df.iterrows():
            scenario = row['scenario']
            scenario_name = {'none': 'Sans panne', 'predictable': 'Prévisible', 'random': 'Aléatoire'}.get(scenario, scenario)
            
            # Degré moyen
            f.write(f"| {scenario_name} | Degré moyen (⟨k⟩) | {row['before_mean_degree']:.3f} | ")
            f.write(f"{row['after_mean_degree']:.3f} | {row['delta_mean_degree']:.2f}% |\n")
            
            # Composante géante
            f.write(f"| {scenario_name} | Taille composante géante (|Gₘₐₓ|/N) | {row['before_giant_component']:.3f} | ")
            f.write(f"{row['after_giant_component']:.3f} | {row['delta_giant_component']:.2f}% |\n")
            
            # Longueur moyenne des chemins
            f.write(f"| {scenario_name} | Long. moy. chemins (ℓ̄) | {row['before_path_length']:.3f} | ")
            f.write(f"{row['after_path_length']:.3f} | {row['delta_path_length']:.2f}% |\n")
            
            # Diamètre
            f.write(f"| {scenario_name} | Diamètre (D) | {row['before_diameter']:.1f} | ")
            f.write(f"{row['after_diameter']:.1f} | {row['delta_diameter']:.2f}% |\n")
            
            # Coefficient de clustering
            f.write(f"| {scenario_name} | Clustering (C) | {row['before_clustering']:.3f} | ")
            f.write(f"{row['after_clustering']:.3f} | {row['delta_clustering']:.2f}% |\n")
        
        f.write("\n\n")
        
        # Analyse des seuils critiques
        f.write("## Analyse des seuils critiques\n\n")
        
        for _, row in metrics_df.iterrows():
            scenario = row['scenario']
            scenario_name = {'none': 'Sans panne', 'predictable': 'Prévisible', 'random': 'Aléatoire'}.get(scenario, scenario)
            
            f.write(f"### Scénario: {scenario_name}\n\n")
            
            # Analyser chaque métrique par rapport aux seuils critiques
            if row['delta_mean_degree'] < -10:
                f.write(f"- **Perte de redondance significative**: Le degré moyen a chuté de {abs(row['delta_mean_degree']):.2f}%, ")
                f.write("dépassant le seuil critique de 10%. Cela indique une forte diminution de la redondance des liens.\n")
            
            if row['after_giant_component'] < 0.5:
                f.write(f"- **Fragmentation majeure**: La taille de la composante géante est tombée à {row['after_giant_component']:.3f}, ")
                f.write("sous le seuil critique de 0.5. Le réseau est maintenant fortement fragmenté.\n")
            
            if row['delta_path_length'] > 100:  # Multiplié par 2
                f.write(f"- **Stretching du réseau**: La longueur moyenne des chemins a augmenté de {row['delta_path_length']:.2f}%, ")
                f.write("indiquant un étirement significatif du réseau (seuil critique: 100%).\n")
            
            if row['delta_diameter'] > 100:  # Multiplié par 2
                f.write(f"- **Augmentation critique du diamètre**: Le diamètre a augmenté de {row['delta_diameter']:.2f}%, ")
                f.write("dépassant le seuil critique de 100%. Les communications de bout en bout sont gravement affectées.\n")
            
            if row['delta_clustering'] < -20:
                f.write(f"- **Rupture des maillages locaux**: Le coefficient de clustering a diminué de {abs(row['delta_clustering']):.2f}%, ")
                f.write("dépassant le seuil critique de 20%. Cela révèle une désagrégation significative des triangles locaux.\n")
            
            f.write("\n")
    
    print(f"  - Métriques avancées exportées vers {csv_path} et {report_path}")
    
    return csv_path, report_path


def analyze_advanced_robustness(swarms, matrixes, scenarios=None):
    """
    Fonction principale pour analyser la robustesse du réseau avec les métriques avancées.
    
    Args:
        swarms: Dictionnaire des swarms par scénario et temps
        matrixes: Dictionnaire des matrices pondérées par scénario et temps
        scenarios: Liste des scénarios à analyser (par défaut: tous)
        
    Returns:
        Tuple: (DataFrame des métriques, Chemin du CSV, Chemin du rapport)
    """
    if scenarios is None:
        scenarios = list(swarms.keys())
    
    print(f"\nConfiguration: T_PRED={T_PRED}")
    
    # Dictionnaire pour stocker toutes les métriques
    all_metrics = {}
    
    # Stocker les graphes aux instants critiques pour analyse détaillée
    key_graphs = {}
    
    # Calculer les métriques pour chaque scénario et temps
    for scenario in scenarios:
        all_metrics[scenario] = {}
        print(f"\n  --- Analyse du scénario {scenario} ---")
        
        # Conserver les graphes aux instants t=49 et t=50 pour analyse ultérieure
        key_graphs[scenario] = {}
        
        for t in sorted(swarms[scenario].keys()):
            # Convertir en graphe NetworkX
            G = nx.Graph()
            
            # Ajouter les nœuds et arêtes depuis la matrice
            n = len(matrixes[scenario][t])
            for i in range(n):
                G.add_node(i)
                for j in range(i):
                    if matrixes[scenario][t][i][j] > 0:
                        G.add_edge(i, j, weight=matrixes[scenario][t][i][j])
            
            # Conserver les graphes aux instants clés
            if t in [49, 50]:
                key_graphs[scenario][t] = G.copy()
            
            # Afficher des informations de debugging pour les instants importants
            if t in [49, 50]:
                print(f"  - t={t}: Graphe construit avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes")
                
                # Vérifier les composantes connexes
                components = list(nx.connected_components(G))
                print(f"    - Nombre de composantes connexes: {len(components)}")
                if components:
                    giant_size = len(max(components, key=len))
                    print(f"    - Taille de la composante géante: {giant_size} ({giant_size/G.number_of_nodes()*100:.1f}%)")
            
            # Calculer les métriques avancées
            all_metrics[scenario][t] = compute_advanced_metrics(G)
            
            # Afficher des informations sur les métriques clés pour les instants importants
            if t in [49, 50]:
                metrics = all_metrics[scenario][t]
                print(f"    - Métriques: <k>={metrics.mean_degree:.2f}, |Gₘₐₓ|/N={metrics.giant_component_size:.3f}, ℓ̄={metrics.avg_path_length:.3f}, D={metrics.diameter}, C={metrics.clustering_coefficient:.3f}")
    
    # Analyse supplémentaire des composantes pour les scénarios avec pannes
    try:
        from simulation.component_analysis import compare_fragmentation_patterns, analyze_critical_nodes
        
        print("\n  --- Analyse détaillée de la fragmentation du réseau ---")
        
        fragmentation_results = {}
        for scenario in scenarios:
            if scenario != 'none' and 49 in key_graphs[scenario] and 50 in key_graphs[scenario]:
                print(f"  - Analyse de fragmentation pour le scénario '{scenario}'...")
                fragmentation_results[scenario] = compare_fragmentation_patterns(
                    key_graphs[scenario][49], key_graphs[scenario][50], scenario
                )
                
                # Identification des nœuds critiques dans le graphe après panne
                print(f"  - Identification des nœuds critiques pour le scénario '{scenario}'...")
                critical_nodes = analyze_critical_nodes(key_graphs[scenario][50], scenario)
                if critical_nodes:
                    print(f"    - {len(critical_nodes)} nœuds critiques identifiés")
                    for i, node in enumerate(critical_nodes[:5]):  # Top 5
                        node_type = "Point d'articulation" if node['is_articulation'] else "Nœud central"
                        print(f"      {i+1}. {node_type} #{node['node_id']} (score: {node['criticality_score']:.3f})")
    except ImportError as e:
        print(f"  - Note: L'analyse détaillée des composantes n'est pas disponible: {e}")
    
    # Générer le tableau de métriques résumé
    metrics_df = extract_metrics_by_scenario(all_metrics)
    
    # Générer les graphiques
    plot_time_series_metrics(all_metrics)
    plot_component_size_distribution(all_metrics)
    plot_degree_distribution(all_metrics)
    
    # Exporter le tableau de métriques
    csv_path, report_path = export_metrics_table(metrics_df)
    
    return metrics_df, csv_path, report_path
