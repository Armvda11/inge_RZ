# simulation/visualize.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
from config import OUTDIR, WINDOW_SIZE

def plot_time_series(stats):
    """
    Génère un graphique d'évolution temporelle des métriques.
    
    Args:
        stats: Dictionnaire des métriques par temps
    """
    ts = sorted(stats.keys())
    d = [stats[t].MeanDegree for t in ts]
    c = [stats[t].MeanClusterCoef for t in ts]
    k = [stats[t].Connexity for t in ts]
    e = [stats[t].Efficiency for t in ts]

    plt.figure(figsize=(8, 4))
    plt.plot(ts, d, label='Degré moyen')
    plt.plot(ts, c, label='Clustering')
    plt.plot(ts, k, label='Connexité')
    plt.plot(ts, e, label='Efficience')
    plt.legend()
    plt.xlabel('Temps')
    plt.ylabel('Valeur')
    plt.title('Évolution des métriques')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/time_series.png")
    plt.close()

def plot_snapshot(t, positions, matrixes, num_sats):
    """
    Génère un snapshot de la topologie à un instant t.
    
    Args:
        t: Instant de la simulation
        positions: Dictionnaire des positions des nœuds
        matrixes: Dictionnaire des matrices pondérées
        num_sats: Nombre de satellites
    """
    nodes, M = positions[t], matrixes[t]
    G = nx.Graph()
    
    for i, n in nodes.items():
        G.add_node(i, pos=(n.pos[0], n.pos[1]))
    
    for i in range(num_sats):
        for j in range(i):
            if i < len(M) and j < len(M[i]) and M[i][j] > 0:
                G.add_edge(i, j, weight=M[i][j])
    
    pos = nx.get_node_attributes(G, 'pos')
    deg = dict(G.degree())
    sizes = [deg.get(i, 0) * 5 for i in G.nodes()]
    weights = [G[u][v].get('weight', 0) for u, v in G.edges()]
    
    # Créer une mapping de couleurs basé sur les poids
    edge_colors = []
    for w in weights:
        if w == 1:
            edge_colors.append('blue')
        elif w == 2:
            edge_colors.append('green')
        elif w == 3:
            edge_colors.append('red')
        else:
            edge_colors.append('black')

    plt.figure(figsize=(7, 7))
    
    # Créer un objet Axes spécifique pour le dessin du graphe
    ax = plt.gca()
    nx.draw(G, pos,
            node_size=sizes,
            edge_color=edge_colors,
            with_labels=False,
            ax=ax)
    
    # Ajouter une légende pour les couleurs des arêtes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='MIN_RANGE'),
        Line2D([0], [0], color='green', lw=2, label='MID_RANGE'),
        Line2D([0], [0], color='red', lw=2, label='MAX_RANGE')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f'Topologie t={t}')
    ax.axis('off')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(f"{OUTDIR}/snapshot_t{t}.png")
    plt.close()

def plot_protocol_comparison(no_failure_results, predictable_results, random_results):
    """
    Génère un graphique comparatif des performances des protocoles DTN dans différents scénarios.
    
    Args:
        no_failure_results: Résultats sans panne
        predictable_results: Résultats avec pannes prévisibles
        random_results: Résultats avec pannes aléatoires
    """
    idx_sw = next((i for i, res in enumerate(no_failure_results['Spray-and-Wait']['values']) 
                 if res['param_value'] == 10), 0)
    idx_pr = next((i for i, res in enumerate(no_failure_results['Prophet']['values']) 
                 if res['param_value'] == 0.5), 0)
    
    # Extraire les delivery ratios
    protocols = ['Spray-and-Wait', 'Epidemic', 'Prophet']
    dr_no_failure = [
        no_failure_results['Spray-and-Wait']['values'][idx_sw]['delivery_ratio'],
        no_failure_results['Epidemic']['values'][0]['delivery_ratio'],
        no_failure_results['Prophet']['values'][idx_pr]['delivery_ratio']
    ]
    
    dr_predictable = [
        predictable_results['Spray-and-Wait']['values'][idx_sw]['delivery_ratio'],
        predictable_results['Epidemic']['values'][0]['delivery_ratio'],
        predictable_results['Prophet']['values'][idx_pr]['delivery_ratio']
    ]
    
    dr_random = [
        random_results['Spray-and-Wait']['values'][idx_sw]['delivery_ratio'],
        random_results['Epidemic']['values'][0]['delivery_ratio'],
        random_results['Prophet']['values'][idx_pr]['delivery_ratio']
    ]
    
    dd_no_failure = [
        no_failure_results['Spray-and-Wait']['values'][idx_sw]['delivery_delay'],
        no_failure_results['Epidemic']['values'][0]['delivery_delay'],
        no_failure_results['Prophet']['values'][idx_pr]['delivery_delay']
    ]
    
    dd_predictable = [
        predictable_results['Spray-and-Wait']['values'][idx_sw]['delivery_delay'],
        predictable_results['Epidemic']['values'][0]['delivery_delay'],
        predictable_results['Prophet']['values'][idx_pr]['delivery_delay']
    ]
    
    dd_random = [
        random_results['Spray-and-Wait']['values'][idx_sw]['delivery_delay'],
        random_results['Epidemic']['values'][0]['delivery_delay'],
        random_results['Prophet']['values'][idx_pr]['delivery_delay']
    ]
    
    # Générer graphique de comparaison
    plt.figure(figsize=(12, 5))
    
    # Delivery Ratio
    plt.subplot(1, 2, 1)
    x = np.arange(len(protocols))
    width = 0.25
    
    plt.bar(x - width, dr_no_failure, width, label='Sans panne')
    plt.bar(x, dr_predictable, width, label='Prévisible')
    plt.bar(x + width, dr_random, width, label='Aléatoire')
    
    plt.xlabel('Protocole DTN')
    plt.ylabel('Delivery Ratio')
    plt.title('Impact des pannes sur le Delivery Ratio')
    plt.xticks(x, protocols)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Delivery Delay
    plt.subplot(1, 2, 2)
    
    # Limiter les valeurs infinies à une valeur max pour l'affichage
    max_delay = max([d for d in dd_no_failure + dd_predictable + dd_random if d != float('inf')], default=100)
    dd_no_failure = [min(d, max_delay * 1.2) for d in dd_no_failure]
    dd_predictable = [min(d, max_delay * 1.2) for d in dd_predictable]
    dd_random = [min(d, max_delay * 1.2) for d in dd_random]
    
    plt.bar(x - width, dd_no_failure, width, label='Sans panne')
    plt.bar(x, dd_predictable, width, label='Prévisible')
    plt.bar(x + width, dd_random, width, label='Aléatoire')
    
    plt.xlabel('Protocole DTN')
    plt.ylabel('Delivery Delay')
    plt.title('Impact des pannes sur le Delivery Delay')
    plt.xticks(x, protocols)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/protocol_robustness.png")
    plt.close()

def plot_protocol_parameter_analysis(protocol_name, results, failure_type):
    """
    Génère des graphiques montrant l'impact des paramètres sur les performances du protocole.
    
    Args:
        protocol_name: Nom du protocole
        results: Résultats des simulations
        failure_type: Type de panne
    """
    if protocol_name == 'Epidemic':
        return  # Epidemic n'a pas de paramètre à analyser
    
    proto_data = results[protocol_name]
    param_name = proto_data['param_name']
    param_values = [res['param_value'] for res in proto_data['values']]
    delivery_ratios = [res['delivery_ratio'] for res in proto_data['values']]
    delivery_delays = [res['delivery_delay'] for res in proto_data['values']]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(param_values, delivery_ratios, 'o-')
    plt.xlabel(param_name)
    plt.ylabel('Delivery Ratio')
    plt.title(f'{protocol_name}: Delivery Ratio vs {param_name}')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(param_values, delivery_delays, 'o-')
    plt.xlabel(param_name)
    plt.ylabel('Delivery Delay')
    plt.title(f'{protocol_name}: Delivery Delay vs {param_name}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{protocol_name}_{failure_type}.png")
    plt.close()

def plot_delay_histograms(metrics_df, output_dir=OUTDIR):
    """
    Trace des histogrammes des délais par protocole et scénario.
    
    Args:
        metrics_df: DataFrame contenant les métriques par paquet
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Créer un sous-répertoire pour les analyses
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Pour chaque protocole
    for protocol in metrics_df['protocol'].unique():
        proto_df = metrics_df[metrics_df['protocol'] == protocol]
        
        # Créer une figure avec plusieurs sous-graphiques (un par scénario)
        fig, axes = plt.subplots(1, len(proto_df['scenario'].unique()), figsize=(15, 5))
        
        for i, scenario in enumerate(sorted(proto_df['scenario'].unique())):
            data = proto_df[proto_df['scenario'] == scenario]['delay']
            
            # Si un seul scénario, axes est un objet unique plutôt qu'une liste
            if len(proto_df['scenario'].unique()) == 1:
                ax = axes
            else:
                ax = axes[i]
                
            ax.hist(data, bins=20, alpha=0.7)
            ax.set_title(f'{scenario}')
            ax.set_xlabel('Délai (unités de temps)')
            ax.set_ylabel('Nombre de paquets')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution des délais - {protocol}')
        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/{protocol}_delay_histograms.png")
        plt.close()
    
    print(f"  - Histogrammes des délais générés dans {analysis_dir}/")

def plot_boxplots(metrics_df, output_dir=OUTDIR):
    """
    Trace des boxplots de délais et nombre de sauts par protocole.
    
    Args:
        metrics_df: DataFrame contenant les métriques par paquet
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Créer un sous-répertoire pour les analyses
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Boxplot des délais par protocole
    plt.figure(figsize=(10, 6))
    boxplot = metrics_df.boxplot(column=['delay'], by=['protocol', 'scenario'], rot=45)
    plt.title('Délais par protocole et scénario')
    plt.suptitle('')  # Supprimer le titre par défaut
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/delay_boxplots.png")
    plt.close()
    
    # Boxplot des sauts par protocole
    plt.figure(figsize=(10, 6))
    boxplot = metrics_df.boxplot(column=['num_hops'], by=['protocol', 'scenario'], rot=45)
    plt.title('Nombre de sauts par protocole et scénario')
    plt.suptitle('')  # Supprimer le titre par défaut
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/hops_boxplots.png")
    plt.close()
    
    print(f"  - Boxplots générés dans {analysis_dir}/")

def plot_delivery_ratio_over_time(metrics_df, window_size=WINDOW_SIZE, output_dir=OUTDIR):
    """
    Trace des courbes temporelles du delivery ratio par fenêtre temporelle.
    
    Args:
        metrics_df: DataFrame contenant les métriques par paquet
        window_size: Taille de la fenêtre de temps pour le calcul du ratio
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Créer un sous-répertoire pour les analyses
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Pour chaque protocole
    for protocol in metrics_df['protocol'].unique():
        proto_df = metrics_df[metrics_df['protocol'] == protocol]
        
        plt.figure(figsize=(10, 6))
        
        for scenario in sorted(proto_df['scenario'].unique()):
            scenario_df = proto_df[proto_df['scenario'] == scenario]
            
            # Définir les fenêtres temporelles
            max_time = scenario_df['t_recv'].max()
            time_windows = np.arange(0, max_time + window_size, window_size)
            delivery_ratios = []
            
            # Pour chaque fenêtre temporelle
            for i in range(len(time_windows) - 1):
                start_time = time_windows[i]
                end_time = time_windows[i+1]
                
                # Compter les paquets émis et reçus dans cette fenêtre
                packets_received = scenario_df[(scenario_df['t_recv'] >= start_time) & 
                                              (scenario_df['t_recv'] < end_time)].shape[0]
                packets_sent = scenario_df[(scenario_df['t_emit'] >= start_time) & 
                                         (scenario_df['t_emit'] < end_time)].shape[0]
                
                # Calculer le ratio de livraison pour cette fenêtre
                ratio = packets_received / max(packets_sent, 1)  # Éviter la division par zéro
                delivery_ratios.append(ratio)
            
            # Tracer la courbe pour ce scénario
            plt.plot(time_windows[1:], delivery_ratios, label=scenario, marker='o')
        
        plt.title(f'Évolution du Delivery Ratio au cours du temps - {protocol}')
        plt.xlabel('Temps')
        plt.ylabel('Delivery Ratio par fenêtre')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/{protocol}_delivery_ratio_over_time.png")
        plt.close()
    
    print(f"  - Courbes de delivery ratio générées dans {analysis_dir}/")