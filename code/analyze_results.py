#!/usr/bin/env python3
# analyze_results.py
"""
Script d'analyse détaillée des résultats de simulation de réseaux satellites.
Ce script charge les fichiers de logs de paquets et calcule des statistiques avancées.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import OUTDIR, WINDOW_SIZE, NUM_RUNS
from simulation.visualize import (
    plot_delay_histograms,
    plot_boxplots,
    plot_delivery_ratio_over_time
)

def format_change(value):
    """Formatte un pourcentage de changement pour éviter les inf%"""
    if np.isinf(value):
        return "pas de changement"
    else:
        return f"{value:.1f}%"

def perform_descriptive_statistics(metrics_df):
    """Calcule des statistiques descriptives pour les métriques par paquet."""
    print("### Statistiques descriptives ###")
    
    # Créer un DataFrame pour stocker les statistiques
    stats_df = pd.DataFrame(columns=[
        'protocol', 'scenario', 'delay_mean', 'delay_std', 
        'delay_min', 'delay_max', 'num_hops_mean', 'num_hops_std'
    ])
    
    # Pour chaque protocole et scénario, calculer les statistiques
    for protocol in metrics_df['protocol'].unique():
        for scenario in metrics_df['scenario'].unique():
            # Filtrer les données
            subset = metrics_df[(metrics_df['protocol'] == protocol) & 
                               (metrics_df['scenario'] == scenario)]
            
            # Si des paquets ont été reçus
            if not subset.empty:
                # Calculer les statistiques
                stats_row = {
                    'protocol': protocol,
                    'scenario': scenario,
                    'delay_mean': subset['delay'].mean(),
                    'delay_std': subset['delay'].std(),
                    'delay_min': subset['delay'].min(),
                    'delay_max': subset['delay'].max(),
                    'num_hops_mean': subset['num_hops'].mean(),
                    'num_hops_std': subset['num_hops'].std()
                }
                
                # Ajouter la ligne au DataFrame
                stats_df = pd.concat([stats_df, pd.DataFrame([stats_row])], ignore_index=True)
    
    # Afficher les statistiques
    print(stats_df)
    
    # Exporter les statistiques
    stats_path = os.path.join(OUTDIR, "analysis", "descriptive_stats.csv")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    print(f"  - Statistiques descriptives exportées vers {stats_path}")
    
    return stats_df

def perform_anova(metrics_df):
    """Réalise une ANOVA pour tester l'effet du protocole et du scénario sur les métriques."""
    print("\n### Tests ANOVA ###")
    
    # ANOVA à un facteur : Effet du protocole sur le délai
    protocols = metrics_df['protocol'].unique()
    protocol_groups = [metrics_df[metrics_df['protocol'] == p]['delay'] for p in protocols]
    
    f_val, p_val = stats.f_oneway(*protocol_groups)
    print(f"ANOVA - Effet du protocole sur le délai: F={f_val:.3f}, p={p_val:.4f}")
    significant = "significatif" if p_val < 0.05 else "non significatif"
    print(f"  → Effet {significant} du protocole sur le délai")
    
    # ANOVA à un facteur : Effet du scénario sur le délai
    scenarios = metrics_df['scenario'].unique()
    scenario_groups = [metrics_df[metrics_df['scenario'] == s]['delay'] for s in scenarios]
    
    f_val, p_val = stats.f_oneway(*scenario_groups)
    print(f"ANOVA - Effet du scénario sur le délai: F={f_val:.3f}, p={p_val:.4f}")
    significant = "significatif" if p_val < 0.05 else "non significatif"
    print(f"  → Effet {significant} du scénario sur le délai")
    
    # Exporter les résultats
    anova_path = os.path.join(OUTDIR, "analysis", "anova_results.txt")
    os.makedirs(os.path.dirname(anova_path), exist_ok=True)
    
    with open(anova_path, 'w') as f:
        f.write("Résultats des tests ANOVA\n")
        f.write("=========================\n\n")
        f.write(f"ANOVA - Effet du protocole sur le délai: F={f_val:.3f}, p={p_val:.4f}\n")
        f.write(f"  → Effet {significant} du protocole sur le délai\n\n")
        
        f.write(f"ANOVA - Effet du scénario sur le délai: F={f_val:.3f}, p={p_val:.4f}\n")
        f.write(f"  → Effet {significant} du scénario sur le délai\n")
    
    print(f"  - Résultats des tests ANOVA exportés vers {anova_path}")

def plot_delay_distribution_comparison(metrics_df):
    """Génère des graphiques comparant la distribution des délais entre protocoles et scénarios."""
    analysis_dir = os.path.join(OUTDIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Comparaison des délais par protocole (tous scénarios confondus)
    plt.figure(figsize=(10, 6))
    for protocol in metrics_df['protocol'].unique():
        data = metrics_df[metrics_df['protocol'] == protocol]['delay']
        plt.hist(data, bins=20, alpha=0.5, label=protocol)
    
    plt.title('Distribution des délais par protocole')
    plt.xlabel('Délai (unités de temps)')
    plt.ylabel('Nombre de paquets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{analysis_dir}/delay_distribution_by_protocol.png")
    plt.close()
    
    # Comparaison des délais par scénario (tous protocoles confondus)
    plt.figure(figsize=(10, 6))
    for scenario in sorted(metrics_df['scenario'].unique()):
        data = metrics_df[metrics_df['scenario'] == scenario]['delay']
        plt.hist(data, bins=20, alpha=0.5, label=scenario)
    
    plt.title('Distribution des délais par scénario')
    plt.xlabel('Délai (unités de temps)')
    plt.ylabel('Nombre de paquets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{analysis_dir}/delay_distribution_by_scenario.png")
    plt.close()
    
    print(f"  - Graphiques des distributions de délais générés dans {analysis_dir}/")

def main():
    """Point d'entrée principal du script d'analyse."""
    print("### Analyse des résultats de simulation ###")
    
    # Créer le répertoire d'analyse si nécessaire
    analysis_dir = os.path.join(OUTDIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Chemins des fichiers à analyser
    packet_logs_path = os.path.join(OUTDIR, "packet_logs.csv")
    per_packet_metrics_path = os.path.join(OUTDIR, "per_packet_metrics.csv")
    
    # Vérifier que les fichiers existent
    if not os.path.exists(packet_logs_path):
        print(f"Erreur: Le fichier {packet_logs_path} n'existe pas.")
        print("Exécutez d'abord main.py pour générer les logs de paquets.")
        return
    
    if not os.path.exists(per_packet_metrics_path):
        print(f"Erreur: Le fichier {per_packet_metrics_path} n'existe pas.")
        print("Exécutez d'abord main.py pour générer les métriques par paquet.")
        return
    
    # Charger les données
    print(f"Chargement des fichiers...")
    packet_logs = pd.read_csv(packet_logs_path)
    metrics_df = pd.read_csv(per_packet_metrics_path)
    
    print(f"  - {len(packet_logs)} logs de paquets chargés")
    print(f"  - {len(metrics_df)} métriques par paquet chargées")
    
    # Réaliser les analyses
    perform_descriptive_statistics(metrics_df)
    perform_anova(metrics_df)
    
    # Générer les visualisations
    print("\n### Génération des visualisations ###")
    plot_delay_histograms(metrics_df)
    plot_boxplots(metrics_df)
    plot_delivery_ratio_over_time(metrics_df, window_size=WINDOW_SIZE)
    plot_delay_distribution_comparison(metrics_df)
    
    print("\n### Analyse terminée ###")
    print(f"Les résultats sont disponibles dans le dossier {analysis_dir}/")

if __name__ == "__main__":
    main()
