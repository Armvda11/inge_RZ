#!/usr/bin/env python3
# analyze_robustness.py
"""
Script spécialisé pour l'analyse avancée de la robustesse du réseau satellite.
Ce script calcule et visualise des métriques de robustesse pour quantifier l'impact des pannes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy
from config import PATH, MAXTEMPS, MAX_RANGE, MID_RANGE, MIN_RANGE, OUTDIR, T_PRED, N_PRED, P_FAIL, CONFIG
from data.loader import load_data
from models.swarm import Swarm
from simulation.metrics import get_weighted_matrix
from simulation.failure import NodeFailureManager
from simulation.advanced_metrics import analyze_advanced_robustness
from analyze_results import format_change

def main():
    """Point d'entrée principal pour l'analyse de robustesse."""
    print("### Analyse avancée de la robustesse du réseau satellite ###")
    
    # Chargement des données
    print("Chargement des données...")
    positions, swarms, matrixes, adjacency, num_sats = load_data(PATH, MAXTEMPS, {
        'min': MIN_RANGE,
        'mid': MID_RANGE,
        'max': MAX_RANGE
    })
    print(f"  - Données chargées pour {MAXTEMPS} instants et {num_sats} satellites")
    
    # Identifier le meilleur instant en termes d'efficacité pour les pannes prévisibles
    from simulation.metrics import analyze_single_graph
    print("Identification du meilleur instant pour les pannes prévisibles...")
    stats = {t: analyze_single_graph(swarms[t], matrixes[t]) for t in range(MAXTEMPS)}
    connected_instants = [t for t, m in stats.items() if m.Connexity > 0.9]
    
    if connected_instants:
        best_t = max(connected_instants, key=lambda t: stats[t].Efficiency)
    else:
        best_t = 0
    
    print(f"  - Meilleur instant identifié: t={best_t}")
    
    # Préparer les données pour l'analyse avancée (structure par scénario)
    print("Préparation des données par scénario de panne...")
    advanced_swarms = {
        'none': {t: swarms[t] for t in range(MAXTEMPS)},        # Swarms originaux (sans panne)
        'predictable': {},                                       # Swarms avec pannes prévisibles
        'random': {}                                             # Swarms avec pannes aléatoires
    }
    
    advanced_matrixes = {
        'none': {t: matrixes[t] for t in range(MAXTEMPS)},       # Matrices originales (sans panne)
        'predictable': {},                                       # Matrices avec pannes prévisibles
        'random': {}                                             # Matrices avec pannes aléatoires
    }
    
    # Récupérer les swarms et matrices modifiés pour les scénarios avec pannes
    for t in range(MAXTEMPS):
        # Simuler les pannes prévisibles à ce temps
        failure_mgr_pred = NodeFailureManager(dict(positions))
        failure_mgr_pred.setup_predictable_failures(T_PRED, N_PRED, 'centralite', 
                                              matrixes[best_t], swarms[best_t])
        
        # Appliquer les pannes prévisibles
        nodes_pred = copy.deepcopy(positions[t])
        failure_mgr_pred.apply_failures(t, nodes_pred)
        active_nodes_pred = failure_mgr_pred.get_active_nodes(nodes_pred)
        
        # Reconstruire swarm et matrice pour les pannes prévisibles
        sw_pred = Swarm(MAX_RANGE, list(active_nodes_pred.values()))
        advanced_swarms['predictable'][t] = sw_pred
        advanced_matrixes['predictable'][t] = get_weighted_matrix(sw_pred, MIN_RANGE, MID_RANGE, MAX_RANGE)
        
        # Log spécial pour t=49 et t=50 (avant et après T_PRED)
        if t == 49 or t == 50:
            print(f"  - Predictable snapshot t={t}: {len(active_nodes_pred)} nœuds actifs, {len(sw_pred.edges)} liens")
        
        # Simuler les pannes aléatoires à ce temps
        failure_mgr_rand = NodeFailureManager(dict(positions))
        # Configuration explicite des pannes aléatoires avec la probabilité depuis config.py
        failure_mgr_rand.setup_random_failures(P_FAIL)
        
        # Pour uniformité des logs, afficher les paramètres
        if t == 49:
            print(f"  - Configuration de pannes aléatoires avec p={P_FAIL}")
        
        # Appliquer les pannes aléatoires
        nodes_rand = copy.deepcopy(positions[t])
        failure_mgr_rand.apply_failures(t, nodes_rand)
        active_nodes_rand = failure_mgr_rand.get_active_nodes(nodes_rand)
        
        # Reconstruire swarm et matrice pour les pannes aléatoires
        sw_rand = Swarm(MAX_RANGE, list(active_nodes_rand.values()))
        advanced_swarms['random'][t] = sw_rand
        advanced_matrixes['random'][t] = get_weighted_matrix(sw_rand, MIN_RANGE, MID_RANGE, MAX_RANGE)
        
        # Log spécial pour t=49 et t=50 (avant et après T_PRED)
        if t == 49 or t == 50:
            print(f"  - Random snapshot t={t}: {len(active_nodes_rand)} nœuds actifs, {len(sw_rand.edges)} liens")
    
    # Exécuter l'analyse avancée de robustesse
    print("\nCalcul des métriques avancées de robustesse...")
    try:
        metrics_df, csv_path, report_path = analyze_advanced_robustness(
            advanced_swarms, advanced_matrixes
        )
        
        # Afficher un résumé des résultats clés
        print("\n  - Résumé des impacts critiques par scénario:")
        
        scenarios = ['none', 'predictable', 'random']
        scenario_names = {'none': 'Sans panne', 'predictable': 'Pannes prévisibles', 'random': 'Pannes aléatoires'}
        
        for scenario in scenarios:
            row = metrics_df[metrics_df['scenario'] == scenario].iloc[0]
            scenario_name = scenario_names.get(scenario, scenario)
            print(f"\n    {scenario_name}:")
            
            # Degré moyen (redondance)
            print(f"      - Degré moyen: {row['before_mean_degree']:.3f} → {row['after_mean_degree']:.3f} ({format_change(row['delta_mean_degree'])})")
            
            # Taille de la composante géante (fragmentation)
            print(f"      - Composante géante: {row['before_giant_component']:.3f} → {row['after_giant_component']:.3f} ({format_change(row['delta_giant_component'])})")
            
            # Longueur moyenne des chemins (stretching)
            print(f"      - Longueur des chemins: {row['before_path_length']:.3f} → {row['after_path_length']:.3f} ({format_change(row['delta_path_length'])})")
            
            # Diamètre (worst case)
            print(f"      - Diamètre: {row['before_diameter']:.1f} → {row['after_diameter']:.1f} ({format_change(row['delta_diameter'])})")
            
            # Coefficient de clustering (maillage local)
            print(f"      - Clustering: {row['before_clustering']:.3f} → {row['after_clustering']:.3f} ({format_change(row['delta_clustering'])})")
        
        print(f"\n  - Détails complets disponibles dans: {report_path}")
        print(f"  - Données exportées au format CSV: {csv_path}")
        print(f"  - Graphiques générés dans le dossier: {OUTDIR}/advanced/")
        
    except Exception as e:
        print(f"  - Erreur lors de l'analyse avancée: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n### Analyse terminée ###")


if __name__ == "__main__":
    main()
