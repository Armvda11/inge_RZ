#!/usr/bin/env python3
# test_prophet_baseline.py
"""
Script de test pour établir une baseline du protocole PROPHET sans aucune panne.
Ce test permet de vérifier les performances optimales du protocole avant d'introduire
des pannes, et d'analyser l'impact des différents paramètres sur les performances.
"""
import sys
import os
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocols.prophet import Prophet
from config import OUTDIR

def create_dynamic_network(num_nodes, base_connectivity, time_window):
    """
    Crée une série de réseaux dynamiques qui changent au fil du temps.
    
    Args:
        num_nodes: Nombre de nœuds dans le réseau
        base_connectivity: Connectivité de base entre les nœuds
        time_window: Nombre de pas de temps à simuler
    
    Returns:
        list: Liste des dictionnaires d'adjacence pour chaque pas de temps
    """
    networks = []
    
    # Pour chaque pas de temps, générer une nouvelle topologie
    for t in range(time_window):
        # Varier légèrement la connectivité pour simuler le mouvement des nœuds
        current_connectivity = base_connectivity * (0.8 + 0.4 * random.random())
        
        # Utiliser un graphe non orienté pour assurer la symétrie des liens
        G = nx.erdos_renyi_graph(num_nodes, current_connectivity, seed=42+t)
        
        # Convertir en dictionnaire d'adjacence
        adjacency = {i: set(G.neighbors(i)) for i in range(num_nodes)}
        networks.append(adjacency)
    
    return networks

def analyze_prophet_parameters():
    """
    Analyse l'impact des différents paramètres de PROPHET sur les performances.
    Cette fonction teste plusieurs combinaisons de p_init, gamma et beta.
    """
    # Paramètres de simulation
    num_nodes = 40
    source = 0
    destination = num_nodes - 1
    time_window = 50
    base_connectivity = 0.15  # Connectivité réduite pour tester la robustesse
    
    # Paramètres à tester
    p_init_values = [0.5, 0.7, 0.9]
    gamma_values = [0.98, 0.99, 0.995]
    beta_values = [0.25, 0.5]
    
    # Générer les topologies de réseau (identiques pour tous les tests)
    print("Génération des topologies de réseau...")
    networks = create_dynamic_network(num_nodes, base_connectivity, time_window)
    
    # Résultats
    results = []
    
    # Tester toutes les combinaisons de paramètres
    for p_init in p_init_values:
        for gamma in gamma_values:
            for beta in beta_values:
                print(f"Test avec p_init={p_init}, gamma={gamma}, beta={beta}")
                
                # Créer l'instance PROPHET
                prophet = Prophet(num_nodes, p_init, destination, source, gamma, beta)
                prophet.set_debug_mode(False)  # Désactiver le debug pour éviter trop de sortie
                
                # Exécuter la simulation
                delivery_time = None
                
                for t in range(time_window):
                    # Exécuter un pas de simulation
                    prophet.step(t, networks[t])
                    
                    # Vérifier si le message a été livré
                    if prophet.message_delivered and delivery_time is None:
                        delivery_time = t
                        # Continuer la simulation pour voir l'évolution complète
                
                # Analyser l'état final des probabilités
                prob_stats = prophet.analyze_probability_state()
                
                # Collecter les résultats
                result = {
                    "p_init": p_init,
                    "gamma": gamma,
                    "beta": beta,
                    "delivered": prophet.message_delivered,
                    "delivery_time": delivery_time if delivery_time is not None else "Non livré",
                    "hops": prophet.num_hops.get(destination, "N/A") if prophet.message_delivered else "N/A",
                    "copies": prophet.total_copies_created,
                    "overhead": prophet.overhead_ratio(),
                    "avg_prob": prob_stats["avg_to_dest"],
                    "max_prob": prob_stats["max_to_dest"],
                    "nodes_above_threshold": prob_stats["nodes_above_threshold"]
                }
                results.append(result)
    
    # Afficher les résultats sous forme de tableau
    print("\n" + "=" * 80)
    print("RÉSULTATS DE L'ANALYSE DES PARAMÈTRES PROPHET")
    print("=" * 80 + "\n")
    
    # Trier par délai de livraison puis par overhead
    def sort_key(r):
        time = r["delivery_time"] if isinstance(r["delivery_time"], int) else float('inf')
        return (not r["delivered"], time, r["overhead"])
    
    sorted_results = sorted(results, key=sort_key)
    
    # Préparer le tableau
    table_data = []
    for r in sorted_results:
        table_data.append([
            f"{r['p_init']:.2f}",
            f"{r['gamma']:.3f}",
            f"{r['beta']:.2f}",
            "✅" if r["delivered"] else "❌",
            r["delivery_time"],
            r["hops"],
            r["copies"],
            f"{r['overhead']:.2f}" if isinstance(r["overhead"], float) else r["overhead"],
            f"{r['avg_prob']:.4f}",
            f"{r['max_prob']:.4f}",
            r["nodes_above_threshold"]
        ])
    
    headers = ["P_init", "Gamma", "Beta", "Livré", "Délai", "Sauts", "Copies", "Overhead", 
               "Prob. moy", "Prob. max", "Nœuds > seuil"]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Sauvegarder les résultats
    output_dir = os.path.join(OUTDIR, "protocols", "prophet_parameter_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder le tableau en CSV
    with open(os.path.join(output_dir, "prophet_parameters_baseline.csv"), "w") as f:
        f.write(",".join(headers) + "\n")
        for row in table_data:
            f.write(",".join(str(cell) for cell in row) + "\n")
    
    print(f"\nRésultats sauvegardés dans {os.path.join(output_dir, 'prophet_parameters_baseline.csv')}")
    
    # Créer un graphique pour visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Filtrer les résultats livrés seulement
    delivered_results = [r for r in sorted_results if r["delivered"]]
    
    if delivered_results:
        # Ajouter un graphique pour le délai vs. overhead
        plt.subplot(1, 2, 1)
        for r in delivered_results:
            label = f"P={r['p_init']}, γ={r['gamma']}, β={r['beta']}"
            plt.scatter(r["delivery_time"], r["overhead"], s=100, alpha=0.7, label=label)
        
        plt.xlabel("Délai de livraison")
        plt.ylabel("Overhead ratio")
        plt.title("Délai vs Overhead")
        plt.grid(True, alpha=0.3)
        
        # Ajouter un graphique pour le nombre de copies vs. probabilité moyenne
        plt.subplot(1, 2, 2)
        for r in delivered_results:
            label = f"P={r['p_init']}, γ={r['gamma']}, β={r['beta']}"
            plt.scatter(r["copies"], r["avg_prob"], s=100, alpha=0.7)
        
        plt.xlabel("Nombre de copies")
        plt.ylabel("Probabilité moyenne vers destination")
        plt.title("Copies vs Probabilité moyenne")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prophet_parameter_analysis.png"))
        plt.close()
        
        print(f"Graphiques sauvegardés dans {os.path.join(output_dir, 'prophet_parameter_analysis.png')}")
    else:
        print("Aucun message n'a été livré, impossible de créer des graphiques.")

if __name__ == "__main__":
    analyze_prophet_parameters()
