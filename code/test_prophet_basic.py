#!/usr/bin/env python3
# test_prophet_basic.py
"""
Script de test pour vérifier le fonctionnement de base du protocole PRoPHET.
Ce test est conçu pour valider que la propagation des messages fonctionne correctement
sans aucune panne, avant d'ajouter la complexité des défaillances dynamiques.
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

def create_static_network(num_nodes, connectivity):
    """
    Crée un réseau statique avec la connectivité spécifiée.
    
    Args:
        num_nodes: Nombre de nœuds dans le réseau
        connectivity: Probabilité de connexion entre deux nœuds (0-1)
    
    Returns:
        dict: Dictionnaire d'adjacence représentant le réseau
    """
    # Utiliser un graphe non orienté pour assurer la symétrie des liens
    G = nx.erdos_renyi_graph(num_nodes, connectivity)
    
    # Convertir en dictionnaire d'adjacence
    adjacency = {i: set(G.neighbors(i)) for i in range(num_nodes)}
    return adjacency

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
    for _ in range(time_window):
        # Varier légèrement la connectivité pour simuler le mouvement des nœuds
        current_connectivity = base_connectivity * (0.8 + 0.4 * random.random())
        networks.append(create_static_network(num_nodes, current_connectivity))
    
    return networks

def run_test_basic_prophet():
    """
    Exécute un test de base du protocole PRoPHET sans pannes.
    """
    # Paramètres de simulation
    num_nodes = 20
    source = 0
    destination = num_nodes - 1
    time_window = 30
    base_connectivity = 0.2  # 20% de chance que deux nœuds soient connectés
    
    # Générer les topologies de réseau
    print("Génération des topologies de réseau...")
    networks = create_dynamic_network(num_nodes, base_connectivity, time_window)
    
    # Paramètres PRoPHET
    p_init = 0.7
    gamma = 0.98
    beta = 0.25
    
    # Créer l'instance PRoPHET avec le mode debug activé
    print(f"Initialisation du protocole PRoPHET avec p_init={p_init}, gamma={gamma}, beta={beta}")
    prophet = Prophet(num_nodes, p_init, destination, source, gamma, beta)
    prophet.set_debug_mode(True)  # Activer le mode debug
    
    # Exécuter la simulation
    print("Exécution de la simulation...")
    delivery_time = None
    
    copies_per_step = []
    
    for t in range(time_window):
        # Exécuter un pas de simulation
        prophet.step(t, networks[t])
        
        # Compter les copies actives
        active_copies = sum(1 for c in prophet.copies.values() if c > 0)
        copies_per_step.append(active_copies)
        
        # Vérifier si le message a été livré
        if prophet.message_delivered and delivery_time is None:
            delivery_time = t
            print(f"Message livré à t={t}!")
            # Continuer la simulation pour voir l'évolution complète
    
    # Afficher les résultats
    print("\n" + "=" * 50)
    print("RÉSULTATS DU TEST DE BASE PROPHET")
    print("=" * 50)
    
    # Tableau des métriques
    metrics = [
        ["Taux de livraison", "100%" if prophet.message_delivered else "0%"],
        ["Temps de livraison", str(delivery_time) if delivery_time is not None else "Non livré"],
        ["Nombre de sauts", prophet.num_hops.get(destination, "N/A") if prophet.message_delivered else "N/A"],
        ["Overhead ratio", f"{prophet.overhead_ratio():.2f}"],
        ["Total copies créées", prophet.total_copies_created]
    ]
    print(tabulate(metrics, headers=["Métrique", "Valeur"], tablefmt="grid"))
    
    # Afficher l'historique des copies
    print("\nHistorique des copies:")
    copies_data = []
    for entry in prophet.copies_history:
        t = entry['t']
        copies = entry['copies']
        active_nodes = [node for node, has_copy in copies.items() if has_copy > 0]
        copies_data.append([t, len(active_nodes), str(active_nodes)])
    
    print(tabulate(copies_data, headers=["Temps", "Nombre de copies", "Nœuds actifs"], tablefmt="grid"))
    
    # Visualiser l'évolution des copies
    plt.figure(figsize=(10, 6))
    plt.plot(range(time_window), copies_per_step, 'b-', marker='o')
    plt.xlabel('Temps')
    plt.ylabel('Nombre de copies actives')
    plt.title('Évolution du nombre de copies dans le réseau')
    plt.grid(True)
    
    # Ajouter une ligne verticale pour le temps de livraison si le message a été livré
    if delivery_time is not None:
        plt.axvline(x=delivery_time, color='r', linestyle='--', 
                   label=f'Livraison à t={delivery_time}')
        plt.legend()
    
    # Sauvegarder le graphique
    plt.savefig(os.path.join(OUTDIR, "prophet_basic_test_copies.png"))
    plt.close()
    
    print(f"\nGraphique sauvegardé dans {os.path.join(OUTDIR, 'prophet_basic_test_copies.png')}")
    
    # Visualiser la matrice de probabilité finale
    plt.figure(figsize=(10, 8))
    plt.imshow(prophet.probability_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Probabilité')
    plt.xlabel('Nœud destination')
    plt.ylabel('Nœud source')
    plt.title('Matrice de probabilité finale PRoPHET')
    
    # Sauvegarder la matrice
    plt.savefig(os.path.join(OUTDIR, "prophet_probability_matrix.png"))
    plt.close()
    
    print(f"Matrice de probabilité sauvegardée dans {os.path.join(OUTDIR, 'prophet_probability_matrix.png')}")

if __name__ == "__main__":
    run_test_basic_prophet()
