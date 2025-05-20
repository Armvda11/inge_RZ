#!/usr/bin/env python3
# test_spray_and_wait.py
"""
Script de test pour le protocole Spray-and-Wait.
Ce script démontre l'utilisation du protocole dans un petit réseau dynamique.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTDIR
from protocols.spray_and_wait import SprayAndWait

def create_test_network(t: int, num_nodes: int = 10) -> dict[int, set[int]]:
    """
    Crée un réseau test pour simuler des connexions dynamiques.
    
    Args:
        t (int): L'instant de temps (pour varier les connexions)
        num_nodes (int): Nombre de nœuds dans le réseau
        
    Returns:
        dict[int, set[int]]: Dictionnaire d'adjacence
    """
    # Initialiser le réseau vide
    adjacency = {i: set() for i in range(num_nodes)}
    
    # Connecter des nœuds en fonction du temps
    # On utilise une méthode simple basée sur t pour avoir des connexions variables
    for i in range(num_nodes):
        # Créer des connexions variables en fonction du temps
        for j in range(num_nodes):
            if i != j:
                # Condition pour créer une connexion basée sur le temps (exemple: modulo)
                # Cette formule crée des modèles répétitifs basés sur le temps
                if (i + j + t) % 3 == 0:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
    
    return adjacency

def test_spray_and_wait():
    """
    Test du protocole Spray-and-Wait avec simulation des résultats.
    """
    print("=== Test du protocole Spray-and-Wait ===")
    
    # Paramètres de simulation
    num_nodes = 10
    max_steps = 20
    L_values = [2, 4, 8]  # Nombre initial de copies
    
    # Dossier de sortie
    output_dir = f"{OUTDIR}/protocols/spray_and_wait_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Stocker les résultats pour chaque valeur de L
    results = []
    
    # Tester différentes valeurs de L
    for L in L_values:
        # Initialiser protocole Spray-and-Wait
        source = 0
        destination = 9  # Nœud de destination
        protocol = SprayAndWait(num_nodes, L, destination, source, binary=True)
        
        # Suivi de la distribution des copies
        copies_history = []
        
        # Exécuter la simulation
        print(f"\nTest avec L={L}:")
        print(f"Source: {source}, Destination: {destination}")
        print(f"{'t':>3} | {'Copies par nœud':^40} | {'Livré':<7}")
        print("-" * 55)
        
        for t in range(max_steps):
            # Générer le réseau pour l'instant t
            adjacency = create_test_network(t, num_nodes)
            
            # Avant de faire un pas, enregistrer l'état actuel
            copies_history.append(protocol.copies.copy())
            
            # Exécuter un pas de simulation
            protocol.step(t, adjacency)
            
            # Afficher la distribution des copies
            copies_str = " ".join([f"{i}:{protocol.copies.get(i, 0)}" for i in range(num_nodes)])
            delivered = destination in protocol.delivered_at
            delivered_str = f"Oui (t={protocol.delivered_at.get(destination, 'N/A')})" if delivered else "Non"
            print(f"{t:3d} | {copies_str:<40} | {delivered_str:<7}")
            
            # Arrêter si le message est livré
            if delivered:
                # Ajouter une ligne finale pour l'historique
                copies_history.append(protocol.copies.copy())
                break

        # Calculer les métriques finales
        delivery_ratio = protocol.delivery_ratio()
        delivery_delay = protocol.delivery_delay()
        
        if delivery_delay != float('inf'):
            total_copies = sum(protocol.copies.values())
            results.append({
                'L': L,
                'delivered': destination in protocol.delivered_at,
                'delivery_time': protocol.delivered_at.get(destination, float('inf')),
                'total_copies': total_copies,
                'overhead': total_copies / (1 if delivery_ratio == 0 else delivery_ratio)
            })
            
            print(f"\nRésultats pour L={L}:")
            print(f"  - Message livré: {delivery_ratio > 0}")
            print(f"  - Délai de livraison: {delivery_delay}")
            print(f"  - Nombre total de copies créées: {total_copies}")
            
            # Visualiser l'évolution des copies dans le temps
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Convertir l'historique des copies en dataframe pour visualisation
            df = pd.DataFrame(copies_history)
            
            # Créer un graphique à lignes pour chaque nœud
            for i in range(num_nodes):
                if i in df.columns:
                    ax.plot(df.index, df[i], marker='o', label=f"Nœud {i}")
            
            ax.set_title(f"Distribution des copies dans le temps (L={L})")
            ax.set_xlabel("Temps (t)")
            ax.set_ylabel("Nombre de copies")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Sauvegarder la figure
            plt.tight_layout()
            plt.savefig(f"{output_dir}/copies_distribution_L{L}.png", dpi=300)
            print(f"Figure sauvegardée dans {output_dir}/copies_distribution_L{L}.png")
        else:
            print(f"\nRésultats pour L={L}:")
            print(f"  - Message non livré dans le délai imparti")
    
    # Créer un tableau comparatif
    if results:
        df_results = pd.DataFrame(results)
        print("\nTableau comparatif:")
        print(df_results.to_string(index=False))
        
        # Sauvegarder les résultats en CSV
        df_results.to_csv(f"{output_dir}/resultats.csv", index=False)
        print(f"Résultats sauvegardés dans {output_dir}/resultats.csv")
        
        # Créer un graphique comparatif
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graphique du délai de livraison
        ax1.bar(df_results['L'].astype(str), df_results['delivery_time'])
        ax1.set_title('Délai de livraison')
        ax1.set_xlabel('L (nombre initial de copies)')
        ax1.set_ylabel('Temps')
        ax1.grid(True, alpha=0.3)
        
        # Graphique de l'overhead
        ax2.bar(df_results['L'].astype(str), df_results['total_copies'])
        ax2.set_title('Total des copies créées')
        ax2.set_xlabel('L (nombre initial de copies)')
        ax2.set_ylabel('Nombre de copies')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparaison_spray_and_wait.png", dpi=300)
        print(f"Graphique comparatif sauvegardé dans {output_dir}/comparaison_spray_and_wait.png")

if __name__ == "__main__":
    test_spray_and_wait()
    print("\nTest terminé avec succès!")
