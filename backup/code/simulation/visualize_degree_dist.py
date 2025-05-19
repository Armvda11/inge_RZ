# simulation/visualize_degree_dist.py
import matplotlib.pyplot as plt
import numpy as np
import os
from config import OUTDIR

def plot_degree_distribution(degree_dist, title="Distribution des degrés", filename=None):
    """
    Trace la distribution des degrés d'un graphe.
    
    Args:
        degree_dist: Dictionnaire {degré: fréquence}
        title: Titre du graphique
        filename: Nom du fichier pour sauvegarde (sans extension)
    """
    if not degree_dist:
        print("Aucune distribution de degrés à afficher")
        return
    
    # Trier les degrés par ordre croissant
    degrees = sorted(degree_dist.keys())
    frequencies = [degree_dist[d] for d in degrees]
    
    # Calcul des statistiques
    total_nodes = sum(frequencies)
    
    plt.figure(figsize=(10, 6))
    
    # Tracer la distribution en barres
    plt.bar(degrees, frequencies, width=0.7, alpha=0.7)
    
    # Configurer les axes et titres
    plt.xlabel('Degré (k)')
    plt.ylabel('Nombre de nœuds')
    plt.title(f"{title}\n{total_nodes} nœuds au total")
    
    # Configurer l'axe X pour qu'il affiche uniquement les valeurs entières
    plt.xticks(np.arange(min(degrees), max(degrees)+1, 1.0))
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Sauvegarder le graphique si un nom de fichier est fourni
    if filename:
        output_path = f"{OUTDIR}/{filename}.png"
        plt.savefig(output_path)
        print(f"  - Distribution des degrés sauvegardée dans {output_path}")
    
    plt.close()

def compare_degree_distributions(before_dist, after_dist, event_name="pannes", filename=None):
    """
    Compare deux distributions de degrés (avant/après un événement).
    
    Args:
        before_dist: Dictionnaire {degré: fréquence} avant l'événement
        after_dist: Dictionnaire {degré: fréquence} après l'événement
        event_name: Description de l'événement (par exemple "pannes")
        filename: Nom du fichier pour sauvegarde (sans extension)
    """
    # Initialiser des dictionnaires vides s'ils sont None
    before_dist = before_dist or {}
    after_dist = after_dist or {}
    
    # Vérifier si les distributions sont vraiment vides (pas de valeurs)
    if len(before_dist) == 0 and len(after_dist) == 0:
        print(f"Les deux distributions sont vides, impossible de comparer ({event_name})")
        return
    
    # Si une des distributions est vide mais pas l'autre, on peut quand même faire une comparaison
    if len(before_dist) == 0:
        print(f"Distribution 'avant' vide, seule la distribution 'après' sera affichée ({event_name})")
    
    if len(after_dist) == 0:
        print(f"Distribution 'après' vide, seule la distribution 'avant' sera affichée ({event_name})")
    
    # Créer l'ensemble de tous les degrés présents dans les deux distributions
    all_degrees = sorted(set(list(before_dist.keys()) + list(after_dist.keys())))
    
    # Obtenir les fréquences (utiliser 0 si le degré n'existe pas dans la distribution)
    before_freq = [before_dist.get(d, 0) for d in all_degrees]
    after_freq = [after_dist.get(d, 0) for d in all_degrees]
    
    # Calcul des statistiques
    total_before = sum(before_freq)
    total_after = sum(after_freq)
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Distribution absolue
    ax1.bar(np.array(all_degrees)-0.2, before_freq, width=0.4, alpha=0.7, label='Avant')
    ax1.bar(np.array(all_degrees)+0.2, after_freq, width=0.4, alpha=0.7, label='Après')
    
    ax1.set_xlabel('Degré (k)')
    ax1.set_ylabel('Nombre de nœuds')
    ax1.set_title(f"Distribution des degrés avant/après {event_name}\n(Avant: {total_before} nœuds, Après: {total_after} nœuds)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: Distribution relative (pourcentage)
    before_pct = [f / total_before * 100 if total_before > 0 else 0 for f in before_freq]
    after_pct = [f / total_after * 100 if total_after > 0 else 0 for f in after_freq]
    
    ax2.bar(np.array(all_degrees)-0.2, before_pct, width=0.4, alpha=0.7, label='Avant')
    ax2.bar(np.array(all_degrees)+0.2, after_pct, width=0.4, alpha=0.7, label='Après')
    
    ax2.set_xlabel('Degré (k)')
    ax2.set_ylabel('Pourcentage de nœuds (%)')
    ax2.set_title(f"Distribution relative des degrés avant/après {event_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Sauvegarder la comparaison si un nom de fichier est fourni
    if filename:
        output_path = f"{OUTDIR}/{filename}.png"
        plt.savefig(output_path)
        print(f"  - Comparaison des distributions sauvegardée dans {output_path}")
    
    plt.close()
