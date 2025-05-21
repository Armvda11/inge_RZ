#!/usr/bin/env python3
# analyze_topologie.py
"""
Script simplifié pour l'analyse topologique du réseau de satellites.
Se concentre sur les 4 métriques clés pour l'évaluation de la robustesse du réseau:
    1. Degré moyen ⟨k⟩
    2. Taille de la composante géante |Gₘₐₓ|/N
    3. Longueur moyenne des plus courts chemins ℓ̄
    4. Coefficient de clustering C
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy
import traceback
from config import PATH, MAXTEMPS, MAX_RANGE, MID_RANGE, MIN_RANGE, OUTDIR, T_PRED, N_PRED, P_FAIL
from data.loader import load_data
from models.swarm import Swarm
from models.node import Node

# Fonction pour remplacer get_weighted_matrix pour être indépendant
def get_weighted_matrix(swarm, min_range, mid_range, max_range):
    """
    Génère une matrice pondérée en fonction de la portée.
    Reproduit la fonctionnalité de simulation.metrics.get_weighted_matrix
    
    Args:
        swarm: L'essaim de satellites
        min_range: Portée minimale
        mid_range: Portée moyenne
        max_range: Portée maximale
    
    Returns:
        numpy.ndarray: Matrice pondérée
    """
    n = len(swarm.nodes)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                node_i = swarm.nodes[i]
                node_j = swarm.nodes[j]
                distance = node_i.distance_to(node_j)
                
                if distance <= min_range:
                    matrix[i][j] = 3  # Lien fort
                elif distance <= mid_range:
                    matrix[i][j] = 2  # Lien moyen
                elif distance <= max_range:
                    matrix[i][j] = 1  # Lien faible
    
    return matrix

# Assurez-vous que le dossier de sortie existe
os.makedirs(f"{OUTDIR}/topologie", exist_ok=True)

def calculer_metriques(G):
    """
    Calcule les 4 métriques clés pour un graphe donné.
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        dict: Dictionnaire contenant les métriques
    """
    # Vérifier si le graphe est vide
    if G.number_of_nodes() == 0:
        return {
            "degre_moyen": 0,
            "taille_composante_geante": 0,
            "longueur_moyenne_chemins": 0,
            "coefficient_clustering": 0
        }
    
    # 1. Degré moyen ⟨k⟩
    degre_moyen = sum(d for _, d in G.degree()) / G.number_of_nodes()
    
    # 2. Taille de la composante géante |Gₘₐₓ|/N
    composantes = list(nx.connected_components(G))
    if not composantes:
        taille_composante_geante = 0
    else:
        taille_plus_grande = len(max(composantes, key=len))
        taille_composante_geante = taille_plus_grande / G.number_of_nodes()
    
    # 3. Longueur moyenne des plus courts chemins ℓ̄
    if nx.is_connected(G):
        longueur_moyenne_chemins = nx.average_shortest_path_length(G)
    else:
        # Calcul sur la composante géante uniquement si le graphe est fragmenté
        if composantes:
            composante_geante = max(composantes, key=len)
            sous_graphe = G.subgraph(composante_geante)
            longueur_moyenne_chemins = nx.average_shortest_path_length(sous_graphe)
        else:
            longueur_moyenne_chemins = float('inf')
    
    # 4. Coefficient de clustering C
    coefficient_clustering = nx.average_clustering(G)
    
    return {
        "degre_moyen": degre_moyen,
        "taille_composante_geante": taille_composante_geante,
        "longueur_moyenne_chemins": longueur_moyenne_chemins,
        "coefficient_clustering": coefficient_clustering
    }


class GestionnaireFailures:
    """
    Classe pour gérer les pannes des satellites selon le protocole d'expérience:
    - Aléatoire : suppression de p% de nœuds tirés au sort
    - Ciblée (prévisible) : suppression de nœuds prévus (batteries vides)
    - Betweenness : suppression ciblée des nœuds avec la plus forte betweenness centrality
    """
    def __init__(self):
        self.noeud_predictable_failures = set()  # Nœuds qui vont tomber en panne de manière prévisible
        self.noeud_betweenness_failures = set()  # Nœuds à forte betweenness qui seront ciblés
        self.prob_random_failure = 0.0          # Probabilité de panne aléatoire
        self.failed_nodes = set()               # Nœuds qui ont déjà tombé en panne
        self.random_nodes = set()               # Nœuds sélectionnés pour les pannes aléatoires (pour cohérence entre instants)
    
    def config_failures_predictable(self, num_nodes, nodes_list, centrality_measure=None):
        """
        Configure les nœuds qui tomberont en panne de manière prévisibles (ciblée)
        Utilise la centralité pour cibler les nœuds les plus importants
        
        Args:
            num_nodes: Nombre de nœuds à sélectionner
            nodes_list: Liste de tous les nœuds disponibles
            centrality_measure: Mesure de centralité (dictionnaire {node_id: centralité})
        """
        if centrality_measure is not None:
            # Trier les nœuds par centralité et prendre les N_PRED plus centraux (ciblage stratégique)
            sorted_nodes = sorted(centrality_measure.items(), key=lambda x: x[1], reverse=True)
            self.noeud_predictable_failures = {node_id for node_id, _ in sorted_nodes[:num_nodes]}
            print(f"  - Pannes prévisibles configurées pour {len(self.noeud_predictable_failures)} nœuds les plus centraux")
        else:
            # Sélectionner aléatoirement N_PRED nœuds
            all_nodes = list(nodes_list.keys())
            self.noeud_predictable_failures = set(np.random.choice(all_nodes, num_nodes, replace=False))
            print(f"  - Pannes prévisibles configurées pour {len(self.noeud_predictable_failures)} nœuds aléatoires")
    
    def config_failures_random(self, prob):
        """
        Configure la probabilité de pannes aléatoires
        
        Args:
            prob: Probabilité qu'un nœud tombe en panne à chaque instant
        """
        self.prob_random_failure = prob
        
        # Réinitialiser les nœuds aléatoires
        self.random_nodes = set()
    
    def config_failures_betweenness(self, num_nodes, nodes_list, betweenness_centrality=None):
        """
        Configure les nœuds qui tomberont en panne selon une attaque ciblée basée sur la betweenness centrality
        
        Args:
            num_nodes: Nombre de nœuds à sélectionner
            nodes_list: Liste de tous les nœuds disponibles
            betweenness_centrality: Mesure de betweenness centrality (dictionnaire {node_id: betweenness})
        """
        if betweenness_centrality is not None:
            # Trier les nœuds par betweenness et prendre les plus importants (ciblage stratégique)
            sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
            self.noeud_betweenness_failures = {node_id for node_id, _ in sorted_nodes[:num_nodes]}
            print(f"  - Attaque ciblée configurée pour {len(self.noeud_betweenness_failures)} nœuds à forte betweenness")
        else:
            # Si pas de mesure de betweenness, sélectionner aléatoirement (fallback)
            all_nodes = list(nodes_list.keys())
            self.noeud_betweenness_failures = set(np.random.choice(all_nodes, num_nodes, replace=False))
            print(f"  - Attaque ciblée configurée pour {len(self.noeud_betweenness_failures)} nœuds aléatoires (fallback)")
    
    def apply_failures(self, t, nodes_dict, failure_type):
        """
        Applique les pannes selon le type de scénario
        
        Args:
            t: Temps actuel
            nodes_dict: Dictionnaire des nœuds
            failure_type: Type de panne ('none', 'predictable', 'random', 'betweenness')
        
        Returns:
            dict: Dictionnaire des nœuds actifs
        """
        # Copie des nœuds pour ne pas modifier l'original
        active_nodes = copy.deepcopy(nodes_dict)
        
        # Scénario sans panne
        if failure_type == 'none':
            return active_nodes
        
        # Scénario de pannes prévisibles (ciblées)
        elif failure_type == 'predictable' and t >= T_PRED:
            # À l'instant T_PRED, on supprime les nœuds prévisibles (batteries vides)
            for node_id in self.noeud_predictable_failures:
                if node_id in active_nodes:
                    del active_nodes[node_id]
            
            # Afficher des statistiques au moment de la panne
            if t == T_PRED:
                print(f"  - Instant t={t}: {len(self.noeud_predictable_failures)} nœuds prévisibles ont été supprimés")
        
        # Scénario d'attaque ciblée basée sur la betweenness centrality
        elif failure_type == 'betweenness' and t >= T_PRED:
            # À l'instant T_PRED, on supprime les nœuds à forte betweenness (attaque ciblée)
            for node_id in self.noeud_betweenness_failures:
                if node_id in active_nodes:
                    del active_nodes[node_id]
            
            # Afficher des statistiques au moment de l'attaque
            if t == T_PRED:
                print(f"  - Instant t={t}: {len(self.noeud_betweenness_failures)} nœuds à forte betweenness ont été supprimés (attaque ciblée)")
        
        # Scénario de pannes aléatoires
        elif failure_type == 'random':
            # Pour préserver la cohérence entre les instants, on maintient une liste des nœuds
            # qui sont déjà tombés en panne dans ce scénario
            
            # Déterminer quels nouveaux nœuds tombent en panne à cet instant
            if t == 0:
                # À t=0, on sélectionne aléatoirement p% des nœuds qui tomberont en panne
                # pour avoir une comparaison équitable avec le scénario prévisible
                total_failures = int(len(nodes_dict) * self.prob_random_failure * MAXTEMPS)
                total_failures = min(total_failures, len(nodes_dict) - 1)  # Éviter de supprimer tous les nœuds
                
                all_nodes = list(nodes_dict.keys())
                self.random_nodes = set(np.random.choice(all_nodes, total_failures, replace=False))
                
                # Pour t=0, aucun nœud ne tombe encore en panne
                print(f"  - Pannes aléatoires: {len(self.random_nodes)} nœuds sélectionnés sur {len(nodes_dict)}")
            else:
                # À chaque instant t>0, une fraction des nœuds aléatoires tombe en panne
                # proportionnellement au temps qui passe
                nodes_to_fail = int(len(self.random_nodes) * (t / MAXTEMPS))
                nodes_to_fail_now = list(self.random_nodes)[:nodes_to_fail]
                
                # Supprimer ces nœuds
                for node_id in nodes_to_fail_now:
                    if node_id in active_nodes:
                        del active_nodes[node_id]
        
        return active_nodes


def analyser_topologie():
    """
    Analyse topologique du réseau satellite avec les métriques clés.
    """
    try:
        print(f"### Analyse topologique du réseau satellite ###")
        
        # Vérifier les dossiers de sortie
        os.makedirs(f"{OUTDIR}/topologie", exist_ok=True)
        print(f"Dossier de sortie: {OUTDIR}/topologie")
        
        # Chargement des données
        print("Chargement des données...")
        positions, swarms, matrixes, adjacency, num_sats = load_data(PATH, MAXTEMPS, {
            'min': MIN_RANGE,
            'mid': MID_RANGE,
            'max': MAX_RANGE
        })
        print(f"  - Données chargées pour {MAXTEMPS} instants et {num_sats} satellites")
    
        # Identifier le moment où le réseau est le plus efficace pour les pannes prévisibles
        t_efficace = T_PRED - 1  # Par défaut, juste avant le moment de panne prévisible
    
        # Initialisation du gestionnaire de pannes
        gestionnaire_pannes = GestionnaireFailures()
    
        # Convertir le swarm en graphe pour calculer les mesures de centralité
        swarm_graph = swarms[t_efficace].swarm_to_nxgraph() if hasattr(swarms[t_efficace], "swarm_to_nxgraph") else nx.Graph()
        
        # Mesure de centralité basée sur le degré
        centralite = nx.degree_centrality(swarm_graph)
        
        # Mesure de centralité basée sur la betweenness
        betweenness = nx.betweenness_centrality(swarm_graph)
        
        # Configuration des pannes prévisibles basées sur la centralité
        print(f"Configuration des pannes prévisibles pour {N_PRED} nœuds...")
        gestionnaire_pannes.config_failures_predictable(N_PRED, positions[t_efficace], centralite)
    
        # Configuration de l'attaque ciblée basée sur la betweenness
        print(f"Configuration de l'attaque ciblée basée sur la betweenness pour {N_PRED} nœuds...")
        gestionnaire_pannes.config_failures_betweenness(N_PRED, positions[t_efficace], betweenness)
        
        # Configuration des pannes aléatoires
        print(f"Configuration des pannes aléatoires avec probabilité {P_FAIL*100:.2f}%...")
        gestionnaire_pannes.config_failures_random(P_FAIL)
    
        # Structure pour stocker les métriques par scénario
        metriques = {
            'none': [],       # Pas de panne
            'predictable': [], # Pannes prévisibles
            'random': [],      # Pannes aléatoires
            'betweenness': []  # Attaque ciblée basée sur la betweenness
        }
    
        # Analyse pour chaque instant de temps
        print("Analyse topologique pour chaque instant de temps et scénario...")
        for t in range(MAXTEMPS):
            # Pour chaque scénario de panne
            for scenario in ['none', 'predictable', 'random', 'betweenness']:
                # Appliquer les pannes selon le scénario
                nodes_actifs = gestionnaire_pannes.apply_failures(t, positions[t], scenario)
            
                # Construire le swarm avec les nœuds actifs
                swarm = Swarm(MAX_RANGE, list(nodes_actifs.values()))
            
                # Convertir en graphe NetworkX pour l'analyse
                G = swarm.swarm_to_nxgraph() if hasattr(swarm, "swarm_to_nxgraph") else nx.Graph()
            
                # Calculer les métriques
                metriques_t = calculer_metriques(G)
            
                # Stocker avec le temps
                metriques[scenario].append({
                    't': t,
                    'nodes': len(nodes_actifs),
                    'edges': len(swarm.edges) if hasattr(swarm, "edges") else 0,
                    **metriques_t
                })
    
        # Conversion en DataFrame pour faciliter la visualisation
        df_none = pd.DataFrame(metriques['none'])
        df_predictable = pd.DataFrame(metriques['predictable'])
        df_random = pd.DataFrame(metriques['random'])
        df_betweenness = pd.DataFrame(metriques['betweenness'])
    
        # Analyse pour identifier les instants critiques
        print("Recherche des instants critiques...")
        
        # Calcul des changements relatifs dans les métriques pour détecter les instants critiques
        critical_moments = []
        
        # Examiner les changements dans la composante géante et la longueur des chemins
        for t in range(1, MAXTEMPS):
            # Vérifier si la taille de la composante géante a chuté significativement
            if (df_predictable['taille_composante_geante'][t] < 0.9 * df_predictable['taille_composante_geante'][t-1] or 
                df_predictable['longueur_moyenne_chemins'][t] > 1.5 * df_predictable['longueur_moyenne_chemins'][t-1]):
                critical_moments.append(t)
                print(f"  - Instant critique détecté à t={t} pour le scénario de panne prévisible")
        
        # Ajouter T_PRED aux moments critiques s'il n'y est pas déjà
        if T_PRED not in critical_moments:
            critical_moments.append(T_PRED)
        
        # Visualisation des métriques
        print("Génération des visualisations...")
        
        # Style des tracés
        styles = {
            'none': {'color': 'blue', 'linestyle': '-', 'label': 'Sans panne', 'linewidth': 2},
            'predictable': {'color': 'red', 'linestyle': '-', 'label': 'Panne prévisible', 'linewidth': 2},
            'random': {'color': 'green', 'linestyle': '-', 'label': 'Panne aléatoire', 'linewidth': 2},
            'betweenness': {'color': 'purple', 'linestyle': '-', 'label': 'Attaque betweenness', 'linewidth': 2}
        }
        
        # Fond pour la zone critique
        critical_zone_color = 'lightyellow'
        
        # Paramètres de figure communs
        plt.figure(figsize=(12, 12))  # Taille ajustée pour 4 graphiques
        plt.suptitle("Métriques topologiques d'un réseau de nanosatellites sous différents scénarios de panne et d'attaque", 
                    fontsize=16, fontweight='bold')
        
        # 1. Degré moyen ⟨k⟩ : combien de liens en moyenne par satellite
        plt.subplot(2, 2, 1)
        
        # Fond pour les moments critiques
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
        
        # Tracer les courbes
        plt.plot(df_none['t'], df_none['degre_moyen'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['degre_moyen'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['degre_moyen'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['degre_moyen'], **styles['betweenness'])
        
        # Ligne verticale pour T_PRED
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Instant de panne prévisible (T={T_PRED})')
        
        plt.title('Degré moyen ⟨k⟩', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Degré moyen')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # 2. Taille de la composante géante |Gₘₐₓ|/N : fraction de nœuds toujours connectés
        plt.subplot(2, 2, 2)
        
        # Fond pour les moments critiques
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
        
        plt.plot(df_none['t'], df_none['taille_composante_geante'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['taille_composante_geante'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['taille_composante_geante'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['taille_composante_geante'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5)
        
        plt.title('Taille relative de la composante géante |Gₘₐₓ|/N', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Fraction de nœuds dans la composante géante')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # 3. Longueur moyenne des plus courts chemins ℓ̄ : proxy pour la latence minimale
        plt.subplot(2, 2, 3)
        
        # Fond pour les moments critiques
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
            
        plt.plot(df_none['t'], df_none['longueur_moyenne_chemins'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['longueur_moyenne_chemins'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['longueur_moyenne_chemins'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['longueur_moyenne_chemins'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5)
        
        plt.title('Longueur moyenne des plus courts chemins ℓ̄', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Longueur moyenne')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # 4. Coefficient de clustering C : mesure la redondance locale (triangles)
        plt.subplot(2, 2, 4)
        
        # Fond pour les moments critiques
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
            
        plt.plot(df_none['t'], df_none['coefficient_clustering'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['coefficient_clustering'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['coefficient_clustering'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['coefficient_clustering'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5)
        
        plt.title('Coefficient de clustering C', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Coefficient de clustering')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Note: Les graphiques d'efficacité ont été supprimés
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Ajuster pour le titre principal
        
        # Sauvegarder la figure
        plt.savefig(f"{OUTDIR}/topologie/metriques_topologiques.png", dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée dans {OUTDIR}/topologie/metriques_topologiques.png")
    
        # Sauvegarder les résultats en CSV pour analyse ultérieure
        df_none['scenario'] = 'sans_panne'
        df_predictable['scenario'] = 'panne_previsible'
        df_random['scenario'] = 'panne_aleatoire'
        df_betweenness['scenario'] = 'attaque_betweenness'
    
        df_all = pd.concat([df_none, df_predictable, df_random, df_betweenness])
        df_all.to_csv(f"{OUTDIR}/topologie/metriques_topologiques.csv", index=False)
        print(f"Données sauvegardées dans {OUTDIR}/topologie/metriques_topologiques.csv")
        
        # Analyse comparative avant/après panne prévisible
        print("\n### Analyse comparative avant/après panne ###")
        t_avant = T_PRED - 1
        t_apres = T_PRED
        
        # Créer un dataframe pour le rapport
        rapport_data = []
        
        # Analyser les changements pour chaque métrique et chaque scénario
        for scenario, df in [
            ('Sans panne', df_none), 
            ('Panne prévisible', df_predictable), 
            ('Panne aléatoire', df_random),
            ('Attaque betweenness', df_betweenness)
        ]:
            # Récupérer les valeurs avant et après
            avant = df.iloc[t_avant]
            apres = df.iloc[t_apres]
            
            # Calculer les changements pour chaque métrique
            for metrique, nom_fr in [
                ('degre_moyen', 'Degré moyen ⟨k⟩'),
                ('taille_composante_geante', 'Taille composante géante |Gₘₐₓ|/N'),
                ('longueur_moyenne_chemins', 'Longueur moyenne chemins ℓ̄'),
                ('coefficient_clustering', 'Coefficient de clustering C')
            ]:
                # Éviter division par zéro
                if avant[metrique] == 0:
                    pct_changement = float('inf') if apres[metrique] > 0 else 0
                else:
                    pct_changement = 100 * (apres[metrique] - avant[metrique]) / avant[metrique]
                
                rapport_data.append({
                    'Scénario': scenario,
                    'Métrique': nom_fr,
                    'Avant (t={})'.format(t_avant): avant[metrique],
                    'Après (t={})'.format(t_apres): apres[metrique],
                    'Variation (%)': pct_changement
                })
        
        # Créer et afficher le tableau de rapport
        rapport_df = pd.DataFrame(rapport_data)
        print("\nTableau comparatif des métriques avant/après panne (t={} → t={}):".format(t_avant, t_apres))
        
        # Formater le tableau pour un affichage lisible
        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 3)
        print(rapport_df)
        
        # Sauvegarder le rapport
        rapport_df.to_csv(f"{OUTDIR}/topologie/rapport_avant_apres_panne.csv", index=False)
        print(f"\nRapport détaillé sauvegardé dans {OUTDIR}/topologie/rapport_avant_apres_panne.csv")
        
        # Créer un graphique de comparaison avant/après
        plt.figure(figsize=(12, 8))
        
        scenarios = rapport_df['Scénario'].unique()
        metriques = rapport_df['Métrique'].unique();
        
        # Utiliser des barres groupées pour montrer les variations
        x = np.arange(len(metriques))
        width = 0.25  # largeur des barres
        
        # Couleurs par scénario
        colors = {
            'Sans panne': 'blue', 
            'Panne prévisible': 'red', 
            'Panne aléatoire': 'green',
            'Attaque betweenness': 'purple'
        }
        
        # Tracer les barres pour chaque scénario
        for i, scenario in enumerate(scenarios):
            data = rapport_df[rapport_df['Scénario'] == scenario]
            variations = [data[data['Métrique'] == m]['Variation (%)'].values[0] for m in metriques]
            plt.bar(x + (i-1.5)*width, variations, width, label=scenario, color=colors[scenario], alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Métrique')
        plt.ylabel('Variation (%)')
        plt.title('Impact des pannes sur les métriques topologiques (t={} → t={})'.format(t_avant, t_apres))
        plt.xticks(x, metriques, rotation=15)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/topologie/variations_metriques.png", dpi=300)
        print(f"Graphique des variations sauvegardé dans {OUTDIR}/topologie/variations_metriques.png")
        
        print("\n### Analyse terminée ###")
        return True
    except Exception as e:
        print(f"Erreur pendant l'analyse: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = analyser_topologie()
    # Retourner un code de sortie approprié pour le script shell
    sys.exit(0 if success else 1)
