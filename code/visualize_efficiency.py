#!/usr/bin/env python3
# visualize_efficiency.py
"""
Script dédié à la visualisation des métriques d'efficacité du réseau de satellites.
Se concentre sur les 2 métriques d'efficacité:
    1. Efficacité globale Eglob(G) = 1/(n(n-1)) * ∑(i≠j) 1/dij
    2. Efficacité locale Eloc(G) = 1/n * ∑(v=1...n) Eglob(G[N(v)])
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from config import PATH, MAXTEMPS, MAX_RANGE, MID_RANGE, MIN_RANGE, OUTDIR, T_PRED, N_PRED, P_FAIL

def visualiser_efficacite():
    """
    Visualisation dédiée aux métriques d'efficacité du réseau satellite.
    Utilise les données précédemment calculées par analyze_topologie.py.
    """
    try:
        print(f"### Visualisation des métriques d'efficacité du réseau satellite ###")
        
        # Vérifier les dossiers de sortie
        output_dir = f"{OUTDIR}/topologie/efficacite"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Dossier de sortie: {output_dir}")
        
        # Chargement des données précalculées
        csv_path = f"{OUTDIR}/topologie/metriques_topologiques.csv"
        if not os.path.exists(csv_path):
            print(f"Erreur: Le fichier de données {csv_path} n'existe pas.")
            print("Veuillez d'abord exécuter analyze_topologie.py pour générer les données.")
            return False
        
        print(f"Chargement des données depuis {csv_path}...")
        df_all = pd.read_csv(csv_path)
        
        # Séparation des données par scénario
        df_none = df_all[df_all['scenario'] == 'sans_panne']
        df_predictable = df_all[df_all['scenario'] == 'panne_previsible']
        df_random = df_all[df_all['scenario'] == 'panne_aleatoire']
        df_betweenness = df_all[df_all['scenario'] == 'attaque_betweenness']
        
        print(f"  - Données chargées pour {len(df_all['t'].unique())} instants")
        
        # Récupération des instants critiques (si disponibles dans les données)
        # Sinon, on prend juste l'instant T_PRED comme référence
        critical_moments = [T_PRED]
        
        # Examiner les changements dans l'efficacité globale pour détecter d'autres instants critiques
        for t in range(1, len(df_predictable)):
            # Détection de chute significative de l'efficacité globale
            if 'efficacite_globale' in df_predictable.columns:
                if (df_predictable['efficacite_globale'].iloc[t] < 
                    0.8 * df_predictable['efficacite_globale'].iloc[t-1]):
                    if t not in critical_moments:
                        critical_moments.append(t)
                        print(f"  - Instant critique détecté à t={t} (chute d'efficacité globale)")
        
        # Style des tracés
        styles = {
            'none': {'color': 'blue', 'linestyle': '-', 'label': 'Sans panne', 'linewidth': 2},
            'predictable': {'color': 'red', 'linestyle': '-', 'label': 'Panne prévisible', 'linewidth': 2},
            'random': {'color': 'green', 'linestyle': '-', 'label': 'Panne aléatoire', 'linewidth': 2},
            'betweenness': {'color': 'purple', 'linestyle': '-', 'label': 'Attaque betweenness', 'linewidth': 2}
        }
        
        # Fond pour la zone critique
        critical_zone_color = 'lightyellow'
        
        print("Génération des visualisations...")
        
        # Figure pour l'efficacité globale
        plt.figure(figsize=(12, 6))
        plt.suptitle("Efficacité globale du réseau de nanosatellites", 
                    fontsize=16, fontweight='bold')
        
        # Fond pour les moments critiques
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
            
        plt.plot(df_none['t'], df_none['efficacite_globale'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['efficacite_globale'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['efficacite_globale'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['efficacite_globale'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Instant de panne prévisible (T={T_PRED})')
        
        plt.title('Efficacité globale Eglob(G) = 1/(n(n-1)) * ∑(i≠j) 1/dij', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Efficacité globale')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Annotations explicatives
        plt.annotate('Rapidité moyenne de transmission \nà l\'échelle du réseau', 
                     xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficacite_globale.png", dpi=300, bbox_inches='tight')
        print(f"  - Figure d'efficacité globale sauvegardée dans {output_dir}/efficacite_globale.png")
        
        # Figure pour l'efficacité locale
        plt.figure(figsize=(12, 6))
        plt.suptitle("Efficacité locale du réseau de nanosatellites", 
                    fontsize=16, fontweight='bold')
        
        # Fond pour les moments critiques
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
            
        plt.plot(df_none['t'], df_none['efficacite_locale'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['efficacite_locale'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['efficacite_locale'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['efficacite_locale'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Instant de panne prévisible (T={T_PRED})')
        
        plt.title('Efficacité locale Eloc(G) = 1/n * ∑(v=1...n) Eglob(G[N(v)])', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Efficacité locale')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Annotations explicatives
        plt.annotate('Résilience locale autour \nde chaque satellite', 
                     xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficacite_locale.png", dpi=300, bbox_inches='tight')
        print(f"  - Figure d'efficacité locale sauvegardée dans {output_dir}/efficacite_locale.png")
        
        # Figure combinée pour comparaison
        plt.figure(figsize=(14, 10))
        plt.suptitle("Comparaison des métriques d'efficacité du réseau de nanosatellites", 
                    fontsize=16, fontweight='bold')
        
        # Efficacité globale
        plt.subplot(2, 1, 1)
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
            
        plt.plot(df_none['t'], df_none['efficacite_globale'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['efficacite_globale'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['efficacite_globale'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['efficacite_globale'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5)
        
        plt.title('Efficacité globale Eglob(G)', fontweight='bold')
        plt.ylabel('Efficacité globale')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Efficacité locale
        plt.subplot(2, 1, 2)
        for t in critical_moments:
            plt.axvspan(t-0.5, t+0.5, color=critical_zone_color, alpha=0.3)
            
        plt.plot(df_none['t'], df_none['efficacite_locale'], **styles['none'])
        plt.plot(df_predictable['t'], df_predictable['efficacite_locale'], **styles['predictable'])
        plt.plot(df_random['t'], df_random['efficacite_locale'], **styles['random'])
        plt.plot(df_betweenness['t'], df_betweenness['efficacite_locale'], **styles['betweenness'])
        plt.axvline(x=T_PRED, color='black', linestyle='--', linewidth=1.5,
                   label=f'Instant de panne prévisible (T={T_PRED})')
        
        plt.title('Efficacité locale Eloc(G)', fontweight='bold')
        plt.xlabel('Temps (t)')
        plt.ylabel('Efficacité locale')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparaison_efficacite.png", dpi=300, bbox_inches='tight')
        print(f"  - Figure de comparaison d'efficacité sauvegardée dans {output_dir}/comparaison_efficacite.png")
        
        # Analyse comparative avant/après panne prévisible pour les métriques d'efficacité
        print("\n### Analyse comparative avant/après panne pour l'efficacité ###")
        t_avant = T_PRED - 1
        t_apres = T_PRED
        
        # Créer un dataframe pour le rapport d'efficacité
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
            
            # Calculer les changements pour l'efficacité
            for metrique, nom_fr in [
                ('efficacite_globale', 'Efficacité globale Eglob(G)'),
                ('efficacite_locale', 'Efficacité locale Eloc(G)')
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
        print("\nTableau comparatif des métriques d'efficacité avant/après panne (t={} → t={}):".format(t_avant, t_apres))
        
        # Formater le tableau pour un affichage lisible
        pd.set_option('display.width', 120)
        pd.set_option('display.precision', 3)
        print(rapport_df)
        
        # Sauvegarder le rapport
        rapport_df.to_csv(f"{output_dir}/rapport_efficacite.csv", index=False)
        print(f"\nRapport d'efficacité sauvegardé dans {output_dir}/rapport_efficacite.csv")
        
        # Créer un graphique de comparaison avant/après pour l'efficacité
        plt.figure(figsize=(10, 6))
        
        scenarios = rapport_df['Scénario'].unique()
        metriques = rapport_df['Métrique'].unique()
        
        # Utiliser des barres groupées pour montrer les variations
        x = np.arange(len(metriques))
        width = 0.2  # largeur des barres
        
        # Couleurs par scénario (cohérentes avec le reste)
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
        plt.xlabel('Métrique d\'efficacité')
        plt.ylabel('Variation (%)')
        plt.title('Impact des pannes sur les métriques d\'efficacité (t={} → t={})'.format(t_avant, t_apres))
        plt.xticks(x, metriques)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(f"{output_dir}/variations_efficacite.png", dpi=300)
        print(f"Graphique des variations d'efficacité sauvegardé dans {output_dir}/variations_efficacite.png")
        
        print("\n### Visualisation d'efficacité terminée ###")
        
        # Retourner True si tout s'est bien passé
        return True
        
    except Exception as e:
        print(f"Erreur pendant la visualisation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = visualiser_efficacite()
    # Retourner un code de sortie approprié pour le script shell
    import sys
    sys.exit(0 if success else 1)
