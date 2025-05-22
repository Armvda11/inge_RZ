#!/usr/bin/env python3
# performance_metrics.py
"""
Module pour le calcul de métriques de performance pour les protocoles de routage DTN,
spécialement adapté au contexte de pannes dynamiques.

Ce module fournit des classes et fonctions permettant de:
1. Suivre l'évolution des métriques de performance au cours du temps
2. Calculer des statistiques avancées sur les délais et débits
3. Générer des rapports de performance détaillés sous forme de tableaux
4. Exporter les résultats au format HTML pour une meilleure visualisation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from tabulate import tabulate
import statistics

# Imports optionnels pour l'export HTML
try:
    import jinja2
except ImportError:
    jinja2 = None

class PerformanceTracker:
    """
    Classe pour suivre et calculer les métriques de performance d'un protocole DTN
    dans un contexte de pannes dynamiques sur la durée d'une simulation.
    """
    
    def __init__(self, total_nodes: int, source: int, destination: int):
        """
        Initialise le tracker de performance.
        
        Args:
            total_nodes (int): Nombre total de nœuds dans le réseau
            source (int): ID du nœud source
            destination (int): ID du nœud destination
        """
        self.total_nodes = total_nodes
        self.source = source
        self.destination = destination
        
        # Métriques suivies au fil du temps
        self.metrics_by_time = []
        
        # Données de base pour chaque pas de temps
        self.copies_by_time = []         # Nombre de copies à chaque pas de temps
        self.active_nodes_by_time = []   # Nombre de nœuds actifs à chaque pas
        self.failed_nodes_by_time = []   # Nombre de nœuds en panne à chaque pas
        
        # État de la livraison
        self.delivery_time = None
        self.delivery_occurred = False
        self.delivery_hops = None
        
        # Statistiques finales (calculées à la fin de la simulation)
        self.final_stats = {}
        
    def record_step(self, t: int, protocol: Any, active_nodes: int, failed_nodes: int):
        """
        Enregistre les métriques pour un pas de temps donné.
        
        Args:
            t (int): Pas de temps actuel
            protocol: Instance du protocole contenant les données
            active_nodes (int): Nombre de nœuds actifs
            failed_nodes (int): Nombre de nœuds en panne
        """
        # Extraire les données du protocole
        total_copies = sum(protocol.copies.values())
        nodes_with_copies = sum(1 for n, c in protocol.copies.items() if c > 0)
        delivered = self.destination in protocol.delivered_at
        
        # Vérifier si la livraison vient de se produire
        if delivered and not self.delivery_occurred:
            self.delivery_occurred = True
            self.delivery_time = protocol.delivered_at[self.destination]
            self.delivery_hops = protocol.num_hops.get(self.destination, None)
        
        # Calculer le taux de couverture (pourcentage de nœuds ayant reçu au moins une copie)
        coverage_rate = nodes_with_copies / max(1, active_nodes)
        
        # Calculer le débit instantané avec fenêtre glissante pour plus de stabilité
        if not hasattr(self, 'previous_copies'):
            self.previous_copies = 0
            self.previous_time = 0
            self.throughput_window = []
        
        # Calculer le débit instantané (nouvelles copies créées depuis la dernière mesure)
        if t > self.previous_time:
            instant_throughput = (protocol.total_copies_created - self.previous_copies) / (t - self.previous_time)
            # Mettre à jour les valeurs pour la prochaine itération
            self.previous_copies = protocol.total_copies_created
            self.previous_time = t
        else:
            instant_throughput = 0
        
        # Utiliser une fenêtre glissante pour lisser le débit
        self.throughput_window.append(instant_throughput)
        if len(self.throughput_window) > 5:  # Conserver une fenêtre de 5 points
            self.throughput_window.pop(0)
        
        # Moyenne mobile pour le débit
        smooth_throughput = sum(self.throughput_window) / len(self.throughput_window)
        # Débit global moyen
        avg_throughput = protocol.total_copies_created / max(1, t) if t > 0 else 0
        
        # Enregistrer les métriques pour ce pas de temps
        metrics = {
            't': t,
            'total_copies': total_copies,
            'nodes_with_copies': nodes_with_copies,
            'active_nodes': active_nodes,
            'failed_nodes': failed_nodes,
            'failure_rate': failed_nodes / self.total_nodes,
            'coverage_rate': coverage_rate,
            'delivered': delivered,
            'throughput': smooth_throughput,  # Débit lissé
            'avg_throughput': avg_throughput, # Débit moyen global
            'copies_created': protocol.total_copies_created
        }
        
        self.metrics_by_time.append(metrics)
        
        # Mise à jour des listes temporelles
        self.copies_by_time.append(total_copies)
        self.active_nodes_by_time.append(active_nodes)
        self.failed_nodes_by_time.append(failed_nodes)
    
    def calculate_final_stats(self, protocol: Any, max_steps: int):
        """
        Calcule les statistiques finales à la fin de la simulation.
        
        Args:
            protocol: Instance du protocole avec les métriques finales
            max_steps (int): Nombre maximum de pas de temps
        """
        # Métriques du protocole
        delivery_ratio = protocol.delivery_ratio()
        delivery_delay = protocol.delivery_delay()
        overhead_ratio = protocol.overhead_ratio()
        hop_stats = protocol.get_hop_stats()
        
        # Calcul du débit moyen (copies transmises par unité de temps)
        avg_throughput = protocol.total_copies_created / max(1, max_steps)
        
        # Efficacité de la propagation: combien de copies nécessaires pour livrer le message
        delivery_efficiency = 1.0 / protocol.total_copies_created if protocol.total_copies_created > 0 else 0
        
        # Impact des pannes: calculer le taux moyen de pannes à partir des données enregistrées
        avg_failure_rate = statistics.mean([m['failure_rate'] for m in self.metrics_by_time])
        
        # Résilience: capacité à livrer malgré les pannes
        resilience_score = delivery_ratio / max(0.01, avg_failure_rate) if avg_failure_rate > 0 else 0
        
        # Stocker toutes les statistiques finales
        self.final_stats = {
            'delivery_ratio': delivery_ratio,
            'delivery_delay': delivery_delay if delivery_ratio > 0 else float('inf'),
            'overhead_ratio': overhead_ratio,
            'total_copies_created': protocol.total_copies_created,
            'hop_count': hop_stats.get('destination', None),
            'avg_throughput': avg_throughput,
            'delivery_efficiency': delivery_efficiency,
            'avg_failure_rate': avg_failure_rate,
            'resilience_score': resilience_score
        }
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Convertit les métriques en DataFrame pandas pour analyse.
        
        Returns:
            pd.DataFrame: DataFrame contenant toutes les métriques par pas de temps
        """
        return pd.DataFrame(self.metrics_by_time)
    
    def print_progress_table(self, t: int, window_size: int = 5):
        """
        Affiche un tableau de progression pour les derniers pas de temps.
        
        Args:
            t (int): Pas de temps actuel
            window_size (int): Nombre de lignes à afficher
        """
        # Sélectionner les dernières entrées
        start_idx = max(0, len(self.metrics_by_time) - window_size)
        recent_metrics = self.metrics_by_time[start_idx:]
        
        # Créer le tableau avec les colonnes sélectionnées
        headers = ['Temps', 'Copies', 'Nœuds actifs', 'Nœuds en panne', 'Taux échec', 'Livré']
        table_data = []
        
        for m in recent_metrics:
            delivered_str = f"Oui (t={self.delivery_time})" if m['delivered'] else "Non"
            
            row = [
                m['t'],
                m['total_copies'],
                m['active_nodes'],
                m['failed_nodes'],
                f"{m['failure_rate']*100:.1f}%",
                delivered_str
            ]
            table_data.append(row)
        
        # Afficher le tableau
        print("\nTableau de progression (derniers pas de temps):")
        print(tabulate(table_data, headers=headers, tablefmt='pretty'))
    
    def print_final_report(self):
        """
        Affiche un rapport détaillé des métriques de performance à la fin de la simulation.
        """
        if not self.final_stats:
            print("⚠️ Aucune statistique finale n'a été calculée. Exécutez d'abord calculate_final_stats().")
            return
        
        print("\n" + "="*60)
        print("RAPPORT DÉTAILLÉ DE PERFORMANCE")
        print("="*60)
        
        # Section 1: Résumé de livraison
        print("\n📊 RÉSUMÉ DE LIVRAISON")
        print("-" * 40)
        
        if self.final_stats['delivery_ratio'] > 0:
            print(f"✅ Livraison réussie en {self.final_stats['delivery_delay']:.1f} unités de temps")
            print(f"✅ Nombre de sauts: {self.final_stats['hop_count']}")
        else:
            print("❌ Échec de livraison")
            print(f"❌ Causes possibles: {self.final_stats['avg_failure_rate']*100:.1f}% des nœuds en panne")
        
        # Section 2: Statistiques de copies
        print("\n📊 STATISTIQUES DE COPIES")
        print("-" * 40)
        print(f"📈 Copies créées au total: {self.final_stats['total_copies_created']}")
        print(f"📈 Overhead ratio: {self.final_stats['overhead_ratio']:.2f}")
        print(f"📈 Efficacité de livraison: {self.final_stats['delivery_efficiency']*100:.4f}%")
        
        # Section 3: Métriques de performance
        print("\n📊 MÉTRIQUES DE PERFORMANCE")
        print("-" * 40)
        print(f"📈 Débit moyen: {self.final_stats['avg_throughput']:.2f} copies/pas de temps")
        print(f"📈 Score de résilience: {self.final_stats['resilience_score']:.2f}")
        
        # Section 4: Statistiques sur les pannes
        print("\n📊 IMPACT DES PANNES")
        print("-" * 40)
        print(f"📉 Taux moyen de pannes: {self.final_stats['avg_failure_rate']*100:.1f}%")
        
        # Résumé final
        print("\n" + "="*60)
        verdict = "RÉUSSITE" if self.final_stats['delivery_ratio'] > 0 else "ÉCHEC"
        print(f"VERDICT FINAL: {verdict}")
        print("="*60 + "\n")

def generate_comparative_table(results_list: List[Dict[str, Any]], table_format: str = 'pretty') -> str:
    """
    Génère un tableau comparatif des résultats de plusieurs simulations.
    
    Args:
        results_list (List[Dict]): Liste des résultats de différentes simulations
        table_format (str): Format du tableau ('pretty', 'html', 'markdown', etc.)
        
    Returns:
        str: Tableau formaté prêt à être affiché ou enregistré
    """
    # Sélectionner les colonnes d'intérêt pour le tableau comparatif
    headers = [
        'L', 'Livré', 'Délai', 'Sauts', 'Total copies', 
        'Overhead', 'Débit', 'Résilience', 'Taux échec'
    ]
    
    table_data = []
    
    for result in results_list:
        delivered_str = "✅" if result.get('delivered', False) else "❌"
        delay = result.get('delivery_delay', float('inf'))
        delay_str = f"{delay:.1f}" if delay != float('inf') else "N/A"
        
        row = [
            result['L'],
            delivered_str,
            delay_str,
            result.get('hop_count', 'N/A'),
            result.get('total_copies', 0),
            f"{result.get('overhead_ratio', float('inf')):.2f}",
            f"{result.get('avg_throughput', 0):.2f}",
            f"{result.get('resilience_score', 0):.2f}",
            f"{result.get('avg_failure_rate', 0)*100:.1f}%"
        ]
        table_data.append(row)
    
    # Trier par L
    table_data.sort(key=lambda x: x[0])
    
    # Générer le tableau
    return tabulate(table_data, headers=headers, tablefmt=table_format)


def export_results_to_html(results_list: List[Dict[str, Any]], metrics_df: pd.DataFrame, 
                          protocol_info: Dict[str, Any], output_path: str):
    """
    Exporte les résultats complets d'une simulation au format HTML.
    
    Args:
        results_list (List[Dict]): Liste des résultats de différentes simulations
        metrics_df (pd.DataFrame): DataFrame contenant les métriques détaillées
        protocol_info (Dict): Informations sur le protocole et les paramètres
        output_path (str): Chemin où sauvegarder le rapport HTML
    """
    try:
        import jinja2
        import matplotlib
        matplotlib.use('Agg')  # Mode non-interactif
    except ImportError:
        print("⚠️ Les modules jinja2 ou matplotlib ne sont pas installés.")
        print("   Installation nécessaire pour générer des rapports HTML.")
        print("   Exécutez: pip install jinja2 matplotlib")
        return
    
    # Générer le tableau HTML
    table_html = generate_comparative_table(results_list, 'html')
    
    # Générer un graphique des métriques clés
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['t'], metrics_df['throughput'], 'r-', label="Débit")
    plt.plot(metrics_df['t'], metrics_df['coverage_rate'], 'b-', label="Couverture")
    plt.plot(metrics_df['t'], metrics_df['failure_rate'], 'g-', label="Taux de panne")
    plt.xlabel('Temps')
    plt.ylabel('Valeur')
    plt.title('Évolution des métriques clés')
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder le graphique
    metrics_chart_path = f"{output_path}_metrics_chart.png"
    plt.savefig(metrics_chart_path)
    plt.close()
    
    # Créer le template HTML
    html_template = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport de Simulation - Protocol Spray and Wait</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metrics-container { margin-top: 30px; }
            .chart { width: 100%; max-width: 800px; margin: 20px auto; }
            .summary-box { background-color: #f8f9fa; border: 1px solid #e9ecef; 
                          border-radius: 5px; padding: 15px; margin-bottom: 20px; }
            .parameter-table { width: auto; min-width: 50%; }
            .success { color: green; }
            .failure { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport de Simulation - Spray and Wait avec Pannes Dynamiques</h1>
            
            <div class="summary-box">
                <h2>Paramètres de Simulation</h2>
                <table class="parameter-table">
                    <tr>
                        <th>Paramètre</th>
                        <th>Valeur</th>
                    </tr>
                    <tr>
                        <td>Mode de panne</td>
                        <td>{{ protocol_info.failure_mode }}</td>
                    </tr>
                    <tr>
                        <td>Nombre de nœuds</td>
                        <td>{{ protocol_info.num_nodes }}</td>
                    </tr>
                    <tr>
                        <td>TTL</td>
                        <td>{{ protocol_info.ttl_value }}</td>
                    </tr>
                    <tr>
                        <td>Dilution du réseau</td>
                        <td>{{ protocol_info.network_dilution }}</td>
                    </tr>
                    <tr>
                        <td>Taux de distribution</td>
                        <td>{{ protocol_info.distribution_rate }}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Tableau Comparatif des Résultats</h2>
            {{ table_html|safe }}
            
            <div class="metrics-container">
                <h2>Évolution des Métriques Clés</h2>
                <img src="{{ metrics_chart_path }}" alt="Graphique des métriques" class="chart">
            </div>
            
            <h2>Conclusion</h2>
            <p>
                La simulation avec le mode de panne <strong>{{ protocol_info.failure_mode }}</strong> 
                a été exécutée avec différentes valeurs de L (nombre initial de copies).
                {% if all_delivered %}
                <span class="success">Toutes les configurations ont réussi à livrer le message à destination.</span>
                {% else %}
                <span class="failure">Certaines configurations n'ont pas réussi à livrer le message à destination.</span>
                {% endif %}
            </p>
            <p>
                Observations principales:
                <ul>
                    <li>Meilleur délai de livraison: L={{ best_delay.L }} ({{ best_delay.delay }} pas de temps)</li>
                    <li>Meilleure efficacité: L={{ best_efficiency.L }} ({{ best_efficiency.efficiency }}% d'efficacité)</li>
                    <li>Meilleure résilience: L={{ best_resilience.L }} (score de {{ best_resilience.score }})</li>
                </ul>
            </p>
        </div>
    </body>
    </html>
    """
    
    # Analyser les meilleurs résultats
    delivered_results = [r for r in results_list if r.get('delivered', False)]
    
    if delivered_results:
        best_delay = min(delivered_results, key=lambda x: x.get('delivery_delay', float('inf')))
        best_efficiency = max(delivered_results, key=lambda x: 1.0/x.get('total_copies', 1))
        best_resilience = max(delivered_results, key=lambda x: x.get('resilience_score', 0))
    else:
        best_delay = {'L': 'N/A', 'delivery_delay': 'N/A'}
        best_efficiency = {'L': 'N/A', 'efficiency': 'N/A'}
        best_resilience = {'L': 'N/A', 'resilience_score': 'N/A'}
    
    # Données à injecter dans le template
    template_data = {
        'protocol_info': protocol_info,
        'table_html': table_html,
        'metrics_chart_path': os.path.basename(metrics_chart_path),
        'all_delivered': all(r.get('delivered', False) for r in results_list),
        'best_delay': {
            'L': best_delay.get('L', 'N/A'),
            'delay': best_delay.get('delivery_delay', 'N/A')
        },
        'best_efficiency': {
            'L': best_efficiency.get('L', 'N/A'),
            'efficiency': round(100/best_efficiency.get('total_copies', 1), 2)
        },
        'best_resilience': {
            'L': best_resilience.get('L', 'N/A'),
            'score': round(best_resilience.get('resilience_score', 0), 2)
        }
    }
    
    # Générer le HTML
    try:
        template = jinja2.Template(html_template)
        html_output = template.render(**template_data)
        
        # Sauvegarder le HTML
        with open(f"{output_path}.html", 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        print(f"\n✅ Rapport HTML généré avec succès: {output_path}.html")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération du rapport HTML: {str(e)}")
        return None
