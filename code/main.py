# main.py
import numpy as np
import random
import pandas as pd
import os
import sys
import argparse
from config import PATH, MAXTEMPS, MAX_RANGE, MID_RANGE, MIN_RANGE, OUTDIR, T_PRED, N_PRED, CONFIG
from data.loader import load_data
from protocols.spray_and_wait import SprayAndWait
from protocols.epidemic import Epidemic
from protocols.prophet import Prophet
from simulation.metrics import analyze_single_graph, compute_per_packet_metrics
from simulation.failure import simulate_with_failures

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Simulation de robustesse d'un essaim de nano-satellites")
    parser.add_argument(
        '--categorie-panne', 
        type=str, 
        choices=['legere', 'moyenne', 'lourde'],
        default='moyenne',
        help='Catégorie de probabilité de pannes (legere, moyenne, lourde)'
    )
    
    return parser.parse_args()

def main():
    """Point d'entrée principal du programme."""
    # Analyser les arguments de ligne de commande
    args = parse_arguments()
    
    # Mettre à jour la catégorie de panne dans la configuration
    print(f"### Analyse réseau satellite - Catégorie de panne: {args.categorie_panne} ###")
    CONFIG['failure']['active_category'] = args.categorie_panne
    
    # Mettre à jour les variables globales de panne selon la catégorie choisie
    global N_PRED, P_FAIL
    N_PRED = CONFIG['failure']['categories'][args.categorie_panne]['predictable_nodes']
    P_FAIL = CONFIG['failure']['categories'][args.categorie_panne]['random_prob']
    
    print(f"  - Probabilité de panne aléatoire: {P_FAIL*100:.2f}%")
    print(f"  - Nombre de nœuds avec pannes prévisibles: {N_PRED}")
    
    # Chargement des données
    positions, swarms, matrixes, adjacency, num_sats = load_data(PATH, MAXTEMPS, {
        'min': MIN_RANGE,
        'mid': MID_RANGE,
        'max': MAX_RANGE
    })
    
    # Destination pour les protocoles DTN (arbitrairement au milieu du réseau)
    DEST = num_sats // 2
    
    # Analyse topologique
    print("### Analyse topologique par instant ###")
    stats = {t: analyze_single_graph(swarms[t], matrixes[t]) for t in range(MAXTEMPS)}
    
    # Identification des meilleurs/pires instants en termes d'efficacité
    connected_instants = [t for t, m in stats.items() if m.Connexity == 1.0]
    if connected_instants:
        best_t = max(connected_instants, key=lambda t: stats[t].Efficiency)
        worst_t = min(connected_instants, key=lambda t: stats[t].Efficiency)
    else:
        best_t = 0
        worst_t = 0
    
    # Métriques moyennes
    print("### Métriques moyennes globales ###")
    global_metric_values = {
        'Degré moyen': np.mean([m.MeanDegree for m in stats.values()]),
        'Clustering': np.mean([m.MeanClusterCoef for m in stats.values()]),
        'Connexité': np.mean([m.Connexity for m in stats.values()]),
        'Efficience': np.mean([m.Efficiency for m in stats.values() if m.Connexity == 1.0])
    }
    
    for metric, value in global_metric_values.items():
        print(f"{metric}: {value:.3f}")
    
    # Simulation avec pannes de nœuds
    print("\n### Simulation avec pannes de nœuds ###")
    
    print("  - Simulation sans panne...")
    no_failure_metrics, no_failure_results = simulate_with_failures(
        positions, swarms, matrixes, adjacency, num_sats, DEST, 'none', best_t)
    
    print("  - Simulation avec pannes prévisibles...")
    predictable_metrics, predictable_results = simulate_with_failures(
        positions, swarms, matrixes, adjacency, num_sats, DEST, 'predictable', best_t)
    
    print("  - Simulation avec pannes aléatoires...")
    random_metrics, random_results = simulate_with_failures(
        positions, swarms, matrixes, adjacency, num_sats, DEST, 'random', best_t)
    
    # Comparaison des métriques topologiques
    print("\n### Comparaison des métriques topologiques ###")
    print(f"{'Scénario':<15} {'Degré moyen':<12} {'Clustering':<12} {'Connexité':<12} {'Efficience':<12}")
    print(f"{'-'*65}")
    
    print(f"{'Sans panne':<15} {no_failure_metrics.MeanDegree:<12.3f} "
          f"{no_failure_metrics.MeanClusterCoef:<12.3f} {no_failure_metrics.Connexity:<12.3f} "
          f"{no_failure_metrics.Efficiency:<12.3f}")
    
    print(f"{'Prévisible':<15} {predictable_metrics.MeanDegree:<12.3f} "
          f"{predictable_metrics.MeanClusterCoef:<12.3f} {predictable_metrics.Connexity:<12.3f} "
          f"{predictable_metrics.Efficiency:<12.3f}")
    
    print(f"{'Aléatoire':<15} {random_metrics.MeanDegree:<12.3f} "
          f"{random_metrics.MeanClusterCoef:<12.3f} {random_metrics.Connexity:<12.3f} "
          f"{random_metrics.Efficiency:<12.3f}")
    
    # Comparaison des performances DTN entre les scénarios
    print("\n### Comparaison des performances DTN ###")
    for proto_name in ['Spray-and-Wait', 'Epidemic', 'Prophet']:
        print(f"\nProtocole: {proto_name}")
        
        # Trouver le paramètre approprié pour chaque protocole
        if proto_name == 'Spray-and-Wait':
            param_idx = next((i for i, res in enumerate(no_failure_results[proto_name]['values']) 
                             if res['param_value'] == 10), 0)
        elif proto_name == 'Prophet':
            param_idx = next((i for i, res in enumerate(no_failure_results[proto_name]['values']) 
                             if res['param_value'] == 0.5), 0)
        else:  # Epidemic
            param_idx = 0
        
        # Extraire les résultats
        no_failure_dr = no_failure_results[proto_name]['values'][param_idx]['delivery_ratio']
        no_failure_dd = no_failure_results[proto_name]['values'][param_idx]['delivery_delay']
        
        predictable_dr = predictable_results[proto_name]['values'][param_idx]['delivery_ratio']
        predictable_dd = predictable_results[proto_name]['values'][param_idx]['delivery_delay']
        
        random_dr = random_results[proto_name]['values'][param_idx]['delivery_ratio']
        random_dd = random_results[proto_name]['values'][param_idx]['delivery_delay']
        
        # Afficher la comparaison
        print(f"{'Scénario':<15} {'Delivery Ratio':<15} {'Delivery Delay':<15}")
        print(f"{'-'*50}")
        print(f"{'Sans panne':<15} {no_failure_dr:<15.3f} {no_failure_dd:<15.1f}")
        print(f"{'Prévisible':<15} {predictable_dr:<15.3f} {predictable_dd:<15.1f}")
        print(f"{'Aléatoire':<15} {random_dr:<15.3f} {random_dd:<15.1f}")
        
        # Calculer les variations par rapport au scénario sans panne
        dr_pred_var = ((predictable_dr - no_failure_dr) / no_failure_dr) * 100 if no_failure_dr > 0 else float('inf')
        dr_rand_var = ((random_dr - no_failure_dr) / no_failure_dr) * 100 if no_failure_dr > 0 else float('inf')
        
        dd_pred_var = ((predictable_dd - no_failure_dd) / no_failure_dd) * 100 if no_failure_dd > 0 else float('inf')
        dd_rand_var = ((random_dd - no_failure_dd) / no_failure_dd) * 100 if no_failure_dd > 0 else float('inf')
        
        print(f"\nVariation du delivery ratio:")
        print(f"  Prévisible: {dr_pred_var:.1f}%")
        print(f"  Aléatoire: {dr_rand_var:.1f}%")
        
        print(f"Variation du delivery delay:")
        print(f"  Prévisible: {dd_pred_var:.1f}%")
        print(f"  Aléatoire: {dd_rand_var:.1f}%")
    
    # Collecte et export des logs de paquets
    print("\n### Collecte et export des logs de paquets ###")
    packet_logs = []
    
    # Déterminer les meilleurs paramètres pour chaque protocole et scénario
    best_params = {}
    for scenario, results in [("Sans panne", no_failure_results), 
                             ("Prévisible", predictable_results), 
                             ("Aléatoire", random_results)]:
        best_params[scenario] = {}
        for proto_name in ["Spray-and-Wait", "Epidemic", "Prophet"]:
            # Pour Epidemic, toujours utiliser le paramètre par défaut
            if proto_name == "Epidemic":
                best_params[scenario][proto_name] = 0
                continue
                
            proto_results = results[proto_name]['values']
            # Trouver le paramètre donnant le meilleur delivery ratio
            best_idx = max(range(len(proto_results)), 
                          key=lambda i: proto_results[i]['delivery_ratio'])
            best_params[scenario][proto_name] = best_idx
            print(f"  - {scenario}, {proto_name}: Meilleur paramètre = {proto_results[best_idx]['param_value']}")
    
    # Parcourir tous les protocoles et résultats (uniquement les meilleures configurations)
    for scenario, results in [("Sans panne", no_failure_results), 
                             ("Prévisible", predictable_results), 
                             ("Aléatoire", random_results)]:
        for proto_name in ["Spray-and-Wait", "Epidemic", "Prophet"]:
            # Utiliser le meilleur paramètre trouvé pour ce protocole et ce scénario
            best_idx = best_params[scenario][proto_name]
            result = results[proto_name]['values'][best_idx]
            
            # Récupérer l'instance du protocole
            protocol = result['protocol_instance']
            param_value = result['param_value']
            
            # Ajouter plus de métadonnées au log pour analyses futures
            for log in protocol.packet_logs:
                log['scenario'] = scenario
                log['param_value'] = param_value
                
            # Collecter les logs
            packet_logs.extend(protocol.packet_logs)
    
    # Exporter les logs en CSV
    if packet_logs:
        # Créer le DataFrame
        df = pd.DataFrame(packet_logs)
        
        # Définir le chemin d'export
        os.makedirs(OUTDIR, exist_ok=True)  # S'assurer que le répertoire existe
        csv_path = os.path.join(OUTDIR, "packet_logs.csv")
        
        # Exporter en CSV
        df.to_csv(csv_path, index=False)
        print(f"  - {len(packet_logs)} logs de paquets exportés vers {csv_path}")
        
        # Calculer et exporter les métriques par paquet
        metrics_path = os.path.join(OUTDIR, "per_packet_metrics.csv")
        compute_per_packet_metrics(csv_path, metrics_path)
    else:
        print("  - Aucun log de paquet collecté")
    
    print("\n### Analyse terminée ###")
    
if __name__ == "__main__":
    main()