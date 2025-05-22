#!/usr/bin/env python3
# batch_runs.py
"""
Script d'automatisation pour les tests par lots des protocoles de routage DTN
(Spray & Wait et PROPHET) sous différentes conditions de pannes dynamiques.

Ce script permet de:
1. Exécuter N runs pour chaque configuration de protocole et mode de panne
2. Collecter les métriques de performance (taux de livraison, délai, overhead...)
3. Calculer des statistiques agrégées (moyenne, écart-type)
4. Générer un fichier CSV de résultats pour analyse ultérieure

Usage:
    python batch_runs.py --protocol [spray_and_wait|prophet|all] --mode [continuous|cascade|targeted_dynamic] 
                         --failure-rate [0-1] --runs [N] --output-csv [fichier.csv]
"""

import os
import sys
import argparse
import subprocess
import json
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from tqdm import tqdm  # Pour l'affichage de barres de progression

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Analyse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments analysés
    """
    parser = argparse.ArgumentParser(
        description="Tests par lots pour les protocoles de routage DTN sous conditions de pannes"
    )
    
    # Arguments principaux
    parser.add_argument('--protocol', type=str, required=True, 
                      choices=['spray_and_wait', 'prophet', 'all'],
                      help='Protocole à tester')
    
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['continuous', 'cascade', 'targeted_dynamic'],
                      help='Mode de panne dynamique')
    
    parser.add_argument('--failure-rate', type=float, required=True,
                      help='Taux de panne (entre 0 et 1)')
    
    parser.add_argument('--runs', type=int, required=True,
                      help='Nombre de répétitions à exécuter')
    
    parser.add_argument('--output-csv', type=str, required=True,
                      help='Chemin du fichier CSV de sortie')
    
    # Options supplémentaires
    parser.add_argument('--parallel', type=int, default=os.cpu_count(),
                      help='Nombre de processus en parallèle (défaut: nombre de CPU)')
    
    parser.add_argument('--dry-run', action='store_true',
                      help="Mode test: n'exécute pas les simulations mais génère des données aléatoires")
    
    args = parser.parse_args()
    
    # Validation des arguments
    if args.failure_rate < 0 or args.failure_rate > 1:
        parser.error("Le taux de panne doit être entre 0 et 1")
    
    if args.runs <= 0:
        parser.error("Le nombre de runs doit être positif")
    
    return args

def run_simulation(protocol: str, mode: str, failure_rate: float, run_id: int, dry_run: bool) -> Dict:
    """
    Exécute une simulation pour un protocole et un mode de panne donnés.
    
    Args:
        protocol: Nom du protocole ('spray_and_wait' ou 'prophet')
        mode: Mode de panne ('continuous', 'cascade', 'targeted_dynamic')
        failure_rate: Taux de panne (0-1)
        run_id: Identifiant du run
        dry_run: Si True, génère des données aléatoires au lieu d'exécuter la simulation
        
    Returns:
        Dict: Résultats de la simulation
    """
    # Pour le mode dry-run, générer des données aléatoires
    if dry_run:
        import random
        
        # Simuler un temps d'exécution variable selon le protocole pour plus de réalisme
        sleep_time = random.uniform(0.05, 0.2)
        time.sleep(sleep_time)
        
        # Afficher des informations détaillées sur le run courant (désactivé en mode parallèle)
        if random.random() < 0.05:  # Afficher seulement 5% des runs pour éviter de surcharger le terminal
            print(f"[Dry-Run #{run_id:04d}] Simulation {protocol} - mode {mode} - taux {failure_rate}")
        
        # Générer des données aléatoires réalistes selon le protocole et le mode
        # Les taux de livraison et autres métriques varient selon le protocole et le taux de panne
        delivered_chance = 0.8 - (failure_rate * (0.7 if protocol == 'prophet' else 0.6))
        delivered = random.random() < delivered_chance
        
        # Valeurs qui ont du sens pour chaque métrique et protocole
        if delivered:
            # Spray & Wait est généralement plus rapide mais moins efficient en nombre de copies
            if protocol == 'spray_and_wait':
                delay = random.uniform(5, 15 + failure_rate * 10)  
                hops = random.randint(2, 6)
                copies = random.randint(4, 10 + int(failure_rate * 15))
            else:  # Prophet
                delay = random.uniform(8, 20 + failure_rate * 15)  
                hops = random.randint(3, 8)
                copies = random.randint(5, 8 + int(failure_rate * 10))
                
            overhead = copies / (hops or 1)  # Éviter division par zéro
        else:
            delay = None
            hops = None
            copies = random.randint(3, 12) if protocol == 'spray_and_wait' else random.randint(4, 15)
            overhead = float('inf')  # Infini si non livré
        
        # Résultat avec des métriques supplémentaires pour plus d'informations
        return {
            'run_id': run_id,
            'protocol': protocol,
            'mode': mode,
            'failure_rate': failure_rate,
            'delivered': delivered,
            'delay': delay,
            'hops': hops, 
            'copies': copies,
            'overhead': overhead,
            'failure': 0 if delivered else 1,
            'simulation_time': sleep_time,
            'timestamp': time.time()
        }
    
    # Mode réel: exécuter la simulation
    script_name = f"test_{protocol}_dynamic_failures.py"
    output_file = f"tmp_result_{protocol}_{mode}_{run_id}_{int(time.time())}.json"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    
    # Construire la commande
    cmd = [
        "python",
        script_name,
        "--mode", mode,
        "--failure-rate", str(failure_rate),
        "--output", output_path,
        "--batch-mode"
    ]
    
    try:
        # Exécuter la commande
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Erreur lors de l'exécution de {script_name} (run {run_id}):")
            logger.error(stderr.decode('utf-8'))
            return {
                'run_id': run_id,
                'protocol': protocol,
                'mode': mode,
                'failure_rate': failure_rate,
                'error': stderr.decode('utf-8')
            }
        
        # Attendre que le fichier de sortie soit écrit
        timeout = 5  # secondes
        while not os.path.exists(output_path) and timeout > 0:
            time.sleep(0.5)
            timeout -= 0.5
            
        # Lire les résultats
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                results = json.load(f)
                
            # Nettoyer le fichier temporaire
            try:
                os.remove(output_path)
            except:
                pass
                
            # Ajouter les informations de contexte
            results['run_id'] = run_id
            results['protocol'] = protocol
            results['mode'] = mode
            results['failure_rate'] = failure_rate
            
            # Calculer failure = 1 - (delivered ? 1 : 0)
            results['failure'] = 0 if results.get('delivered', False) else 1
            
            return results
        else:
            logger.error(f"Fichier de résultats non trouvé pour le run {run_id}")
            return {
                'run_id': run_id,
                'protocol': protocol,
                'mode': mode,
                'failure_rate': failure_rate,
                'error': "Fichier de résultats non trouvé"
            }
            
    except Exception as e:
        logger.error(f"Exception lors du run {run_id}: {str(e)}")
        return {
            'run_id': run_id,
            'protocol': protocol,
            'mode': mode,
            'failure_rate': failure_rate,
            'error': str(e)
        }

def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """
    Agrège les résultats de plusieurs runs pour calculer les statistiques.
    
    Args:
        results: Liste des résultats de tous les runs
        
    Returns:
        pd.DataFrame: DataFrame avec les résultats agrégés
    """
    # Filtrer les résultats avec erreur
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        logger.error("Aucun résultat valide à agréger")
        return pd.DataFrame()
    
    # Regrouper par protocole et mode
    grouped_results = {}
    
    for result in valid_results:
        key = (result['protocol'], result['mode'], result['failure_rate'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Métriques à agréger
    metrics = ['delivered', 'delay', 'copies', 'overhead', 'hops', 'failure']
    
    # Construire le DataFrame de résultats
    rows = []
    
    for (protocol, mode, failure_rate), group in grouped_results.items():
        # Nombre de runs dans ce groupe
        num_runs = len(group)
        
        for metric in metrics:
            # Collecter les valeurs non-None pour cette métrique
            values = [r.get(metric) for r in group if r.get(metric) is not None]
            
            # Sauter si aucune valeur disponible
            if not values:
                continue
                
            # Calculer moyenne et écart-type
            mean_val = float(np.mean(values))
            std_val = float(np.std(values)) if len(values) > 1 else 0.0
            
            # Ajouter une ligne au DataFrame
            rows.append({
                'protocol': protocol,
                'mode': mode,
                'failure_rate': failure_rate,
                'runs': num_runs,
                'metric': metric,
                'mean': mean_val,
                'std': std_val
            })
    
    return pd.DataFrame(rows)

def save_to_csv(df: pd.DataFrame, output_file: str):
    """
    Sauvegarde les résultats agrégés dans un fichier CSV.
    
    Args:
        df: DataFrame contenant les résultats agrégés
        output_file: Chemin du fichier CSV de sortie
    """
    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Sauvegarder en CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Résultats sauvegardés dans {output_file}")

def main():
    """
    Fonction principale du script.
    """
    # Analyser les arguments
    args = parse_arguments()
    
    # Déterminer les protocoles à tester
    protocols = []
    if args.protocol == 'all':
        protocols = ['spray_and_wait', 'prophet']
    else:
        protocols = [args.protocol]
    
    # Afficher les paramètres d'exécution
    logger.info(f"=== Configuration des tests par lots ===")
    logger.info(f"Protocole(s): {protocols}")
    logger.info(f"Mode de panne: {args.mode}")
    logger.info(f"Taux de panne: {args.failure_rate}")
    logger.info(f"Nombre de runs: {args.runs}")
    logger.info(f"Fichier de sortie: {args.output_csv}")
    if args.dry_run:
        logger.info("Mode: DRY RUN (simulation)")
    logger.info(f"===================================")
    
    # Préparer les tâches à exécuter
    tasks = []
    for protocol in protocols:
        for run_id in range(args.runs):
            tasks.append((protocol, args.mode, args.failure_rate, run_id, args.dry_run))
    
    # Exécuter les simulations (en parallèle si demandé)
    logger.info(f"Démarrage de {len(tasks)} simulations...")
    start_time = time.time()
    
    # Créer un rapport initial sur la tâche à accomplir
    print("\n" + "=" * 80)
    print(f"EXÉCUTION DES TESTS PAR LOTS")
    print(f"Protocole(s): {', '.join(protocols)}")
    print(f"Mode de panne: {args.mode}")
    print(f"Taux de panne: {args.failure_rate}")
    print(f"Nombre total de simulations: {len(tasks)}")
    print("=" * 80 + "\n")
    
    results = []
    if args.parallel > 1:
        # Exécution parallèle
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(run_simulation, protocol, mode, failure_rate, run_id, args.dry_run)
                for protocol, mode, failure_rate, run_id, _ in tasks
            ]
            
            # Collecter les résultats avec une barre de progression
            print(f"Exécution en parallèle avec {args.parallel} processus:")
            
            # Afficher une barre de progression visuelle
            with tqdm(total=len(tasks), desc="Progression", unit="sim") as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)
                    
                    # Mettre à jour le titre de la barre de progression
                    if len(results) > 0:
                        delivered = sum(1 for r in results if r.get('delivered', False))
                        pbar.set_postfix({"Livraison": f"{delivered}/{len(results)} ({delivered/len(results)*100:.1f}%)"})
    else:
        # Exécution séquentielle
        print("Exécution séquentielle:")
        
        # Barre de progression visuelle
        with tqdm(total=len(tasks), desc="Progression", unit="sim") as pbar:
            for i, (protocol, mode, failure_rate, run_id, dry_run) in enumerate(tasks):
                result = run_simulation(protocol, mode, failure_rate, run_id, dry_run)
                results.append(result)
                pbar.update(1)
                
                # Mettre à jour le titre de la barre de progression
                if (i+1) % 10 == 0 or (i+1) == len(tasks):
                    delivered = sum(1 for r in results if r.get('delivered', False))
                    pbar.set_postfix({"Livraison": f"{delivered}/{len(results)} ({delivered/len(results)*100:.1f}%)"})
    
    # Calculer le temps d'exécution
    execution_time = time.time() - start_time
    minutes = int(execution_time // 60)
    seconds = execution_time % 60
    
    print("\n" + "=" * 80)
    print(f"RÉSULTATS DE L'EXÉCUTION")
    print(f"Temps d'exécution total: {minutes} minutes et {seconds:.1f} secondes")
    
    # Filtrer les résultats avec erreur
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"ATTENTION: {len(errors)} simulations ont échoué sur {len(results)} ({len(errors)/len(results)*100:.1f}%)")
    
    # Calculer des statistiques préliminaires pour l'affichage
    successes = [r for r in results if r.get('delivered', False)]
    success_rate = len(successes) / len(results) * 100 if results else 0
    
    print(f"Taux de livraison global: {success_rate:.1f}%")
    print("=" * 80 + "\n")
    
    # Agréger les résultats avec une barre de progression
    print("Agrégation des résultats et calcul des statistiques...")
    with tqdm(total=1, desc="Traitement statistique", bar_format="{desc}: {bar} {percentage:3.0f}%") as pbar:
        aggregated_df = aggregate_results(results)
        pbar.update(1)
    
    # Sauvegarder en CSV
    save_to_csv(aggregated_df, args.output_csv)
    
    # Afficher un résumé détaillé des résultats
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES RÉSULTATS PAR PROTOCOLE")
    print("=" * 80)
    
    # Tableau récapitulatif
    print(f"{'Protocole':<15} {'Livrés (%)':<15} {'Délai moyen':<15} {'Overhead':<15}")
    print("-" * 60)
    
    for protocol in protocols:
        protocol_results = aggregated_df[aggregated_df['protocol'] == protocol]
        
        delivered = protocol_results[protocol_results['metric'] == 'delivered']
        delivery_rate = delivered.iloc[0]['mean'] * 100 if not delivered.empty else 0
        
        delay = protocol_results[protocol_results['metric'] == 'delay']
        delay_mean = delay.iloc[0]['mean'] if not delay.empty else "N/A"
        
        overhead = protocol_results[protocol_results['metric'] == 'overhead']
        overhead_mean = overhead.iloc[0]['mean'] if not overhead.empty else "N/A"
        
        print(f"{protocol:<15} {delivery_rate:>6.1f}%{'':<8} {delay_mean:<15.2f} {overhead_mean:<15.2f}")
    
    print("\n" + "=" * 80)
    print(f"Résultats complets disponibles dans: {args.output_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
