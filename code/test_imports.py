#!/usr/bin/env python3
# test_imports.py
"""
Script de test pour vérifier que tous les imports fonctionnent correctement.
"""

print("Vérification des imports...")

try:
    import config
    print("- config: OK")
except ImportError as e:
    print(f"- config: ERREUR - {e}")

try:
    from data.loader import load_data
    print("- data.loader: OK")
except ImportError as e:
    print(f"- data.loader: ERREUR - {e}")

try:
    from models.node import Node
    from models.swarm import Swarm
    print("- models: OK")
except ImportError as e:
    print(f"- models: ERREUR - {e}")

try:
    from simulation.metrics import analyze_single_graph, get_weighted_matrix
    print("- simulation.metrics: OK")
except ImportError as e:
    print(f"- simulation.metrics: ERREUR - {e}")

try:
    from simulation.failure import NodeFailureManager, simulate_with_failures
    print("- simulation.failure: OK")
except ImportError as e:
    print(f"- simulation.failure: ERREUR - {e}")
    
try:
    from protocols.base import DTNProtocol
    print("- protocols.base: OK")
except ImportError as e:
    print(f"- protocols.base: ERREUR - {e}")

try:
    from protocols.spray_and_wait import SprayAndWait
    print("- protocols.spray_and_wait: OK")
except ImportError as e:
    print(f"- protocols.spray_and_wait: ERREUR - {e}")
    
try:
    from protocols.prophet import Prophet
    print("- protocols.prophet: OK")
except ImportError as e:
    print(f"- protocols.prophet: ERREUR - {e}")
    
try:
    from performance_metrics import PerformanceTracker, generate_comparative_table
    print("- performance_metrics: OK")
except ImportError as e:
    print(f"- performance_metrics: ERREUR - {e}")

try:
    from simulation.advanced_metrics import compute_advanced_metrics, analyze_advanced_robustness
    print("- simulation.advanced_metrics: OK")
except ImportError as e:
    print(f"- simulation.advanced_metrics: ERREUR - {e}")

try:
    from analyze_results import format_change
    print("- analyze_results: OK")
except ImportError as e:
    print(f"- analyze_results: ERREUR - {e}")

print("\nVérification terminée.")
