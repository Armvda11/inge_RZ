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
