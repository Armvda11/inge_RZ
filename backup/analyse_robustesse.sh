#!/bin/bash
# filepath: /Users/hermas/Desktop/Projets/inge_RZ/analyse_robustesse.sh

# Script amélioré pour lancer l'analyse avancée de la robustesse du réseau satellite
# Ce script permet de lancer l'analyse complète ou seulement les tests

# Fonction d'aide
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help     Afficher cette aide"
    echo "  -t, --test     Exécuter seulement les tests"
    echo "  -v, --verbose  Mode verbeux avec plus de détails"
    echo "  -c, --clean    Nettoyer les fichiers de résultats avant analyse"
    echo
    echo "Sans option, le script lance l'analyse complète."
    exit 0
}

# Initialiser les variables
RUN_TESTS=false
VERBOSE=false
CLEAN=false

# Traiter les options de ligne de commande
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Option inconnue: $1"
            show_help
            ;;
    esac
done

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { echo "Impossible d'accéder au dossier code"; exit 1; }

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo "Python n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Nettoyer les résultats si demandé
if $CLEAN; then
    echo "Nettoyage des résultats précédents..."
    rm -rf ../data_logs/advanced/*.png ../data_logs/advanced/*.csv ../data_logs/advanced/*.txt
fi

# Exécuter les tests si demandé
if $VERBOSE; then
    echo "Mode verbeux activé."
fi

if $RUN_TESTS; then
    echo "Exécution des tests des métriques avancées..."
    if $VERBOSE; then
        python -u test_advanced_metrics.py
    else
        python test_advanced_metrics.py
    fi
    echo "Tests terminés."
else
    # Exécuter l'analyse complète
    echo "Exécution de l'analyse avancée de robustesse..."
    if $VERBOSE; then
        python -u analyze_robustness.py
    else
        python analyze_robustness.py
    fi
    echo "Analyse complète terminée."
fi

# Ouvrir le dossier des résultats
echo "Ouverture du dossier des résultats..."
cd ..
open ./data_logs/advanced/ 2>/dev/null || xdg-open ./data_logs/advanced/ 2>/dev/null || explorer ./data_logs/advanced/ 2>/dev/null || echo "Impossible d'ouvrir automatiquement le dossier. Veuillez l'ouvrir manuellement: ./data_logs/advanced/"

echo "Opération terminée avec succès!"
