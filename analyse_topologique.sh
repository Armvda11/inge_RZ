#!/bin/bash
# analyse_topologique.sh
# Script simplifié pour lancer l'analyse topologique du réseau satellite

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { echo "Impossible d'accéder au dossier code"; exit 1; }

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo "Python n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Exécuter l'analyse topologique
echo "Exécution de l'analyse topologique..."
python analyze_topologie.py

# Ouvrir le dossier des résultats
echo "Ouverture du dossier des résultats..."
cd ..
open ./data_logs/topologie/ 2>/dev/null || xdg-open ./data_logs/topologie/ 2>/dev/null || explorer ./data_logs/topologie/ 2>/dev/null || echo "Impossible d'ouvrir automatiquement le dossier. Veuillez l'ouvrir manuellement: ./data_logs/topologie/"

echo "Analyse topologique terminée avec succès!"
