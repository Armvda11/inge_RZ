#!/bin/bash
# visualise_efficacite.sh
# Script pour lancer uniquement la visualisation des métriques d'efficacité du réseau satellite

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { echo "Impossible d'accéder au dossier code"; exit 1; }

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo "Python n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Exécuter la visualisation d'efficacité
echo "Exécution de la visualisation des métriques d'efficacité..."
python visualize_efficiency.py

# Ouvrir le dossier des résultats
echo "Ouverture du dossier des résultats..."
cd ..
open ./data_logs/topologie/efficacite/ 2>/dev/null || xdg-open ./data_logs/topologie/efficacite/ 2>/dev/null || explorer ./data_logs/topologie/efficacite/ 2>/dev/null || echo "Impossible d'ouvrir automatiquement le dossier. Veuillez l'ouvrir manuellement: ./data_logs/topologie/efficacite/"

echo "Visualisation d'efficacité terminée avec succès!"
