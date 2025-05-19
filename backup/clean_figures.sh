#!/bin/bash
# clean_figures.sh
# Script pour nettoyer les figures générées par les simulations précédentes

echo "=== Nettoyage des figures et résultats précédents ==="

# Vérifier si le répertoire data_logs existe
if [ -d "data_logs" ]; then
    # Supprimer uniquement les fichiers PNG (figures) et CSV (logs)
    rm -f data_logs/*.png
    rm -f data_logs/*.csv
    echo "  - Figures et logs supprimés"
else
    # Créer le répertoire s'il n'existe pas
    mkdir -p data_logs
    echo "  - Répertoire data_logs créé"
fi

echo "=== Nettoyage terminé ==="
