#!/bin/bash
# Script pour nettoyer les figures existantes et créer le nouveau dossier de logs

# Afficher un message d'avertissement
echo "===== Nettoyage des visualisations ====="
echo "Ce script va supprimer tous les fichiers de visualisation et logs existants."
echo "Les données brutes seront préservées."
echo ""

# Supprimer les fichiers PNG existants
echo "Suppression des visualisations PNG..."
find . -name "*.png" -delete

# Nettoyage complet du dossier figures s'il existe
if [ -d "figures" ]; then
    echo "Suppression du dossier figures..."
    rm -rf figures
fi

# Nettoyage et recréation du dossier data_logs
echo "Recréation du dossier data_logs..."
rm -rf data_logs
mkdir -p data_logs

echo "Nettoyage terminé ! Le dossier data_logs est prêt à recevoir les nouvelles données."
