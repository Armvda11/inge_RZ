#!/bin/bash
# nettoyage_projet.sh
# Script pour nettoyer le projet en supprimant les fichiers non nécessaires

echo "### Nettoyage du projet ###"
echo "Suppression des fichiers non nécessaires à l'analyse topologique..."

# Créer des dossiers de sauvegarde
echo "Création d'un dossier de sauvegarde..."
mkdir -p ./backup/code/protocols
mkdir -p ./backup/code/simulation
mkdir -p ./backup/code
mkdir -p ./backup

# 1. Déplacer les fichiers de protocole dans la sauvegarde
echo "Sauvegarde des fichiers de protocoles..."
mv code/protocols/base.py ./backup/code/protocols/
mv code/protocols/epidemic.py ./backup/code/protocols/
mv code/protocols/prophet.py ./backup/code/protocols/
mv code/protocols/spray_and_wait.py ./backup/code/protocols/
mv code/protocols/__init__.py ./backup/code/protocols/
rmdir code/protocols

# 2. Déplacer les scripts d'analyse complexes
echo "Sauvegarde des scripts d'analyse complexes..."
mv code/analyze_robustness.py ./backup/code/
mv code/analyze_results.py ./backup/code/
mv analyse_robustesse.sh ./backup/
mv clean_figures.sh ./backup/
mv lance.sh ./backup/

# 3. Déplacer les scripts de test
echo "Sauvegarde des scripts de test..."
mv code/test_advanced_metrics.py ./backup/code/

# 4. Déplacer les fichiers de métriques avancées
echo "Sauvegarde des métriques avancées..."
mv code/simulation/advanced_metrics.py ./backup/code/simulation/
mv code/simulation/component_analysis.py ./backup/code/simulation/
mv code/simulation/visualize_degree_dist.py ./backup/code/simulation/
mv code/simulation/visualize.py ./backup/code/simulation/

# 5. Déplacer le fichier main
echo "Sauvegarde du fichier main..."
mv code/main.py ./backup/code/

echo ""
echo "### Nettoyage terminé ###"
echo "Les fichiers supprimés ont été déplacés dans le dossier ./backup"
echo "Le projet est maintenant simplifié pour l'analyse topologique uniquement."
echo ""
echo "Pour exécuter l'analyse topologique, utilisez la commande :"
echo "./analyse_topologique.sh"
