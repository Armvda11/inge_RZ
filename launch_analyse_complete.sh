#!/bin/bash
# launch_analyse_complete.sh
# Script principal pour lancer toutes les analyses topologiques en une seule commande

# Affichage coloré pour améliorer la lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}=== Analyse Topologique Complète du Réseau Satellite ===${NC}"
echo -e "${YELLOW}Date: $(date)${NC}"
echo -e "${YELLOW}Projet: Analyse de la robustesse d'un réseau de nanosatellites${NC}"
echo

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { echo "Impossible d'accéder au dossier code"; exit 1; }

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo -e "${RED}Erreur: Python n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# 1. Exécuter l'analyse topologique principale
echo -e "${GREEN}${BOLD}[1/3] Exécution de l'analyse topologique principale...${NC}"
python analyze_topologie.py
analyse_status=$?

# Vérifier si l'analyse s'est bien déroulée
if [ $analyse_status -ne 0 ]; then
    echo -e "${RED}Erreur lors de l'analyse topologique. Arrêt du processus.${NC}"
    exit 1
fi

# 2. Exécuter l'analyse d'efficacité
echo
echo -e "${GREEN}${BOLD}[2/3] Exécution de l'analyse d'efficacité dédiée...${NC}"
python visualize_efficiency.py
efficacite_status=$?

# Vérifier si l'analyse d'efficacité s'est bien déroulée
if [ $efficacite_status -ne 0 ]; then
    echo -e "${YELLOW}Attention: L'analyse d'efficacité a rencontré un problème, mais on continue.${NC}"
fi

# 3. Générer un rapport récapitulatif qui combine les résultats
echo
echo -e "${GREEN}${BOLD}[3/3] Génération du rapport récapitulatif...${NC}"

# Créer un dossier pour le rapport complet s'il n'existe pas
rapport_dir="../data_logs/topologie/rapport_complet"
mkdir -p "$rapport_dir"

# Générer un rapport HTML simplifié
cat > "$rapport_dir/rapport_analyse_topologique.html" << EOF
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse Topologique du Réseau Satellite</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }
        h1, h2, h3 { color: #2c3e50; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }
        .img-container { text-align: center; margin: 20px 0; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .caption { font-style: italic; color: #666; text-align: center; margin-top: 8px; }
        .highlight { background-color: #fffacd; padding: 15px; border-left: 4px solid #ffd700; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; text-align: center; font-size: 0.9em; color: #777; }
        @media print { 
            body { font-size: 12pt; }
            .section { break-inside: avoid; }
            a { text-decoration: none; color: #000; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rapport d'Analyse Topologique du Réseau Satellite</h1>
        <p>Date de génération: $(date)</p>
        
        <div class="section">
            <h2>1. Métriques Topologiques Générales</h2>
            <p>Vue d'ensemble des 6 métriques clés pour évaluer la robustesse du réseau:</p>
            <div class="img-container">
                <img src="../metriques_topologiques.png" alt="Métriques topologiques">
                <div class="caption">Figure 1: Évolution des métriques topologiques sous différents scénarios de panne</div>
            </div>
        </div>

        <div class="section">
            <h2>2. Focus sur les Métriques d'Efficacité</h2>
            
            <h3>2.1 Efficacité Globale</h3>
            <p>L'efficacité globale mesure la rapidité moyenne de communication à travers l'ensemble du réseau.</p>
            <div class="img-container">
                <img src="../efficacite/efficacite_globale.png" alt="Efficacité globale">
                <div class="caption">Figure 2: Évolution de l'efficacité globale du réseau</div>
            </div>
            
            <h3>2.2 Efficacité Locale</h3>
            <p>L'efficacité locale mesure la résilience du réseau autour de chaque satellite.</p>
            <div class="img-container">
                <img src="../efficacite/efficacite_locale.png" alt="Efficacité locale">
                <div class="caption">Figure 3: Évolution de l'efficacité locale du réseau</div>
            </div>
            
            <h3>2.3 Comparaison des deux métriques d'efficacité</h3>
            <div class="img-container">
                <img src="../efficacite/comparaison_efficacite.png" alt="Comparaison d'efficacité">
                <div class="caption">Figure 4: Comparaison entre l'efficacité globale et locale</div>
            </div>
        </div>
        
        <div class="section">
            <h2>3. Impact des Pannes et Attaques</h2>
            
            <div class="highlight">
                <h3>Analyse comparative avant/après panne</h3>
                <p>Les variations des métriques suite aux différents scénarios de panne:</p>
                <div class="img-container">
                    <img src="../variations_metriques.png" alt="Variations des métriques">
                    <div class="caption">Figure 5: Impact des différents scénarios sur les métriques topologiques</div>
                </div>
                
                <div class="img-container">
                    <img src="../efficacite/variations_efficacite.png" alt="Variations d'efficacité">
                    <div class="caption">Figure 6: Impact spécifique sur les métriques d'efficacité</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Projet: Analyse de la robustesse d'un réseau de nanosatellites</p>
            <p>Rapport généré automatiquement par le script d'analyse topologique</p>
        </div>
    </div>
</body>
</html>
EOF

# Copier les fichiers CSV de résultats dans le dossier du rapport si existants
cp -f ../data_logs/topologie/metriques_topologiques.csv "$rapport_dir/" 2>/dev/null || true
cp -f ../data_logs/topologie/rapport_avant_apres_panne.csv "$rapport_dir/" 2>/dev/null || true
cp -f ../data_logs/topologie/efficacite/rapport_efficacite.csv "$rapport_dir/" 2>/dev/null || true

echo -e "${BLUE}${BOLD}Rapport HTML généré: ${NC}${rapport_dir}/rapport_analyse_topologique.html"

# Ouvrir le rapport HTML dans le navigateur par défaut
echo
echo -e "${GREEN}${BOLD}=== Analyse topologique terminée avec succès ! ===${NC}"
echo -e "${YELLOW}Ouverture du rapport récapitulatif...${NC}"

# Retour au dossier d'origine
cd ..

# Ouvrir le rapport HTML avec le navigateur par défaut
open "${rapport_dir}/rapport_analyse_topologique.html" 2>/dev/null || \
xdg-open "${rapport_dir}/rapport_analyse_topologique.html" 2>/dev/null || \
explorer "${rapport_dir}/rapport_analyse_topologique.html" 2>/dev/null || \
echo -e "${YELLOW}Rapport HTML généré dans: ${rapport_dir}/rapport_analyse_topologique.html${NC}"

# Afficher les liens vers les différents résultats
echo
echo -e "${BOLD}Résultats disponibles dans:${NC}"
echo -e "  - Métriques générales: ${YELLOW}data_logs/topologie/${NC}"
echo -e "  - Métriques d'efficacité: ${YELLOW}data_logs/topologie/efficacite/${NC}"
echo -e "  - Rapport complet: ${YELLOW}data_logs/topologie/rapport_complet/${NC}"

echo
echo -e "${BLUE}${BOLD}Pour relancer l'analyse complète ultérieurement, utilisez simplement:${NC}"
echo -e "${YELLOW}./launch_analyse_complete.sh${NC}"
