#!/bin/bash
# test_spray_and_wait_multihop.sh
# Script pour tester le protocole Spray-and-Wait en scénario multi-sauts

# Affichage coloré pour améliorer la lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}=== Test du Protocole Spray-and-Wait en Scénario Multi-sauts ===${NC}"
echo -e "${YELLOW}Date: $(date)${NC}"
echo

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { echo -e "${RED}Impossible d'accéder au dossier code${NC}"; exit 1; }

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo -e "${RED}Erreur: Python n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Vérifier que networkx est installé
python -c "import networkx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installation du module networkx nécessaire pour les visualisations...${NC}"
    pip install networkx
    if [ $? -ne 0 ]; then
        echo -e "${RED}Erreur: Impossible d'installer networkx. Veuillez l'installer manuellement.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Module networkx installé avec succès!${NC}"
fi

# Exécuter le test Spray-and-Wait multi-sauts
echo -e "${GREEN}${BOLD}Exécution du test du protocole Spray-and-Wait en scénario multi-sauts...${NC}"
echo -e "${YELLOW}Cette simulation peut prendre quelques minutes car elle génère des visualisations.${NC}"
echo

python test_spray_and_wait_multihop.py

test_status=$?

# Vérifier si le test s'est bien déroulé
if [ $test_status -ne 0 ]; then
    echo -e "${RED}Erreur lors du test. Arrêt du processus.${NC}"
    exit 1
fi

# Ouvrir le dossier des résultats
echo
echo -e "${GREEN}${BOLD}=== Test multi-sauts terminé avec succès ! ===${NC}"
echo -e "${YELLOW}Ouverture du dossier des résultats...${NC}"

# Retour au dossier d'origine
cd ..

# Ouvrir le dossier avec le navigateur de fichiers par défaut
open "./data_logs/protocols/spray_and_wait_multihop_test" 2>/dev/null || \
xdg-open "./data_logs/protocols/spray_and_wait_multihop_test" 2>/dev/null || \
explorer "./data_logs/protocols/spray_and_wait_multihop_test" 2>/dev/null || \
echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/protocols/spray_and_wait_multihop_test${NC}"

echo
echo -e "${BLUE}${BOLD}Pour relancer le test multisauts ultérieurement, utilisez simplement:${NC}"
echo -e "${YELLOW}./test_spray_and_wait_multihop.sh${NC}"

echo
echo -e "${BLUE}${BOLD}Points d'intérêt dans les résultats:${NC}"
echo -e "${YELLOW}1. Visualisations du réseau (network_tXX.png): ${NC}Montrent comment les copies se propagent dans le réseau"
echo -e "${YELLOW}2. Carte de chaleur (heatmap_LX.png): ${NC}Illustre la diffusion des copies au fil du temps"
echo -e "${YELLOW}3. Évolution des sauts (hops_evolution_LX.png): ${NC}Montre comment le message se propage en termes de sauts"
echo -e "${YELLOW}4. Évolution des nœuds actifs (active_copies_LX.png): ${NC}Montre l'évolution du nombre de nœuds possédant une copie"
echo -e "${YELLOW}5. Comparaison (comparaison_multihop.png): ${NC}Analyse l'impact du paramètre L sur les performances"
echo
echo -e "${BLUE}${BOLD}Nouvelles fonctionnalités:${NC}"
echo -e "${GREEN}- Time-To-Live (TTL) pour les copies: ${NC}Les copies expirent après un certain temps"
echo -e "${GREEN}- Taux de distribution contrôlé: ${NC}La vitesse de propagation des copies est ralentie"
echo -e "${GREEN}- Simulation complète: ${NC}La simulation continue même après la livraison du message"
echo -e "${GREEN}- Suivi amélioré des copies: ${NC}Historique complet des copies et statistiques détaillées"
echo -e "${GREEN}- Visualisation améliorée: ${NC}Représentation plus claire des clusters et des liens entre nœuds"
