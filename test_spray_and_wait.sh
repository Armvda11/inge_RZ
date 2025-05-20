#!/bin/bash
# test_spray_and_wait.sh
# Script pour tester le protocole Spray-and-Wait

# Affichage coloré pour améliorer la lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}=== Test du Protocole Spray-and-Wait ===${NC}"
echo -e "${YELLOW}Date: $(date)${NC}"
echo

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { echo "Impossible d'accéder au dossier code"; exit 1; }

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo -e "${RED}Erreur: Python n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Exécuter le test Spray-and-Wait
echo -e "${GREEN}${BOLD}Exécution du test du protocole Spray-and-Wait...${NC}"
python test_spray_and_wait.py

test_status=$?

# Vérifier si le test s'est bien déroulé
if [ $test_status -ne 0 ]; then
    echo -e "${RED}Erreur lors du test. Arrêt du processus.${NC}"
    exit 1
fi

# Ouvrir le dossier des résultats
echo
echo -e "${GREEN}${BOLD}=== Test terminé avec succès ! ===${NC}"
echo -e "${YELLOW}Ouverture du dossier des résultats...${NC}"

# Retour au dossier d'origine
cd ..

# Ouvrir le dossier avec le navigateur de fichiers par défaut
open "./data_logs/protocols/spray_and_wait_test" 2>/dev/null || \
xdg-open "./data_logs/protocols/spray_and_wait_test" 2>/dev/null || \
explorer "./data_logs/protocols/spray_and_wait_test" 2>/dev/null || \
echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/protocols/spray_and_wait_test${NC}"

echo
echo -e "${BLUE}${BOLD}Pour relancer le test ultérieurement, utilisez simplement:${NC}"
echo -e "${YELLOW}./test_spray_and_wait.sh${NC}"
