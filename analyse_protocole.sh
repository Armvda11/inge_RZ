#!/bin/zsh
# analyse_protocole.sh
# Script pour l'analyse des protocoles de communication (Spray-and-Wait)

# Affichage coloré
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Activer l'environnement virtuel s'il existe
if [ -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Activation de l'environnement Python...${NC}"
    source .venv/bin/activate
else
    echo -e "${RED}Environnement Python non configuré!${NC}"
    echo -e "${YELLOW}Veuillez d'abord exécuter ./setup.sh${NC}"
    exit 1
fi

# Menu pour choisir le protocole à analyser
echo -e "${BLUE}${BOLD}=== Analyse des protocoles de communication ===${NC}"
echo -e "${YELLOW}Date: $(date)${NC}\n"

echo -e "${BOLD}Choisissez le protocole à analyser:${NC}"
echo -e "  ${GREEN}1.${NC} Protocole Spray-and-Wait (standard)"
echo -e "  ${GREEN}2.${NC} Protocole Spray-and-Wait Multi-hop"
echo -e "  ${GREEN}3.${NC} Retour au menu principal"
echo -e "\n${BOLD}Votre choix:${NC}"
read protocol_choice

# Se déplacer dans le dossier du code
cd "$(dirname "$0")/code" || { 
    echo -e "${RED}Impossible d'accéder au dossier code.${NC}"
    exit 1
}

# Vérifier la commande Python à utiliser
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo -e "${RED}Python n'est pas installé.${NC}"
    exit 1
fi

case $protocol_choice in
    1)
        echo -e "\n${BLUE}${BOLD}Exécution du test du protocole Spray-and-Wait standard...${NC}"
        $PY_CMD test_spray_and_wait.py
        
        if [ $? -eq 0 ]; then
            # Ouvrir le dossier des résultats
            cd ..
            echo -e "\n${GREEN}${BOLD}Test terminé avec succès!${NC}"
            echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/protocols/spray_and_wait_test${NC}"
            open ./data_logs/protocols/spray_and_wait_test 2>/dev/null || \
            xdg-open ./data_logs/protocols/spray_and_wait_test 2>/dev/null || \
            explorer ./data_logs/protocols/spray_and_wait_test 2>/dev/null
        else
            cd ..
            echo -e "\n${RED}Erreur lors de l'exécution du test.${NC}"
        fi
        ;;
    2)
        echo -e "\n${BLUE}${BOLD}Exécution du test du protocole Spray-and-Wait Multi-hop...${NC}"
        $PY_CMD test_spray_and_wait_multihop.py
        
        if [ $? -eq 0 ]; then
            # Ouvrir le dossier des résultats
            cd ..
            echo -e "\n${GREEN}${BOLD}Test terminé avec succès!${NC}"
            echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/protocols/spray_and_wait_multihop_test${NC}"
            open ./data_logs/protocols/spray_and_wait_multihop_test 2>/dev/null || \
            xdg-open ./data_logs/protocols/spray_and_wait_multihop_test 2>/dev/null || \
            explorer ./data_logs/protocols/spray_and_wait_multihop_test 2>/dev/null
        else
            cd ..
            echo -e "\n${RED}Erreur lors de l'exécution du test.${NC}"
        fi
        ;;
    3)
        cd ..
        echo -e "\n${YELLOW}Retour au menu principal.${NC}"
        exit 0
        ;;
    *)
        cd ..
        echo -e "\n${RED}${BOLD}Choix non valide!${NC}"
        exit 1
        ;;
esac
