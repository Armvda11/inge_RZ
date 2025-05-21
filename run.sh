#!/bin/zsh
# run.sh
# Script principal pour lancer les analyses du réseau de nanosatellites

# Affichage coloré
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Vérifier si l'environnement est activé
activate_env() {
    if [ -f ".venv/bin/activate" ]; then
        echo -e "${YELLOW}Activation de l'environnement Python...${NC}"
        source .venv/bin/activate
        return 0
    else
        echo -e "${RED}Environnement Python non configuré.${NC}"
        echo -e "${YELLOW}Exécution de setup.sh pour configurer l'environnement...${NC}"
        ./setup.sh
        if [ $? -ne 0 ]; then
            echo -e "${RED}Échec de la configuration de l'environnement.${NC}"
            return 1
        fi
        source .venv/bin/activate
        return 0
    fi
}

# Menu principal
show_menu() {
    echo -e "${BLUE}${BOLD}=== Analyse du réseau de nanosatellites ===${NC}"
    echo -e "${YELLOW}Date: $(date)${NC}\n"
    echo -e "${BOLD}Choisissez une option :${NC}"
    echo -e "  ${GREEN}1.${NC} Analyse topologique (métriques structurelles)"
    echo -e "  ${GREEN}2.${NC} Analyse du protocole Spray-and-Wait"
    echo -e "  ${GREEN}3.${NC} Configuration de l'environnement"
    echo -e "  ${GREEN}q.${NC} Quitter"
    echo -e "\n${BOLD}Votre choix :${NC}"
}

# Boucle principale
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            activate_env || continue
            echo -e "\n${BLUE}${BOLD}Lancement de l'analyse topologique...${NC}"
            ./analyse_topologique.sh
            ;;
        2)
            activate_env || continue
            echo -e "\n${BLUE}${BOLD}Lancement de l'analyse des protocoles...${NC}"
            ./analyse_protocole.sh
            ;;
        3)
            echo -e "\n${BLUE}${BOLD}Configuration de l'environnement...${NC}"
            ./setup.sh
            ;;
        q|Q)
            echo -e "\n${YELLOW}Au revoir !${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}${BOLD}Choix non valide !${NC}"
            ;;
    esac
    
    echo -e "\n${YELLOW}Appuyez sur Entrée pour continuer...${NC}"
    read
    clear
done
