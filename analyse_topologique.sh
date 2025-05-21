#!/bin/zsh
# analyse_topologique.sh
# Script simplifié pour l'analyse topologique du réseau satellite
# Analyse centrée sur 4 métriques topologiques (sans métriques d'efficacité)

# Affichage coloré
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}=== Analyse topologique du réseau satellite ===${NC}"
echo -e "${YELLOW}Date: $(date)${NC}\n"

# Activer l'environnement virtuel Python s'il existe
if [ -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Activation de l'environnement Python...${NC}"
    source .venv/bin/activate
else
    echo -e "${RED}Environnement Python non configuré!${NC}"
    echo -e "${YELLOW}Veuillez d'abord exécuter ./setup.sh${NC}"
    exit 1
fi

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

# Exécuter l'analyse topologique
echo -e "\n${BLUE}${BOLD}Exécution de l'analyse topologique...${NC}"
$PY_CMD analyze_topologie.py

if [ $? -eq 0 ]; then
    # Ouvrir le dossier des résultats
    cd ..
    echo -e "\n${GREEN}${BOLD}Analyse topologique terminée avec succès!${NC}"
    echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/topologie/${NC}"
    open ./data_logs/topologie/ 2>/dev/null || \
    xdg-open ./data_logs/topologie/ 2>/dev/null || \
    explorer ./data_logs/topologie/ 2>/dev/null
else
    cd ..
    echo -e "\n${RED}Erreur lors de l'analyse topologique.${NC}"
fi
