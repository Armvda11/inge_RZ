#!/bin/zsh
# setup.sh
# Script de configuration et d'activation de l'environnement

# Affichage coloré
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}=== Configuration de l'environnement pour l'analyse du réseau satellite ===${NC}"

# Déterminer quelle commande Python utiliser
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
    echo -e "${GREEN}Python 3 trouvé : $(python3 --version)${NC}"
elif command -v python &> /dev/null; then
    PY_CMD="python"
    echo -e "${GREEN}Python trouvé : $(python --version)${NC}"
else
    echo -e "${RED}Erreur: Python n'est pas installé.${NC}"
    exit 1
fi

# Supprimer l'ancien environnement virtuel s'il existe
if [ -d ".venv" ]; then
    echo -e "\n${YELLOW}Suppression de l'ancien environnement virtuel...${NC}"
    rm -rf .venv
fi

# Créer un nouvel environnement virtuel
echo -e "\n${YELLOW}Création d'un nouvel environnement virtuel Python...${NC}"
$PY_CMD -m venv .venv

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur lors de la création de l'environnement virtuel.${NC}"
    echo -e "${YELLOW}Vérifiez que le module venv est installé.${NC}"
    exit 1
fi

# Activer l'environnement virtuel
echo -e "\n${YELLOW}Activation de l'environnement virtuel...${NC}"
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur lors de l'activation de l'environnement virtuel.${NC}"
    exit 1
fi

# Installer les dépendances
echo -e "\n${YELLOW}Installation des dépendances...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur lors de l'installation des dépendances.${NC}"
    exit 1
fi

# Vérifier les installations
echo -e "\n${YELLOW}Vérification des installations...${NC}"
$PY_CMD -c "import numpy; import pandas; import networkx; import matplotlib; print('Toutes les bibliothèques requises sont installées.')"

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur: Certaines bibliothèques ne sont pas correctement installées.${NC}"
    exit 1
fi

# Créer les dossiers nécessaires s'ils n'existent pas
echo -e "\n${YELLOW}Création des dossiers pour les résultats...${NC}"
mkdir -p ./data_logs/topologie
mkdir -p ./data_logs/protocols/spray_and_wait_test
mkdir -p ./data_logs/protocols/spray_and_wait_multihop_test

echo -e "\n${GREEN}${BOLD}L'environnement a été configuré avec succès !${NC}"
echo -e "${BLUE}Pour activer l'environnement manuellement, exécutez : ${YELLOW}source .venv/bin/activate${NC}"
