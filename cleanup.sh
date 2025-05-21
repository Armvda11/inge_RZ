#!/bin/zsh
# cleanup.sh
# Script pour nettoyer le projet et ne garder que les scripts essentiels

# Affichage coloré
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}=== Nettoyage du projet et simplification de l'architecture ===${NC}"

# Créer un répertoire d'archivage pour sauvegarder les scripts obsolètes
echo -e "${YELLOW}Création d'un dossier d'archive pour les scripts obsolètes...${NC}"
mkdir -p ./archived_scripts

# Liste des scripts à conserver
SCRIPTS_A_CONSERVER=(
    "setup.sh"                  # Nouveau script unique pour la configuration
    "run.sh"                    # Nouveau script principal pour tout lancer
    "analyse_topologique.sh"    # Script pour l'analyse topologique
    "analyse_protocole.sh"      # Script pour l'analyse des protocoles
)

# 1. Archiver les scripts à supprimer
echo -e "${YELLOW}Archivage des scripts obsolètes...${NC}"
for script in *.sh; do
    # Vérifier si le script doit être conservé
    keep=false
    for s in "${SCRIPTS_A_CONSERVER[@]}"; do
        if [ "$script" = "$s" ]; then
            keep=true
            break
        fi
    done
    
    # Si ce n'est pas un script à conserver et qu'il n'est pas le script actuel
    if [ "$keep" = false ] && [ "$script" != "cleanup.sh" ]; then
        echo -e "  - Archivage de ${YELLOW}$script${NC}"
        mv "$script" ./archived_scripts/
    fi
done

# 2. Créer les nouveaux scripts essentiels
echo -e "${GREEN}Création des scripts essentiels...${NC}"

# Script principal qui lance tout
echo -e "  - Création de ${GREEN}run.sh${NC}"
cat > run.sh << 'EOF'
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
    echo -e "  ${GREEN}1.${NC} Analyse topologique (métriques structurelles du réseau)"
    echo -e "  ${GREEN}2.${NC} Analyse des protocoles de communication"
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
EOF

# Script de configuration
echo -e "  - Création de ${GREEN}setup.sh${NC}"
cat > setup.sh << 'EOF'
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
mkdir -p ./data_logs/topologie/efficacite
mkdir -p ./data_logs/protocols/spray_and_wait_test

echo -e "\n${GREEN}${BOLD}L'environnement a été configuré avec succès !${NC}"
echo -e "${BLUE}Pour activer l'environnement manuellement, exécutez : ${YELLOW}source .venv/bin/activate${NC}"
EOF

# Script d'analyse des protocoles (combinaison de tous les scripts de test)
echo -e "  - Création de ${GREEN}analyse_protocole.sh${NC}"
cat > analyse_protocole.sh << 'EOF'
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
echo -e "  ${GREEN}2.${NC} Protocole Spray-and-Wait avec pannes"
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
            echo -e "\n${RED}Erreur lors de l'exécution du test.${NC}"
        fi
        ;;
    2)
        echo -e "\n${BLUE}${BOLD}Exécution du test du protocole Spray-and-Wait avec pannes...${NC}"
        $PY_CMD test_spray_and_wait_with_failures.py
        
        if [ $? -eq 0 ]; then
            # Ouvrir le dossier des résultats
            cd ..
            echo -e "\n${GREEN}${BOLD}Test terminé avec succès!${NC}"
            echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/protocols/spray_and_wait_failures_test${NC}"
        else
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
EOF

# Script d'analyse topologique (mise à jour)
echo -e "  - Mise à jour de ${GREEN}analyse_topologique.sh${NC}"
cat > analyse_topologique.sh << 'EOF'
#!/bin/zsh
# analyse_topologique.sh
# Script simplifié pour l'analyse topologique du réseau satellite

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

# Menu pour choisir l'analyse topologique
echo -e "${BOLD}Choisissez l'analyse topologique à effectuer:${NC}"
echo -e "  ${GREEN}1.${NC} Analyse complète (métriques structurelles)"
echo -e "  ${GREEN}2.${NC} Visualisation de l'efficacité"
echo -e "  ${GREEN}3.${NC} Retour au menu principal"
echo -e "\n${BOLD}Votre choix:${NC}"
read topo_choice

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

case $topo_choice in
    1)
        echo -e "\n${BLUE}${BOLD}Exécution de l'analyse topologique complète...${NC}"
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
            echo -e "\n${RED}Erreur lors de l'analyse topologique.${NC}"
        fi
        ;;
    2)
        echo -e "\n${BLUE}${BOLD}Exécution de la visualisation d'efficacité...${NC}"
        $PY_CMD visualize_efficiency.py
        
        if [ $? -eq 0 ]; then
            # Ouvrir le dossier des résultats
            cd ..
            echo -e "\n${GREEN}${BOLD}Visualisation d'efficacité terminée avec succès!${NC}"
            echo -e "${YELLOW}Résultats disponibles dans: ./data_logs/topologie/efficacite/${NC}"
            open ./data_logs/topologie/efficacite/ 2>/dev/null || \
            xdg-open ./data_logs/topologie/efficacite/ 2>/dev/null || \
            explorer ./data_logs/topologie/efficacite/ 2>/dev/null
        else
            echo -e "\n${RED}Erreur lors de la visualisation d'efficacité.${NC}"
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
EOF

# Rendre les scripts exécutables
echo -e "${YELLOW}Attribution des droits d'exécution aux scripts...${NC}"
chmod +x run.sh
chmod +x setup.sh
chmod +x analyse_topologique.sh
chmod +x analyse_protocole.sh

echo -e "\n${GREEN}${BOLD}Nettoyage terminé avec succès!${NC}"
echo -e "${BLUE}La nouvelle architecture du projet comporte désormais 4 scripts principaux:${NC}"
echo -e "  ${YELLOW}1.${NC} ${GREEN}run.sh${NC} - Script principal pour lancer toutes les analyses"
echo -e "  ${YELLOW}2.${NC} ${GREEN}setup.sh${NC} - Configuration de l'environnement"
echo -e "  ${YELLOW}3.${NC} ${GREEN}analyse_topologique.sh${NC} - Analyse de la topologie du réseau"
echo -e "  ${YELLOW}4.${NC} ${GREEN}analyse_protocole.sh${NC} - Analyse des protocoles de communication"
echo -e "\n${BLUE}Les autres scripts ont été archivés dans le dossier ${YELLOW}./archived_scripts/${NC}"
echo -e "\n${YELLOW}Pour commencer, exécutez: ${GREEN}./run.sh${NC}"
