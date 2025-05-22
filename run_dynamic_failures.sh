#!/bin/zsh
# Script pour exécuter le test de pannes dynamiques avec l'environnement virtuel

# Aller au répertoire du projet
cd "$(dirname "$0")"

# Activer l'environnement virtuel
source venv/bin/activate

# Vérifier et installer les dépendances nécessaires
echo "Vérification des dépendances..."

# Pour les tableaux formatés
if ! pip list | grep -q tabulate; then
    echo "Installation de tabulate pour les tableaux formatés..."
    pip install tabulate
fi

# Pour l'export HTML
if ! pip list | grep -q jinja2; then
    echo "Installation de jinja2 pour l'export HTML..."
    pip install jinja2
fi

# Exécuter le script
cd code
python test_spray_and_wait_dynamic_failures.py

# Désactiver l'environnement virtuel à la fin
deactivate
