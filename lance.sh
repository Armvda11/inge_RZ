# Depuis le dossier du projet
# Cette version nettoie les figures existantes et exécute le projet avec le nouvel environnement

# Nettoyage des visualisations
./clean_figures.sh

# Configuration de l'environnement
echo "=== Configuration de l'environnement virtuel ==="
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Exécution du programme principal
echo -e "\n=== Exécution du programme principal ===\n"
cd code

# Récupération de la catégorie de panne depuis les arguments
CATEGORIE_PANNE=${1:-moyenne}  # Par défaut 'moyenne' si aucun argument n'est fourni

# Vérifier que la catégorie est valide
if [[ "$CATEGORIE_PANNE" != "legere" && "$CATEGORIE_PANNE" != "moyenne" && "$CATEGORIE_PANNE" != "lourde" ]]; then
  echo "Catégorie de panne non valide: $CATEGORIE_PANNE"
  echo "Utilisation: ./lance.sh [legere|moyenne|lourde]"
  exit 1
fi

echo "Catégorie de panne sélectionnée: $CATEGORIE_PANNE"
python3 main.py --categorie-panne $CATEGORIE_PANNE