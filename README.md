# Banana & Mastercard — tableau de bord Streamlit

Application de démonstration réunissant deux sujets :

- **Bananes** : analyse exploratoire et classification d'images par un ResNet18 avec transfert d'apprentissage.
- **Mastercard (MA)** : comparaison et prévision du cours avec NeuralProphet, sur le calendrier des séances NYSE.

## Déployer sur Streamlit Community Cloud

1. Publier ce dépôt GitHub en gardant `app.py` et `requirements.txt` à la racine.
2. Dans Streamlit Community Cloud, créer une application depuis ce dépôt et sélectionner `app.py` comme fichier principal.
3. Choisir Python 3.11 dans les paramètres de déploiement.
4. Déployer. Les modèles et les petits artefacts nécessaires sont déjà versionnés dans `models/` et `artifacts/`.

Le jeu complet de 13 478 images n'est pas inclus. Les statistiques EDA sont lues depuis `artifacts/eda_summary.json` et quatre images exemples sont fournies dans `assets/samples/`. La classification reste interactive via le téléversement d'une image.

## Déploiement local

```bash
python -m venv .venv
# Windows : .venv\Scripts\Activate.ps1
# macOS/Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Les prévisions boursières utilisent le modèle NeuralProphet exporté et l'historique présent dans `artifacts/`. Une connexion réseau peut être nécessaire pour récupérer des données Yahoo Finance si l'historique local est absent.

## Fichiers conservés dans le dépôt

- `app.py`, `src/`, `.streamlit/config.toml` : application et configuration visuelle.
- `models/image_model.pt`, `models/image_metadata.json` : modèle d'image utilisé en prédiction.
- `models/mastercard_neuralprophet.np` : modèle de prévision boursière.
- `artifacts/` : statistiques EDA, métriques et séries de validation/prévision affichées par l'application.
- `assets/samples/` : une image d'exemple pour chacune des quatre classes.

Le dataset complet, les notebooks, les environnements virtuels et les poids des modèles d'entraînement non chargés par l'application sont exclus par `.gitignore`.

Dataset source : [Banana Ripeness Classification Dataset](https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset).
