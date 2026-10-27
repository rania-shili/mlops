
# ===== 1. Importer les bibliothèques =====
from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===== 2. Créer l'application FastAPI =====
app = FastAPI(title="API Churn")

# ===== 3. Charger le modèle champion au démarrage =====
# Le modèle est chargé une seule fois, pas à chaque requête
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# ===== 4. Décrire et valider les informations du client =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# ===== 5. Créer la route GET /health =====
# Cette route vérifie si l'API répond
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 6. Créer la route POST /predict =====
# Le client envoie ses données au format JSON
@app.post("/predict")
def predict(client: Client):

    # Convertir les données du client en tableau pandas
    ligne = pd.DataFrame([client.model_dump()])

    # Transformer l'abonnement en colonnes numériques
    ligne = pd.get_dummies(ligne)

    # Garder les colonnes et leur ordre attendus par le modèle
    # Les colonnes absentes reçoivent la valeur 0
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Obtenir la prédiction : 1 = départ, 0 = reste
    resultat = int(model.predict(ligne)[0])

    # Renvoyer le résultat en JSON
    return {
        "churn": resultat,
        "message": "Le client va partir"
        if resultat == 1 else "Le client reste"
    }
