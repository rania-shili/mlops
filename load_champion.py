
import mlflow
import mlflow.sklearn
import pandas as pd

# Se connecter à la base du projet
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Charger le modèle via son alias, sans préciser son numéro de version
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Modele charge :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# Construire un exemple de client
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# Réorganiser les colonnes selon celles du modèle
client = client[list(model.feature_names_in_)]

# Prédire : 1 = départ, 0 = reste
prediction = model.predict(client)[0]

print("Le client va-t-il partir ? ->",
      "OUI" if prediction == 1 else "NON")
