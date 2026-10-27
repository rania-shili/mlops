
import mlflow
from mlflow import MlflowClient

# Utiliser la base du projet
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Retrouver le run nommé essai-2
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

if runs.empty:
    raise ValueError("Le run essai-2 est introuvable.")

run_id = runs.iloc[0]["run_id"]

# Enregistrer ce run comme une nouvelle version de churn-model
v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)
print("Nouvelle version creee :", v.version)

# Mémoriser la version actuellement désignée par champion
actuel = client.get_model_version_by_alias(
    "churn-model", "champion"
)
print("Champion actuel : version", actuel.version)

# Déplacer temporairement l'alias vers la nouvelle version
client.set_registered_model_alias(
    "churn-model", "champion", v.version
)
print("Apres deplacement : champion -> version",
      client.get_model_version_by_alias(
          "churn-model", "champion"
      ).version)

# Revenir à la version initiale
client.set_registered_model_alias(
    "churn-model", "champion", actuel.version
)
print("Apres retour : champion -> version",
      client.get_model_version_by_alias(
          "churn-model", "champion"
      ).version)
