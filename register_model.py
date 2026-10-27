
# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Utiliser la même base MLflow que le TP3 =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Chercher le meilleur run selon le score F1 =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Le premier run du tableau est celui qui a le meilleur score F1
if runs.empty:
    raise ValueError("Aucun run trouvé dans l'expérience churn-prediction.")

best = runs.iloc[0]
run_id = best["run_id"]

print("Meilleur run :", best["tags.mlflow.runName"],
      "| id :", run_id[:8], "...")

# ===== 4. Enregistrer le modèle dans le Registry =====
# Le nom churn-model sera créé s'il n'existe pas encore
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Modele enregistre : churn-model | version", result.version)

# ===== 5. Désigner cette version comme champion =====
client = MlflowClient()
client.set_registered_model_alias(
    "churn-model", "champion", result.version
)

print("Alias 'champion' -> version", result.version)
