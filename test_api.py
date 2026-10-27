
import requests

# Adresse de base de l'API
URL = "http://localhost:8000"

# ===== 1. Vérifier que l'API répond =====
response = requests.get(URL + "/health")
print("GET /health ->", response.json())

# ===== 2. Tester un client avec une faible fidélité =====
client = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

response = requests.post(URL + "/predict", json=client)
print("POST /predict (client 1) ->",
      response.status_code, response.json())

# ===== 3. Tester un client fidèle =====
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

response = requests.post(URL + "/predict", json=client2)
print("POST /predict (client 2) ->",
      response.status_code, response.json())

# ===== 4. Tester une donnée invalide =====
# On copie client et on remplace l'âge par du texte
mauvais = dict(client, age="abc")

response = requests.post(URL + "/predict", json=mauvais)
print("POST /predict (age='abc') ->", response.status_code)

# Afficher le message d'erreur envoyé par FastAPI
print("Message :", response.json()["detail"][0]["msg"])
