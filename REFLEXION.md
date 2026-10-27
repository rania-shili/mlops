\# Réflexion TP4 — MLOps Registry et API FastAPI



\## 1. Comment désigner officiellement le modèle à utiliser ?



Le Model Registry de MLflow conserve les différentes versions du modèle. L'alias `champion` indique la version que l'application doit utiliser. On peut déplacer cet alias vers une autre version sans modifier le code de l'application.



\## 2. Comment une application peut-elle utiliser le modèle ?



L'application utilise une API FastAPI. Elle envoie les informations du client au format JSON à la route `/predict`. L'API utilise le modèle MLflow pour effectuer une prédiction et renvoie le résultat.



\## 3. Que se passe-t-il quand on change de champion ?



On déplace l'alias `champion` vers la version souhaitée du modèle. L'application doit recharger le modèle pour utiliser cette nouvelle version, par exemple en redémarrant l'API. Le code du client qui appelle l'API reste inchangé.

