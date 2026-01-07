# Application MNIST avec boucle de feedback utilisateur

Ce projet met en œuvre une application complète de test en conditions réelles d’un modèle de classification MNIST, intégrant une boucle de feedback humaine.
L’objectif est de collecter des données corrigées en production afin d’améliorer progressivement le modèle via des cycles de réentraînement automatisés (étapes suivantes du projet).

## Prérequis
- Docker + Docker Compose

---

## Lancer avec Docker Compose
Depuis la racine :

```bash
docker compose up --build
````

## Accès aux différentes interfaces :

### Frontend MNIST (interface utilisateur) : http://localhost:8501

- Vous pouvez dessiner des chiffres et obtenir des prédictions
- Envoyer des corrections si nécessaire

### Backend API (FastAPI) : http://localhost:8000/docs
- Documentation interactive de l'API
- Testez les endpoints /predict et /correct

### Prefect UI (monitoring MLOps) : http://localhost:4200
- Visualisez les flows exécutés
- Surveillez le pipeline de détection de dérive
- Voir les entraînements automatiques

### MLflow UI (tracking des entraînements)

👉 http://localhost:5000

Permet de :

- Visualiser tous les entraînements du modèle

- Comparer les métriques (accuracy, loss, etc.)

- Consulter les paramètres d’entraînement

- Télécharger les artefacts (modèles, courbes)

### PostgreSQL : Port 5436 (mappé depuis 5432)
- Base de données Prefect : prefect
- Base de données MNIST : mnist
- Connexion base de données mnist : docker exec -it postgres psql -U app_user -d mnist

---

## Utilisation
1. Tester la classification MNIST

    1. Ouvrir http://localhost:8501

    2. Dessiner un chiffre dans la zone de dessin

    3. Cliquer sur "🔍 Prédire"

    4. Vérifier la prédiction affichée

2. Améliorer le modèle

    1. Si la prédiction est incorrecte :
      - Sélectionner le chiffre correct dans la liste déroulante
      - Cliquer sur "Envoyer correction"

    2. Le feedback est stocké en base pour améliorations futures

3. Monitorer le pipeline

    1. Ouvrir http://localhost:4200

    2. Naviguer vers "Deployments" → "fastia-drift-monitoring"

    3. Observer :

        - Les runs automatiques toutes les 30 secondes

        - Les logs de détection de dérive

        - Les réentraînements déclenchés

        - Les retries en cas d'échec simulé

