# DST Airlines - Système de Prédiction des Retards de Vols

Bienvenue sur le dépôt du projet DST Airlines.
Ce projet de bout en bout (Data Engineering & MLOps) vise à prédire les retards de vols en combinant des données historiques (Batch) et des flux de données en temps réel (Streaming).

# Fonctionnalités Principales

- ETL Batch : Nettoyage et agrégation des données historiques dans PostgreSQL.
- Machine Learning : Entraînement d'un modèle XGBoost de classification (suivi avec MLflow).
- Streaming Temps Réel : Ingestion de vols simulés via Apache Kafka.
- Microservices API : Exposition des prédictions et des données sécurisées via FastAPI.
- Dashboarding : Interface utilisateur interactive développée avec Dash.

# Stack Technique

- Langage : Python 3.10
- Bases de données : PostgreSQL (Données froides/Gold), MongoDB (Données chaudes/Temps réel)
- Streaming : Apache Kafka & Zookeeper
- Backend / API : FastAPI, Uvicorn, Pydantic, PyJWT (Authentification & Rate Limiting)
- Machine Learning : Scikit-Learn, XGBoost, MLflow
- Déploiement : Docker & Docker Compose

# Architecture du Projet

Le projet est structuré selon les standards de l'industrie (Clean Architecture / Data Engineering) :
```
dst_airlines_project/
├── .github/workflows/     # CI/CD (Pipelines de test et déploiement automatique)
├── docs/                  # Documentation et images (diagrammes, métriques)
├── infra/                 # Scripts d'infrastructure et configurations
├── src/                   # Code source principal du projet
│   ├── api/               # Backend sécurisé avec FastAPI
│   ├── dashboard/         # Interface utilisateur Dash et assets (CSS)
│   ├── dags/              # Orchestration des pipelines (Airflow / Tâches)
│   ├── ingestion/         # Ingestion streaming (Kafka) et batch (Bronze/Silver/Gold)
│   └── ml_engine/         # Entraînement des modèles et tracking MLflow
├── tests/                 # Scripts de tests unitaires et vérification de la qualité
├── notebooks/             # Notebooks Jupyter de recherche et d'exploration
├── .env.example           # Template des variables d'environnement
├── docker-compose.yml     # Orchestration de l'infrastructure locale
└── Dockerfile             # Conteneurisation de l'application
```

# Sécurité de l'API (BFF)

L'API Backend (/api/backend.py) sert de point d'entrée unique pour le Dashboard. Elle implémente :

- Validation des données : Pydantic (Regex pour les emails et mots de passe forts).
- Rate Limiting : Protection contre les attaques par force brute (SlowAPI).
- Authentification : Utilisation de JWT (JSON Web Tokens) avec le protocole HTTPBearer.
- RBAC (Role-Based Access Control) : Routes protégées selon le rôle (user ou admin).

# Installation et Lancement (Local)

1. Démarrer l'infrastructure

Assurez-vous d'avoir Docker installé, puis lancez les bases de données et Kafka :
```
docker-compose up -d
```

2. Lancer l'API Backend (Dockerisée)
```
cd api
docker build -t dst_backend_api .
docker run -p 8001:8001 dst_backend_api
```

L'API sera accessible sur http://localhost:8001. La documentation interactive (Swagger UI) est disponible sur http://localhost:8001/docs.

Projet réalisé dans le cadre de la certification Data Engineer.