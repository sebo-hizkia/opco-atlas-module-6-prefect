# FastIA – Template modulaire (Frontend + Backend) avec Docker & CI

## Objectif
Template minimaliste et reproductible :
- Frontend Streamlit : envoie un entier
- Backend FastAPI : valide l'entrée avec Pydantic et renvoie le carré
- Loguru : logs lisibles dans les deux services
- Pytest : test unitaire sur la logique métier
- GitHub Actions : exécute les tests à chaque push / PR
- Docker & Docker Compose : environnement isolé

## Architecture
- `frontend/app.py` : UI Streamlit + appel REST
- `backend/main.py` : API FastAPI (3 routes)
- `backend/modules/calcul.py` : logique métier (calcul du carré)
- `backend/tests/test_calcul.py` : tests pytest
- `docker-compose.yml` : lance uniquement frontend + backend
- `.github/workflows/test.yml` : CI

## Arborescence du projet

.
├── docker-compose.yml
├── README.md
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── modules/
│   │   ├── __init__.py
│   │   └── calcul.py
│   └── tests/
│       ├── __init__.py
│       └── test_calcul.py
└── .github/
    └── workflows/
        └── test.yml


## Routes API
- `GET /` : message de statut
- `GET /health` : healthcheck
- `POST /calcul` : body JSON `{ "n": 5 }` → `{ "n": 5, "carre": 25 }`

## Lancer en local (Docker)
```bash
docker compose up --build
