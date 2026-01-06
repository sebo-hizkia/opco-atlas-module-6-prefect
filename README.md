# FastIA — Monitoring de dérive avec Prefect (serve + worker + UI)

Objectif : simuler un pipeline de supervision IA orchestré par Prefect :
- Exécution automatique toutes les X secondes (via `flow.serve(interval=X)`)
- Détection de dérive aléatoire
- Déclenchement conditionnel d’un "réentraînement"
- Tasks avec retries + retry_delay_seconds
- Logs visibles dans l’UI Prefect (http://localhost:4200)
- Exécutable en local OU dans Docker sans modifier le code Python

## Prérequis
- Docker + Docker Compose

---

## Lancer avec Docker Compose
Depuis la racine :

```bash
docker compose up --build
````

Puis ouvrir :

* UI Prefect : [http://localhost:4200](http://localhost:4200)

### Ce que vous devez voir dans l’UI

* Un flow `fastia-drift-monitoring`
* Des runs qui apparaissent toutes les `SERVE_INTERVAL_SECONDS` secondes
* Des logs clairs avec deux cas :

  * `OK: pas de dérive...`
  * `RÉENTRAÎNEMENT déclenché...` puis `RÉENTRAÎNEMENT terminé...`
* En cas d’échec simulé de retrain, vous verrez les retries (et leur délai)

---

## Variables d’environnement

* `SERVE_INTERVAL_SECONDS` (défaut: 10)
* `DRIFT_THRESHOLD` (défaut: 0.5)
* `PREFECT_API_URL` (nécessaire pour pointer vers l’API Prefect)
* `PREFECT_WORK_POOL` (défaut: local-pool)

---

## Notes d’implémentation

* La dérive est simulée par `random.random()`
* Le retrain est déclenché si `drift_score < DRIFT_THRESHOLD`
* `retrain_model` a volontairement une probabilité d’échec pour démontrer `retries` et `retry_delay_seconds`


[1]: https://docs.prefect.io/v3/how-to-guides/self-hosted/docker-compose "How to run the Prefect Server via Docker Compose"
