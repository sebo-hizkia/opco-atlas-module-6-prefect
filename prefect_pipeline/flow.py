import os
import httpx
import random
import time
from prefect import flow, task, get_run_logger

# Intervalle (secondes) pour `serve(interval=...)`
SERVE_INTERVAL_SECONDS = int(os.getenv("SERVE_INTERVAL_SECONDS", "10"))

# Seuil de dérive
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))

def wait_for_prefect_api(api_url: str, timeout: int = 60):
    """
    Permet d'attendre que la base de données et l'API prefect soient démarrées le monitoring
    """
    start = time.time()
    while True:
        try:
            r = httpx.get(f"{api_url}/health")
            if r.status_code == 200:
                print("✅ Prefect API disponible")
                return
        except Exception:
            pass

        if time.time() - start > timeout:
            raise RuntimeError("❌ Prefect API indisponible après attente")

        print("⏳ Attente du serveur Prefect...")
        time.sleep(2)


@task(retries=2, retry_delay_seconds=5)
def fetch_new_batch() -> list[float]:
    """
    Simule l'arrivée d'un nouveau batch de données (ex: métriques / features).
    """
    logger = get_run_logger()
    batch = [random.random() for _ in range(10)]
    logger.info(f"Batch reçu (10 valeurs). Aperçu={batch[:3]}")
    return batch


@task
def detect_drift(batch: list[float]) -> float:
    """
    Simule une métrique de dérive via un tirage aléatoire.
    (On pourrait aussi dériver depuis le batch ; ici c'est volontairement symbolique.)
    """
    logger = get_run_logger()
    drift_score = random.random()
    logger.info(f"Score de dérive simulé = {drift_score:.3f}")
    return drift_score


@task(retries=3, retry_delay_seconds=3)
def retrain_model(drift_score: float) -> str:
    """
    Simule un réentraînement avec une chance d'échec pour démontrer retries + logs.
    """
    logger = get_run_logger()
    logger.warning(
        f"RÉENTRAÎNEMENT déclenché (drift_score={drift_score:.3f} < threshold={DRIFT_THRESHOLD})."
    )

    # Simule un job de retrain
    time.sleep(1.5)

    # Simule un échec aléatoire pour voir les retries dans l'UI
    if random.random() < 0.3:
        raise RuntimeError("Échec simulé du retrain (pour tester retries).")

    model_version = f"model-{int(time.time())}"
    logger.warning(f"RÉENTRAÎNEMENT terminé. Nouvelle version: {model_version}")
    return model_version


@task
def log_ok(drift_score: float) -> None:
    logger = get_run_logger()
    logger.info(
        f"OK: pas de dérive (drift_score={drift_score:.3f} >= threshold={DRIFT_THRESHOLD})."
    )


@flow(name="fastia-drift-monitoring")
def monitoring_flow() -> None:
    logger = get_run_logger()
    logger.info("Démarrage du cycle de monitoring...")

    batch = fetch_new_batch()
    drift_score = detect_drift(batch)

    if drift_score < DRIFT_THRESHOLD:
        _ = retrain_model(drift_score)
    else:
        log_ok(drift_score)

    logger.info("Fin du cycle de monitoring.")


if __name__ == "__main__":
    # Serve crée une "deployment-like" et déclenche des runs à intervalle fixe.
    # Important: on précise le work pool pour exécuter via le worker docker.

    # On attend que l'API soit démarrée
    api_url = os.getenv("PREFECT_API_URL")
    wait_for_prefect_api(api_url)

    monitoring_flow.serve(
        name="fastia-drift-monitoring-serve",
        interval=SERVE_INTERVAL_SECONDS,
        tags=["fastia", "monitoring", "drift"],
    )
