import os
import time
import io
from typing import Dict

import numpy as np
from PIL import Image
from sqlalchemy import create_engine, text

import mlflow
import mlflow.tensorflow

from prefect import flow, task, get_run_logger

# =====================================================
# CONFIG
# =====================================================
SERVE_INTERVAL_SECONDS = int(os.getenv("SERVE_INTERVAL_SECONDS", "3600"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/mnist"
)

MIN_CORRECTIONS = int(os.getenv("MIN_CORRECTIONS", "10"))
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mnist-production")

MODEL_OUTPUT_PATH = "/models/mnist_latest.h5"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# =====================================================
# TASKS
# =====================================================

@task
def compute_production_metrics() -> Dict[str, float]:
    logger = get_run_logger()
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        total_predictions = conn.execute(
            text("SELECT COUNT(*) FROM prediction_log")
        ).scalar() or 0

        total_corrections = conn.execute(
            text("SELECT COUNT(*) FROM feedback")
        ).scalar() or 0

    accuracy = (
        1 - total_corrections / total_predictions
        if total_predictions > 0 else 1.0
    )

    metrics = {
        "total_predictions": total_predictions,
        "total_corrections": total_corrections,
        "production_accuracy": accuracy,
    }

    logger.info(f"Métriques production : {metrics}")
    return metrics


@task
def should_retrain(metrics: Dict[str, float]) -> bool:
    logger = get_run_logger()

    decision = (
        metrics["total_corrections"] >= MIN_CORRECTIONS
        or metrics["production_accuracy"] < ACCURACY_THRESHOLD
    )

    logger.warning(
        f"Retrain={decision} | "
        f"corrections={metrics['total_corrections']} | "
        f"accuracy={metrics['production_accuracy']:.3f}"
    )
    return decision


@task(retries=2, retry_delay_seconds=10)
def retrain_model(metrics: Dict[str, float]) -> str:
    logger = get_run_logger()
    logger.warning("🔁 DÉMARRAGE DU RÉENTRAÎNEMENT (MLflow)")

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    )
    from tensorflow.keras.optimizers import Adam

    # -------------------------------------------------
    # MLflow Run
    # -------------------------------------------------
    with mlflow.start_run(run_name=f"retrain-{int(time.time())}"):

        # Log paramètres
        mlflow.log_param("epochs", 3)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 1e-3)

        # -------------------------------------------------
        # Dataset MNIST
        # -------------------------------------------------
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train / 255.0
        x_train = x_train[..., np.newaxis]

        # -------------------------------------------------
        # Feedback utilisateur
        # -------------------------------------------------
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT image, true_label FROM feedback")
            ).fetchall()

        if rows:
            imgs, labels = [], []
            for img_bytes, label in rows:
                img = Image.open(io.BytesIO(img_bytes)).convert("L")
                imgs.append(np.array(img) / 255.0)
                labels.append(label)

            x_user = np.array(imgs)[..., np.newaxis]
            y_user = np.array(labels)

            x_train = np.concatenate([x_train, x_user])
            y_train = np.concatenate([y_train, y_user])

            mlflow.log_metric("user_samples", len(x_user))
        else:
            mlflow.log_metric("user_samples", 0)

        # -------------------------------------------------
        # Modèle
        # -------------------------------------------------
        model = Sequential([
            Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=3,
            batch_size=64,
            verbose=1,
        )

        final_accuracy = history.history["accuracy"][-1]
        mlflow.log_metric("train_accuracy", final_accuracy)

        # -------------------------------------------------
        # Sauvegarde + artefact MLflow
        # -------------------------------------------------
        os.makedirs("/models", exist_ok=True)
        model.save(MODEL_OUTPUT_PATH)

        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="mnist-cnn"
        )

        # Log métriques production
        for k, v in metrics.items():
            mlflow.log_metric(f"prod_{k}", v)

        logger.warning(
            f"✅ RÉENTRAÎNEMENT TERMINÉ | accuracy={final_accuracy:.3f}"
        )

        return mlflow.active_run().info.run_id


# =====================================================
# FLOW
# =====================================================

@flow(name="mnist-monitoring-mlflow")
def monitoring_flow():
    logger = get_run_logger()
    logger.info("🚀 Monitoring MNIST + MLflow")

    metrics = compute_production_metrics()

    if should_retrain(metrics):
        retrain_model(metrics)
    else:
        logger.info("✅ Modèle OK – pas de retrain")

    logger.info("🏁 Fin du cycle")

if __name__ == "__main__":
    monitoring_flow.serve(
        name="mnist-monitoring-mlflow",
        interval=SERVE_INTERVAL_SECONDS,
        tags=["mnist", "mlflow", "retraining"],
    )
