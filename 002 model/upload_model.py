import os

import mlflow

from rnn_model import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    TEST_SPLIT,
    train,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "RNN-nombres-dinosaurios"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

model, history, vocab, test_metrics = train()

with mlflow.start_run():
    # ── Hyperparameters ───────────────────────────────────────────────────────
    mlflow.log_params({
        "hidden_units": HIDDEN_UNITS,
        "embedding_dim": EMBEDDING_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_split": TEST_SPLIT,
        "vocab_size": len(vocab),
    })

    # ── Per-epoch training metrics ────────────────────────────────────────────
    for epoch, (loss, acc, val_loss, val_acc) in enumerate(
        zip(
            history["loss"],
            history["accuracy"],
            history["val_loss"],
            history["val_accuracy"],
        ),
        start=1,
    ):
        mlflow.log_metrics(
            {
                "train_loss": loss,
                "train_accuracy": acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )

    # ── Final test metrics ────────────────────────────────────────────────────
    mlflow.log_metrics(test_metrics)

    # ── Model artifact ────────────────────────────────────────────────────────
    mlflow.log_artifact(MODEL_SAVE_PATH, artifact_path="rnn_model")

    print("Modelo subido exitosamente a MLflow.")
    print(f"Test loss     : {test_metrics['test_loss']:.4f}")
    print(f"Test accuracy : {test_metrics['test_accuracy']:.4f}")
