import os
import tempfile

import mlflow

from rnn_model import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    SEED,
    TEST_SPLIT,
    generate_name,
    train,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "RNN-nombres-dinosaurios"

EXPERIMENTS = [
    {"top_k": 5,  "top_p": 1.00, "run_name": "top_k5_top_p1.00"},
    {"top_k": 10, "top_p": 1.00, "run_name": "top_k10_top_p1.00"},
    {"top_k": 0,  "top_p": 0.80, "run_name": "top_k0_top_p0.80"},
    {"top_k": 0,  "top_p": 0.95, "run_name": "top_k0_top_p0.95"},
    {"top_k": 10, "top_p": 0.90, "run_name": "top_k10_top_p0.90"},
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("Entrenando modelo...")
model, history, vocab, test_metrics = train()
inv_vocab = {v: k for k, v in vocab.items()}

for cfg in EXPERIMENTS:
    top_k = cfg["top_k"]
    top_p = cfg["top_p"]
    run_name = cfg["run_name"]

    print(f"\nSubiendo run: {run_name}")

    generated = [
        generate_name(model, vocab, inv_vocab, top_k=top_k, top_p=top_p, seed=SEED + i)
        for i in range(10)
    ]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "hidden_units": HIDDEN_UNITS,
            "embedding_dim": EMBEDDING_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "test_split": TEST_SPLIT,
            "vocab_size": len(vocab),
            "top_k": top_k,
            "top_p": top_p,
        })

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

        mlflow.log_metrics(test_metrics)
        mlflow.log_artifact(MODEL_SAVE_PATH, artifact_path="rnn_model")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="generated_names_", delete=False
        ) as f:
            f.write("\n".join(generated))
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="generated")
        os.unlink(tmp_path)

        print(f"  Nombres generados: {generated}")
        print(f"  Run '{run_name}' subido a MLflow.")

print("\nTodos los runs subidos exitosamente.")
print(f"Test loss     : {test_metrics['test_loss']:.4f}")
print(f"Test accuracy : {test_metrics['test_accuracy']:.4f}")
