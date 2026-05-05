import mlflow
import mlflow.pytorch

from rnn_model import rnn_model

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "RNN-nombres-dinosaurios"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.pytorch.log_model(rnn_model, artifact_path="rnn_model")
    print("Modelo subido exitosamente a MLflow.")
