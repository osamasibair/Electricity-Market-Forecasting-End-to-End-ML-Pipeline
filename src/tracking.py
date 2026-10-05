"""Log training runs to MLflow; the settings, the results and the saved model files."""
import os

import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))


def log_run(experiment, params, metrics, files, model=None, registered_name=None, input_example=None):
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics({name: float(value) for name, value in metrics.items()})
        for path in files:
            mlflow.log_artifact(str(path))
        if model is not None:
            mlflow.lightgbm.log_model(model, name="model", input_example=input_example, registered_model_name=registered_name)