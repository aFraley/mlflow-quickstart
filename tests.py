"""Integration tests for mlflow server"""
import logging
import os
import pathlib
import unittest

import mlflow
import pandas as pd

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = "test"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)

MLFLOW_ARTIFACT_PATH = "artifacts"
LOG_LEVEL_ENV_VAR = os.environ.get("LOG_LEVEL", "WARNING")
LOG_LEVEL = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
}.get(LOG_LEVEL_ENV_VAR, logging.WARNING)

if MLFLOW_TRACKING_URI is None:
    raise Exception("No Tracking URI is set, make sure to run `. ./.env`")

logging.basicConfig(level=LOG_LEVEL)


class AddN(mlflow.pyfunc.PythonModel):
    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)


class MLFlowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # set tracking server URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # set experiment to use
        try:
            cls.experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
        except mlflow.exceptions.MlflowException as e:
            logging.info(e.args[0])
            cls.experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)

        # construct and save the model
        add5_model = AddN(n=5)
        mlflow.pyfunc.log_model(
            artifact_path=MLFLOW_ARTIFACT_PATH,
            python_model=add5_model,
            registered_model_name="add5_model",
            input_example=pd.DataFrame([range(10)]),
        )

    # NOTE: I'd like to have a tearDown to delete the generated exp.
    # but mlflow does not support reusing experiment names.
    # logically it makes sense from a scientific approach; you can never
    # reuse an experiment. But it's just a name, in essence here. -__-
    # They may update this in the future, hence thise note to future us.

    def test_can_log_param(self):
        mlflow.log_param("param1", 1)

    def test_can_log_metric(self):
        mlflow.log_metric("metric1", 1)
        mlflow.log_metric("metric1", 2.5)

    def test_can_log_model(self):
        # construct and save the model
        add5_model = AddN(n=5)
        mlflow.pyfunc.log_model(
            artifact_path=MLFLOW_ARTIFACT_PATH,
            python_model=add5_model,
            registered_model_name="add5_model",
            input_example=pd.DataFrame([range(10)]),
        )

    def test_can_load_model(self):
        model_name = "add5_model"
        version = 1

        # load model from tracking api
        loaded_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{version}"
        )

        # evaluate the add5 model
        model_input = pd.DataFrame([range(10)])
        model_output = loaded_model.predict(model_input)
        self.assertTrue(model_output.equals(pd.DataFrame([range(5, 15)])))

    def test_can_log_artifacts(self):

        # Log an artifact (output file)
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")

        with open("artifacts/test.txt", "w") as f:
            f.write("hello decker!")

        with open("artifacts/test2.txt", "w") as f:
            f.write("Science!")

        mlflow.log_artifacts("outputs")

    def test_can_transition_model_stage(self):

        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name="add5_model", version=1, stage="Staging"
        )


if __name__ == "__main__":
    unittest.main()
