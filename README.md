# mlflow-server
Tracking Server and Model Registry Server for MLflow

## Deployment
This repo will build an mlflow server to log experiment artifacts and serve models

### Example .env
```
export MLFLOW_BACKEND_STORE_URI='postgresql://username:password@host:5432/db-name'
export MLFLOW_ARTIFACTS_DESTINATION='s3://artifact-location/'
export MLFLOW_DEFAULT_ARTIFACT_ROOT='s3://artifact-location/'
export MLFLOW_REGISTRY_STORE_URI='s3://model-registry/'
export MLFLOW_TRACKING_TOKEN='a-valid-bearer-token'
export MLFLOW_TRACKING_URI='http://localhost:5555'
export MLFLOW_TRACKING_AWS_SIGV4=True
export MLFLOW_WORKERS=1
export MLFLOW_SERVE_ARTIFACTS=true
export MLFLOW_HOST='0.0.0.0'
export MLFLOW_PORT=5555
export AWS_DEFAULT_REGION=s3-region
export AWS_ACCESS_KEY_ID=key-id
export AWS_SECRET_ACCESS_KEY=secret-key
export DB_NAME=db-name
export DB_USER=user
export DB_PASSWORD=password
export DB_HOST=db-host

############
# R env vars
export MLFLOW_PYTHON_BIN=/Users/alan/dev/mlflow-stuff/venv/bin/python
export MLFLOW_BIN=/Users/alan/dev/mlflow-stuff/venv/bin/mlflow

```

### RUN
Once you have a configured .env file in the base directory:
`docker-compose up --build`

## TESTS
Integration tests can be run `python tests.py`

## Proxy
MLFLOW does not offer an authorization solution out of the box. This is usually accomplished with a proxy. In this case nginx will proxy the Authorization headers to a Django application and proxies on a 200 response, denying access on a 401 or 403.

You'll need to setup a user in the Django application Gatekeeper if you want to test out the secure proxies. Else you can expose the mlflow port directly and interact with the mlflow server directly.# mlflow-quickstart
