library(mlflow)
library(carrier)

readRenviron(".env")
TRACKING_URI <- Sys.getenv("MLFLOW_TRACKING_URI")
MLFLOW_ARTIFACT_PATH <- "artifacts"

# Configure to use tracking service
mlflow_set_tracking_uri(TRACKING_URI)
mlflow_set_experiment("test")

# log parameters
mlflow_log_param("param1", 1)
mlflow_log_param("param2", 2)

# log metrics
mlflow_log_metric("metric1", 1)
mlflow_log_metric("metric1", 2)

# store artifacts
writeLines("hello R", "data.txt")
mlflow_log_artifact("data.txt")

# Generate a model from an R function
data_frame <- data.frame(list(0:9))

model_crate <- crate(
    function(df) {
        return(data.frame(df + 5))
    }
)

mlflow_log_model(model_crate, MLFLOW_ARTIFACT_PATH)
# Update version to Staging


# BUG: there's a bug with archive_existing_versions serialization that causes it to fail 
# if it is not set to NULL
# NOTE: this will fail if model is not already created
mlflow_transition_model_version_stage(name = "addn_model_R",
                                      version = 1,
                                      stage = "Staging",
                                      archive_existing_versions = NULL,
)


run <- mlflow_end_run()
print("RUN ID")
print(run[1,1])
