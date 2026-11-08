import os
import warnings

import mlflow
from mlflow.tracking import MlflowClient

from src.evaluate import EXPERIMENT_NAME, TRACKING_URI, get_latest_run, write_summary

NAME = "karimunjawa-kelayakan"


def main():
    run = get_latest_run()
    run_id = run["run_id"]
    sha = os.getenv("GITHUB_SHA", "lokal")[:7]

    mlflow.set_tracking_uri(TRACKING_URI)
    mv = mlflow.register_model(f"runs:/{run_id}/model", NAME)
    version = str(mv.version)

    client = MlflowClient()
    client.update_model_version(
        NAME, version,
        description=f"Didaftarkan otomatis oleh CI (commit {sha}) setelah lolos validasi threshold.",
    )
    client.set_model_version_tag(NAME, version, "git_sha", sha)
    client.set_model_version_tag(NAME, version, "registered_by", "github-actions")
    client.set_model_version_tag(NAME, version, "run_name", str(run["tags.mlflow.runName"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # API stage ditandai deprecated di MLflow baru
        client.transition_model_version_stage(NAME, version, "Staging")
    client.set_registered_model_alias(NAME, "staging", version)

    final = client.get_model_version(NAME, version)
    msg = (f"Model '{NAME}' versi {version} terdaftar, stage={final.current_stage}, "
           f"alias={sorted(final.aliases)}, run={run['tags.mlflow.runName']}")
    print(msg)
    write_summary(f"## Registry update\n\n{msg}")


if __name__ == "__main__":
    main()
