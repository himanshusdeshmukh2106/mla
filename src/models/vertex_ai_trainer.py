"""
Vertex AI model training implementation
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from google.cloud import aiplatform
from google.cloud.storage import Client as GCSClient
import pandas as pd
import numpy as np

from ..interfaces import IModelTrainer, ModelMetrics
from ..exceptions import TradingSystemError

class VertexAITrainer(IModelTrainer):
    """
    Trains a model using Google Cloud Vertex AI.
    """

    def __init__(self, vertex_config: Dict[str, Any], gcp_config: Dict[str, Any]):
        """
        Initialize the Vertex AI trainer.

        Args:
            vertex_config: Configuration for Vertex AI.
            gcp_config: Configuration for Google Cloud Platform.
        """
        self.vertex_config = vertex_config
        self.gcp_config = gcp_config
        self.logger = logging.getLogger(__name__)

        # Set credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_config['service_account_key_path']

        # Initialize clients
        aiplatform.init(
            project=self.gcp_config['project_id'],
            location=self.gcp_config['region'],
            staging_bucket=f"gs://{self.vertex_config['gcs_bucket_name']}"
        )
        self.gcs_client = GCSClient()

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method is not directly used for Vertex AI training as the data
        is uploaded to GCS and processed by the training container.
        """
        self.logger.warning("prepare_data is a no-op for VertexAITrainer.")
        # This method is part of the interface but not used in the same way.
        # We'll return empty arrays to satisfy the interface.
        return np.array([]), np.array([])

    def _upload_data_to_gcs(self, data: pd.DataFrame, gcs_uri: str):
        """
        Uploads a pandas DataFrame to a GCS bucket as a CSV file.

        Args:
            data: The DataFrame to upload.
            gcs_uri: The GCS URI to upload the data to (e.g., "gs://bucket/path/to/file.csv").
        """
        bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
        blob_name = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
        
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        self.logger.info(f"Uploading data to {gcs_uri}...")
        blob.upload_from_string(data.to_csv(index=False), 'text/csv')
        self.logger.info("Upload complete.")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> str:
        """
        Trains the model on Vertex AI.

        Args:
            X_train: Training features.
            y_train: Training targets.

        Returns:
            The name of the trained model resource.
        """
        # Combine features and target for training
        training_data = pd.concat([y_train, X_train], axis=1)
        
        # Upload data to GCS
        gcs_train_uri = f"gs://{self.vertex_config['gcs_bucket_name']}/data/training.csv"
        self._upload_data_to_gcs(training_data, gcs_train_uri)

        # Define the training job
        job = aiplatform.CustomTrainingJob(
            display_name=self.vertex_config['training_job_display_name'],
            container_uri=self.vertex_config['pre_built_container'],
            script_path=os.path.join(os.path.dirname(__file__), 'training_script.py'),
            model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest",
        )

        # Define model serving container and run the job
        
        self.logger.info("Submitting training job to Vertex AI...")
        model = job.run(
            dataset=aiplatform.TabularDataset.create(gcs_source=[gcs_train_uri]),
            model_display_name=self.vertex_config['model_display_name'],
            args=["--target_column", y_train.name],
            machine_type=self.vertex_config.get("machine_type", "n1-standard-4"), # Default to n1-standard-4
            sync=True # Wait for the job to complete
        )
        self.logger.info(f"Training job completed. Model resource name: {model.resource_name}")
        
        return model.resource_name

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        Evaluation is typically done within Vertex AI using a batch prediction job.
        This method is a placeholder to satisfy the interface.
        """
        self.logger.warning("evaluate_model is a no-op for VertexAITrainer. Evaluation should be done via batch prediction jobs.")
        
        # Return dummy metrics
        return ModelMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=np.array([[0, 0], [0, 0]]),
            feature_importance={}
        )
