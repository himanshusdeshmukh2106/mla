"""
Verifies the connection to Google Cloud Vertex AI and checks credentials.
"""

import sys
import os
import logging

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_manager import ConfigManager
from src.logger import setup_logging

def verify_connection():
    """
    Initializes the Vertex AI client and performs a simple read operation
    to verify the connection and credentials.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Vertex AI connection verification...")

    try:
        # 1. Load Configuration
        logger.info("Loading configuration...")
        config_manager = ConfigManager(config_dir="config")
        vertex_config = config_manager.get_config('vertex_config.yaml')
        gcp_config = vertex_config.get('gcp', {})
        
        if not all([gcp_config.get('project_id'), gcp_config.get('region'), gcp_config.get('service_account_key_path')]):
            logger.error("GCP configuration is incomplete. Please check config/vertex_config.yaml")
            return False

        logger.info(f"Project ID: {gcp_config['project_id']}")
        logger.info(f"Region: {gcp_config['region']}")

        # 2. Set Credentials
        key_path = gcp_config['service_account_key_path']
        if not os.path.exists(key_path):
            logger.error(f"Service account key file not found at: {key_path}")
            return False
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        logger.info("GOOGLE_APPLICATION_CREDENTIALS environment variable set.")

        # 3. Initialize Vertex AI Client
        from google.cloud import aiplatform
        
        logger.info("Initializing Vertex AI client...")
        aiplatform.init(
            project=gcp_config['project_id'],
            location=gcp_config['region'],
        )
        logger.info("Vertex AI client initialized successfully.")

        # 4. Perform a Read-Only Operation
        logger.info("Attempting to list models to verify permissions...")
        models = aiplatform.Model.list()
        logger.info(f"Successfully listed models (found {len(models)}).")
        
        logger.info("Connection to Vertex AI is successful and permissions appear to be correct.")
        return True

    except Exception as e:
        logger.error(f"Vertex AI connection verification failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if verify_connection():
        print("\nVerification successful! Ready to start training on Vertex AI.")
    else:
        print("\nVerification failed. Please check the logs for details.")
