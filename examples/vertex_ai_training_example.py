"""
Example usage of VertexAITrainer for training a model on Google Cloud Vertex AI.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_manager import ConfigManager
from src.models.vertex_ai_trainer import VertexAITrainer
from src.logger import setup_logging

def create_sample_data(n_samples=1000):
    """Create sample financial data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
    sma_20 = pd.Series(price).rolling(20).mean()
    rsi = pd.Series(price).pct_change().rolling(14).apply(
        lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean()))))
    )
    target = (pd.Series(price).shift(-1) > pd.Series(price)).astype(int)
    
    data = pd.DataFrame({
        'sma_20': sma_20,
        'rsi': rsi,
        'target': target
    })
    
    return data.dropna().reset_index(drop=True)

def main():
    """Main example function"""
    setup_logging()
    print("Vertex AI Model Training Example")
    print("=" * 40)

    # Load configuration
    try:
        config_manager = ConfigManager(config_dir="config")
        vertex_config = config_manager.get_config('vertex_config.yaml')
        gcp_config = vertex_config.get('gcp', {})
        vertex_ai_config = vertex_config.get('vertex_ai', {})
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Create sample data
    print("Creating sample data...")
    data = create_sample_data()
    X = data[['sma_20', 'rsi']]
    y = data['target']
    print(f"Created dataset with {len(data)} samples.")

    # Initialize Vertex AI Trainer
    print("Initializing Vertex AI Trainer...")
    trainer = VertexAITrainer(vertex_config=vertex_ai_config, gcp_config=gcp_config)

    # Train model on Vertex AI
    print("Submitting training job to Vertex AI...")
    try:
        model_resource_name = trainer.train_model(X, y)
        print(f"Successfully trained model. Resource Name: {model_resource_name}")
    except Exception as e:
        print(f"An error occurred during Vertex AI training: {e}")

if __name__ == "__main__":
    main()
