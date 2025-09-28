
import pandas as pd
from src.models.vertex_ai_trainer import VertexAITrainer
from src.config_manager import ConfigManager

def main():
    """
    Main function to trigger Vertex AI training.
    """
    # Load configuration
    config_manager = ConfigManager(config_dir='config')
    vertex_config = config_manager.get_vertex_config()
    gcp_config = config_manager.get_gcp_config()

    # Load data
    data_path = "data/reliance_data_5min_full_year.csv"
    data = pd.read_csv(data_path)

    # For the purpose of this script, we'll assume the target column is 'target_binary'
    # and the rest are features. In a real-world scenario, you would perform
    # feature engineering and target generation here.
    
    # Create a dummy target column for demonstration if it doesn't exist
    if 'target_binary' not in data.columns:
        data['target_binary'] = (data['close'].pct_change().shift(-1) > 0).astype(int)
        data = data.dropna()


    X_train = data.drop('target_binary', axis=1)
    y_train = data['target_binary']
    
    # Initialize and run the trainer
    trainer = VertexAITrainer(vertex_config.__dict__, gcp_config.__dict__)
    trainer.train_model(X_train, y_train)

if __name__ == "__main__":
    main()
