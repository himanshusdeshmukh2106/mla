"""
Main application entry point for XGBoost trading system
"""

import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager
from logger import setup_logging, get_logger
from exceptions import TradingSystemError, ConfigurationError


class TradingApplication:
    """Main application orchestrator"""
    
    def __init__(self, config_dir: str = "config", log_dir: str = "logs"):
        self.config_dir = config_dir
        self.log_dir = log_dir
        
        # Initialize logging
        self.logger = setup_logging(log_dir=log_dir)
        self.logger.info("Initializing XGBoost Trading System")
        
        # Initialize configuration
        try:
            self.config_manager = ConfigManager(config_dir)
            self.logger.info("Configuration manager initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize configuration manager", e)
            raise ConfigurationError("Configuration initialization failed") from e
        
        # Component placeholders (will be initialized in later tasks)
        self.data_loader = None
        self.feature_engineer = None
        self.model_trainer = None
        self.predictor = None
        self.risk_manager = None
        self.strategy_engine = None
        self.backtest_engine = None
    
    def validate_environment(self) -> bool:
        """Validate that all required dependencies are available"""
        required_packages = [
            'xgboost', 'pandas', 'numpy', 'sklearn', 
            'pandas_ta', 'backtesting', 'yaml'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            return False
        
        self.logger.info("All required dependencies are available")
        return True
    
    def get_system_info(self) -> dict:
        """Get system information and configuration summary"""
        return {
            'config_dir': str(self.config_dir),
            'log_dir': str(self.log_dir),
            'data_config': self.config_manager.get_data_config().__dict__,
            'model_config': self.config_manager.get_model_config().__dict__,
            'risk_config': self.config_manager.get_risk_config().__dict__,
            'environment_valid': self.validate_environment()
        }
    
    def run_pipeline(self):
        """Run the complete trading pipeline (placeholder for future implementation)"""
        self.logger.info("Pipeline execution not yet implemented - awaiting component development")
        raise NotImplementedError("Pipeline execution will be implemented in subsequent tasks")


def main():
    """Main entry point"""
    try:
        app = TradingApplication()
        
        # Validate environment
        if not app.validate_environment():
            print("Environment validation failed. Please install required dependencies.")
            return 1
        
        # Display system info
        info = app.get_system_info()
        print("XGBoost Trading System Initialized Successfully")
        print(f"Configuration directory: {info['config_dir']}")
        print(f"Log directory: {info['log_dir']}")
        print(f"Environment valid: {info['environment_valid']}")
        
        return 0
        
    except Exception as e:
        print(f"Application initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())