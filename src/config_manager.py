"""
Configuration management system with YAML support
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    trend_periods: List[int] = field(default_factory=lambda: [20, 50])
    momentum_periods: Dict[str, int] = field(default_factory=lambda: {
        "rsi": 14, "macd_fast": 12, "macd_slow": 26
    })
    volatility_periods: Dict[str, int] = field(default_factory=lambda: {
        "bb": 20, "atr": 14
    })
    volume_periods: List[int] = field(default_factory=lambda: [20])


@dataclass
class ModelConfig:
    """Model training configuration"""
    test_size: float = 0.2
    cv_splits: int = 5
    hyperparameter_grid: Dict = field(default_factory=dict)
    early_stopping_rounds: int = 10


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.02  # 2% of portfolio
    stop_loss_pct: float = 0.01      # 1% stop loss
    take_profit_pct: float = 0.02    # 2% take profit
    max_drawdown: float = 0.10       # 10% maximum drawdown
    max_positions: int = 5           # Maximum concurrent positions


@dataclass
class DataConfig:
    """Data configuration"""
    timeframe: str = "1min"
    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    max_missing_pct: float = 0.05
    outlier_threshold: float = 3.0


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_cash: float = 10000
    commission: float = 0.001
    slippage: float = 0.0005
    margin: float = 1.0
    trade_on_open: bool = False
    trade_on_close: bool = True


@dataclass
class GCPConfig:
    """GCP configuration"""
    project_id: str = ""
    region: str = ""
    service_account_key_path: str = ""


@dataclass
class VertexConfig:
    """Vertex AI configuration"""
    gcs_bucket_name: str = ""
    training_job_display_name: str = ""
    model_display_name: str = ""
    pre_built_container: str = ""
    machine_type: str = "n1-standard-4"




class ConfigManager:
    """Configuration manager with YAML support"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML configuration files from the config directory."""
        for config_path in self.config_dir.glob("*.yaml"):
            config_name = config_path.stem.replace('_config', '')
            if config_path.exists():
                self._configs[config_name] = self._load_yaml(config_path)
            else:
                # This part of the logic might need adjustment
                # as _get_default_config is based on specific names.
                # For now, we'll just load what's there.
                pass
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            raise ValueError(f"Error loading config file {file_path}: {e}")
    
    def _save_yaml(self, file_path: Path, config: Dict[str, Any]):
        """Save configuration to YAML file"""
        try:
            with open(file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving config file {file_path}: {e}")
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get default configuration for given config type"""
        defaults = {
            'data': {
                'timeframe': '1min',
                'symbols': ['AAPL'],
                'start_date': '2023-01-01',
                'end_date': '2024-01-01',
                'validation': {
                    'max_missing_pct': 0.05,
                    'outlier_threshold': 3.0
                }
            },
            'features': {
                'trend': {
                    'sma_periods': [20, 50, 200],
                    'ema_periods': [12, 26, 50]
                },
                'momentum': {
                    'rsi_period': 14,
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9}
                },
                'volatility': {
                    'bollinger_bands': {'period': 20, 'std': 2},
                    'atr_period': 14
                },
                'volume': {
                    'periods': [20]
                }
            },
            'model': {
                'algorithm': 'xgboost',
                'objective': 'binary:logistic',
                'test_size': 0.2,
                'hyperparameters': {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300]
                },
                'cross_validation': {
                    'method': 'TimeSeriesSplit',
                    'n_splits': 5
                }
            },
            'risk': {
                'position_sizing': {
                    'max_position_pct': 0.02,
                    'confidence_scaling': True
                },
                'stop_loss': {
                    'method': 'percentage',
                    'value': 0.01
                },
                'take_profit': {
                    'method': 'percentage',
                    'value': 0.02
                },
                'portfolio': {
                    'max_drawdown': 0.10,
                    'max_positions': 5
                }
            },
            'backtest': {
                'initial_cash': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'margin': 1.0,
                'trade_on_open': False,
                'trade_on_close': True
            }
        }
        return defaults.get(config_name, {})
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get configuration by name or filename."""
        # If the name ends with .yaml, use it as a key directly (without the extension)
        if config_name.endswith(".yaml"):
            config_key = config_name[:-5] # Remove .yaml
        else:
            config_key = config_name

        if config_key not in self._configs:
            raise ValueError(f"Configuration '{config_key}' not found")
        return self._configs[config_key].copy()
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration as dataclass"""
        config = self.get_config('data')
        return DataConfig(
            timeframe=config.get('timeframe', '1min'),
            symbols=config.get('symbols', ['AAPL']),
            start_date=config.get('start_date', '2023-01-01'),
            end_date=config.get('end_date', '2024-01-01'),
            max_missing_pct=config.get('validation', {}).get('max_missing_pct', 0.05),
            outlier_threshold=config.get('validation', {}).get('outlier_threshold', 3.0)
        )
    
    def get_feature_config(self) -> FeatureConfig:
        """Get feature configuration as dataclass"""
        config = self.get_config('features')
        return FeatureConfig(
            trend_periods=config.get('trend', {}).get('sma_periods', [20, 50]),
            momentum_periods={
                'rsi': config.get('momentum', {}).get('rsi_period', 14),
                'macd_fast': config.get('momentum', {}).get('macd', {}).get('fast', 12),
                'macd_slow': config.get('momentum', {}).get('macd', {}).get('slow', 26)
            },
            volatility_periods={
                'bb': config.get('volatility', {}).get('bollinger_bands', {}).get('period', 20),
                'atr': config.get('volatility', {}).get('atr_period', 14)
            },
            volume_periods=config.get('volume', {}).get('periods', [20])
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as dataclass"""
        config = self.get_config('model')
        return ModelConfig(
            test_size=config.get('test_size', 0.2),
            cv_splits=config.get('cross_validation', {}).get('n_splits', 5),
            hyperparameter_grid=config.get('hyperparameters', {}),
            early_stopping_rounds=config.get('early_stopping_rounds', 10)
        )
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk configuration as dataclass"""
        config = self.get_config('risk')
        return RiskConfig(
            max_position_size=config.get('position_sizing', {}).get('max_position_pct', 0.02),
            stop_loss_pct=config.get('stop_loss', {}).get('value', 0.01),
            take_profit_pct=config.get('take_profit', {}).get('value', 0.02),
            max_drawdown=config.get('portfolio', {}).get('max_drawdown', 0.10),
            max_positions=config.get('portfolio', {}).get('max_positions', 5)
        )
    
    def get_backtest_config(self) -> BacktestConfig:
        """Get backtest configuration as dataclass"""
        config = self.get_config('backtest')
        return BacktestConfig(
            initial_cash=config.get('initial_cash', 10000),
            commission=config.get('commission', 0.001),
            slippage=config.get('slippage', 0.0005),
            margin=config.get('margin', 1.0),
            trade_on_open=config.get('trade_on_open', False),
            trade_on_close=config.get('trade_on_close', True)
        )

    def get_gcp_config(self) -> GCPConfig:
        """Get GCP configuration as dataclass"""
        config = self.get_config('vertex')
        gcp_config = config.get('gcp', {})
        return GCPConfig(
            project_id=gcp_config.get('project_id'),
            region=gcp_config.get('region'),
            service_account_key_path=gcp_config.get('service_account_key_path')
        )

    def get_vertex_config(self) -> VertexConfig:
        """Get Vertex AI configuration as dataclass"""
        config = self.get_config('vertex')
        vertex_config = config.get('vertex_ai', {})
        return VertexConfig(
            gcs_bucket_name=vertex_config.get('gcs_bucket_name'),
            training_job_display_name=vertex_config.get('training_job_display_name'),
            model_display_name=vertex_config.get('model_display_name'),
            pre_built_container=vertex_config.get('pre_built_container'),
            machine_type=vertex_config.get('machine_type')
        )

    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if config_name not in self._configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        self._configs[config_name].update(updates)
        config_file = self.config_dir / f"{config_name}_config.yaml"
        self._save_yaml(config_file, self._configs[config_name])
    
    def reload_configs(self):
        """Reload all configurations from files"""
        self._load_all_configs()