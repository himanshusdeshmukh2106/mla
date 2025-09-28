# Intraday XGBoost Trading Strategy

A complete machine learning-based trading framework using XGBoost for intraday trading strategies.

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data management components
│   ├── models/            # Machine learning models
│   ├── strategies/        # Trading strategies
│   ├── interfaces.py      # Base interfaces and abstract classes
│   ├── config_manager.py  # Configuration management
│   ├── logger.py          # Logging infrastructure
│   ├── exceptions.py      # Custom exception hierarchy
│   └── main.py           # Main application entry point
├── config/               # Configuration files (YAML)
├── tests/               # Test suite
├── logs/                # Log files (created at runtime)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python src/main.py
```

## Configuration

The system uses YAML configuration files in the `config/` directory:

- `data_config.yaml` - Data source and validation settings
- `feature_config.yaml` - Technical indicator parameters
- `model_config.yaml` - XGBoost model and training settings
- `risk_config.yaml` - Risk management parameters
- `backtest_config.yaml` - Backtesting configuration

Configuration files are automatically created with default values on first run.

## Features

- Modular architecture with clear separation of concerns
- Comprehensive logging with multiple log levels
- YAML-based configuration management
- Custom exception hierarchy for error handling
- Abstract interfaces for all major components
- Time-series aware data processing
- Risk management integration
- Backtesting capabilities

## Development Status

This is the initial project structure. Individual components will be implemented in subsequent development phases.