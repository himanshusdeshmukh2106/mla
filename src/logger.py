"""
Logging infrastructure with appropriate log levels
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class TradingLogger:
    """Centralized logging system for trading framework"""
    
    def __init__(self, name: str = "trading_system", log_dir: str = "logs", 
                 log_level: str = "INFO", console_output: bool = True):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        self.console_output = console_output
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler for all logs
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error file handler for errors only
        error_file = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # Console handler if enabled
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True, **kwargs)
        else:
            self.logger.error(message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception"""
        if exception:
            self.logger.critical(f"{message}: {str(exception)}", exc_info=True, **kwargs)
        else:
            self.logger.critical(message, **kwargs)
    
    def log_data_quality(self, symbol: str, total_records: int, missing_records: int, 
                        outliers: int, quality_score: float):
        """Log data quality metrics"""
        self.info(
            f"Data Quality - Symbol: {symbol}, Total: {total_records}, "
            f"Missing: {missing_records}, Outliers: {outliers}, "
            f"Quality Score: {quality_score:.2f}"
        )
    
    def log_model_performance(self, accuracy: float, precision: float, 
                            recall: float, f1_score: float):
        """Log model performance metrics"""
        self.info(
            f"Model Performance - Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1-Score: {f1_score:.4f}"
        )
    
    def log_trade_signal(self, signal: int, confidence: float, symbol: str, 
                        timestamp: datetime):
        """Log trading signal generation"""
        signal_text = {-1: "SELL", 0: "HOLD", 1: "BUY"}.get(signal, "UNKNOWN")
        self.info(
            f"Trade Signal - {signal_text} {symbol} at {timestamp} "
            f"(Confidence: {confidence:.2f})"
        )
    
    def log_risk_event(self, event_type: str, symbol: str, details: str):
        """Log risk management events"""
        self.warning(f"Risk Event - {event_type} for {symbol}: {details}")
    
    def log_backtest_summary(self, total_return: float, sharpe_ratio: float, 
                           max_drawdown: float, total_trades: int):
        """Log backtest summary results"""
        self.info(
            f"Backtest Summary - Return: {total_return:.2f}%, "
            f"Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2f}%, "
            f"Trades: {total_trades}"
        )
    
    def set_level(self, level: str):
        """Change logging level"""
        new_level = getattr(logging, level.upper())
        self.logger.setLevel(new_level)
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(new_level)


# Global logger instance
_global_logger: Optional[TradingLogger] = None


def get_logger(name: str = "trading_system", **kwargs) -> TradingLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TradingLogger(name, **kwargs)
    return _global_logger


def setup_logging(log_level: str = "INFO", log_dir: str = "logs", 
                 console_output: bool = True) -> TradingLogger:
    """Setup global logging configuration"""
    global _global_logger
    _global_logger = TradingLogger(
        name="trading_system",
        log_dir=log_dir,
        log_level=log_level,
        console_output=console_output
    )
    return _global_logger