"""
Unit tests for the RiskManager risk management system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.risk_manager import (
    RiskManager, RiskConfig, Position, PositionType, RiskMetrics, 
    RiskViolation, RiskLimitType, RiskManagementError
)
from src.interfaces import TradingSignal


class TestRiskManager:
    """Test cases for RiskManager class"""
    
    @pytest.fixture
    def risk_config(self):
        """Create test risk configuration"""
        return RiskConfig(
            max_position_size_pct=0.02,  # 2%
            stop_loss_pct=0.01,  # 1%
            take_profit_pct=0.02,  # 2%
            max_drawdown_pct=0.10,  # 10%
            max_positions=5,
            emergency_drawdown_pct=0.15  # 15%
        )
        
    @pytest.fixture
    def risk_manager(self, risk_config):
        """Create risk manager instance for testing"""
        rm = RiskManager(risk_config)
        rm.set_initial_capital(10000.0)  # $10,000 initial capital
        return rm
        
    @pytest.fixture
    def sample_signal(self):
        """Create sample trading signal"""
        return TradingSignal(
            signal=1,
            confidence=0.8,
            timestamp=datetime.now(),
            features={'RSI': 65.0, 'MACD': 0.5}
        )
        
    @pytest.fixture
    def sample_position(self):
        """Create sample position"""
        return Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=150.0,
            quantity=100,
            entry_time=datetime.now(),
            stop_loss_price=148.5,  # 1% stop loss
            take_profit_price=153.0  # 2% take profit
        )
        
    def test_risk_manager_initialization(self, risk_config):
        """Test risk manager initialization"""
        rm = RiskManager(risk_config)
        
        assert rm.config == risk_config
        assert rm.positions == {}
        assert rm.portfolio_value == 0.0
        assert rm.emergency_stop_triggered is False
        assert rm.risk_violations == []
        
    def test_risk_manager_initialization_default_config(self):
        """Test risk manager initialization with default config"""
        rm = RiskManager()
        
        assert isinstance(rm.config, RiskConfig)
        assert rm.config.max_position_size_pct == 0.02  # Default value
        
    def test_set_initial_capital(self, risk_manager):
        """Test setting initial capital"""
        initial_capital = 10000.0
        
        assert risk_manager.portfolio_value == initial_capital
        assert risk_manager.initial_portfolio_value == initial_capital
        assert risk_manager.peak_portfolio_value == initial_capital
        assert risk_manager.cash_balance == initial_capital
        
    def test_calculate_position_size_basic(self, risk_manager, sample_signal):
        """Test basic position size calculation"""
        current_price = 100.0
        
        position_size = risk_manager.calculate_position_size(
            sample_signal, 
            risk_manager.portfolio_value, 
            current_price
        )
        
        # Expected: 2% of portfolio * 0.8 confidence / $100 = 1.6 shares
        expected_size = (10000 * 0.02 * 0.8) / 100
        assert position_size == expected_size
        
    def test_calculate_position_size_without_confidence_scaling(self, risk_config):
        """Test position size calculation without confidence scaling"""
        risk_config.confidence_scaling = False
        rm = RiskManager(risk_config)
        rm.set_initial_capital(10000.0)
        
        signal = TradingSignal(1, 0.5, datetime.now(), {})  # Low confidence
        current_price = 100.0
        
        position_size = rm.calculate_position_size(signal, rm.portfolio_value, current_price)
        
        # Expected: 2% of portfolio / $100 = 2.0 shares (no confidence scaling)
        expected_size = (10000 * 0.02) / 100
        assert position_size == expected_size
        
    def test_calculate_position_size_minimum_size(self, risk_manager):
        """Test position size calculation with minimum size constraint"""
        # Very low confidence signal
        signal = TradingSignal(1, 0.01, datetime.now(), {})
        current_price = 100.0
        
        position_size = risk_manager.calculate_position_size(
            signal, 
            risk_manager.portfolio_value, 
            current_price
        )
        
        # Should be at least minimum position size
        min_shares = risk_manager.config.min_position_size / current_price
        assert position_size >= min_shares
        
    def test_calculate_position_size_atr_based(self, risk_config):
        """Test ATR-based position size calculation"""
        risk_config.stop_loss_method = "atr"
        risk_config.atr_multiplier = 2.0
        rm = RiskManager(risk_config)
        rm.set_initial_capital(10000.0)
        
        signal = TradingSignal(1, 0.8, datetime.now(), {})
        current_price = 100.0
        atr = 2.0
        
        position_size = rm.calculate_position_size(
            signal, 
            rm.portfolio_value, 
            current_price,
            atr
        )
        
        # Should be risk-adjusted based on ATR
        assert position_size > 0
        assert isinstance(position_size, float)
        
    def test_calculate_stop_loss_price_percentage(self, risk_manager):
        """Test percentage-based stop loss calculation"""
        entry_price = 100.0
        
        # Long position
        stop_price_long = risk_manager.calculate_stop_loss_price(
            entry_price, PositionType.LONG
        )
        expected_long = entry_price * (1 - risk_manager.config.stop_loss_pct)
        assert stop_price_long == expected_long
        
        # Short position
        stop_price_short = risk_manager.calculate_stop_loss_price(
            entry_price, PositionType.SHORT
        )
        expected_short = entry_price * (1 + risk_manager.config.stop_loss_pct)
        assert stop_price_short == expected_short
        
    def test_calculate_stop_loss_price_atr(self, risk_config):
        """Test ATR-based stop loss calculation"""
        risk_config.stop_loss_method = "atr"
        risk_config.atr_multiplier = 2.0
        rm = RiskManager(risk_config)
        
        entry_price = 100.0
        atr = 1.5
        
        # Long position
        stop_price_long = rm.calculate_stop_loss_price(
            entry_price, PositionType.LONG, atr
        )
        expected_long = entry_price - (atr * risk_config.atr_multiplier)
        assert stop_price_long == expected_long
        
        # Short position
        stop_price_short = rm.calculate_stop_loss_price(
            entry_price, PositionType.SHORT, atr
        )
        expected_short = entry_price + (atr * risk_config.atr_multiplier)
        assert stop_price_short == expected_short
        
    def test_calculate_take_profit_price_percentage(self, risk_manager):
        """Test percentage-based take profit calculation"""
        entry_price = 100.0
        stop_loss_price = 99.0
        
        # Long position
        tp_price_long = risk_manager.calculate_take_profit_price(
            entry_price, PositionType.LONG, stop_loss_price
        )
        expected_long = entry_price * (1 + risk_manager.config.take_profit_pct)
        assert tp_price_long == expected_long
        
        # Short position
        tp_price_short = risk_manager.calculate_take_profit_price(
            entry_price, PositionType.SHORT, stop_loss_price
        )
        expected_short = entry_price * (1 - risk_manager.config.take_profit_pct)
        assert tp_price_short == expected_short
        
    def test_calculate_take_profit_price_risk_reward(self, risk_config):
        """Test risk-reward ratio based take profit calculation"""
        risk_config.take_profit_method = "risk_reward"
        risk_config.risk_reward_ratio = 2.0
        rm = RiskManager(risk_config)
        
        entry_price = 100.0
        stop_loss_price = 98.0  # $2 risk
        
        # Long position - should target $4 profit (2:1 ratio)
        tp_price_long = rm.calculate_take_profit_price(
            entry_price, PositionType.LONG, stop_loss_price
        )
        expected_long = entry_price + (2.0 * 2.0)  # entry + (risk * ratio)
        assert tp_price_long == expected_long
        
    def test_check_risk_limits_max_positions(self, risk_manager, sample_signal):
        """Test maximum positions risk limit"""
        # Create maximum number of positions
        current_positions = {}
        for i in range(risk_manager.config.max_positions):
            pos = Position(
                symbol=f"STOCK{i}",
                position_type=PositionType.LONG,
                entry_price=100.0,
                quantity=10,
                entry_time=datetime.now()
            )
            current_positions[f"STOCK{i}"] = pos
            
        # Try to add one more position
        result = risk_manager.check_risk_limits(
            current_positions, sample_signal, 10, 100.0
        )
        
        assert result is False  # Should be rejected
        assert len(risk_manager.risk_violations) > 0
        assert any(v.violation_type == RiskLimitType.MAX_POSITIONS for v in risk_manager.risk_violations)
        
    def test_check_risk_limits_position_size(self, risk_manager, sample_signal):
        """Test position size risk limit"""
        # Try to create oversized position
        oversized_position = 1000  # Much larger than 2% of portfolio
        current_price = 100.0
        
        result = risk_manager.check_risk_limits(
            {}, sample_signal, oversized_position, current_price
        )
        
        assert result is False  # Should be rejected
        assert len(risk_manager.risk_violations) > 0
        assert any(v.violation_type == RiskLimitType.POSITION_SIZE for v in risk_manager.risk_violations)
        
    def test_check_risk_limits_emergency_stop(self, risk_manager, sample_signal):
        """Test emergency stop risk limit"""
        # Trigger emergency stop
        risk_manager.emergency_stop_triggered = True
        
        result = risk_manager.check_risk_limits(
            {}, sample_signal, 10, 100.0
        )
        
        assert result is False  # Should be rejected
        assert any(v.violation_type == RiskLimitType.MAX_DRAWDOWN for v in risk_manager.risk_violations)
        
    def test_apply_stop_loss_long_position(self, sample_position):
        """Test stop loss application for long position"""
        rm = RiskManager()
        
        # Price above stop loss - should not trigger
        result = rm.apply_stop_loss(sample_position, 149.0)
        assert result is False
        
        # Price at stop loss - should trigger
        result = rm.apply_stop_loss(sample_position, 148.5)
        assert result is True
        
        # Price below stop loss - should trigger
        result = rm.apply_stop_loss(sample_position, 148.0)
        assert result is True
        
    def test_apply_stop_loss_short_position(self):
        """Test stop loss application for short position"""
        rm = RiskManager()
        
        short_position = Position(
            symbol="AAPL",
            position_type=PositionType.SHORT,
            entry_price=150.0,
            quantity=100,
            entry_time=datetime.now(),
            stop_loss_price=151.5  # 1% stop loss for short
        )
        
        # Price below stop loss - should not trigger
        result = rm.apply_stop_loss(short_position, 151.0)
        assert result is False
        
        # Price at stop loss - should trigger
        result = rm.apply_stop_loss(short_position, 151.5)
        assert result is True
        
        # Price above stop loss - should trigger
        result = rm.apply_stop_loss(short_position, 152.0)
        assert result is True
        
    def test_apply_take_profit_long_position(self, sample_position):
        """Test take profit application for long position"""
        rm = RiskManager()
        
        # Price below take profit - should not trigger
        result = rm.apply_take_profit(sample_position, 152.0)
        assert result is False
        
        # Price at take profit - should trigger
        result = rm.apply_take_profit(sample_position, 153.0)
        assert result is True
        
        # Price above take profit - should trigger
        result = rm.apply_take_profit(sample_position, 154.0)
        assert result is True
        
    def test_apply_take_profit_short_position(self):
        """Test take profit application for short position"""
        rm = RiskManager()
        
        short_position = Position(
            symbol="AAPL",
            position_type=PositionType.SHORT,
            entry_price=150.0,
            quantity=100,
            entry_time=datetime.now(),
            take_profit_price=147.0  # 2% take profit for short
        )
        
        # Price above take profit - should not trigger
        result = rm.apply_take_profit(short_position, 148.0)
        assert result is False
        
        # Price at take profit - should trigger
        result = rm.apply_take_profit(short_position, 147.0)
        assert result is True
        
        # Price below take profit - should trigger
        result = rm.apply_take_profit(short_position, 146.0)
        assert result is True
        
    def test_update_trailing_stop_long_position(self, risk_config):
        """Test trailing stop update for long position"""
        risk_config.enable_trailing_stop = True
        risk_config.trailing_stop_pct = 0.005  # 0.5%
        rm = RiskManager(risk_config)
        
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=100.0,
            quantity=100,
            entry_time=datetime.now(),
            stop_loss_price=99.0  # Initial stop
        )
        
        # Price moves up - should update trailing stop
        new_stop = rm.update_trailing_stop(position, 105.0)
        expected_stop = 105.0 * (1 - 0.005)  # 104.475
        
        assert new_stop is not None
        assert position.stop_loss_price == round(expected_stop, 4)
        
        # Price moves down - should not update trailing stop
        old_stop = position.stop_loss_price
        new_stop = rm.update_trailing_stop(position, 104.0)
        
        assert new_stop is None
        assert position.stop_loss_price == old_stop
        
    def test_check_drawdown_limit_normal(self, risk_manager):
        """Test normal drawdown limit check"""
        # 5% drawdown - should not trigger limit
        result = risk_manager.check_drawdown_limit(9500.0, 10000.0)
        
        assert result is False  # Within limits
        
    def test_check_drawdown_limit_exceeded(self, risk_manager):
        """Test drawdown limit exceeded"""
        # 12% drawdown - should exceed 10% limit
        result = risk_manager.check_drawdown_limit(8800.0, 10000.0)
        
        assert result is True  # Limit exceeded
        assert len(risk_manager.risk_violations) > 0
        
    def test_check_drawdown_limit_emergency_stop(self, risk_manager):
        """Test emergency stop trigger"""
        # 20% drawdown - should trigger emergency stop (15% threshold)
        result = risk_manager.check_drawdown_limit(8000.0, 10000.0)
        
        assert result is True
        assert risk_manager.emergency_stop_triggered is True
        assert any(v.severity == "critical" for v in risk_manager.risk_violations)
        
    def test_update_portfolio_value(self, risk_manager):
        """Test portfolio value update"""
        new_value = 11000.0
        
        risk_manager.update_portfolio_value(new_value)
        
        assert risk_manager.portfolio_value == new_value
        assert risk_manager.peak_portfolio_value == new_value  # New peak
        assert len(risk_manager.equity_curve) == 1
        
        # Update with lower value
        risk_manager.update_portfolio_value(10500.0)
        
        assert risk_manager.portfolio_value == 10500.0
        assert risk_manager.peak_portfolio_value == 11000.0  # Peak unchanged
        assert len(risk_manager.equity_curve) == 2
        
    def test_get_risk_metrics(self, risk_manager):
        """Test risk metrics calculation"""
        # Add some positions
        risk_manager.positions["AAPL"] = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=150.0,
            quantity=50,
            entry_time=datetime.now()
        )
        
        # Update portfolio value to create drawdown
        risk_manager.update_portfolio_value(9000.0)  # 10% drawdown
        
        metrics = risk_manager.get_risk_metrics()
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.current_drawdown == 0.1  # 10%
        assert metrics.position_count == 1
        assert metrics.largest_position_pct > 0
        
    def test_reset_emergency_stop(self, risk_manager):
        """Test emergency stop reset"""
        # Trigger emergency stop
        risk_manager.emergency_stop_triggered = True
        
        # Reset
        risk_manager.reset_emergency_stop()
        
        assert risk_manager.emergency_stop_triggered is False
        
    def test_get_recent_violations(self, risk_manager):
        """Test getting recent violations"""
        # Add some violations
        old_violation = RiskViolation(
            violation_type=RiskLimitType.POSITION_SIZE,
            current_value=0.05,
            limit_value=0.02,
            severity="warning",
            message="Old violation",
            timestamp=datetime.now() - timedelta(days=2)
        )
        
        recent_violation = RiskViolation(
            violation_type=RiskLimitType.MAX_POSITIONS,
            current_value=6,
            limit_value=5,
            severity="critical",
            message="Recent violation"
        )
        
        risk_manager.risk_violations = [old_violation, recent_violation]
        
        # Get recent violations (last 24 hours)
        recent = risk_manager.get_recent_violations(24)
        
        assert len(recent) == 1
        assert recent[0].message == "Recent violation"
        
    def test_clear_old_violations(self, risk_manager):
        """Test clearing old violations"""
        # Add old and new violations
        old_violation = RiskViolation(
            violation_type=RiskLimitType.POSITION_SIZE,
            current_value=0.05,
            limit_value=0.02,
            severity="warning",
            message="Old violation",
            timestamp=datetime.now() - timedelta(days=10)
        )
        
        new_violation = RiskViolation(
            violation_type=RiskLimitType.MAX_POSITIONS,
            current_value=6,
            limit_value=5,
            severity="critical",
            message="New violation"
        )
        
        risk_manager.risk_violations = [old_violation, new_violation]
        
        # Clear violations older than 7 days
        risk_manager.clear_old_violations(7)
        
        assert len(risk_manager.risk_violations) == 1
        assert risk_manager.risk_violations[0].message == "New violation"
        
    def test_export_risk_report(self, risk_manager):
        """Test risk report export"""
        # Add a position
        risk_manager.positions["AAPL"] = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=150.0,
            quantity=50,
            entry_time=datetime.now(),
            stop_loss_price=148.5,
            take_profit_price=153.0
        )
        
        # Add a violation
        risk_manager.risk_violations.append(RiskViolation(
            violation_type=RiskLimitType.POSITION_SIZE,
            current_value=0.05,
            limit_value=0.02,
            severity="warning",
            message="Test violation"
        ))
        
        report = risk_manager.export_risk_report()
        
        assert "timestamp" in report
        assert "portfolio_value" in report
        assert "risk_metrics" in report
        assert "risk_limits" in report
        assert "recent_violations" in report
        assert "positions" in report
        
        assert report["portfolio_value"] == 10000.0
        assert len(report["positions"]) == 1
        assert len(report["recent_violations"]) == 1
        
    def test_position_creation(self):
        """Test Position dataclass creation"""
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=150.0,
            quantity=100,
            entry_time=datetime.now()
        )
        
        assert position.symbol == "AAPL"
        assert position.position_type == PositionType.LONG
        assert position.entry_price == 150.0
        assert position.quantity == 100
        assert position.position_value == 15000.0  # 150 * 100
        assert position.unrealized_pnl == 0.0
        
    def test_risk_config_defaults(self):
        """Test RiskConfig default values"""
        config = RiskConfig()
        
        assert config.max_position_size_pct == 0.02
        assert config.stop_loss_pct == 0.01
        assert config.take_profit_pct == 0.02
        assert config.max_drawdown_pct == 0.10
        assert config.max_positions == 5
        assert config.emergency_drawdown_pct == 0.15
        assert config.confidence_scaling is True
        assert config.enable_trailing_stop is False


class TestRiskManagerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_calculate_position_size_zero_portfolio(self):
        """Test position size calculation with zero portfolio value"""
        rm = RiskManager()
        signal = TradingSignal(1, 0.8, datetime.now(), {})
        
        with pytest.raises(RiskManagementError):
            rm.calculate_position_size(signal, 0.0, 100.0)
            
    def test_calculate_stop_loss_invalid_method(self):
        """Test stop loss calculation with invalid method"""
        config = RiskConfig(stop_loss_method="invalid")
        rm = RiskManager(config)
        
        with pytest.raises(RiskManagementError):
            rm.calculate_stop_loss_price(100.0, PositionType.LONG)
            
    def test_apply_stop_loss_no_stop_price(self):
        """Test stop loss application with no stop price set"""
        rm = RiskManager()
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=150.0,
            quantity=100,
            entry_time=datetime.now()
            # No stop_loss_price set
        )
        
        result = rm.apply_stop_loss(position, 140.0)
        assert result is False
        
    def test_apply_take_profit_no_tp_price(self):
        """Test take profit application with no TP price set"""
        rm = RiskManager()
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=150.0,
            quantity=100,
            entry_time=datetime.now()
            # No take_profit_price set
        )
        
        result = rm.apply_take_profit(position, 160.0)
        assert result is False
        
    def test_trailing_stop_disabled(self):
        """Test trailing stop when disabled"""
        config = RiskConfig(enable_trailing_stop=False)
        rm = RiskManager(config)
        
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=100.0,
            quantity=100,
            entry_time=datetime.now(),
            stop_loss_price=99.0
        )
        
        result = rm.update_trailing_stop(position, 105.0)
        assert result is None
        
    def test_drawdown_calculation_zero_peak(self):
        """Test drawdown calculation with zero peak value"""
        rm = RiskManager()
        
        result = rm.check_drawdown_limit(1000.0, 0.0)
        assert result is False  # Should handle gracefully
        
    def test_risk_metrics_empty_portfolio(self):
        """Test risk metrics with empty portfolio"""
        rm = RiskManager()
        
        metrics = rm.get_risk_metrics()
        
        assert metrics.current_drawdown == 0.0
        assert metrics.position_count == 0
        assert metrics.largest_position_pct == 0.0


class TestIntegrationScenarios:
    """Integration test scenarios for risk management"""
    
    def test_complete_position_lifecycle(self):
        """Test complete position lifecycle with risk management"""
        rm = RiskManager()
        rm.set_initial_capital(10000.0)
        
        # Create signal
        signal = TradingSignal(1, 0.8, datetime.now(), {})
        
        # Calculate position size
        current_price = 100.0
        position_size = rm.calculate_position_size(signal, rm.portfolio_value, current_price)
        
        # Check risk limits
        risk_ok = rm.check_risk_limits({}, signal, position_size, current_price)
        assert risk_ok is True
        
        # Create position
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            entry_price=current_price,
            quantity=position_size,
            entry_time=datetime.now()
        )
        
        # Set stop loss and take profit
        position.stop_loss_price = rm.calculate_stop_loss_price(
            current_price, PositionType.LONG
        )
        position.take_profit_price = rm.calculate_take_profit_price(
            current_price, PositionType.LONG, position.stop_loss_price
        )
        
        # Add to portfolio
        rm.positions["AAPL"] = position
        
        # Test price movements
        # Price goes up - should not trigger stops
        new_price = 105.0
        assert rm.apply_stop_loss(position, new_price) is False
        assert rm.apply_take_profit(position, new_price) is True
        
        # Price hits take profit
        tp_price = position.take_profit_price
        assert rm.apply_take_profit(position, tp_price) is True
        
    def test_portfolio_drawdown_scenario(self):
        """Test portfolio drawdown management scenario"""
        rm = RiskManager()
        rm.set_initial_capital(10000.0)
        
        # Simulate portfolio decline
        values = [10000, 9500, 9000, 8500, 8000, 7500]  # 25% drawdown
        
        for value in values:
            rm.update_portfolio_value(value)
            
        # Should have triggered emergency stop
        assert rm.emergency_stop_triggered is True
        
        # Should have violations
        violations = rm.get_recent_violations()
        assert len(violations) > 0
        assert any(v.severity == "critical" for v in violations)
        
    def test_multiple_positions_risk_management(self):
        """Test risk management with multiple positions"""
        rm = RiskManager()
        rm.set_initial_capital(10000.0)
        
        # Create multiple positions
        positions = {}
        for i in range(3):
            position = Position(
                symbol=f"STOCK{i}",
                position_type=PositionType.LONG,
                entry_price=100.0,
                quantity=20,  # $2000 each
                entry_time=datetime.now()
            )
            positions[f"STOCK{i}"] = position
            
        rm.positions = positions
        
        # Check risk metrics
        metrics = rm.get_risk_metrics()
        
        assert metrics.position_count == 3
        assert metrics.total_portfolio_risk == 0.6  # 60% of portfolio
        assert metrics.largest_position_pct == 0.2  # 20% each
        
        # Try to add another large position - should be rejected
        signal = TradingSignal(1, 0.8, datetime.now(), {})
        risk_ok = rm.check_risk_limits(positions, signal, 50, 100.0)  # $5000 position
        
        assert risk_ok is False  # Should be rejected due to portfolio risk