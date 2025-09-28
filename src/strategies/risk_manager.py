"""
Risk management system with position sizing, stop-loss, take-profit, and drawdown controls
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
from enum import Enum

from ..interfaces import IRiskManager, TradingSignal
from ..exceptions import TradingSystemError


class PositionType(Enum):
    """Position type enumeration"""
    LONG = "long"
    SHORT = "short"


class RiskLimitType(Enum):
    """Risk limit type enumeration"""
    POSITION_SIZE = "position_size"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_POSITIONS = "max_positions"


@dataclass
class Position:
    """Trading position data structure"""
    symbol: str
    position_type: PositionType
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    position_value: float = 0.0
    
    def __post_init__(self):
        """Calculate initial position value"""
        self.position_value = abs(self.entry_price * self.quantity)


@dataclass
class RiskConfig:
    """Configuration for risk management"""
    # Position sizing
    max_position_size_pct: float = 0.02  # 2% of portfolio per trade
    min_position_size: float = 100.0  # Minimum position size in currency
    confidence_scaling: bool = True  # Scale position size by signal confidence
    
    # Stop loss configuration
    stop_loss_pct: float = 0.01  # 1% stop loss
    stop_loss_method: str = "percentage"  # "percentage", "atr", "fixed"
    atr_multiplier: float = 2.0  # ATR multiplier for ATR-based stops
    
    # Take profit configuration
    take_profit_pct: float = 0.02  # 2% take profit
    take_profit_method: str = "percentage"  # "percentage", "atr", "fixed"
    risk_reward_ratio: float = 2.0  # Risk-reward ratio
    
    # Portfolio limits
    max_drawdown_pct: float = 0.10  # 10% maximum drawdown
    max_positions: int = 5  # Maximum concurrent positions
    max_portfolio_risk_pct: float = 0.10  # Maximum total portfolio risk
    
    # Risk monitoring
    enable_trailing_stop: bool = False  # Enable trailing stop loss
    trailing_stop_pct: float = 0.005  # 0.5% trailing stop
    enable_position_scaling: bool = False  # Enable position scaling
    
    # Emergency controls
    emergency_stop_enabled: bool = True  # Enable emergency stop
    emergency_drawdown_pct: float = 0.15  # 15% emergency stop drawdown


@dataclass
class RiskMetrics:
    """Risk metrics and statistics"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    total_portfolio_risk: float = 0.0
    position_count: int = 0
    largest_position_pct: float = 0.0
    risk_adjusted_return: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    
    
@dataclass
class RiskViolation:
    """Risk limit violation information"""
    violation_type: RiskLimitType
    current_value: float
    limit_value: float
    severity: str  # "warning", "critical"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class RiskManagementError(TradingSystemError):
    """Raised when risk management operations fail"""
    pass


class RiskManager(IRiskManager):
    """
    Risk management system with position sizing, stop-loss, take-profit, and drawdown controls
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager
        
        Args:
            config: Risk management configuration
        """
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.portfolio_value: float = 0.0
        self.initial_portfolio_value: float = 0.0
        self.peak_portfolio_value: float = 0.0
        self.cash_balance: float = 0.0
        
        # Risk tracking
        self.risk_violations: List[RiskViolation] = []
        self.emergency_stop_triggered: bool = False
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        self.logger.info("RiskManager initialized with risk management system")
        
    def set_initial_capital(self, initial_capital: float):
        """
        Set initial portfolio capital
        
        Args:
            initial_capital: Initial capital amount
        """
        self.portfolio_value = initial_capital
        self.initial_portfolio_value = initial_capital
        self.peak_portfolio_value = initial_capital
        self.cash_balance = initial_capital
        
        self.logger.info(f"Initial capital set to {initial_capital}")
        
    def calculate_position_size(self, 
                              signal: TradingSignal, 
                              portfolio_value: float,
                              current_price: float,
                              atr: Optional[float] = None) -> float:
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            signal: Trading signal with confidence
            portfolio_value: Current portfolio value
            current_price: Current asset price
            atr: Average True Range for ATR-based sizing
            
        Returns:
            Position size in shares/units
        """
        try:
            if portfolio_value <= 0 or current_price <= 0:
                raise RiskManagementError("Portfolio value and current price must be positive")
            
            # Base position size as percentage of portfolio
            base_position_value = portfolio_value * self.config.max_position_size_pct
            
            # Apply confidence scaling if enabled
            if self.config.confidence_scaling:
                confidence_multiplier = signal.confidence
                base_position_value *= confidence_multiplier
                
            # Ensure minimum position size
            if base_position_value < self.config.min_position_size:
                base_position_value = self.config.min_position_size
                
            # Calculate position size in shares
            position_size = base_position_value / current_price
            
            # Apply risk-based position sizing if stop loss is percentage-based
            if self.config.stop_loss_method == "percentage":
                # Adjust position size based on stop loss distance
                risk_per_share = current_price * self.config.stop_loss_pct
                max_risk = portfolio_value * self.config.max_position_size_pct
                risk_adjusted_size = max_risk / risk_per_share
                position_size = min(position_size, risk_adjusted_size)
                
            elif self.config.stop_loss_method == "atr" and atr is not None:
                # ATR-based position sizing
                risk_per_share = atr * self.config.atr_multiplier
                max_risk = portfolio_value * self.config.max_position_size_pct
                risk_adjusted_size = max_risk / risk_per_share
                position_size = min(position_size, risk_adjusted_size)
                
            # Round to appropriate precision
            position_size = round(position_size, 2)
            
            self.logger.debug(f"Calculated position size: {position_size} shares "
                            f"(value: {position_size * current_price:.2f})")
            
            return position_size
            
        except Exception as e:
            raise RiskManagementError(f"Position size calculation failed: {str(e)}")
            
    def calculate_stop_loss_price(self, 
                                entry_price: float, 
                                position_type: PositionType,
                                atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price based on configuration
        
        Args:
            entry_price: Entry price of the position
            position_type: Long or short position
            atr: Average True Range for ATR-based stops
            
        Returns:
            Stop loss price
        """
        try:
            if self.config.stop_loss_method == "percentage":
                if position_type == PositionType.LONG:
                    stop_price = entry_price * (1 - self.config.stop_loss_pct)
                else:  # SHORT
                    stop_price = entry_price * (1 + self.config.stop_loss_pct)
                    
            elif self.config.stop_loss_method == "atr" and atr is not None:
                stop_distance = atr * self.config.atr_multiplier
                if position_type == PositionType.LONG:
                    stop_price = entry_price - stop_distance
                else:  # SHORT
                    stop_price = entry_price + stop_distance
                    
            elif self.config.stop_loss_method == "fixed":
                # Fixed dollar amount stop loss
                fixed_amount = entry_price * self.config.stop_loss_pct  # Use percentage as fixed amount
                if position_type == PositionType.LONG:
                    stop_price = entry_price - fixed_amount
                else:  # SHORT
                    stop_price = entry_price + fixed_amount
                    
            else:
                raise RiskManagementError(f"Unknown stop loss method: {self.config.stop_loss_method}")
                
            return round(stop_price, 4)
            
        except Exception as e:
            raise RiskManagementError(f"Stop loss calculation failed: {str(e)}")
            
    def calculate_take_profit_price(self, 
                                  entry_price: float, 
                                  position_type: PositionType,
                                  stop_loss_price: float,
                                  atr: Optional[float] = None) -> float:
        """
        Calculate take profit price based on configuration
        
        Args:
            entry_price: Entry price of the position
            position_type: Long or short position
            stop_loss_price: Stop loss price for risk-reward calculation
            atr: Average True Range for ATR-based targets
            
        Returns:
            Take profit price
        """
        try:
            if self.config.take_profit_method == "percentage":
                if position_type == PositionType.LONG:
                    tp_price = entry_price * (1 + self.config.take_profit_pct)
                else:  # SHORT
                    tp_price = entry_price * (1 - self.config.take_profit_pct)
                    
            elif self.config.take_profit_method == "atr" and atr is not None:
                tp_distance = atr * self.config.atr_multiplier * self.config.risk_reward_ratio
                if position_type == PositionType.LONG:
                    tp_price = entry_price + tp_distance
                else:  # SHORT
                    tp_price = entry_price - tp_distance
                    
            else:
                # Risk-reward ratio based on stop loss distance
                stop_distance = abs(entry_price - stop_loss_price)
                tp_distance = stop_distance * self.config.risk_reward_ratio
                
                if position_type == PositionType.LONG:
                    tp_price = entry_price + tp_distance
                else:  # SHORT
                    tp_price = entry_price - tp_distance
                    
            return round(tp_price, 4)
            
        except Exception as e:
            raise RiskManagementError(f"Take profit calculation failed: {str(e)}")
            
    def check_risk_limits(self, 
                         current_positions: Dict[str, Position], 
                         new_signal: TradingSignal,
                         new_position_size: float,
                         current_price: float) -> bool:
        """
        Check if new signal violates risk limits
        
        Args:
            current_positions: Current open positions
            new_signal: New trading signal
            new_position_size: Proposed position size
            current_price: Current asset price
            
        Returns:
            True if signal passes risk checks, False otherwise
        """
        try:
            violations = []
            
            # Check maximum number of positions
            if len(current_positions) >= self.config.max_positions:
                violations.append(RiskViolation(
                    violation_type=RiskLimitType.MAX_POSITIONS,
                    current_value=len(current_positions),
                    limit_value=self.config.max_positions,
                    severity="critical",
                    message=f"Maximum positions limit reached: {len(current_positions)}/{self.config.max_positions}"
                ))
                
            # Check position size limit
            position_value = new_position_size * current_price
            max_position_value = self.portfolio_value * self.config.max_position_size_pct
            
            if position_value > max_position_value:
                violations.append(RiskViolation(
                    violation_type=RiskLimitType.POSITION_SIZE,
                    current_value=position_value / self.portfolio_value,
                    limit_value=self.config.max_position_size_pct,
                    severity="critical",
                    message=f"Position size exceeds limit: {position_value:.2f} > {max_position_value:.2f}"
                ))
                
            # Check total portfolio risk
            current_risk = self._calculate_total_portfolio_risk(current_positions)
            new_position_risk = position_value / self.portfolio_value
            total_risk = current_risk + new_position_risk
            
            if total_risk > self.config.max_portfolio_risk_pct:
                violations.append(RiskViolation(
                    violation_type=RiskLimitType.POSITION_SIZE,
                    current_value=total_risk,
                    limit_value=self.config.max_portfolio_risk_pct,
                    severity="warning",
                    message=f"Total portfolio risk exceeds limit: {total_risk:.3f} > {self.config.max_portfolio_risk_pct:.3f}"
                ))
                
            # Check emergency stop
            if self.emergency_stop_triggered:
                violations.append(RiskViolation(
                    violation_type=RiskLimitType.MAX_DRAWDOWN,
                    current_value=self._calculate_current_drawdown(),
                    limit_value=self.config.emergency_drawdown_pct,
                    severity="critical",
                    message="Emergency stop is active - no new positions allowed"
                ))
                
            # Store violations
            self.risk_violations.extend(violations)
            
            # Log violations
            for violation in violations:
                if violation.severity == "critical":
                    self.logger.error(f"Risk violation: {violation.message}")
                else:
                    self.logger.warning(f"Risk warning: {violation.message}")
                    
            # Return True only if no critical violations
            critical_violations = [v for v in violations if v.severity == "critical"]
            return len(critical_violations) == 0
            
        except Exception as e:
            raise RiskManagementError(f"Risk limit check failed: {str(e)}")
            
    def apply_stop_loss(self, position: Position, current_price: float) -> bool:
        """
        Check if stop loss should be triggered for a position
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            True if stop loss should be triggered
        """
        try:
            if position.stop_loss_price is None:
                return False
                
            should_trigger = False
            
            if position.position_type == PositionType.LONG:
                should_trigger = current_price <= position.stop_loss_price
            else:  # SHORT
                should_trigger = current_price >= position.stop_loss_price
                
            if should_trigger:
                self.logger.info(f"Stop loss triggered for {position.symbol}: "
                               f"current={current_price}, stop={position.stop_loss_price}")
                
            return should_trigger
            
        except Exception as e:
            raise RiskManagementError(f"Stop loss check failed: {str(e)}")
            
    def apply_take_profit(self, position: Position, current_price: float) -> bool:
        """
        Check if take profit should be triggered for a position
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            True if take profit should be triggered
        """
        try:
            if position.take_profit_price is None:
                return False
                
            should_trigger = False
            
            if position.position_type == PositionType.LONG:
                should_trigger = current_price >= position.take_profit_price
            else:  # SHORT
                should_trigger = current_price <= position.take_profit_price
                
            if should_trigger:
                self.logger.info(f"Take profit triggered for {position.symbol}: "
                               f"current={current_price}, target={position.take_profit_price}")
                
            return should_trigger
            
        except Exception as e:
            raise RiskManagementError(f"Take profit check failed: {str(e)}")
            
    def update_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """
        Update trailing stop loss price
        
        Args:
            position: Position to update
            current_price: Current market price
            
        Returns:
            New stop loss price if updated, None otherwise
        """
        try:
            if not self.config.enable_trailing_stop or position.stop_loss_price is None:
                return None
                
            new_stop_price = None
            
            if position.position_type == PositionType.LONG:
                # For long positions, trail stop up as price increases
                potential_stop = current_price * (1 - self.config.trailing_stop_pct)
                if potential_stop > position.stop_loss_price:
                    new_stop_price = potential_stop
                    
            else:  # SHORT
                # For short positions, trail stop down as price decreases
                potential_stop = current_price * (1 + self.config.trailing_stop_pct)
                if potential_stop < position.stop_loss_price:
                    new_stop_price = potential_stop
                    
            if new_stop_price is not None:
                old_stop = position.stop_loss_price
                position.stop_loss_price = round(new_stop_price, 4)
                self.logger.debug(f"Trailing stop updated for {position.symbol}: "
                                f"{old_stop} -> {position.stop_loss_price}")
                                
            return new_stop_price
            
        except Exception as e:
            raise RiskManagementError(f"Trailing stop update failed: {str(e)}")
            
    def check_drawdown_limit(self, portfolio_value: float, peak_value: float) -> bool:
        """
        Check if maximum drawdown limit is exceeded
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            
        Returns:
            True if drawdown limit is exceeded
        """
        try:
            if peak_value <= 0:
                return False  # No drawdown if peak is not positive
            
            current_drawdown = (peak_value - portfolio_value) / peak_value
            
            # Check regular drawdown limit
            if current_drawdown > self.config.max_drawdown_pct:
                self.logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.3f} > {self.config.max_drawdown_pct:.3f}")
                
                # Add violation
                self.risk_violations.append(RiskViolation(
                    violation_type=RiskLimitType.MAX_DRAWDOWN,
                    current_value=current_drawdown,
                    limit_value=self.config.max_drawdown_pct,
                    severity="warning",
                    message=f"Maximum drawdown exceeded: {current_drawdown:.1%}"
                ))
                
            # Check emergency stop
            if (self.config.emergency_stop_enabled and 
                current_drawdown > self.config.emergency_drawdown_pct):
                
                self.emergency_stop_triggered = True
                self.logger.error(f"EMERGENCY STOP TRIGGERED: Drawdown {current_drawdown:.1%} > {self.config.emergency_drawdown_pct:.1%}")
                
                # Add critical violation
                self.risk_violations.append(RiskViolation(
                    violation_type=RiskLimitType.MAX_DRAWDOWN,
                    current_value=current_drawdown,
                    limit_value=self.config.emergency_drawdown_pct,
                    severity="critical",
                    message=f"Emergency stop triggered: {current_drawdown:.1%}"
                ))
                
                return True
                
            return current_drawdown > self.config.max_drawdown_pct
            
        except Exception as e:
            raise RiskManagementError(f"Drawdown check failed: {str(e)}")
            
    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value and track equity curve
        
        Args:
            new_value: New portfolio value
        """
        self.portfolio_value = new_value
        
        # Update peak value
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value
            
        # Add to equity curve
        self.equity_curve.append((datetime.now(), new_value))
        
        # Check drawdown limits
        self.check_drawdown_limit(new_value, self.peak_portfolio_value)
        
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current risk metrics
        
        Returns:
            RiskMetrics object with current statistics
        """
        try:
            current_drawdown = self._calculate_current_drawdown()
            max_drawdown = self._calculate_max_drawdown()
            total_risk = self._calculate_total_portfolio_risk(self.positions)
            
            # Calculate largest position percentage
            largest_position_pct = 0.0
            if self.positions and self.portfolio_value > 0:
                largest_position_value = max(pos.position_value for pos in self.positions.values())
                largest_position_pct = largest_position_value / self.portfolio_value
                
            return RiskMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                total_portfolio_risk=total_risk,
                position_count=len(self.positions),
                largest_position_pct=largest_position_pct
            )
            
        except Exception as e:
            raise RiskManagementError(f"Risk metrics calculation failed: {str(e)}")
            
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_portfolio_value <= 0:
            return 0.0
        return (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0
            
        values = [point[1] for point in self.equity_curve]
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
                
        return max_dd
        
    def _calculate_total_portfolio_risk(self, positions: Dict[str, Position]) -> float:
        """Calculate total portfolio risk as percentage"""
        if not positions or self.portfolio_value <= 0:
            return 0.0
            
        total_position_value = sum(pos.position_value for pos in positions.values())
        return total_position_value / self.portfolio_value
        
    def reset_emergency_stop(self):
        """Reset emergency stop (use with caution)"""
        self.emergency_stop_triggered = False
        self.logger.info("Emergency stop reset")
        
    def get_recent_violations(self, hours: int = 24) -> List[RiskViolation]:
        """
        Get recent risk violations
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent violations
        """
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        return [v for v in self.risk_violations if v.timestamp >= cutoff_time]
        
    def clear_old_violations(self, days: int = 7):
        """
        Clear old risk violations
        
        Args:
            days: Number of days to keep violations
        """
        cutoff_time = datetime.now() - pd.Timedelta(days=days)
        self.risk_violations = [v for v in self.risk_violations if v.timestamp >= cutoff_time]
        
    def export_risk_report(self) -> Dict[str, Any]:
        """
        Export comprehensive risk report
        
        Returns:
            Dictionary with risk analysis
        """
        metrics = self.get_risk_metrics()
        recent_violations = self.get_recent_violations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self.portfolio_value,
            "initial_value": self.initial_portfolio_value,
            "peak_value": self.peak_portfolio_value,
            "emergency_stop_active": self.emergency_stop_triggered,
            "risk_metrics": {
                "current_drawdown": metrics.current_drawdown,
                "max_drawdown": metrics.max_drawdown,
                "total_portfolio_risk": metrics.total_portfolio_risk,
                "position_count": metrics.position_count,
                "largest_position_pct": metrics.largest_position_pct
            },
            "risk_limits": {
                "max_position_size_pct": self.config.max_position_size_pct,
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "max_positions": self.config.max_positions,
                "emergency_drawdown_pct": self.config.emergency_drawdown_pct
            },
            "recent_violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity,
                    "message": v.message,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in recent_violations
            ],
            "positions": [
                {
                    "symbol": pos.symbol,
                    "type": pos.position_type.value,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "position_value": pos.position_value,
                    "stop_loss": pos.stop_loss_price,
                    "take_profit": pos.take_profit_price,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for pos in self.positions.values()
            ]
        }