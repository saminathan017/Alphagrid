"""
risk/portfolio.py
Portfolio-level risk management: position sizing, stop-loss, drawdown limits.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np
from loguru import logger
from core.config import settings
from core.events import event_bus, Event, EventType


@dataclass
class RiskParameters:
    """Parameters for a single trade."""
    symbol: str
    direction: str          # LONG | SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: float
    risk_amount: float      # $ risk on this trade
    risk_pct: float         # % of portfolio at risk


class RiskManager:
    """
    Central risk management controller.
    Responsibilities:
      1. Position sizing (Kelly/fixed fractional)
      2. Stop-loss computation (ATR-based + trailing)
      3. Portfolio-level limits (max positions, max drawdown, daily loss)
      4. Pre-trade risk checks
    """

    def __init__(self) -> None:
        self._cfg  = settings.get("risk", {})
        self._portfolio_value: float = settings["backtest"]["initial_capital"]
        self._cash: float = self._portfolio_value
        self._daily_pnl: float = 0.0
        self._daily_high: float = self._portfolio_value
        self._peak_value: float = self._portfolio_value
        self._open_positions: dict[str, dict] = {}
        self._trading_halted: bool = False

        # Subscribe to fill/close events
        event_bus.subscribe(EventType.ORDER_FILLED,    self._on_fill)
        event_bus.subscribe(EventType.POSITION_CLOSED, self._on_close)

    # ─── Position Sizing ─────────────────────────────────────────────────

    def compute_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
    ) -> float:
        """
        Calculate position size in shares/units.
        Uses Kelly Fractional or Fixed Fractional depending on config.
        """
        method = self._cfg.get("position_sizing", {}).get("method", "fixed")
        risk_pct = self._cfg.get("max_portfolio_risk", 0.02)

        if method == "kelly_fractional" and win_rate is not None:
            kelly_frac = self._cfg.get("position_sizing", {}).get("kelly_fraction", 0.25)
            # Kelly formula: f* = (p*(b+1) - 1) / b
            b = avg_win_loss_ratio or self._cfg.get("take_profit", {}).get("risk_reward_ratio", 2.5)
            p = win_rate
            full_kelly = (p * (b + 1) - 1) / (b + 1e-9)
            full_kelly = max(0.0, full_kelly)
            risk_pct = min(full_kelly * kelly_frac, risk_pct)
        else:
            risk_pct = self._cfg.get("position_sizing", {}).get("fixed_pct", risk_pct)

        risk_amount = self._portfolio_value * risk_pct
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance < 1e-9:
            logger.warning(f"Stop distance near zero for {symbol}, defaulting to min stop")
            stop_distance = entry_price * self._cfg.get("stop_loss", {}).get("min_stop_pct", 0.005)

        qty = risk_amount / stop_distance

        # Apply max position size cap
        max_position_value = self._portfolio_value * self._cfg.get("max_position_size", 0.10)
        max_qty = max_position_value / (entry_price + 1e-9)
        qty = min(qty, max_qty)

        # Ensure we have enough cash
        required_cash = qty * entry_price
        if required_cash > self._cash:
            qty = self._cash / (entry_price + 1e-9)

        return max(0.0, qty)

    # ─── Stop-Loss Computation ───────────────────────────────────────────

    def compute_stop_loss(
        self,
        direction: str,
        entry_price: float,
        atr: float,
    ) -> float:
        """
        Compute ATR-based stop-loss price.
        LONG:  stop = entry - 2×ATR
        SHORT: stop = entry + 2×ATR
        """
        sl_cfg = self._cfg.get("stop_loss", {})
        atr_mult    = sl_cfg.get("atr_multiplier", 2.0)
        min_stop    = sl_cfg.get("min_stop_pct", 0.005)

        stop_distance = max(atr * atr_mult, entry_price * min_stop)

        if direction == "LONG":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_take_profit(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """
        Compute take-profit based on risk:reward ratio.
        Default: 2.5:1
        """
        rr = self._cfg.get("take_profit", {}).get("risk_reward_ratio", 2.5)
        stop_distance = abs(entry_price - stop_loss)
        if direction == "LONG":
            return entry_price + stop_distance * rr
        else:
            return entry_price - stop_distance * rr

    def update_trailing_stop(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        current_stop: float,
        entry_price: float,
        atr: float,
    ) -> float:
        """
        Update trailing stop-loss. Activates after price moves
        `trailing_activation` % in our favor.
        """
        sl_cfg = self._cfg.get("stop_loss", {})
        if not sl_cfg.get("trailing", True):
            return current_stop

        activation_pct = sl_cfg.get("trailing_activation", 0.015)
        atr_mult = sl_cfg.get("atr_multiplier", 2.0)

        if direction == "LONG":
            profit_pct = (current_price - entry_price) / (entry_price + 1e-9)
            if profit_pct >= activation_pct:
                new_stop = current_price - atr * atr_mult
                return max(new_stop, current_stop)  # Only move stop up
        else:
            profit_pct = (entry_price - current_price) / (entry_price + 1e-9)
            if profit_pct >= activation_pct:
                new_stop = current_price + atr * atr_mult
                return min(new_stop, current_stop)  # Only move stop down

        return current_stop

    # ─── Pre-Trade Checks ────────────────────────────────────────────────

    def pre_trade_check(self, symbol: str, signal_strength: float) -> tuple[bool, str]:
        """
        Perform all pre-trade risk checks.
        Returns (approved: bool, reason: str)
        """
        if self._trading_halted:
            return False, "Trading halted (risk limit breach)"

        # Daily loss limit
        daily_loss_limit = self._cfg.get("max_daily_loss", 0.03)
        if self._daily_pnl / (self._portfolio_value + 1e-9) < -daily_loss_limit:
            self._trading_halted = True
            asyncio.create_task(event_bus.publish(Event(
                event_type=EventType.RISK_BREACH,
                source="risk_manager",
                data={"reason": "daily_loss_limit", "pnl": self._daily_pnl},
            )))
            return False, f"Daily loss limit hit: {self._daily_pnl:.2f}"

        # Max drawdown
        drawdown = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-9)
        max_dd = self._cfg.get("max_drawdown", 0.15)
        if drawdown >= max_dd:
            self._trading_halted = True
            return False, f"Max drawdown hit: {drawdown:.1%}"

        # Max open positions
        max_pos = self._cfg.get("max_open_positions", 10)
        if len(self._open_positions) >= max_pos and symbol not in self._open_positions:
            return False, f"Max positions reached ({max_pos})"

        # Minimum signal strength
        if signal_strength < 0.3:
            return False, f"Signal too weak: {signal_strength:.3f}"

        # Already have position in this symbol
        if symbol in self._open_positions:
            return False, f"Already holding {symbol}"

        return True, "OK"

    def compute_full_risk_params(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
        win_rate: Optional[float] = None,
        avg_rr: Optional[float] = None,
    ) -> Optional[RiskParameters]:
        """
        Compute all risk parameters for a potential trade.
        Returns None if position size is too small.
        """
        stop_loss   = self.compute_stop_loss(direction, entry_price, atr)
        take_profit = self.compute_take_profit(direction, entry_price, stop_loss)
        qty = self.compute_position_size(symbol, entry_price, stop_loss, win_rate, avg_rr)

        if qty < 0.001:
            return None

        risk_amount = abs(entry_price - stop_loss) * qty
        risk_pct    = risk_amount / (self._portfolio_value + 1e-9)

        return RiskParameters(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            qty=qty,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
        )

    # ─── Portfolio State ─────────────────────────────────────────────────

    def update_portfolio_value(self, value: float) -> None:
        """Update current portfolio value and track peak/drawdown."""
        self._portfolio_value = value
        if value > self._peak_value:
            self._peak_value = value

    @property
    def drawdown(self) -> float:
        return (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-9)

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def is_trading_allowed(self) -> bool:
        return not self._trading_halted

    def reset_daily(self) -> None:
        """Reset daily stats at market open."""
        self._daily_pnl  = 0.0
        self._daily_high = self._portfolio_value
        self._trading_halted = False  # Reset daily halt
        logger.info("Daily risk stats reset.")

    # ─── Event Handlers ──────────────────────────────────────────────────

    async def _on_fill(self, event: Event) -> None:
        data = event.data
        symbol = data.get("symbol")
        side   = data.get("side")
        qty    = data.get("filled_qty", 0)
        price  = data.get("filled_avg_price", 0)

        if side == "BUY":
            self._cash -= qty * price
            self._open_positions[symbol] = {
                "qty": qty, "entry_price": price,
                "side": side, "opened_at": datetime.utcnow(),
            }
        elif side == "SELL" and symbol in self._open_positions:
            pos = self._open_positions.pop(symbol)
            pnl = (price - pos["entry_price"]) * pos["qty"]
            self._daily_pnl += pnl
            self._cash += qty * price

    async def _on_close(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        if symbol in self._open_positions:
            self._open_positions.pop(symbol)
