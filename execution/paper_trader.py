"""
execution/paper_trader.py
High-fidelity paper trading simulator with realistic order fill modeling,
slippage, commissions, and portfolio tracking.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
import numpy as np
from loguru import logger
from core.config import settings
from core.events import event_bus, Event, EventType


class PaperTrader:
    """
    Simulated broker for paper trading and backtesting.
    Models:
      - Market order slippage (configurable bps)
      - Commission (percentage per trade)
      - Partial fills (for large orders)
      - Order book state
    """

    def __init__(self, initial_capital: Optional[float] = None) -> None:
        bt_cfg  = settings.get("backtest", {})
        self.initial_capital  = initial_capital or bt_cfg.get("initial_capital", 100_000)
        self.commission_pct   = bt_cfg.get("commission_pct", 0.001)
        self.slippage_pct     = bt_cfg.get("slippage_pct", 0.0005)
        self.forex_spread_pip = bt_cfg.get("forex_spread_pips", 1.0)

        self.cash     = float(self.initial_capital)
        self.equity   = 0.0
        self.positions: dict[str, dict] = {}  # symbol → position
        self.orders:    list[dict] = []
        self.trades:    list[dict] = []
        self._current_prices: dict[str, float] = {}

    # ─── Price Update ────────────────────────────────────────────────────

    def update_price(self, symbol: str, price: float) -> None:
        """Update current market price for a symbol."""
        self._current_prices[symbol] = price
        self._check_stop_take_profit(symbol, price)

    def _check_stop_take_profit(self, symbol: str, price: float) -> None:
        """Check if any SL/TP levels have been hit."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        if pos["side"] == "LONG":
            if pos.get("stop_price") and price <= pos["stop_price"]:
                self._close_position(symbol, price, reason="stop_loss")
            elif pos.get("take_profit") and price >= pos["take_profit"]:
                self._close_position(symbol, price, reason="take_profit")
        else:  # SHORT
            if pos.get("stop_price") and price >= pos["stop_price"]:
                self._close_position(symbol, price, reason="stop_loss")
            elif pos.get("take_profit") and price <= pos["take_profit"]:
                self._close_position(symbol, price, reason="take_profit")

    # ─── Order Execution ─────────────────────────────────────────────────

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        market: str = "us_equities",
    ) -> dict:
        """
        Simulate a bracket order fill with slippage.
        """
        client_order_id = f"ag_paper_{uuid.uuid4().hex[:12]}"
        current_price = self._current_prices.get(symbol, 0)

        if current_price <= 0:
            return {"status": "REJECTED", "reason": "No price available"}

        # Simulate slippage
        if side.upper() == "BUY":
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)

        fill_price = round(fill_price, 4)
        commission = fill_price * qty * self.commission_pct
        total_cost = fill_price * qty + commission

        if side.upper() == "BUY":
            if self.cash < total_cost:
                qty = (self.cash * 0.99) / (fill_price + commission / qty)
                qty = max(0, qty)
                if qty < 0.001:
                    return {"status": "REJECTED", "reason": "Insufficient funds"}
                total_cost = fill_price * qty + commission

            self.cash -= total_cost
            self.positions[symbol] = {
                "symbol":       symbol,
                "market":       market,
                "side":         "LONG",
                "qty":          qty,
                "entry_price":  fill_price,
                "current_price": fill_price,
                "stop_price":   stop_loss,
                "take_profit":  take_profit,
                "unrealised_pnl": 0.0,
                "commission":   commission,
                "opened_at":    datetime.utcnow(),
            }
        else:  # SELL / SHORT
            # For simplicity, only close longs (no naked shorting in paper mode)
            if symbol not in self.positions:
                return {"status": "REJECTED", "reason": "No position to sell"}
            self._close_position(symbol, fill_price, reason="signal")
            return {
                "status": "FILLED",
                "client_order_id": client_order_id,
                "filled_avg_price": fill_price,
                "filled_qty": qty,
                "commission": commission,
            }

        order = {
            "status": "FILLED",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "filled_qty": qty,
            "filled_avg_price": fill_price,
            "stop_price": stop_loss,
            "take_profit": take_profit,
            "commission": commission,
            "filled_at": datetime.utcnow().isoformat(),
        }
        self.orders.append(order)

        logger.info(
            f"[PAPER] {side.upper()} {qty:.2f} {symbol} @ {fill_price:.4f} | "
            f"SL={stop_loss:.4f} TP={take_profit:.4f} | commission={commission:.2f}"
        )
        return order

    def _close_position(self, symbol: str, exit_price: float, reason: str = "manual") -> None:
        """Close a position and record the trade."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        entry = pos["entry_price"]
        qty   = pos["qty"]

        if pos["side"] == "LONG":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        commission = exit_price * qty * self.commission_pct
        net_pnl    = pnl - commission - pos.get("commission", 0)
        self.cash  += exit_price * qty - commission

        holding_mins = int(
            (datetime.utcnow() - pos["opened_at"]).total_seconds() / 60
        )

        trade = {
            "symbol":    symbol,
            "market":    pos.get("market", "us_equities"),
            "side":      pos["side"],
            "qty":       qty,
            "entry_price": entry,
            "exit_price":  exit_price,
            "pnl":       net_pnl,
            "pnl_pct":   net_pnl / (entry * qty + 1e-9),
            "commission": commission + pos.get("commission", 0),
            "opened_at": pos["opened_at"].isoformat(),
            "closed_at": datetime.utcnow().isoformat(),
            "holding_minutes": holding_mins,
            "close_reason": reason,
        }
        self.trades.append(trade)

        logger.info(
            f"[PAPER] CLOSED {symbol} @ {exit_price:.4f} | "
            f"PnL={net_pnl:+.2f} ({trade['pnl_pct']:+.2%}) | reason={reason}"
        )

    def close_position(self, symbol: str) -> bool:
        """Manually close a position at current price."""
        price = self._current_prices.get(symbol)
        if price and symbol in self.positions:
            self._close_position(symbol, price, reason="manual")
            return True
        return False

    def close_all(self) -> None:
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            self.close_position(symbol)

    # ─── Portfolio State ─────────────────────────────────────────────────

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + open positions)."""
        equity = sum(
            pos["qty"] * self._current_prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + equity

    @property
    def unrealised_pnl(self) -> float:
        """Total unrealised P&L across all open positions."""
        total = 0.0
        for sym, pos in self.positions.items():
            price = self._current_prices.get(sym, pos["entry_price"])
            if pos["side"] == "LONG":
                total += (price - pos["entry_price"]) * pos["qty"]
            else:
                total += (pos["entry_price"] - price) * pos["qty"]
        return total

    def get_account(self) -> dict:
        return {
            "cash":            round(self.cash, 2),
            "equity":          round(self.portfolio_value - self.cash, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "unrealised_pnl":  round(self.unrealised_pnl, 2),
            "n_positions":     len(self.positions),
            "n_trades":        len(self.trades),
        }

    def get_positions(self) -> list[dict]:
        result = []
        for sym, pos in self.positions.items():
            price = self._current_prices.get(sym, pos["entry_price"])
            upnl  = (price - pos["entry_price"]) * pos["qty"] if pos["side"] == "LONG" \
                    else (pos["entry_price"] - price) * pos["qty"]
            result.append({**pos, "current_price": price, "unrealised_pnl": upnl})
        return result

    def get_trade_history(self) -> list[dict]:
        return self.trades.copy()
