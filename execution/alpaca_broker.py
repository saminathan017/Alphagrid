"""
execution/alpaca_broker.py
Alpaca Markets order execution for US equities.
Handles order submission, status tracking, and position management.
"""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime
from typing import Optional
from loguru import logger
from core.config import settings
from core.events import event_bus, Event, EventType


class AlpacaBroker:
    """
    Alpaca Markets execution adapter.
    Supports market, limit, and bracket orders.
    """

    def __init__(self) -> None:
        self._cfg = settings.get("alpaca", {})
        self._client = None
        self._mode   = settings.get("trading", {}).get("mode", "paper")

    def _get_client(self):
        if self._client is None:
            try:
                import alpaca_trade_api as tradeapi
                self._client = tradeapi.REST(
                    key_id=self._cfg["api_key"],
                    secret_key=self._cfg["secret_key"],
                    base_url=self._cfg["base_url"],
                )
                logger.info(f"AlpacaBroker connected ({self._mode} mode)")
            except Exception as e:
                logger.error(f"Alpaca connection failed: {e}")
                raise
        return self._client

    # ─── Order Submission ────────────────────────────────────────────────

    async def submit_bracket_order(
        self,
        symbol: str,
        side: str,           # buy | sell
        qty: float,
        stop_loss: float,
        take_profit: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Submit a bracket order (entry + stop-loss + take-profit).
        """
        client_order_id = f"ag_{uuid.uuid4().hex[:12]}"
        try:
            client = self._get_client()
            order_kwargs = {
                "symbol": symbol,
                "qty": max(1, int(qty)),
                "side": side.lower(),
                "type": order_type,
                "time_in_force": "gtc",
                "client_order_id": client_order_id,
                "order_class": "bracket",
                "stop_loss": {"stop_price": round(stop_loss, 2)},
                "take_profit": {"limit_price": round(take_profit, 2)},
            }
            if order_type == "limit" and limit_price:
                order_kwargs["limit_price"] = round(limit_price, 2)

            order = client.submit_order(**order_kwargs)

            order_data = {
                "client_order_id": client_order_id,
                "broker_order_id": order.id,
                "symbol": symbol,
                "market": "us_equities",
                "side": side.upper(),
                "order_type": order_type.upper(),
                "qty": qty,
                "stop_price": stop_loss,
                "take_profit": take_profit,
                "status": "SUBMITTED",
                "submitted_at": datetime.utcnow().isoformat(),
            }

            await event_bus.publish(Event(
                event_type=EventType.ORDER_SUBMITTED,
                source="alpaca_broker",
                data=order_data,
            ))
            logger.info(
                f"Order submitted: {side} {qty:.0f} {symbol} | "
                f"SL={stop_loss:.2f} TP={take_profit:.2f} | ID={client_order_id}"
            )
            return order_data

        except Exception as e:
            logger.error(f"Order submission failed for {symbol}: {e}")
            await event_bus.publish(Event(
                event_type=EventType.ORDER_REJECTED,
                source="alpaca_broker",
                data={"symbol": symbol, "error": str(e), "client_order_id": client_order_id},
            ))
            return None

    async def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Optional[dict]:
        """Submit a simple market order."""
        return await self.submit_bracket_order(
            symbol, side, qty,
            stop_loss=0, take_profit=float("inf"),  # No bracket
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self._get_client().cancel_order(broker_order_id)
            await event_bus.publish(Event(
                event_type=EventType.ORDER_CANCELLED,
                source="alpaca_broker",
                data={"broker_order_id": broker_order_id},
            ))
            return True
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            return False

    async def close_position(self, symbol: str) -> bool:
        """Close an entire position at market."""
        try:
            self._get_client().close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Close position failed for {symbol}: {e}")
            return False

    # ─── Account & Position Queries ──────────────────────────────────────

    def get_account(self) -> Optional[dict]:
        """Get account balance and portfolio value."""
        try:
            acct = self._get_client().get_account()
            return {
                "cash": float(acct.cash),
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "day_trade_count": int(acct.daytrade_count),
            }
        except Exception as e:
            logger.error(f"Get account failed: {e}")
            return None

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        try:
            positions = self._get_client().list_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": "LONG" if float(p.qty) > 0 else "SHORT",
                    "market_value": float(p.market_value),
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealised_pnl": float(p.unrealized_pl),
                    "unrealised_pnl_pct": float(p.unrealized_plpc),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return []

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        try:
            orders = self._get_client().list_orders(status="open")
            return [
                {
                    "order_id": o.id,
                    "symbol": o.symbol,
                    "side": o.side.upper(),
                    "type": o.type,
                    "qty": float(o.qty),
                    "status": o.status,
                    "submitted_at": str(o.submitted_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Get open orders failed: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        try:
            clock = self._get_client().get_clock()
            return clock.is_open
        except Exception:
            return False
