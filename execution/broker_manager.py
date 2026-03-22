"""
execution/broker_manager.py  —  AlphaGrid v4
=============================================
Multi-broker abstraction layer.

Supported brokers:
  • Alpaca Markets  — US equities (free paper + live)
  • OANDA           — Forex (free demo + live)
  • Interactive Brokers (IBKR) — skeleton, requires TWS/Gateway running
  • Paper           — built-in simulator (no credentials needed)

BrokerManager:
  - Holds N broker connections simultaneously
  - Routes orders to the correct broker based on asset class
  - Aggregates account/position data across all connected brokers
  - Broadcasts connection status via AppState.health
"""
from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional

from loguru import logger


# ── Enums ─────────────────────────────────────────────────────────────────────

class BrokerID(str, Enum):
    ALPACA     = "alpaca"
    OANDA      = "oanda"
    ROBINHOOD  = "robinhood"
    IBKR       = "ibkr"
    PAPER      = "paper"


class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"


class BrokerStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING   = "connecting"
    CONNECTED    = "connected"
    ERROR        = "error"


# ── Order dataclass ───────────────────────────────────────────────────────────

class Order:
    def __init__(
        self,
        symbol:      str,
        side:        OrderSide,
        qty:         float,
        order_type:  OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
        broker_id:   Optional[BrokerID] = None,
    ):
        self.client_id   = f"ag_{uuid.uuid4().hex[:10]}"
        self.symbol      = symbol
        self.side        = side
        self.qty         = qty
        self.order_type  = order_type
        self.limit_price = limit_price
        self.stop_loss   = stop_loss
        self.take_profit = take_profit
        self.broker_id   = broker_id
        self.broker_order_id: Optional[str] = None
        self.status      = "pending"
        self.filled_qty  = 0.0
        self.filled_price: Optional[float] = None
        self.submitted_at = datetime.utcnow().isoformat()
        self.filled_at:   Optional[str] = None
        self.error:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "client_id":     self.client_id,
            "broker_order_id": self.broker_order_id,
            "broker":        self.broker_id.value if self.broker_id else "unknown",
            "symbol":        self.symbol,
            "side":          self.side.value,
            "qty":           self.qty,
            "order_type":    self.order_type.value,
            "limit_price":   self.limit_price,
            "stop_loss":     self.stop_loss,
            "take_profit":   self.take_profit,
            "status":        self.status,
            "filled_qty":    self.filled_qty,
            "filled_price":  self.filled_price,
            "submitted_at":  self.submitted_at,
            "filled_at":     self.filled_at,
            "error":         self.error,
        }


# ── Abstract broker base ──────────────────────────────────────────────────────

class BrokerBase(ABC):
    broker_id:  BrokerID
    status:     BrokerStatus = BrokerStatus.DISCONNECTED
    error_msg:  str = ""

    @abstractmethod
    async def connect(self, credentials: dict) -> bool: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_account(self) -> Optional[dict]: ...

    @abstractmethod
    async def get_positions(self) -> list[dict]: ...

    @abstractmethod
    async def get_orders(self, status: str = "open") -> list[dict]: ...

    @abstractmethod
    async def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool: ...

    @abstractmethod
    async def close_position(self, symbol: str) -> bool: ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    def health(self) -> dict:
        return {
            "broker":  self.broker_id.value,
            "status":  self.status.value,
            "error":   self.error_msg,
        }


# ── Alpaca broker ─────────────────────────────────────────────────────────────

class AlpacaBroker(BrokerBase):
    broker_id = BrokerID.ALPACA

    def __init__(self):
        self._client = None
        self.status  = BrokerStatus.DISCONNECTED

    async def connect(self, credentials: dict) -> bool:
        """
        credentials = {
            "api_key":    "...",
            "secret_key": "...",
            "paper":      True   # True = paper trading URL
        }
        """
        self.status = BrokerStatus.CONNECTING
        try:
            import alpaca_trade_api as tradeapi  # pip install alpaca-trade-api
            base_url = (
                "https://paper-api.alpaca.markets"
                if credentials.get("paper", True)
                else "https://api.alpaca.markets"
            )
            self._client = tradeapi.REST(
                key_id=credentials["api_key"],
                secret_key=credentials["secret_key"],
                base_url=base_url,
            )
            # Validate by fetching account
            acct = self._client.get_account()
            self.status = BrokerStatus.CONNECTED
            mode = "paper" if credentials.get("paper", True) else "LIVE"
            logger.info(f"Alpaca connected ({mode}) — equity=${float(acct.equity):,.2f}")
            return True
        except ImportError:
            self.error_msg = "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
            self.status = BrokerStatus.ERROR
            return False
        except Exception as e:
            self.error_msg = str(e)
            self.status = BrokerStatus.ERROR
            logger.error(f"Alpaca connect failed: {e}")
            return False

    async def disconnect(self) -> None:
        self._client = None
        self.status  = BrokerStatus.DISCONNECTED

    async def get_account(self) -> Optional[dict]:
        if not self._client:
            return None
        try:
            a = self._client.get_account()
            return {
                "broker":        "alpaca",
                "cash":          round(float(a.cash), 2),
                "equity":        round(float(a.equity), 2),
                "buying_power":  round(float(a.buying_power), 2),
                "portfolio_value": round(float(a.portfolio_value), 2),
                "day_trade_count": int(a.daytrade_count),
                "pattern_day_trader": bool(a.pattern_day_trader),
                "trading_blocked": bool(a.trading_blocked),
                "currency":      "USD",
            }
        except Exception as e:
            logger.error(f"Alpaca get_account: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        if not self._client:
            return []
        try:
            return [{
                "broker":       "alpaca",
                "symbol":       p.symbol,
                "qty":          float(p.qty),
                "side":         "LONG" if float(p.qty) > 0 else "SHORT",
                "avg_entry":    round(float(p.avg_entry_price), 4),
                "current_price":round(float(p.current_price), 4),
                "market_value": round(float(p.market_value), 2),
                "unrealised_pnl": round(float(p.unrealized_pl), 2),
                "unrealised_pct": round(float(p.unrealized_plpc) * 100, 3),
                "asset_class":  p.asset_class,
            } for p in self._client.list_positions()]
        except Exception as e:
            logger.error(f"Alpaca get_positions: {e}")
            return []

    async def get_orders(self, status: str = "open") -> list[dict]:
        if not self._client:
            return []
        try:
            orders = self._client.list_orders(status=status, limit=50)
            return [{
                "broker":       "alpaca",
                "order_id":     o.id,
                "symbol":       o.symbol,
                "side":         o.side.upper(),
                "type":         o.type,
                "qty":          float(o.qty or 0),
                "filled_qty":   float(o.filled_qty or 0),
                "limit_price":  float(o.limit_price) if o.limit_price else None,
                "stop_price":   float(o.stop_price) if o.stop_price else None,
                "status":       o.status,
                "submitted_at": str(o.submitted_at),
            } for o in orders]
        except Exception as e:
            logger.error(f"Alpaca get_orders: {e}")
            return []

    async def submit_order(self, order: Order) -> Order:
        if not self._client:
            order.status = "rejected"
            order.error  = "Alpaca not connected"
            return order
        try:
            kwargs: dict = {
                "symbol": order.symbol,
                "qty":    max(1, int(order.qty)),
                "side":   order.side.value,
                "type":   order.order_type.value,
                "time_in_force": "gtc",
                "client_order_id": order.client_id,
            }
            if order.order_type == OrderType.LIMIT and order.limit_price:
                kwargs["limit_price"] = round(order.limit_price, 2)
            # Bracket: only if both SL and TP specified
            if order.stop_loss and order.take_profit:
                kwargs["order_class"]  = "bracket"
                kwargs["stop_loss"]    = {"stop_price": round(order.stop_loss, 2)}
                kwargs["take_profit"]  = {"limit_price": round(order.take_profit, 2)}

            resp = self._client.submit_order(**kwargs)
            order.broker_order_id = resp.id
            order.status          = "submitted"
            logger.info(f"[Alpaca] Order submitted: {order.side.value.upper()} "
                        f"{order.qty} {order.symbol} | id={resp.id}")
        except Exception as e:
            order.status = "rejected"
            order.error  = str(e)
            logger.error(f"[Alpaca] Order failed for {order.symbol}: {e}")
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            self._client.cancel_order(broker_order_id)
            return True
        except Exception as e:
            logger.error(f"Alpaca cancel_order: {e}")
            return False

    async def close_position(self, symbol: str) -> bool:
        try:
            self._client.close_position(symbol)
            logger.info(f"[Alpaca] Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Alpaca close_position: {e}")
            return False

    def is_market_open(self) -> bool:
        try:
            return self._client.get_clock().is_open if self._client else False
        except Exception:
            return False


# ── OANDA broker (Forex) ──────────────────────────────────────────────────────

class OANDABroker(BrokerBase):
    broker_id = BrokerID.OANDA

    def __init__(self):
        self._api      = None
        self._acct_id  = None
        self.status    = BrokerStatus.DISCONNECTED

    async def connect(self, credentials: dict) -> bool:
        """
        credentials = {
            "api_key":    "...",
            "account_id": "...",
            "practice":   True
        }
        """
        self.status = BrokerStatus.CONNECTING
        try:
            import oandapyV20  # pip install oandapyV20
            from oandapyV20 import API as OandaAPI
            env = "practice" if credentials.get("practice", True) else "live"
            self._api     = OandaAPI(access_token=credentials["api_key"], environment=env)
            self._acct_id = credentials["account_id"]
            # Validate
            acct = await self.get_account()
            if acct:
                self.status = BrokerStatus.CONNECTED
                logger.info(f"OANDA connected ({env}) — balance={acct.get('balance','?')}")
                return True
            self.status   = BrokerStatus.ERROR
            self.error_msg = "Account fetch returned None"
            return False
        except ImportError:
            self.error_msg = "oandapyV20 not installed. Run: pip install oandapyV20"
            self.status    = BrokerStatus.ERROR
            return False
        except Exception as e:
            self.error_msg = str(e)
            self.status    = BrokerStatus.ERROR
            logger.error(f"OANDA connect failed: {e}")
            return False

    async def disconnect(self) -> None:
        self._api    = None
        self.status  = BrokerStatus.DISCONNECTED

    async def get_account(self) -> Optional[dict]:
        if not self._api:
            return None
        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            r = AccountSummary(self._acct_id)
            self._api.request(r)
            a = r.response["account"]
            return {
                "broker":       "oanda",
                "balance":      round(float(a.get("balance", 0)), 2),
                "nav":          round(float(a.get("NAV", 0)), 2),
                "unrealised_pnl": round(float(a.get("unrealizedPL", 0)), 2),
                "margin_used":  round(float(a.get("marginUsed", 0)), 2),
                "margin_avail": round(float(a.get("marginAvailable", 0)), 2),
                "currency":     a.get("currency", "USD"),
                "open_trades":  int(a.get("openTradeCount", 0)),
            }
        except Exception as e:
            logger.error(f"OANDA get_account: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        if not self._api:
            return []
        try:
            from oandapyV20.endpoints.positions import OpenPositions
            r = OpenPositions(self._acct_id)
            self._api.request(r)
            positions = []
            for p in r.response.get("positions", []):
                long_u  = int(p.get("long",  {}).get("units", 0))
                short_u = int(p.get("short", {}).get("units", 0))
                units   = long_u if long_u else short_u
                avg_p   = float(p.get("long",{}).get("averagePrice",
                          p.get("short",{}).get("averagePrice", 0)))
                upnl    = float(p.get("unrealizedPL", 0))
                positions.append({
                    "broker":        "oanda",
                    "symbol":        p["instrument"],
                    "qty":           abs(units),
                    "side":          "LONG" if long_u else "SHORT",
                    "avg_entry":     round(avg_p, 5),
                    "unrealised_pnl": round(upnl, 2),
                })
            return positions
        except Exception as e:
            logger.error(f"OANDA get_positions: {e}")
            return []

    async def get_orders(self, status: str = "open") -> list[dict]:
        if not self._api:
            return []
        try:
            from oandapyV20.endpoints.orders import OrdersPending
            r = OrdersPending(self._acct_id)
            self._api.request(r)
            return [{
                "broker": "oanda",
                "order_id": o.get("id"),
                "type": o.get("type"),
                "instrument": o.get("instrument"),
                "units": o.get("units"),
                "status": o.get("state"),
            } for o in r.response.get("orders", [])]
        except Exception as e:
            logger.error(f"OANDA get_orders: {e}")
            return []

    async def submit_order(self, order: Order) -> Order:
        if not self._api:
            order.status = "rejected"
            order.error  = "OANDA not connected"
            return order
        try:
            from oandapyV20.endpoints.orders import OrderCreate
            # OANDA uses positive units for BUY, negative for SELL
            units = int(order.qty) if order.side == OrderSide.BUY else -int(order.qty)
            body = {"order": {
                "type": "MARKET",
                "instrument": order.symbol,
                "units": str(units),
            }}
            if order.stop_loss:
                body["order"]["stopLossOnFill"] = {"price": f"{order.stop_loss:.5f}"}
            if order.take_profit:
                body["order"]["takeProfitOnFill"] = {"price": f"{order.take_profit:.5f}"}

            r = OrderCreate(self._acct_id, data=body)
            self._api.request(r)
            resp = r.response
            trade_id = resp.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
            order.broker_order_id = trade_id or resp.get("orderCreateTransaction",{}).get("id")
            order.status          = "submitted"
            logger.info(f"[OANDA] Order submitted: {order.side.value} "
                        f"{order.qty} {order.symbol}")
        except Exception as e:
            order.status = "rejected"
            order.error  = str(e)
            logger.error(f"[OANDA] Order failed: {e}")
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            from oandapyV20.endpoints.orders import OrderCancel
            r = OrderCancel(self._acct_id, broker_order_id)
            self._api.request(r)
            return True
        except Exception as e:
            logger.error(f"OANDA cancel_order: {e}")
            return False

    async def close_position(self, symbol: str) -> bool:
        try:
            from oandapyV20.endpoints.positions import PositionClose
            body = {"longUnits": "ALL", "shortUnits": "ALL"}
            r    = PositionClose(self._acct_id, symbol, data=body)
            self._api.request(r)
            logger.info(f"[OANDA] Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"OANDA close_position: {e}")
            return False

    def is_market_open(self) -> bool:
        # Forex is open 24/5 — closed Saturday UTC
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Friday 22:00 UTC → Sunday 22:00 UTC = closed
        day  = now.weekday()   # 0=Mon, 6=Sun
        hour = now.hour
        if day == 5:  # Saturday — always closed
            return False
        if day == 6 and hour < 22:  # Sunday before 22:00 UTC
            return False
        if day == 4 and hour >= 22:  # Friday after 22:00 UTC
            return False
        return True



# ── Robinhood broker ──────────────────────────────────────────────────────────

class RobinhoodBroker(BrokerBase):
    """
    Robinhood brokerage integration via robin_stocks (unofficial API).

    Install: pip install robin_stocks pyotp

    Authentication:
      Standard:  username + password  (Robinhood prompts SMS/device approval)
      TOTP:      username + password + mfa_secret (authenticator app secret key)
                 Get mfa_secret: Robinhood → Account → Security → 2FA → Authenticator App
                 → when it shows QR code, also shows the text key — save that string.

    ⚠  Important warnings:
      - Robinhood has NO official trading API. robin_stocks uses a reverse-engineered
        private REST API that Robinhood can change or block at any time.
      - As of late 2024 Robinhood tightened their auth flow — login may require
        device approval via the mobile app on first use from a new IP address.
      - NO fractional shares via API (API accepts whole shares only).
      - NO stop-loss bracket orders — only simple market/limit orders supported.
        AlphaGrid handles SL/TP monitoring in software when using Robinhood.
      - Pattern Day Trader (PDT) rules apply: <$25k account → max 3 day trades/week.
      - Commission free but PFOF (payment for order flow) applies.
    """
    broker_id = BrokerID.ROBINHOOD

    def __init__(self):
        self._rh         = None   # robin_stocks.robinhood module
        self._logged_in  = False
        self._username   = ""
        self.status      = BrokerStatus.DISCONNECTED
        # Software-managed SL/TP since Robinhood doesn't support bracket orders
        self._sl_tp_monitor: dict[str, dict] = {}

    async def connect(self, credentials: dict) -> bool:
        """
        credentials = {
            "username":   "your@email.com",
            "password":   "yourpassword",
            "mfa_secret": "JBSWY3DPEHPK3PXP"   # optional TOTP secret from authenticator setup
        }
        With mfa_secret: fully automated login (no phone needed).
        Without mfa_secret: Robinhood sends SMS/device approval — user must approve on phone.
        """
        self.status   = BrokerStatus.CONNECTING
        self.error_msg = ""

        try:
            import robin_stocks.robinhood as rh
            self._rh      = rh
            self._username = credentials.get("username","")
            username  = credentials["username"]
            password  = credentials["password"]
            mfa_secret = credentials.get("mfa_secret","").strip()

            # Build TOTP code if secret provided
            mfa_code = None
            if mfa_secret:
                try:
                    import pyotp
                    mfa_code = pyotp.TOTP(mfa_secret).now()
                    logger.info(f"[RH] TOTP generated from secret")
                except ImportError:
                    self.error_msg = "pyotp not installed. Run: pip install pyotp"
                    self.status = BrokerStatus.ERROR
                    return False
                except Exception as e:
                    self.error_msg = f"TOTP error: {e}"
                    self.status = BrokerStatus.ERROR
                    return False

            # Attempt login
            logger.info(f"[RH] Logging in as {username[:4]}***")
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: rh.login(
                    username=username,
                    password=password,
                    mfa_code=mfa_code,
                    store_session=True,
                    expiresIn=86400,        # 24h session
                )
            )

            # Validate login
            if not result:
                self.error_msg = "Login returned empty response — check credentials"
                self.status = BrokerStatus.ERROR
                return False

            # Check for pending device approval (Robinhood auth flow change Dec 2024)
            if isinstance(result, dict):
                if "verification_workflow" in result:
                    self.error_msg = (
                        "Robinhood requires device approval. "
                        "Open your Robinhood app and approve the login request, "
                        "then retry. If using TOTP, ensure mfa_secret is the "
                        "authenticator key (not a one-time code)."
                    )
                    self.status = BrokerStatus.ERROR
                    logger.warning(f"[RH] Device approval required: {result}")
                    return False
                if "detail" in result and "Invalid" in str(result.get("detail","")):
                    self.error_msg = f"Invalid credentials: {result['detail']}"
                    self.status = BrokerStatus.ERROR
                    return False

            # Verify by fetching account
            acct = await self.get_account()
            if not acct:
                self.error_msg = "Logged in but could not fetch account — session may be invalid"
                self.status = BrokerStatus.ERROR
                return False

            self._logged_in = True
            self.status     = BrokerStatus.CONNECTED
            logger.info(f"[RH] Connected — portfolio: ${acct.get('portfolio_value',0):,.2f}")
            return True

        except ImportError:
            self.error_msg = "robin_stocks not installed. Run: pip install robin_stocks"
            self.status = BrokerStatus.ERROR
            return False
        except Exception as e:
            err = str(e)
            # Surface the most common auth failure clearly
            if "verification_workflow" in err or "workflow_status" in err:
                self.error_msg = (
                    "Robinhood requires device approval on first login from this IP. "
                    "Open the Robinhood app and approve the login, then retry."
                )
            elif "challenge_type" in err or "challenge" in err.lower():
                self.error_msg = "Robinhood sent an SMS challenge — approve on phone then retry."
            else:
                self.error_msg = f"Login failed: {err[:200]}"
            self.status = BrokerStatus.ERROR
            logger.error(f"[RH] Connect error: {err[:200]}")
            return False

    async def disconnect(self) -> None:
        if self._rh and self._logged_in:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._rh.logout)
            except Exception:
                pass
        self._logged_in = False
        self.status     = BrokerStatus.DISCONNECTED
        logger.info("[RH] Logged out")

    async def get_account(self) -> Optional[dict]:
        if not self._rh:
            return None
        try:
            profile = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._rh.load_account_profile(info=None)
            )
            if not profile:
                return None

            # Portfolio value
            portfolio = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._rh.load_portfolio_profile()
            )
            equity   = float(portfolio.get("equity",0) if portfolio else 0)
            cash     = float(profile.get("cash","0") or 0)
            bp       = float(profile.get("buying_power","0") or 0)
            day_trades = int(profile.get("day_trade_count","0") or 0)
            pdt      = profile.get("is_pattern_day_trader", False)

            return {
                "broker":            "robinhood",
                "cash":              round(cash, 2),
                "equity":            round(equity, 2),
                "buying_power":      round(bp, 2),
                "portfolio_value":   round(equity, 2),
                "day_trade_count":   day_trades,
                "pattern_day_trader": pdt,
                "currency":          "USD",
                "username":          self._username[:4] + "***",
            }
        except Exception as e:
            logger.error(f"[RH] get_account: {e}")
            return None

    async def get_positions(self) -> list[dict]:
        if not self._rh or not self._logged_in:
            return []
        try:
            positions_raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._rh.get_open_stock_positions()
            )
            if not positions_raw:
                return []
            results = []
            for p in positions_raw:
                try:
                    qty       = float(p.get("quantity","0") or 0)
                    avg_price = float(p.get("average_buy_price","0") or 0)
                    if qty <= 0:
                        continue
                    # Get current price
                    inst_url  = p.get("instrument","")
                    symbol    = ""
                    curr_price = avg_price
                    if inst_url:
                        try:
                            inst_data = await asyncio.get_event_loop().run_in_executor(
                                None, lambda u=inst_url: self._rh.get_instrument_by_url(u)
                            )
                            symbol = inst_data.get("symbol","") if inst_data else ""
                            if symbol:
                                quote = await asyncio.get_event_loop().run_in_executor(
                                    None, lambda s=symbol: self._rh.get_latest_price(s)
                                )
                                curr_price = float(quote[0]) if quote else avg_price
                        except Exception:
                            pass

                    upnl = (curr_price - avg_price) * qty
                    results.append({
                        "broker":          "robinhood",
                        "symbol":          symbol or "UNKNOWN",
                        "qty":             qty,
                        "side":            "LONG",
                        "avg_entry":       round(avg_price, 4),
                        "current_price":   round(curr_price, 4),
                        "market_value":    round(curr_price * qty, 2),
                        "unrealised_pnl":  round(upnl, 2),
                        "unrealised_pct":  round(upnl / (avg_price * qty + 1e-9) * 100, 3),
                        "asset_class":     "equity",
                    })
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.error(f"[RH] get_positions: {e}")
            return []

    async def get_orders(self, status: str = "open") -> list[dict]:
        if not self._rh or not self._logged_in:
            return []
        try:
            orders_raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._rh.get_all_stock_orders(info=None)
            )
            if not orders_raw:
                return []
            # Filter by status
            rh_status = {"open": ["confirmed","queued","unconfirmed","partially_filled"],
                         "closed": ["filled","cancelled","failed"]}.get(status, ["confirmed"])
            results = []
            for o in orders_raw[:50]:
                if o.get("state","") not in rh_status:
                    continue
                results.append({
                    "broker":       "robinhood",
                    "order_id":     o.get("id",""),
                    "symbol":       o.get("symbol") or o.get("instrument","").split("/")[-2],
                    "side":         o.get("side","").upper(),
                    "type":         o.get("type","market"),
                    "qty":          float(o.get("quantity","0") or 0),
                    "filled_qty":   float(o.get("cumulative_quantity","0") or 0),
                    "limit_price":  float(o.get("price","0") or 0) or None,
                    "status":       o.get("state",""),
                    "submitted_at": o.get("created_at",""),
                })
            return results
        except Exception as e:
            logger.error(f"[RH] get_orders: {e}")
            return []

    async def submit_order(self, order: Order) -> Order:
        """
        Submit market or limit order via robin_stocks.
        NOTE: Robinhood does NOT support bracket/stop-loss orders via API.
        AlphaGrid monitors SL/TP in software and submits closing orders when hit.
        """
        if not self._rh or not self._logged_in:
            order.status = "rejected"
            order.error  = "Robinhood not connected"
            return order
        try:
            sym  = order.symbol
            qty  = max(1, int(order.qty))   # Robinhood API: whole shares only
            side = order.side

            if side == OrderSide.BUY:
                if order.order_type == OrderType.LIMIT and order.limit_price:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._rh.order_buy_limit(
                            symbol=sym, quantity=qty, limitPrice=round(order.limit_price, 2)
                        )
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._rh.order_buy_market(symbol=sym, quantity=qty)
                    )
            else:  # SELL
                if order.order_type == OrderType.LIMIT and order.limit_price:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._rh.order_sell_limit(
                            symbol=sym, quantity=qty, limitPrice=round(order.limit_price, 2)
                        )
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._rh.order_sell_market(symbol=sym, quantity=qty)
                    )

            if not result:
                order.status = "rejected"
                order.error  = "Empty response from Robinhood — order may not have been placed"
                return order

            # Check for error in response
            if isinstance(result, dict) and result.get("detail"):
                order.status = "rejected"
                order.error  = str(result["detail"])
                return order

            order.broker_order_id = result.get("id","") if isinstance(result, dict) else ""
            order.status          = "submitted"
            order.filled_qty      = float(result.get("cumulative_quantity","0") or 0) if isinstance(result,dict) else 0

            # Register for SL/TP software monitoring
            if (order.stop_loss or order.take_profit) and order.status == "submitted":
                self._sl_tp_monitor[sym] = {
                    "side":        side.value,
                    "stop_loss":   order.stop_loss,
                    "take_profit": order.take_profit,
                    "qty":         qty,
                    "order_id":    order.broker_order_id,
                }
                logger.info(f"[RH] SL/TP monitoring registered for {sym}: "
                           f"SL={order.stop_loss} TP={order.take_profit}")

            logger.info(f"[RH] Order submitted: {side.value.upper()} {qty} {sym} | id={order.broker_order_id}")
        except Exception as e:
            order.status = "rejected"
            order.error  = str(e)[:200]
            logger.error(f"[RH] submit_order error: {e}")
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        if not self._rh:
            return False
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._rh.cancel_stock_order(broker_order_id)
            )
            return result is not None
        except Exception as e:
            logger.error(f"[RH] cancel_order: {e}")
            return False

    async def close_position(self, symbol: str) -> bool:
        """Close full position by selling all shares at market."""
        if not self._rh or not self._logged_in:
            return False
        try:
            # Get current qty
            positions = await self.get_positions()
            pos = next((p for p in positions if p["symbol"] == symbol), None)
            if not pos or pos["qty"] <= 0:
                return False
            qty = int(pos["qty"])
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._rh.order_sell_market(symbol=symbol, quantity=qty)
            )
            if result:
                self._sl_tp_monitor.pop(symbol, None)
                logger.info(f"[RH] Position closed: {symbol} qty={qty}")
                return True
            return False
        except Exception as e:
            logger.error(f"[RH] close_position: {e}")
            return False

    def is_market_open(self) -> bool:
        """Check via robin_stocks market hours or fall back to time-based check."""
        try:
            if self._rh:
                markets = self._rh.get_market_hours("XNAS")
                if markets and isinstance(markets, dict):
                    return markets.get("is_open", False)
        except Exception:
            pass
        # Fallback: NYC business hours Mon–Fri 9:30–16:00 ET
        from datetime import timezone
        import zoneinfo
        try:
            et   = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            return True  # Can't determine — allow
        now  = datetime.now(et)
        if now.weekday() >= 5:  # weekend
            return False
        t = now.hour * 60 + now.minute
        return 9*60+30 <= t < 16*60

    async def monitor_sl_tp(self, current_prices: dict) -> list[str]:
        """
        Software-based SL/TP monitoring — call periodically with live prices.
        Returns list of symbols where closing orders were triggered.
        """
        triggered = []
        for sym, monitor in list(self._sl_tp_monitor.items()):
            price = current_prices.get(sym, {})
            if not price:
                continue
            curr = float(price.get("price", 0))
            if curr <= 0:
                continue
            sl   = monitor.get("stop_loss")
            tp   = monitor.get("take_profit")
            side = monitor.get("side","buy")

            hit = False
            reason = ""
            if side == "buy":
                if sl and curr <= sl:
                    hit = True; reason = f"Stop-loss hit: {curr:.4f} ≤ {sl:.4f}"
                elif tp and curr >= tp:
                    hit = True; reason = f"Take-profit hit: {curr:.4f} ≥ {tp:.4f}"
            else:  # short
                if sl and curr >= sl:
                    hit = True; reason = f"Stop-loss hit: {curr:.4f} ≥ {sl:.4f}"
                elif tp and curr <= tp:
                    hit = True; reason = f"Take-profit hit: {curr:.4f} ≤ {tp:.4f}"

            if hit:
                logger.info(f"[RH] SL/TP trigger {sym}: {reason} — closing position")
                ok = await self.close_position(sym)
                if ok:
                    triggered.append(sym)
                    self._sl_tp_monitor.pop(sym, None)
        return triggered

# ── Paper broker (built-in, no credentials) ───────────────────────────────────

class PaperBroker(BrokerBase):
    """Internal paper broker backed by AppState positions/trades."""
    broker_id = BrokerID.PAPER

    def __init__(self, app_state=None):
        self._state = app_state
        self.status = BrokerStatus.CONNECTED

    async def connect(self, credentials: dict) -> bool:
        self.status = BrokerStatus.CONNECTED
        logger.info("Paper broker connected (built-in simulator)")
        return True

    async def disconnect(self) -> None:
        pass

    async def get_account(self) -> Optional[dict]:
        if not self._state:
            return None
        p = self._state.portfolio
        return {
            "broker":          "paper",
            "cash":            round(p.get("cash", 0), 2),
            "equity":          round(p.get("equity", 0), 2),
            "portfolio_value": round(p.get("portfolio_value", 0), 2),
            "unrealised_pnl":  round(p.get("unrealised_pnl", 0), 2),
            "currency":        "USD",
        }

    async def get_positions(self) -> list[dict]:
        if not self._state:
            return []
        return [{"broker": "paper", **pos} for pos in self._state.positions.values()]

    async def get_orders(self, status: str = "open") -> list[dict]:
        return []

    async def submit_order(self, order: Order) -> Order:
        """Delegate to AppState paper trading logic."""
        if not self._state:
            order.status = "rejected"; order.error = "No app state"
            return order
        price_data = self._state.prices.get(order.symbol)
        if not price_data or not price_data.get("price"):
            order.status = "rejected"; order.error = f"No live price for {order.symbol}"
            return order
        price  = price_data["price"]
        cost   = price * order.qty * 1.0005
        cash   = self._state.portfolio.get("cash", 0)
        if order.side == OrderSide.BUY and cost > cash:
            order.status = "rejected"; order.error = "Insufficient cash"
            return order
        sym = order.symbol
        if order.side == OrderSide.BUY:
            self._state.positions[sym] = {
                "symbol": sym, "side": "LONG", "qty": order.qty,
                "entry_price": price, "current_price": price,
                "stop_loss": order.stop_loss, "take_profit": order.take_profit,
                "unrealised_pnl": 0.0, "opened_at": datetime.utcnow().isoformat(),
                "broker": "paper",
            }
            self._state.portfolio["cash"] = round(cash - cost, 2)
        elif order.side == OrderSide.SELL:
            pos = self._state.positions.pop(sym, None)
            if pos:
                pnl = (price - pos["entry_price"]) * pos["qty"] - price * pos["qty"] * 0.001
                self._state.portfolio["cash"] = round(cash + price * pos["qty"] - price*pos["qty"]*0.001, 2)
                self._state.portfolio["daily_pnl"] = round(self._state.portfolio.get("daily_pnl",0)+pnl,2)
                self._state.trades.insert(0, {
                    "symbol": sym, "side": pos["side"], "qty": pos["qty"],
                    "entry_price": pos["entry_price"], "exit_price": round(price,4),
                    "pnl": round(pnl,2), "pnl_pct": round(pnl/(pos["entry_price"]*pos["qty"]),4),
                    "closed_at": datetime.utcnow().isoformat(), "broker": "paper",
                })
        order.broker_order_id = f"paper_{uuid.uuid4().hex[:8]}"
        order.status          = "filled"
        order.filled_qty      = order.qty
        order.filled_price    = price
        order.filled_at       = datetime.utcnow().isoformat()
        logger.info(f"[Paper] {order.side.value.upper()} {order.qty} {sym} @ {price:.4f}")
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        return True

    async def close_position(self, symbol: str) -> bool:
        pos = self._state.positions.get(symbol) if self._state else None
        if not pos:
            return False
        price_data = self._state.prices.get(symbol, {})
        price = price_data.get("price", pos["entry_price"])
        close_order = Order(symbol, OrderSide.SELL, pos["qty"])
        self._state.prices[symbol] = {**price_data, "price": price}
        await self.submit_order(close_order)
        return True

    def is_market_open(self) -> bool:
        return True


# ── Broker Manager ────────────────────────────────────────────────────────────

class BrokerManager:
    """
    Manages multiple broker connections simultaneously.
    Routes orders by asset class: equities → Alpaca, forex → OANDA.
    Falls back to Paper for any unconnected asset class.
    """

    def __init__(self, app_state=None):
        self._brokers: dict[BrokerID, BrokerBase] = {}
        self._paper   = PaperBroker(app_state)
        self._brokers[BrokerID.PAPER] = self._paper
        self._order_log: list[dict] = []

    # ── Connection ────────────────────────────────────────────────────────

    async def connect_alpaca(self, api_key: str, secret_key: str, paper: bool = True) -> dict:
        broker = AlpacaBroker()
        ok = await broker.connect({"api_key": api_key, "secret_key": secret_key, "paper": paper})
        if ok:
            self._brokers[BrokerID.ALPACA] = broker
        return {"success": ok, "status": broker.status.value, "error": broker.error_msg}

    async def connect_robinhood(
        self,
        username:   str,
        password:   str,
        mfa_secret: str = "",
    ) -> dict:
        """
        Connect Robinhood account.
        mfa_secret: the TOTP secret key from your authenticator app setup
                    (the text string shown alongside the QR code).
                    Leave empty to use SMS/device approval flow.
        """
        broker = RobinhoodBroker()
        ok = await broker.connect({
            "username":   username,
            "password":   password,
            "mfa_secret": mfa_secret,
        })
        if ok:
            self._brokers[BrokerID.ROBINHOOD] = broker
            logger.info("Robinhood broker registered in BrokerManager")
        return {
            "success": ok,
            "status":  broker.status.value,
            "error":   broker.error_msg,
        }

    async def run_sl_tp_monitor(self, current_prices: dict) -> list[str]:
        """
        Run software SL/TP monitoring for Robinhood (which has no native bracket orders).
        Call this from the price feed loop every tick.
        Returns list of symbols where positions were closed.
        """
        rh = self._brokers.get(BrokerID.ROBINHOOD)
        if rh and isinstance(rh, RobinhoodBroker):
            return await rh.monitor_sl_tp(current_prices)
        return []

    async def connect_oanda(self, api_key: str, account_id: str, practice: bool = True) -> dict:
        broker = OANDABroker()
        ok = await broker.connect({"api_key": api_key, "account_id": account_id, "practice": practice})
        if ok:
            self._brokers[BrokerID.OANDA] = broker
        return {"success": ok, "status": broker.status.value, "error": broker.error_msg}

    async def disconnect(self, broker_id: BrokerID) -> None:
        b = self._brokers.get(broker_id)
        if b:
            await b.disconnect()
            if broker_id != BrokerID.PAPER:
                del self._brokers[broker_id]

    # ── Routing ───────────────────────────────────────────────────────────

    def _route(self, symbol: str) -> BrokerBase:
        """
        Route to correct broker based on symbol type and availability.
        Priority: Alpaca > Robinhood (equities) | OANDA (forex) | Paper fallback
        """
        is_forex = "=X" in symbol or "/" in symbol or (len(symbol) == 6 and "USD" in symbol.upper())
        if is_forex and BrokerID.OANDA in self._brokers:
            return self._brokers[BrokerID.OANDA]
        # Equities: Alpaca preferred, Robinhood fallback
        if not is_forex:
            if BrokerID.ALPACA in self._brokers:
                return self._brokers[BrokerID.ALPACA]
            if BrokerID.ROBINHOOD in self._brokers:
                return self._brokers[BrokerID.ROBINHOOD]
        return self._paper

    # ── Order API ─────────────────────────────────────────────────────────

    async def submit_order(
        self,
        symbol:      str,
        side:        str,       # "buy" | "sell"
        qty:         float,
        order_type:  str = "market",
        limit_price: Optional[float] = None,
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
        broker:      Optional[str]  = None,
    ) -> dict:
        order = Order(
            symbol      = symbol,
            side        = OrderSide(side.lower()),
            qty         = qty,
            order_type  = OrderType(order_type.lower()),
            limit_price = limit_price,
            stop_loss   = stop_loss,
            take_profit = take_profit,
        )
        b = (self._brokers.get(BrokerID(broker)) if broker else None) or self._route(symbol)
        order.broker_id = b.broker_id
        order = await b.submit_order(order)
        self._order_log.insert(0, order.to_dict())
        self._order_log = self._order_log[:200]
        return order.to_dict()

    async def cancel_order(self, broker_id: str, broker_order_id: str) -> bool:
        b = self._brokers.get(BrokerID(broker_id))
        return await b.cancel_order(broker_order_id) if b else False

    async def close_position(self, symbol: str, broker: Optional[str] = None) -> bool:
        b = (self._brokers.get(BrokerID(broker)) if broker else None) or self._route(symbol)
        return await b.close_position(symbol)

    # ── Account aggregation ───────────────────────────────────────────────

    async def get_all_accounts(self) -> list[dict]:
        tasks  = [b.get_account() for b in self._brokers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    async def get_all_positions(self) -> list[dict]:
        tasks  = [b.get_positions() for b in self._brokers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [pos for r in results if isinstance(r, list) for pos in r]

    async def get_all_orders(self) -> list[dict]:
        tasks  = [b.get_orders() for b in self._brokers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [o for r in results if isinstance(r, list) for o in r]

    def order_log(self) -> list[dict]:
        return self._order_log

    # ── Status ────────────────────────────────────────────────────────────

    def connection_status(self) -> dict:
        return {
            bid.value: b.health()
            for bid, b in self._brokers.items()
        }

    def connected_brokers(self) -> list[str]:
        return [
            bid.value for bid, b in self._brokers.items()
            if b.status == BrokerStatus.CONNECTED
        ]
