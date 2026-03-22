"""
core/database.py
SQLAlchemy ORM models for trades, signals, predictions, and portfolio state.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, JSON, Enum as SAEnum, ForeignKey, Index
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from core.config import settings


class Base(DeclarativeBase):
    pass


# ─── ORM Models ──────────────────────────────────────────────────────────────

class Candle(Base):
    """OHLCV price data."""
    __tablename__ = "candles"
    id        = Column(Integer, primary_key=True)
    symbol    = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open      = Column(Float, nullable=False)
    high      = Column(Float, nullable=False)
    low       = Column(Float, nullable=False)
    close     = Column(Float, nullable=False)
    volume    = Column(Float, nullable=False)
    __table_args__ = (
        Index("ix_candles_symbol_tf_ts", "symbol", "timeframe", "timestamp", unique=True),
    )


class Prediction(Base):
    """ML model price predictions."""
    __tablename__ = "predictions"
    id          = Column(Integer, primary_key=True)
    symbol      = Column(String(20), nullable=False)
    model_name  = Column(String(50), nullable=False)
    timestamp   = Column(DateTime, nullable=False, default=datetime.utcnow)
    horizon     = Column(Integer, nullable=False)  # candles ahead
    direction   = Column(String(5))                # UP | DOWN | FLAT
    confidence  = Column(Float)                    # 0–1
    predicted_price = Column(Float)
    features    = Column(JSON)                     # input feature snapshot
    __table_args__ = (
        Index("ix_predictions_symbol_ts", "symbol", "timestamp"),
    )


class Signal(Base):
    """Trading signal from strategy layer."""
    __tablename__ = "signals"
    id            = Column(Integer, primary_key=True)
    symbol        = Column(String(20), nullable=False)
    market        = Column(String(20), nullable=False)   # us_equities | forex
    direction     = Column(String(5), nullable=False)    # LONG | SHORT | FLAT
    strength      = Column(Float)                        # 0–1
    strategy      = Column(String(50))
    timestamp     = Column(DateTime, nullable=False, default=datetime.utcnow)
    ml_score      = Column(Float)
    ta_score      = Column(Float)
    sentiment_score = Column(Float)
    ensemble_score  = Column(Float)
    acted_on      = Column(Boolean, default=False)
    order_id      = Column(String(50), ForeignKey("orders.client_order_id"), nullable=True)
    __table_args__ = (
        Index("ix_signals_symbol_ts", "symbol", "timestamp"),
    )


class Order(Base):
    """Order record."""
    __tablename__ = "orders"
    id              = Column(Integer, primary_key=True)
    client_order_id = Column(String(50), unique=True, nullable=False)
    broker_order_id = Column(String(100))
    symbol          = Column(String(20), nullable=False)
    market          = Column(String(20), nullable=False)
    side            = Column(String(5), nullable=False)    # BUY | SELL
    order_type      = Column(String(20), nullable=False)   # MARKET | LIMIT | STOP
    qty             = Column(Float, nullable=False)
    limit_price     = Column(Float)
    stop_price      = Column(Float)
    status          = Column(String(20), default="PENDING")
    filled_qty      = Column(Float, default=0.0)
    filled_avg_price = Column(Float)
    commission      = Column(Float, default=0.0)
    slippage        = Column(Float, default=0.0)
    submitted_at    = Column(DateTime, default=datetime.utcnow)
    filled_at       = Column(DateTime)
    cancelled_at    = Column(DateTime)
    signal_id       = Column(Integer, ForeignKey("signals.id"), nullable=True)


class Position(Base):
    """Current open positions."""
    __tablename__ = "positions"
    id            = Column(Integer, primary_key=True)
    symbol        = Column(String(20), unique=True, nullable=False)
    market        = Column(String(20), nullable=False)
    side          = Column(String(5), nullable=False)
    qty           = Column(Float, nullable=False)
    entry_price   = Column(Float, nullable=False)
    current_price = Column(Float)
    stop_price    = Column(Float)
    take_profit   = Column(Float)
    unrealised_pnl = Column(Float, default=0.0)
    realised_pnl  = Column(Float, default=0.0)
    opened_at     = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Trade(Base):
    """Completed (closed) trades."""
    __tablename__ = "trades"
    id            = Column(Integer, primary_key=True)
    symbol        = Column(String(20), nullable=False)
    market        = Column(String(20), nullable=False)
    side          = Column(String(5), nullable=False)
    qty           = Column(Float, nullable=False)
    entry_price   = Column(Float, nullable=False)
    exit_price    = Column(Float, nullable=False)
    pnl           = Column(Float, nullable=False)
    pnl_pct       = Column(Float, nullable=False)
    commission    = Column(Float, default=0.0)
    opened_at     = Column(DateTime, nullable=False)
    closed_at     = Column(DateTime, nullable=False)
    holding_minutes = Column(Integer)
    strategy      = Column(String(50))
    tags          = Column(JSON)
    __table_args__ = (
        Index("ix_trades_symbol_closed", "symbol", "closed_at"),
    )


class SentimentRecord(Base):
    """Stored sentiment analysis results."""
    __tablename__ = "sentiment"
    id          = Column(Integer, primary_key=True)
    symbol      = Column(String(20), nullable=False)
    source      = Column(String(30), nullable=False)  # news | reddit | twitter
    timestamp   = Column(DateTime, nullable=False, default=datetime.utcnow)
    score       = Column(Float, nullable=False)       # -1 to +1
    magnitude   = Column(Float)                       # 0 to 1
    headline    = Column(String(500))
    url         = Column(String(500))
    __table_args__ = (
        Index("ix_sentiment_symbol_ts", "symbol", "timestamp"),
    )


class PortfolioSnapshot(Base):
    """Periodic portfolio state snapshots."""
    __tablename__ = "portfolio_snapshots"
    id               = Column(Integer, primary_key=True)
    timestamp        = Column(DateTime, nullable=False, default=datetime.utcnow)
    cash             = Column(Float, nullable=False)
    equity           = Column(Float, nullable=False)
    total_value      = Column(Float, nullable=False)
    unrealised_pnl   = Column(Float, default=0.0)
    daily_pnl        = Column(Float, default=0.0)
    daily_pnl_pct    = Column(Float, default=0.0)
    drawdown         = Column(Float, default=0.0)
    n_positions      = Column(Integer, default=0)
    __table_args__ = (
        Index("ix_portfolio_ts", "timestamp"),
    )


# ─── Engine & Session Factory ─────────────────────────────────────────────────

def create_db_engine():
    db_url = settings.get("database", {}).get("url", "sqlite:///./alphagrid.db")
    echo   = settings.get("database", {}).get("echo", False)

    if db_url.startswith("sqlite"):
        engine = create_engine(
            db_url, echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(db_url, echo=echo, pool_pre_ping=True)

    Base.metadata.create_all(engine)
    return engine


_engine = create_db_engine()
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_session() -> Session:
    """Context-manager-safe session factory."""
    return SessionLocal()
