# Changelog

All notable changes to AlphaGrid are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v7] — 2024

### Added
- LICENSE file (MIT)
- CHANGELOG.md

### Changed
- Bumped project version to v7 across all source files and documentation
- Dockerfile now installs dependencies via `pip install -r requirements.txt` (was
  duplicating a partial package list inline — this ensures the Docker image gets
  the full dependency set including FastAPI, SQLAlchemy, auth libraries, and all
  async networking packages)
- Removed hardcoded local filesystem paths from `CLOUD_SETUP.md` (was referencing
  developer's home directory; replaced with `/path/to/alphagrid`)
- `SETUP.md` folder references updated from `alphagrid-v6` to `alphagrid`

### Removed
- History page from the dashboard (the static historical data browser added visual
  noise without contributing to signal generation or live trading decisions; all
  relevant data is surfaced through the Signals, Universe, and Chart pages)

---

## [v6] — 2024

### Added
- QuantLSTM: TCN front-end + BiLSTM + Multi-Head Attention + SWA + 8-pass TTA
- FinancialTransformer: 6-layer Pre-LN encoder with Rotary Positional Encoding
- LightGBM DART: regime-conditional (3 separate models per volatility regime)
- MetaLearner: AUC-weighted stacking with degeneracy detection
- 7-Gate Signal Filter: confidence, regime, alpha, risk/reward, freshness, portfolio, liquidity
- 80+ quantitative features across 10 families including Fourier spectral and fractal entropy
- JWT authentication with role-based access (Admin, Builder, Trader)
- Paper trading simulator with real P&L tracking
- WebSocket feed for live price and signal streaming
- AWS cloud bootstrap script for EC2 g4dn.xlarge training

---

## [v5] — 2024

### Added
- Regime-conditional LightGBM with volatility state routing
- MetaLearner stacking over base model out-of-fold predictions
- Triple-barrier labeling replacing simple next-bar return labels
- 7-gate signal filter replacing single confidence threshold

---

## [v4] — 2024

### Added
- Broker integration: Alpaca, OANDA, Robinhood, paper trader
- Fractional Kelly position sizing with 3-tier take-profit cascade
- Portfolio-level risk constraints (sector concentration, gross exposure)

---

## [v3] — 2024

### Added
- QuantLSTM initial architecture
- FinancialTransformer initial architecture
- FastAPI dashboard with REST endpoints and WebSocket
- yfinance data pipeline with SQLite caching
