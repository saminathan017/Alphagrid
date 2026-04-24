# Changelog

All notable changes to AlphaGrid are documented here.

---

## [v7.1] — 2026-03-23

### Added
- **Live demo** — one-click demo access on the login page (since removed in later hardening work)
- **Demo account** — a seeded demo user account was added temporarily during that release cycle
- **⚡ FIRE button** — every signal card now executes a real paper trade with qty input
- **Overview panels** — Holdings (positions + allocation %), Top Movers (live gainers/losers), Top Signals (top 12 by confidence)
- **Drawdown curve, Risk Limits, Position Exposure** moved from Risk page into Overview
- **Symbol autocomplete** on Chart page — alphabetical, keyboard navigation, prefix highlight
- **Cumulative P&L per Trade** bar chart on Trades page (replaces duplicate equity curve)
- **Backtest results** — +56% total return, 17.3% CAGR, 0.56 Sharpe over 2022–2024
- **Dockerfile** (python:3.11-slim, no GPU) for lightweight serving
- **deploy/docker-compose.yml** — one-command local deployment with persistent volumes
- **deploy/railway.json** — Railway deployment config (auto-detects Dockerfile, healthcheck)
- **deploy/render.yaml** — Render web service config with persistent disk
- **deploy/Procfile** — Heroku / generic PaaS fallback
- **.dockerignore** — excludes venv/, local DB, .env from Docker image

### Changed
- Overview: removed System Log panel; removed Live Signals mini panel (redundant with Signals page)
- Signals page: removed Ensemble Score and Strategy Distribution charts; raised limit to 200
- Chart page: fixed interval selector (Daily/1H/15M/5M) not applying on change; fixed intraday data fetching to bypass parquet cache and call yfinance directly
- Risk page removed from sidebar — content merged into Overview
- Signal caps removed from WebSocket tick and snapshot broadcasts
- `app.py` default signal limit raised from 20 to 500
- Login page: removed exposed default credentials panel (security fix)
- Login page: replaced "Multi-broker execution — Alpaca, OANDA, Robinhood" with accurate "Paper trading" and "150 assets" bullet points
- README completely rewritten with updated product overview, tech stack, and deploy guides

### Fixed
- `bcrypt>=4.0` incompatibility with `passlib` — pinned `bcrypt<4.0` in Dockerfile; broadened exception catch from `ImportError` to `Exception`
- Local `alphagrid_auth.db` was being baked into Docker image — `.dockerignore` now excludes it; server seeds fresh accounts on every clean deploy
- Demo password length issue fixed during that release cycle
- Chart endpoint returning HTTP 404 on Railway (no parquet cache) — added live yfinance fallback
- Price feed loop loading 0 symbols on Railway — added batch yfinance fallback per symbol
- Cold start time reduced from 5–10 minutes to ~15 seconds — priority 18 symbols now fetched in one `yf.download()` batch call on startup; rest load in background

### Removed
- Risk page from sidebar navigation (content kept, moved to Overview)
- Ensemble Score chart from Signals page
- Strategy Distribution chart from Signals page
- System Log panel from Overview
- Live Signals mini panel from Overview
- Dockerfile.train renamed from Dockerfile (old GPU training image preserved as deploy/Dockerfile.train)

---

## [v7.0] — 2026

### Added
- LICENSE file (MIT)
- docs/CHANGELOG.md

### Changed
- Bumped project version to v7 across all source files and documentation
- Removed hardcoded local filesystem paths from documentation
- `docs/SETUP.md` folder references updated from `alphagrid-v6` to `alphagrid`

### Removed
- History page from the dashboard

---

## [v6] — 2026

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

## [v5] — 2026

### Added
- Regime-conditional LightGBM with volatility state routing
- MetaLearner stacking over base model out-of-fold predictions
- Triple-barrier labelling replacing simple next-bar return labels
- 7-gate signal filter replacing single confidence threshold

---

## [v4] — 2026

### Added
- Alpaca broker integration (paper + live)
- Paper trading simulator
- Fractional Kelly position sizing with 3-tier take-profit cascade
- Portfolio-level risk constraints (sector concentration, gross exposure)

---

## [v3] — 2026

### Added
- QuantLSTM initial architecture
- FinancialTransformer initial architecture
- FastAPI dashboard with REST endpoints and WebSocket
- yfinance data pipeline with SQLite caching
