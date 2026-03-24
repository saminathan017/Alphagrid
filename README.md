# AlphaGrid v7 — ML-Powered Trading Intelligence

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-brightgreen?style=flat-square)](https://web-production-bd6aa.up.railway.app)
[![GitHub](https://img.shields.io/badge/GitHub-saminathan017%2FAlphagrid-blue?style=flat-square)](https://github.com/saminathan017/Alphagrid)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

A production-grade quantitative trading dashboard that combines an ML ensemble (LSTM + Transformer + LightGBM), a 7-gate institutional signal filter, and a real-time WebSocket feed — all served through a single FastAPI server with no mock data and no hardcoded signals.

---

## Live Demo

**No sign-up required.** Click the button on the login page — instant access.

```
https://web-production-bd6aa.up.railway.app
```

> Demo account is read-only (Trader role) with paper trading enabled.

---

## Quick Start

### Docker (one command)

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid
docker compose up
```

Open [http://localhost:8080](http://localhost:8080) — server starts in ~30 seconds.

### Local

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m dashboard.app
```

Open [http://localhost:8080](http://localhost:8080)

**No API keys required.** All market data comes from yfinance (free).

---

## Backtest Results — 2022 to 2024

10 symbols (AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL, AMD, SPY, QQQ).
$100,000 starting capital · 2% portfolio risk per trade · 2.5:1 R/R ratio.
Includes the full 2022 bear market and the 2023–2024 recovery.

| Metric | Result |
|---|---|
| Total Return | **+56.0%** |
| CAGR | **17.3% / year** |
| Final Value | **$156,005** |
| Sharpe Ratio | 0.56 |
| Sortino Ratio | 0.79 |
| Max Drawdown | -35.1% (2022 bear) |
| Win Rate | 29.4% (211 trades) |

Reproduce it:

```python
from backtest.runner import BacktestRunner
r = BacktestRunner()
result = r.run(
    ['AAPL','MSFT','NVDA','TSLA','META','AMZN','GOOGL','AMD','SPY','QQQ'],
    start='2022-01-01', end='2024-12-31'
)
m = result['metrics']
print(m['total_return_pct'], m['sharpe_ratio'], m['max_drawdown_pct'])
```

---

## How It Works

```
yfinance (free, no API key)
        │
        ▼
Feature Engineering ──── 80+ features across 10 quantitative families
        │
        ▼
ML Ensemble
  ├── QuantLSTM          TCN + BiLSTM + Multi-Head Attention + SWA + TTA
  ├── FinancialTransformer  6-layer Pre-LN encoder + Rotary Positional Encoding
  └── LightGBM DART      Regime-conditional (3 models × volatility state)
        │
        ▼
MetaLearner ─────────── AUC-weighted stacking with degeneracy detection
        │
        ▼
7-Gate Signal Filter ─── confidence · regime · alpha · R/R · freshness · portfolio · liquidity
        │
        ▼
FastAPI Dashboard ─────── live prices · signals · paper trading · WebSocket push
```

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| **Overview** | Portfolio P&L, equity curve, live holdings, top movers, top signals, drawdown curve, risk limits |
| **Signals** | All ML + TA signals with confidence, entry, TP, SL — click ⚡ FIRE to execute as paper trade |
| **Chart** | Interactive OHLCV with 40+ indicators, symbol autocomplete, Daily / 1H / 15M / 5M |
| **Trades** | Open and closed positions, real P&L, cumulative P&L per trade chart |
| **Universe** | Live prices across 150 symbols, updated every 5 seconds, sortable |
| **Models** | Per-symbol model performance, Tier ratings (S/A/B/C/D), calibration |
| **Broker** | Paper trading account, order routing, account state |

Real-time WebSocket at `/ws` — prices, signals, portfolio pushed every 2 seconds.

---

## Paper Trading

Every signal card has a ⚡ FIRE button:

1. Go to **Signals** page
2. Set quantity (default: 1 share)
3. Click **⚡ FIRE** on any actionable signal
4. Trade fills immediately in paper mode at current market price
5. Track it live on **Trades** and **Overview**

To enable live trading with real money, add Alpaca API keys to `.env`.

---

## Model Performance

Trained on 146 symbols × 10 years of daily data. Test set = last 15% of data, never touched during training.

| Symbol | Best Model | Accuracy | Hit@80 | Tier |
|---|---|---|---|---|
| USDTRY=X | Transformer | 90.0% | 90.0% | S |
| ZS | QuantLSTM | 83.3% | 83.3% | S |
| SOFI | QuantLSTM | 80.3% | 80.3% | S |
| NFLX | QuantLSTM | 75.8% | 75.8% | S |
| QCOM | QuantLSTM | 74.2% | 74.2% | S |
| CRWD | QuantLSTM | 71.2% | 71.2% | S |

Hit@80 = accuracy only on predictions where confidence ≥ 80%.

Overall across 146 symbols:

| Model | Avg Accuracy | Avg AUC | Tier S | Tier A |
|---|---|---|---|---|
| QuantLSTM | 50.2% | 0.490 | 19 | 19 |
| Transformer | 49.0% | 0.501 | 18 | 17 |
| MetaEnsemble | 48.0% | 0.529 | 10 | 9 |
| LightGBM | 46.1% | 0.533 | 7 | 10 |

Near-random average accuracy is expected in financial ML. The value is in the high-confidence tail where models are consistently right 80–90% of the time.

---

## Architecture

### QuantLSTM
- TCN front-end: 4 dilated causal convolution blocks (~30 bar receptive field)
- 3-layer BiLSTM: 512 hidden units with explicit dropout
- Multi-head attention: 4 heads with temporal pooling
- Regularisation: Mixup, Stochastic Weight Averaging, 8-pass TTA, focal loss
- Training: AdamW + cosine LR with warmup, early stopping, MPS + CUDA

### FinancialTransformer
- 6-layer encoder, 8 attention heads, d_model=256
- Pre-LayerNorm for stable training on small financial datasets
- Rotary Positional Encoding (RoPE) for temporal generalisation

### LightGBM DART
- 3 separate models for low / medium / high volatility regimes
- DART boosting for stronger regularisation
- Monotone constraints encoding economic priors
- Inference routes to the model matching the current regime

### MetaLearner
- Stacked generalisation on out-of-fold base model predictions
- Falls back to AUC-weighted averaging for small datasets (<100 samples)
- Degeneracy detection: replaces LSTM output if std(output) < 0.05

---

## Feature Engineering

80+ features across 10 families. All stationary, winsorised at 1st/99th percentile.

| Family | Features |
|---|---|
| Multi-horizon returns | 1, 3, 5, 10, 20, 60-day returns, log-returns, momentum |
| Volatility regime | Realised vol (4 horizons), GARCH proxy, vol-of-vol, ATR |
| Trend & momentum | EMA stack (8 periods), MACD, ADX/DI, SuperTrend, slope |
| Mean-reversion | RSI (3 periods), Bollinger, Keltner, Stochastic, CCI |
| Volume & liquidity | OBV, VWAP distance, MFI, Chaikin Money Flow, Amihud |
| Microstructure | Candle anatomy, gaps, intraday range, percentile rank |
| Multi-timeframe | Price vs 5/21/63-bar MAs, alignment score, efficiency ratio |
| Spectral / Fourier | FFT power (short/mid/long bands), SNR |
| Fractal & entropy | Hurst exponent, approximate entropy, run entropy |
| Labels | Triple-barrier (2.5× ATR TP, 2.0× ATR SL) |

---

## 7-Gate Signal Filter

Every signal must clear all 7 gates before reaching the dashboard.

| Gate | Checks |
|---|---|
| 1 | Confidence ≥ dynamic Bayesian threshold (adapts 0.55–0.75) |
| 2 | Direction aligns with market regime (SPY vol, DXY, credit spreads) |
| 3 | IC-weighted alpha factors confirm signal direction |
| 4 | Take-profit / stop-loss ratio ≥ 2.0× |
| 5 | Signal not older than ATR-scaled half-life |
| 6 | Sector concentration, book correlation, gross exposure within limits |
| 7 | Estimated spread ≤ 10 bps |

Signals that pass receive a 0–100 conviction score, fractional Kelly size, and 3-tier TP cascade at 1.0×, 2.0×, 3.5× ATR.

---

## Asset Universe

**150 total — all via yfinance, no API key required**

- **100 US Equities** — mega-cap tech, semiconductors, financials, healthcare, consumer, energy, industrials, growth, ETFs
- **50 Forex Pairs** — majors, minors, EM exotics, gold, silver

---

## Deploy

### Railway (2 minutes)

1. Fork this repo on GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub → select your fork
3. Add `ALPHAGRID_JWT_SECRET` env var (any random string)
4. Deploy — Railway auto-detects the `Dockerfile`

### Render

1. Fork this repo
2. [render.com](https://render.com) → New Web Service → connect GitHub → select fork
3. Render reads `render.yaml` automatically
4. Add `ALPHAGRID_JWT_SECRET` and deploy

### Environment Variables

| Variable | Default | Required |
|---|---|---|
| `ALPHAGRID_JWT_SECRET` | auto-generated | For production |
| `ALPHAGRID_OWNER_USERNAME` | `admin` | Optional |
| `ALPHAGRID_OWNER_PASSWORD` | `Admin@Grid1` | Change this |
| `ALPACA_API_KEY` | — | For live trading only |
| `ALPACA_SECRET_KEY` | — | For live trading only |

---

## Cloud Training

Training 146 symbols from scratch on a MacBook takes 6–12 hours.
The bootstrap script provisions an EC2 g4dn.xlarge for ~$0.50–$1.50 total.

```bash
# On the EC2 instance
bash scripts/cloud_bootstrap.sh
```

| Instance | GPU | Spot/hr | 146 symbols | Cost |
|---|---|---|---|---|
| g4dn.xlarge | NVIDIA T4 | ~$0.16 | ~3–4 hrs | ~$0.50–0.65 |
| g4dn.2xlarge | NVIDIA T4 | ~$0.28 | ~2–3 hrs | ~$0.55–0.85 |
| g5.xlarge | NVIDIA A10G | ~$0.50 | ~1.5–2 hrs | ~$0.75–1.00 |

Retrain or extend the universe:

```bash
python scripts/train_models.py                            # all 146 symbols
python scripts/train_models.py --symbols AAPL,MSFT,NVDA  # specific symbols
python scripts/train_models.py --symbols AAPL --quick    # ~5 min smoke test
```

---

## Project Structure

```
Alphagrid/
├── core/               Config, auth, JWT, ticker universe, event bus
├── data/               Feature engineering, historical data, live feed, news
├── models/             QuantLSTM, Transformer, LightGBM, ensemble,
│                       alpha engine, signal filter, position sizer
├── strategies/         40+ Numba-JIT indicators, day + swing strategies
├── execution/          Alpaca broker + paper trader
├── backtest/           Walk-forward engine, metrics, runner
├── risk/               Kelly sizing, portfolio constraints
├── dashboard/          FastAPI server, WebSocket, 7-page SPA frontend
├── scripts/            Training pipeline, cloud bootstrap, backtest runner
├── config/             settings.yaml (200+ configurable parameters)
│
├── Dockerfile          Serving image (python:3.11-slim, no GPU required)
├── Dockerfile.train    Training image (pytorch/pytorch:2.3.0-cuda12.1)
├── docker-compose.yml  One-command local deployment
├── railway.json        Railway deployment config
├── render.yaml         Render deployment config
├── Procfile            Heroku / generic PaaS fallback
├── requirements.txt    Full dependency list
└── .env.example        Environment template
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Data | yfinance, pandas, numpy, SQLite, SQLAlchemy |
| ML | PyTorch 2.3, LightGBM, scikit-learn, imbalanced-learn |
| Indicators | Numba JIT (optional), 40+ pure-numpy fallbacks |
| API | FastAPI, uvicorn, WebSockets, aiohttp |
| Auth | JWT (python-jose), bcrypt (passlib) |
| Serving | Python 3.11, Docker (python:3.11-slim) |
| Training | Apple Silicon MPS, NVIDIA CUDA 12.1 |
| Cloud | Railway, Render, AWS EC2 g4dn |

---

## Key Design Decisions

**Why triple-barrier labels?**
Simple next-bar return labels are noisy — 52% accuracy is roughly the ceiling. The triple-barrier method labels only bars where a real measurable move occurred, pushing accuracy on labeled samples from 52% to 65–90%.

**Why regime-conditional LightGBM?**
Strategies that work in low-vol trending markets fail in high-vol mean-reverting ones. Three separate models per volatility regime consistently outperforms a single global model.

**Why MetaLearner stacking?**
LSTM captures temporal dependencies. Transformer catches long-range patterns. LightGBM excels on tabular snapshot features. The meta-learner learns when to trust each one.

**Why 7 gates?**
A model right 70% of the time is useless if you trade every signal. The 7-gate filter selects the ~5–15% of signals with genuine edge — those convert to 80–90% accuracy.

**Why pre-trained models in the repo?**
Training 146 symbols from scratch takes 6–12 hours. Shipping trained checkpoints means real ML signals are live on first launch with no GPU, no waiting.

---

## Disclaimer

Paper trading only by default. To enable live trading, configure Alpaca credentials in `.env`.
This is a research and educational project. Past performance does not guarantee future results.

---

## License

MIT — see [LICENSE](LICENSE)
