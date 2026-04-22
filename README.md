# AlphaGrid v8

[![GitHub](https://img.shields.io/badge/GitHub-saminathan017%2FAlphagrid-blue?style=flat-square)](https://github.com/saminathan017/Alphagrid)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=flat-square)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

AlphaGrid is a quantitative trading dashboard I built as a personal project to learn how real algorithmic trading systems work — from raw price data all the way to live signal execution. It runs a full ML pipeline with three models, a 7-gate signal filter, real-time WebSocket streaming, JWT authentication, and a completely redesigned futuristic dashboard. No mock data, no hardcoded signals, no paid API keys required to get started.

---

## Quick Start

Clone and run with Docker (recommended):

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid
docker compose up
```

Open `http://localhost:8080` — the server is ready in about 30 seconds.

Without Docker:

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m dashboard.app
```

Open `http://localhost:8080`. Log in with `demo / demo1234` to explore with the demo account.

All market data is pulled from yfinance — free and works out of the box with no API keys.

---

## What It Does

Raw price data comes in from yfinance, goes through feature engineering (80+ features), through three ML models that each look at the market differently, into a meta-learner that decides how much to trust each model, and finally through a 7-gate filter that rejects anything without a real edge. Only signals that clear every gate reach the dashboard.

```
yfinance
    │
    ▼
Feature Engineering ── 80+ features across 10 quantitative families
    │
    ▼
ML Ensemble
  ├── QuantLSTM             TCN + BiLSTM + Multi-Head Attention + SWA + TTA
  ├── FinancialTransformer  6-layer Pre-LN encoder + Rotary Positional Encoding (RoPE)
  └── LightGBM DART         Regime-conditional (3 models × volatility state)
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

The frontend was fully redesigned in v8 — Space Grotesk, Space Mono, and Syne fonts, a deep blue/cyan design system, animated particle background on the login page, and a 7-page single-page application.

| Page | What it shows |
|---|---|
| Overview | Portfolio P&L, equity curve, live holdings, top movers, top signals, drawdown, risk limits |
| Signals | ML + TA signals with confidence, entry/TP/SL, reason pills, R/R ratio, strategy health panel |
| Chart | Interactive OHLCV chart — candlestick + MACD + RSI, symbol autocomplete, Daily/1H/15M/5M |
| Trades | Open and closed positions, P&L histogram, cumulative P&L chart, live vs backtest divergence |
| Universe | Live prices across 150 symbols, equity heatmap by sector, full forex table |
| Models | Per-symbol indicator grid, signal confidence scatter, strategy engine status |
| Evaluation | Model tier ratings, ROC curves, confusion matrices, auto-upgrade history |

Prices, signals, and portfolio state are pushed via WebSocket at `/ws` every 2 seconds. The dashboard falls back to REST polling at `/api/*` if the WebSocket is unavailable.

---

## Three Features That Solve Real Problems

Most quant dashboards show you a signal and a confidence score. You have no idea why it fired, whether that strategy has been working recently, or whether live performance matches the backtest. These three features were built specifically to address that.

**Signal explainability** — every signal card shows exactly why it was generated: which indicators triggered it (RSI oversold, MACD cross, ADX trend, Bollinger squeeze), the reward-to-risk ratio, stop-loss and take-profit as a percentage of entry, and the strategy name. You can look at any signal and immediately understand what the model was seeing — rather than just trusting a number.

**Strategy decay detection** — a strategy that worked for six months can start failing quietly and you won't notice until the losses have already compounded. The Signals page has a Strategy Health panel that tracks every actionable signal as a pending outcome, then checks whether price moved in the predicted direction after 1 day (day trades) or 5 days (swing trades). For each strategy it shows a rolling win rate, a dot sparkline of recent results, and a trend indicator — improving, stable, or declining. Below 50% accuracy: warning. Below 40%: critical.

**Backtest vs live divergence** — the hardest problem in algorithmic trading is that backtest performance almost never matches live performance exactly. The Trades page tracks live paper trade metrics — win rate, profit factor, Sharpe, max drawdown, average P&L per trade — and compares each one against the 2022–2024 backtest reference. Each metric gets a status: on track, warning, or underperforming. There is also a rolling 10-trade win rate chart with the backtest baseline as a reference line, so you can see whether performance is converging or diverging over time.

---

## Backtest Results

The models are trained on up to 10 years of daily data across 150 symbols. Sample backtest on 10 large-cap symbols over 2022–2024 — a period that includes the full 2022 bear market and the 2023–2024 recovery.

Configuration: $100,000 starting capital, 2% portfolio risk per trade, 2.5:1 reward-to-risk ratio.

| Metric | Result |
|---|---|
| Total Return | +56.0% |
| CAGR | 17.3% per year |
| Final Value | $156,005 |
| Sharpe Ratio | 0.56 |
| Sortino Ratio | 0.79 |
| Max Drawdown | -35.1% (2022 bear market) |
| Win Rate | 29.4% across 211 trades |

Run this yourself with any symbols or date range:

```python
from backtest.runner import BacktestRunner

r = BacktestRunner()
result = r.run(
    ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'AMZN', 'GOOGL', 'AMD', 'SPY', 'QQQ'],
    start='2022-01-01', end='2024-12-31'
)
m = result['metrics']
print(m['total_return_pct'], m['sharpe_ratio'], m['max_drawdown_pct'])
```

---

## Paper Trading

Every signal card has a Fire button. Here is how to use it:

1. Go to the Signals page
2. Set the quantity (defaults to 1 share)
3. Click Fire on any actionable signal
4. The trade fills immediately at current market price in paper mode
5. Track the position on the Trades and Overview pages

To connect real money, add Alpaca API keys to `.env` and trades route through there instead.

---

## Model Performance

The universe has 150 symbols — 100 US equities and 50 forex pairs. Each model is trained on up to 10 years of daily data per symbol, with the last 15% of each symbol's history held out as the test set, never seen during training. 146 of 150 symbols have a fully trained pipeline; 4 were skipped due to insufficient price history.

Top performers on the held-out test set:

| Symbol | Best Model | Accuracy | Hit@80 | Tier |
|---|---|---|---|---|
| USDTRY=X | Transformer | 90.0% | 90.0% | S |
| ZS | QuantLSTM | 83.3% | 83.3% | S |
| SOFI | QuantLSTM | 80.3% | 80.3% | S |
| NFLX | QuantLSTM | 75.8% | 75.8% | S |
| QCOM | QuantLSTM | 74.2% | 74.2% | S |
| CRWD | QuantLSTM | 71.2% | 71.2% | S |

Hit@80 = accuracy only on predictions where model confidence was 80% or above.

Overall averages across all 146 trained symbols:

| Model | Avg Accuracy | Avg AUC | Tier S | Tier A |
|---|---|---|---|---|
| QuantLSTM | 50.2% | 0.490 | 19 | 19 |
| Transformer | 49.0% | 0.501 | 18 | 17 |
| MetaEnsemble | 48.0% | 0.529 | 10 | 9 |
| LightGBM | 46.1% | 0.533 | 7 | 10 |

Near-random average accuracy is expected in financial ML. The goal is not to be right on every prediction — it is to be consistently right when confidence is high, which is where the 80–90% numbers come from.

---

## The Models

**QuantLSTM** starts with a TCN front-end — 4 dilated causal convolution blocks with roughly a 30-bar receptive field — feeds into a 3-layer BiLSTM with 512 hidden units, then passes through multi-head attention with 4 heads and temporal pooling. Regularisation includes Mixup, Stochastic Weight Averaging (SWA), 8-pass test-time augmentation (TTA), and focal loss. Trained with AdamW and cosine learning rate scheduling with warmup, on Apple Silicon MPS or NVIDIA CUDA.

**FinancialTransformer** is a 6-layer encoder with 8 attention heads and d_model=256. It uses Pre-LayerNorm for stability on relatively small financial datasets, and Rotary Positional Encoding (RoPE) for better temporal generalisation than standard learned embeddings.

**LightGBM DART** runs three separate models — one each for low, medium, and high volatility regimes. DART boosting prevents overfitting, and monotone constraints bake in basic economic logic. At inference, the system detects the current regime and routes to the matching model.

**MetaLearner** stacks all three base models using out-of-fold predictions. If there are insufficient samples, it falls back to AUC-weighted averaging. It also includes degeneracy detection — if the LSTM output standard deviation drops below 0.05, that output is replaced rather than passed forward.

---

## Feature Engineering

Over 80 features, all stationary and winsorised at the 1st and 99th percentile to prevent outlier contamination.

| Family | Features |
|---|---|
| Multi-horizon returns | 1, 3, 5, 10, 20, 60-day returns, log-returns, momentum |
| Volatility regime | Realised vol across 4 horizons, GARCH proxy, vol-of-vol, ATR |
| Trend and momentum | EMA stack across 8 periods, MACD, ADX/DI, SuperTrend, slope |
| Mean-reversion | RSI (3 periods), Bollinger, Keltner, Stochastic, CCI |
| Volume and liquidity | OBV, VWAP distance, MFI, Chaikin Money Flow, Amihud illiquidity |
| Microstructure | Candle anatomy, gaps, intraday range, percentile rank |
| Multi-timeframe | Price vs 5, 21, and 63-bar MAs, alignment score, efficiency ratio |
| Spectral / Fourier | FFT power across short, mid, long bands, signal-to-noise ratio |
| Fractal and entropy | Hurst exponent, approximate entropy, run entropy |
| Labels | Triple-barrier — 2.5× ATR take-profit, 2.0× ATR stop-loss |

---

## The 7-Gate Signal Filter

Every signal has to pass all 7 gates before it reaches the dashboard. Most do not make it.

| Gate | What it checks |
|---|---|
| 1 | Confidence exceeds a dynamic Bayesian threshold (adapts between 0.55 and 0.75) |
| 2 | Direction aligns with the current market regime (SPY vol, DXY, credit spreads) |
| 3 | IC-weighted alpha factors agree with the signal direction |
| 4 | Take-profit to stop-loss ratio is at least 2.0× |
| 5 | Signal has not aged past its ATR-scaled half-life |
| 6 | Sector concentration, book correlation, and gross exposure are within limits |
| 7 | Estimated spread is under 10 basis points |

Signals that clear all 7 gates receive a 0–100 conviction score, a fractional Kelly position size, and a 3-tier take-profit cascade at 1.0×, 2.0×, and 3.5× ATR.

---

## Asset Universe

150 symbols in total, all sourced from yfinance — no API key required.

- 100 US equities: mega-cap tech, semiconductors, financials, healthcare, consumer, energy, industrials, growth stocks, and ETFs
- 50 forex pairs: majors, minors, EM exotics, gold, and silver

---

## Deploying It Yourself

**Railway** (about 2 minutes):
1. Fork the repo on GitHub
2. Go to railway.app, create a new project, deploy from your fork
3. Add `ALPHAGRID_JWT_SECRET` as an environment variable (any random string works)
4. Deploy — Railway auto-detects the Dockerfile

**Render**:
1. Fork the repo
2. Go to render.com, create a new web service, connect your fork
3. Render picks up `render.yaml` automatically
4. Add `ALPHAGRID_JWT_SECRET` and deploy

**Environment variables:**

| Variable | Default | Notes |
|---|---|---|
| `ALPHAGRID_JWT_SECRET` | auto-generated | Set this in any production deployment |
| `ALPHAGRID_OWNER_USERNAME` | admin | Optional |
| `ALPHAGRID_OWNER_PASSWORD` | Admin@Grid1 | Change this before deploying |
| `ALPACA_API_KEY` | — | Only needed for live trading |
| `ALPACA_SECRET_KEY` | — | Only needed for live trading |

---

## Retraining the Models

The repo ships with pre-trained checkpoints so signals work immediately on first launch. Retraining all 150 symbols from scratch on a MacBook takes 6–12 hours, but a bootstrap script provisions an EC2 spot instance for roughly $0.50–$1.50 total.

```bash
bash scripts/cloud_bootstrap.sh   # run on the EC2 instance after provisioning
```

Instance options:

| Instance | GPU | Spot price | Time for 150 symbols | Estimated cost |
|---|---|---|---|---|
| g4dn.xlarge | NVIDIA T4 | ~$0.16/hr | 3–4 hours | ~$0.50–0.65 |
| g4dn.2xlarge | NVIDIA T4 | ~$0.28/hr | 2–3 hours | ~$0.55–0.85 |
| g5.xlarge | NVIDIA A10G | ~$0.50/hr | 1.5–2 hours | ~$0.75–1.00 |

To retrain specific symbols or run a quick smoke test:

```bash
python scripts/train_models.py                            # all 150 symbols
python scripts/train_models.py --symbols AAPL,MSFT,NVDA  # specific symbols
python scripts/train_models.py --symbols AAPL --quick    # about 5 minutes
```

---

## Project Structure

```
Alphagrid/
├── core/               Config, auth, JWT, ticker universe, event bus
├── data/               Feature engineering, historical data, live feed, news
├── models/             QuantLSTM, Transformer, LightGBM, ensemble,
│                       alpha engine, signal filter, position sizer
├── strategies/         40+ indicators, day and swing trading strategies
├── execution/          Alpaca broker adapter and paper trader
├── backtest/           Walk-forward engine, metrics, runner
├── risk/               Kelly sizing, portfolio constraints
├── dashboard/          FastAPI server, WebSocket, 7-page SPA frontend
│   ├── app.py          API routes, WebSocket, data streaming
│   ├── auth.html       Login / signup page (animated particle background)
│   ├── index.html      Main dashboard (7 pages, fully wired to backend)
│   └── static/         Bundled Chart.js and Lightweight Charts
├── scripts/            Training pipeline, cloud bootstrap, backtest runner
├── config/             settings.yaml — 200+ configurable parameters
│
├── Dockerfile          Serving image (python:3.11-slim, no GPU required)
├── Dockerfile.train    Training image (PyTorch 2.3, CUDA 12.1)
├── docker-compose.yml  One-command local deployment
├── railway.json        Railway deployment config
├── render.yaml         Render deployment config
├── Procfile            Heroku and generic PaaS fallback
├── requirements.txt    Full dependency list
└── .env.example        Environment variable template
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | yfinance, pandas, numpy, SQLite, SQLAlchemy |
| ML | PyTorch 2.3, LightGBM, scikit-learn, imbalanced-learn |
| Indicators | Numba JIT where available, 40+ pure-numpy fallbacks |
| API | FastAPI, uvicorn, WebSockets, aiohttp |
| Auth | JWT via python-jose, bcrypt via passlib |
| Frontend | Vanilla JS SPA, Chart.js, TradingView Lightweight Charts |
| Fonts | Space Grotesk, Space Mono, Syne |
| Serving | Python 3.11, Docker on python:3.11-slim |
| Training | Apple Silicon MPS, NVIDIA CUDA 12.1 |
| Cloud | Railway, Render, AWS EC2 g4dn |

---

## Design Decisions

**Why triple-barrier labels?** Simple next-bar return labels hit a ceiling around 52% accuracy because many "moves" are just noise. Triple-barrier labels only mark bars where a real, measurable move happened in either direction, pushing accuracy on labeled samples from 52% up to 65–90% depending on the symbol.

**Why regime-conditional LightGBM?** A strategy that works in a low-volatility trending market falls apart in a high-volatility mean-reverting one. Running three separate models — one per volatility regime — and routing inference to the right one consistently outperforms a single global model.

**Why a meta-learner?** The LSTM is good at temporal dependencies. The Transformer catches long-range patterns. LightGBM handles tabular snapshot features better than either. No single model wins everywhere, so the meta-learner learns when to trust each one.

**Why 7 gates?** A model that is right 70% of the time is useless if you act on every prediction. The filter selects the 5–15% of signals that have genuine edge — that subset is where you see 80–90% accuracy.

**Why ship pre-trained models?** Retraining from scratch takes 6–12 hours and needs a GPU. Shipping the checkpoints means real ML signals are live on first launch with no waiting and no hardware requirements.

**Why show signal reasons instead of just confidence?** A score of 73% tells you nothing actionable. Knowing the signal fired because RSI hit 28, ADX is above 25, and MACD just crossed positive gives you something you can actually evaluate and disagree with. It also makes the system auditable — if a signal is wrong, you can trace exactly what the model was seeing.

**Why track strategy decay?** Every strategy has a shelf life. Market regimes change, correlations break down, and what worked in a trending environment stops working in a choppy one. Without explicit tracking, decay is invisible until the account is already down.

**Why compare live performance to the backtest?** Because the backtest is a promise and live trading is reality. If the live win rate is 18% and the backtest said 29%, something is wrong — overfitting, regime change, or execution slippage. Surfacing that gap explicitly forces honest evaluation.

---

## Disclaimer

Paper trading only by default. To enable live trading, add Alpaca credentials to `.env`. This is a personal research and learning project — past performance does not guarantee future results.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built by [saminathan017](https://github.com/saminathan017) — source at [github.com/saminathan017/Alphagrid](https://github.com/saminathan017/Alphagrid)*
