# AlphaGrid v7

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-brightgreen?style=flat-square)](https://web-production-bd6aa.up.railway.app)
[![GitHub](https://img.shields.io/badge/GitHub-saminathan017%2FAlphagrid-blue?style=flat-square)](https://github.com/saminathan017/Alphagrid)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

AlphaGrid is a quantitative trading dashboard I built that runs a full ML pipeline — three models, a signal filter, and a live WebSocket feed — all from a single server with no mock data, no hardcoded signals, and no API keys needed to get started.

---

## Try it live

No sign-up, no setup. Just open the link and log in with the demo account.

```
https://web-production-bd6aa.up.railway.app
```

The demo account is read-only (Trader role) with paper trading enabled so you can fire signals and watch the P&L update in real time.

---

## Running it locally

If you have Docker, one command is all it takes:

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid
docker compose up
```

Then open http://localhost:8080. The server is ready in about 30 seconds.

If you prefer to run it without Docker:

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m dashboard.app
```

No API keys required. All market data comes from yfinance, which is free and works out of the box.

---

## Backtest results

The models are trained on up to 10 years of daily data across 150 symbols. To show how the strategy performs end-to-end, here is a sample backtest run on 10 large-cap symbols over 2022–2024 — a period that includes the full 2022 bear market and the 2023–2024 recovery, so it covers both a losing and a winning environment.

Configuration: $100,000 starting capital, 2% portfolio risk per trade, 2.5:1 reward-to-risk ratio.

| Metric | Result |
|---|---|
| Total Return | +56.0% |
| CAGR | 17.3% per year |
| Final Value | $156,005 |
| Sharpe Ratio | 0.56 |
| Sortino Ratio | 0.79 |
| Max Drawdown | -35.1% (2022 bear) |
| Win Rate | 29.4% across 211 trades |

You can run this yourself with any symbols or date range:

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

## How it works

Raw price data comes in from yfinance. From there, it goes through feature engineering (80+ features), into three models that each look at the market differently, then a meta-learner that decides how much to trust each model, and finally a 7-gate filter that rejects anything that doesn't have a real edge. What makes it onto the dashboard are only the signals that cleared every gate.

```
yfinance
    │
    ▼
Feature Engineering ── 80+ features across 10 quantitative families
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

## What's in the dashboard

| Page | What it shows |
|---|---|
| Overview | Portfolio P&L, equity curve, live holdings, top movers, top signals, drawdown curve, risk limits |
| Signals | ML and TA signals with confidence, entry, TP, SL, reason pills, R/R ratio, and strategy health panel |
| Chart | Interactive OHLCV chart with 40+ indicators, symbol search, Daily / 1H / 15M / 5M timeframes |
| Trades | Open and closed positions, P&L chart, and live vs backtest divergence tracker |
| Universe | Live prices across 150 symbols, updated every 5 seconds, fully sortable |
| Models | Per-symbol model performance, Tier ratings (S through D), calibration stats |
| Broker | Paper trading account, order routing, current account state |

Prices, signals, and portfolio state are all pushed via WebSocket at `/ws` every 2 seconds.

---

## Three features that address real problems in live algo trading

Most quant dashboards show you a signal and a confidence score and leave it there. You have no idea why the signal fired, whether that strategy has been working recently, or whether the live performance actually matches what you saw in the backtest. These three features were built specifically to fix that.

Signal explainability — every signal card now shows the exact reasons it was generated: which indicators triggered it (RSI oversold, MACD cross, ADX trend confirmed, Bollinger squeeze, etc.), the reward-to-risk ratio, the stop-loss and take-profit as a percentage of entry, and the strategy name. You can look at a signal and immediately understand what the market is doing and why the system thinks it has edge, rather than just trusting a number.

Strategy decay detection — this is the one most people miss. A strategy that worked for the past 6 months can start failing quietly and you won't notice until the losses pile up. The Signals page has a Strategy Health panel that tracks every actionable signal as a pending outcome, then checks whether price moved in the predicted direction after 1 day (for day trades) or 5 days (for swing trades). For each strategy it shows a rolling win rate, a dot sparkline of recent results, and a trend indicator — improving, stable, or declining. If a strategy's live accuracy drops below 50% it flags as warning, below 40% it goes critical. You see it happening in real time before it costs you.

Backtest vs live divergence — the hardest problem in algorithmic trading is that backtest performance almost never matches live performance exactly. Overfitting, lookahead bias, transaction costs, and changing market regimes all cause the gap. The Trades page has a comparison panel that tracks your live paper trade metrics — win rate, profit factor, Sharpe, max drawdown, average trade P&L — and compares each one against the backtest reference numbers from the 2022–2024 run. Each metric gets a status: on track, warning, or underperforming. There's also a rolling 10-trade win rate chart with the backtest baseline drawn as a reference line, so you can see whether performance is converging or diverging over time.

---

## Paper trading

Every signal card has a Fire button. Here's how to use it:

1. Go to the Signals page
2. Set the quantity (defaults to 1 share)
3. Click Fire on any signal you like
4. The trade fills immediately at current market price in paper mode
5. Track it on the Trades and Overview pages

If you want to connect real money, you can add Alpaca API keys to `.env` and it routes through there.

---

## Model performance

The universe has 150 symbols (100 US equities + 50 forex pairs). Each model is trained on up to 10 years of daily data per symbol, with the last 15% of each symbol's history held out as the test set — never seen during training. 146 of the 150 symbols have a fully trained pipeline; 4 were skipped due to insufficient price history.

Top performers on the held-out test set:

| Symbol | Best Model | Accuracy | Hit@80 | Tier |
|---|---|---|---|---|
| USDTRY=X | Transformer | 90.0% | 90.0% | S |
| ZS | QuantLSTM | 83.3% | 83.3% | S |
| SOFI | QuantLSTM | 80.3% | 80.3% | S |
| NFLX | QuantLSTM | 75.8% | 75.8% | S |
| QCOM | QuantLSTM | 74.2% | 74.2% | S |
| CRWD | QuantLSTM | 71.2% | 71.2% | S |

Hit@80 means accuracy only on predictions where the model's confidence was 80% or higher.

Overall averages across all 146 trained symbols:

| Model | Avg Accuracy | Avg AUC | Tier S symbols | Tier A symbols |
|---|---|---|---|---|
| QuantLSTM | 50.2% | 0.490 | 19 | 19 |
| Transformer | 49.0% | 0.501 | 18 | 17 |
| MetaEnsemble | 48.0% | 0.529 | 10 | 9 |
| LightGBM | 46.1% | 0.533 | 7 | 10 |

Near-random average accuracy is completely normal in financial ML. The point isn't to be right on every prediction — it's to be consistently right when confidence is high, which is where the 80–90% accuracy numbers come from.

---

## The models

QuantLSTM starts with a TCN front-end (4 dilated causal convolution blocks, roughly a 30-bar receptive field), feeds into a 3-layer BiLSTM with 512 hidden units, then passes through multi-head attention with 4 heads and temporal pooling. Regularisation includes Mixup, Stochastic Weight Averaging, 8-pass test-time augmentation, and focal loss. Trained with AdamW and cosine learning rate scheduling with warmup, on Apple Silicon MPS or NVIDIA CUDA.

FinancialTransformer is a 6-layer encoder with 8 attention heads and d_model=256. It uses Pre-LayerNorm so it stays stable on the relatively small financial datasets we're working with, and Rotary Positional Encoding (RoPE) for better temporal generalisation than standard learned embeddings.

LightGBM DART runs three separate models — one for low volatility regimes, one for medium, one for high. DART boosting keeps it from overfitting, and monotone constraints bake in some economic common sense. At inference time, the system looks at the current regime and routes to the matching model.

MetaLearner stacks the three base models using out-of-fold predictions. If there aren't enough samples, it falls back to AUC-weighted averaging. It also has degeneracy detection — if the LSTM output standard deviation drops below 0.05, that output gets replaced rather than passed forward.

---

## Feature engineering

Over 80 features, all stationary and winsorised at the 1st and 99th percentile to avoid outlier contamination.

| Family | What's included |
|---|---|
| Multi-horizon returns | 1, 3, 5, 10, 20, 60-day returns, log-returns, momentum |
| Volatility regime | Realised vol across 4 horizons, GARCH proxy, vol-of-vol, ATR |
| Trend and momentum | EMA stack across 8 periods, MACD, ADX/DI, SuperTrend, slope |
| Mean-reversion | RSI across 3 periods, Bollinger, Keltner, Stochastic, CCI |
| Volume and liquidity | OBV, VWAP distance, MFI, Chaikin Money Flow, Amihud illiquidity |
| Microstructure | Candle anatomy, gaps, intraday range, percentile rank |
| Multi-timeframe | Price vs 5, 21, and 63-bar MAs, alignment score, efficiency ratio |
| Spectral / Fourier | FFT power across short, mid, and long bands, signal-to-noise ratio |
| Fractal and entropy | Hurst exponent, approximate entropy, run entropy |
| Labels | Triple-barrier with 2.5x ATR take-profit and 2.0x ATR stop-loss |

---

## The 7-gate signal filter

Every signal has to pass all 7 gates before it reaches the dashboard. Most don't make it.

| Gate | What it checks |
|---|---|
| 1 | Confidence is above a dynamic Bayesian threshold that adapts between 0.55 and 0.75 |
| 2 | Direction aligns with the current market regime (SPY vol, DXY, credit spreads) |
| 3 | IC-weighted alpha factors agree with the signal direction |
| 4 | Take-profit to stop-loss ratio is at least 2.0x |
| 5 | Signal hasn't aged past its ATR-scaled half-life |
| 6 | Sector concentration, book correlation, and gross exposure are all within limits |
| 7 | Estimated spread is under 10 basis points |

Signals that clear all 7 gates get a 0–100 conviction score, a fractional Kelly position size, and a 3-tier take-profit cascade at 1.0x, 2.0x, and 3.5x ATR.

---

## Asset universe

150 symbols in total, all pulled from yfinance — no API key needed.

100 US equities covering mega-cap tech, semiconductors, financials, healthcare, consumer, energy, industrials, growth stocks, and ETFs.

50 forex pairs covering majors, minors, EM exotics, gold, and silver.

---

## Deploying it yourself

On Railway (takes about 2 minutes):
1. Fork the repo on GitHub
2. Go to railway.app, create a new project, and deploy from your fork
3. Add `ALPHAGRID_JWT_SECRET` as an environment variable (any random string works)
4. Deploy — Railway auto-detects the Dockerfile

On Render:
1. Fork the repo
2. Go to render.com, create a new web service, and connect your fork
3. Render picks up `render.yaml` automatically
4. Add `ALPHAGRID_JWT_SECRET` and deploy

Environment variables:

| Variable | Default | Notes |
|---|---|---|
| `ALPHAGRID_JWT_SECRET` | auto-generated | Set this for any production deployment |
| `ALPHAGRID_OWNER_USERNAME` | admin | Optional |
| `ALPHAGRID_OWNER_PASSWORD` | Admin@Grid1 | Change this |
| `ALPACA_API_KEY` | — | Only needed for live trading |
| `ALPACA_SECRET_KEY` | — | Only needed for live trading |

---

## Retraining the models

Training all 150 symbols from scratch on a MacBook takes 6–12 hours. The repo ships with pre-trained checkpoints so signals work on first launch, but if you want to retrain or add new symbols, a bootstrap script provisions an EC2 g4dn.xlarge spot instance for roughly $0.50–$1.50 total.

```bash
# On the EC2 instance after running the bootstrap script
bash scripts/cloud_bootstrap.sh
```

Instance options if you want to tune cost vs. speed:

| Instance | GPU | Spot price | Time for 150 symbols | Total cost |
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

## Project structure

```
Alphagrid/
├── core/               Config, auth, JWT, ticker universe, event bus
├── data/               Feature engineering, historical data, live feed, news
├── models/             QuantLSTM, Transformer, LightGBM, ensemble,
│                       alpha engine, signal filter, position sizer
├── strategies/         40+ indicators, day and swing strategies
├── execution/          Alpaca broker and paper trader
├── backtest/           Walk-forward engine, metrics, runner
├── risk/               Kelly sizing, portfolio constraints
├── dashboard/          FastAPI server, WebSocket, 7-page frontend
├── scripts/            Training pipeline, cloud bootstrap, backtest runner
├── config/             settings.yaml with 200+ configurable parameters
│
├── Dockerfile          Serving image (python:3.11-slim, no GPU needed)
├── Dockerfile.train    Training image (pytorch 2.3, CUDA 12.1)
├── docker-compose.yml  One-command local deployment
├── railway.json        Railway deployment config
├── render.yaml         Render deployment config
├── Procfile            Heroku and generic PaaS fallback
├── requirements.txt    Full dependency list
└── .env.example        Environment variable template
```

---

## Tech stack

| Layer | What I used |
|---|---|
| Data | yfinance, pandas, numpy, SQLite, SQLAlchemy |
| ML | PyTorch 2.3, LightGBM, scikit-learn, imbalanced-learn |
| Indicators | Numba JIT where available, 40+ pure-numpy fallbacks |
| API | FastAPI, uvicorn, WebSockets, aiohttp |
| Auth | JWT via python-jose, bcrypt via passlib |
| Serving | Python 3.11, Docker on python:3.11-slim |
| Training | Apple Silicon MPS, NVIDIA CUDA 12.1 |
| Cloud | Railway, Render, AWS EC2 g4dn |

---

## A few design decisions worth explaining

Why triple-barrier labels? Simple next-bar return labels hit a ceiling around 52% accuracy because the labels are too noisy — a lot of the "moves" are just noise. Triple-barrier labels only mark bars where a real, measurable move happened in either direction, which pushes accuracy on labeled samples from 52% up to 65–90% depending on the symbol.

Why regime-conditional LightGBM? A strategy that works well in a low-volatility trending market falls apart in a high-volatility mean-reverting one. Running three separate models — one per volatility regime — and routing inference to the right one consistently beats a single global model.

Why a meta-learner? The LSTM is good at temporal dependencies. The Transformer catches long-range patterns. LightGBM handles tabular snapshot features better than either. No single model wins everywhere, so the meta-learner learns when to trust each one.

Why 7 gates? A model that's right 70% of the time is useless if you act on every prediction. The filter selects the 5–15% of signals that have genuine edge — that subset is where you see 80–90% accuracy.

Why ship pre-trained models? Retraining from scratch takes 6–12 hours and needs a GPU. Shipping the checkpoints means real ML signals are live on the first launch with no waiting and no hardware requirements.

Why show signal reasons instead of just confidence? A confidence score of 73% tells you nothing actionable. Knowing the signal fired because RSI hit 28, ADX is above 25, and MACD just crossed positive gives you something you can actually evaluate and disagree with. It also makes the system auditable — if a signal is wrong, you can trace back exactly what the model was seeing.

Why track strategy decay? Every strategy has a shelf life. Market regimes change, correlations break down, and what worked in a trending environment stops working in a choppy one. Without explicit tracking, decay is invisible until your account is already down. Monitoring the rolling accuracy per strategy means you can rotate out of failing strategies before the damage compounds.

Why compare live performance to the backtest? Because the backtest is a promise and live trading is reality. If your live win rate is 18% and the backtest said 29%, something is wrong — maybe overfitting, maybe regime change, maybe execution slippage. Surfacing that gap explicitly forces honest evaluation instead of hoping the numbers will eventually catch up.

---

## Disclaimer

Paper trading only by default. To enable live trading, add Alpaca credentials to `.env`. This is a research and educational project — past performance does not guarantee future results.

---

## License

MIT — see [LICENSE](LICENSE)
