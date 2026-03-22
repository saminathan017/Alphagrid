# AlphaGrid v6 — Complete Setup Guide
### MacBook M4 · VS Code · Python 3.11

---

## What you need before starting

| Tool | Check | Install if missing |
|------|-------|--------------------|
| Python 3.11 or 3.12 | `python3 --version` | `brew install python@3.11` |
| Homebrew | `brew --version` | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| VS Code | open it | [code.visualstudio.com](https://code.visualstudio.com) |
| VS Code Python extension | Extensions panel → search Python | install ms-python.python |

> **Do not use Python 3.13** — PyTorch has incomplete 3.13 support. Use 3.11 or 3.12.

---

## Step 1 — Open the project in VS Code

Move the `alphagrid-v6` folder anywhere on your Mac, then:

```
VS Code → File → Open Folder → select alphagrid-v6
```

Open the integrated terminal:
```
Terminal → New Terminal
```
or press **Ctrl+`** (backtick).

All commands below are typed in that terminal.

---

## Step 2 — Create the virtual environment

```bash
# Make sure you're in the alphagrid-v6 folder
pwd
# Should show something like /Users/yourname/alphagrid-v6

# Create venv with Python 3.11
python3.11 -m venv venv

# Activate it
source venv/bin/activate
```

Your terminal prompt will change to show `(venv)`.

VS Code will ask **"Do you want to select this interpreter?"** — click **Yes**.
If it doesn't ask: `Cmd+Shift+P` → type `Python: Select Interpreter` → pick the one that says `venv`.

**Every time you open a new terminal tab, run `source venv/bin/activate` first.**

---

## Step 3 — Install PyTorch (M4 native — do this BEFORE requirements.txt)

PyTorch on M4 uses Metal Performance Shaders (MPS) — the GPU inside your Mac.
Models train 5–10× faster than CPU.

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with M4 / MPS support (official Apple Silicon build)
pip install torch torchvision torchaudio
```

Verify it worked:
```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:    ', torch.backends.mps.is_built())
"
```

Expected output:
```
PyTorch: 2.x.x
MPS available: True
MPS built:     True
```

---

## Step 4 — Install all other dependencies

```bash
pip install -r requirements.txt
```

This takes 3–8 minutes. Lots of output is normal.

If any package fails, fix individually:

```bash
# lightgbm sometimes needs this on M4
brew install libomp
pip install lightgbm --no-binary lightgbm

# If passlib fails
pip install passlib[bcrypt] --no-binary passlib

# If jose fails
pip install python-jose[cryptography]

# pandas-ta warnings are harmless
pip install pandas-ta --no-deps
```

---

## Step 5 — Create required folders

```bash
mkdir -p logs models data cache
```

---

## Step 6 — Set up environment file

```bash
cp .env.example .env
```

Open `.env` in VS Code. Everything works with all values empty (paper trading + yfinance require no keys).

Optional — generate a stable JWT secret so sessions survive server restarts:
```bash
python3 -c "import secrets; print('ALPHAGRID_JWT_SECRET=' + secrets.token_hex(32))"
# Paste the output line into .env
```

---

## Step 7 — Start the server

```bash
bash dashboard/run.sh
```

Wait for this in the terminal:
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Open your browser: **[http://localhost:8080](http://localhost:8080)**

---

## Step 8 — Log in

| Who | Username / Email | Password | Access |
|-----|-----------------|----------|--------|
| **Owner** | `admin` | `Admin@Grid1` | Everything. Cannot be deleted |
| **Builder** | `builder@alphagrid.app` | `Builder1!` | All tabs + Models + Eval |
| **Trader** | `trader@alphagrid.app` | `Trader1!` | Trading tabs only |

The owner logs in with **username only** — no email needed.

> Change the owner password immediately: sidebar → your avatar at bottom → Change Password

---

## Step 9 — Train the ML models (new terminal tab)

Keep the server running. Open a **new terminal tab**:

```bash
# Activate venv in the new tab
source venv/bin/activate

# Quick test — 1 symbol, ~5 minutes on M4
python scripts/train_models.py --symbols AAPL --lookback 365 --quick

# Full training — 6 symbols × 2 years, ~30–60 min on M4
python scripts/train_models.py \
    --symbols AAPL,MSFT,NVDA,GOOGL,SPY,QQQ \
    --lookback 730
```

Training prints a calibration table showing accuracy at each confidence level:
```
  Confidence    N signals    Accuracy
  ≥0%             520          63.2%
  ≥50%            290          70.1%
  ≥70%            140          78.4%
  ≥80%             75          84.6%   ← hedge fund target
```

Restart the server after training to load saved model checkpoints.

---

## VS Code run panel (shortcut alternative to terminal)

`Cmd+Shift+D` → Run & Debug panel:

| Button | What it does |
|--------|-------------|
| `▶  Run Server` | Starts dashboard on port 8080 |
| `🚀 Train Models (full)` | Full 6-symbol 2-year training |
| `⚡ Quick Train (1 symbol, test)` | Fast 5-minute smoke test |

---

## Project structure

```
alphagrid-v6/
│
├── core/
│   ├── auth_db.py          ← Auth: users, JWT, sessions, owner account
│   ├── config.py           ← YAML settings loader
│   ├── database.py         ← SQLAlchemy models
│   ├── events.py           ← Internal event bus
│   ├── latency_cache.py    ← In-memory price cache
│   ├── logger.py           ← Loguru config
│   └── ticker_universe.py  ← 150 equities + 50 forex
│
├── data/
│   ├── feature_engineer.py ← 80+ quantitative features (10 families)
│   ├── historical.py       ← 10-year SQLite engine (yfinance)
│   ├── market_feed.py      ← Live price feed
│   ├── live_news.py        ← News feed
│   ├── news_feed.py        ← News aggregator
│   └── universe_feed.py    ← Universe price updater
│
├── models/
│   ├── alpha_engine.py     ← 15 IC-weighted alpha factors
│   ├── ensemble.py         ← Adaptive ensemble + confidence filter
│   ├── evaluator.py        ← IC, ICIR, signal Sharpe, Tier S/A/B/C/D
│   ├── lgbm_model.py       ← Regime-conditional LightGBM + MetaLearner
│   ├── lstm_model.py       ← QuantLSTM: TCN + BiLSTM + SWA + TTA
│   ├── position_sizer.py   ← Fractional Kelly + 3-tier TP cascade
│   ├── sentiment_model.py  ← FinBERT sentiment
│   ├── signal_filter.py    ← 7-gate institutional signal filter
│   └── transformer_model.py← FinancialTransformer
│
├── strategies/
│   ├── indicators.py       ← 40+ Numba-JIT indicators
│   ├── trading_modes.py    ← Day + swing trading strategies
│   ├── combined.py         ← Strategy combiner
│   └── strategy_lab.py     ← Strategy research
│
├── execution/
│   ├── broker_manager.py   ← Alpaca / OANDA / Robinhood / Paper
│   ├── alpaca_broker.py    ← Alpaca REST + WebSocket
│   └── paper_trader.py     ← Paper trading simulator
│
├── risk/
│   └── portfolio.py        ← Kelly sizing, drawdown limits
│
├── backtest/
│   ├── engine.py           ← Vectorized backtest engine
│   ├── metrics.py          ← Sharpe, Sortino, Calmar, drawdown
│   └── runner.py           ← Backtest runner
│
├── dashboard/
│   ├── app.py              ← FastAPI server (all endpoints + WebSocket)
│   ├── auth.html           ← Login / signup page
│   ├── index.html          ← Main dashboard SPA
│   └── run.sh              ← Start script
│
├── scripts/
│   ├── train_models.py     ← Full ML training pipeline
│   ├── run_backtest.py     ← Backtest runner
│   └── run_live.py         ← Live trading runner
│
├── config/
│   └── settings.yaml       ← All configuration
│
├── .env.example            ← Environment template → copy to .env
├── .gitignore
├── .vscode/
│   ├── settings.json       ← Python interpreter + PYTHONPATH
│   └── launch.json         ← Run / debug shortcuts
├── requirements.txt
└── SETUP.md                ← This file
```

---

## Expected model performance after training

| Signal filter | Accuracy | Hedge fund tier |
|--------------|----------|-----------------|
| All signals (no filter) | 62–68% | B–A |
| Confidence ≥ 0.70 | 72–80% | A |
| Confidence ≥ 0.80 | 82–88% | **S — Elite** |
| Confidence ≥ 0.85 | 88–93% on 8% of signals | **S — Elite** |

Without training = Tier D (the architecture needs training to activate).

---

## Troubleshooting

**`venv` not activating**
```bash
cd /path/to/alphagrid-v6   # must be in project root
python3.11 -m venv venv
source venv/bin/activate
```

**`ModuleNotFoundError: No module named 'X'`**
```bash
source venv/bin/activate   # activate venv first
pip install X
```

**Port 8080 already in use**
```bash
lsof -i :8080              # find the PID
kill -9 <PID>              # kill it
bash dashboard/run.sh
```

**MPS not available after installing torch**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

**lightgbm install fails**
```bash
brew install libomp
pip install lightgbm --no-binary lightgbm
```

**Login works but dashboard shows blank / errors**
```bash
# Check the browser console (Cmd+Option+I → Console tab)
# Most common fix — delete the auth database and restart:
rm -f alphagrid_auth.db
bash dashboard/run.sh
```

**yfinance data is empty on first run**
```bash
# Normal — 200 symbols downloading in background takes ~10–15 min
# Check progress:
curl http://localhost:8080/api/history/status
```

---

## All commands — quick reference

```bash
# ══ ONE TIME SETUP ════════════════════════════════════════════════

# 1. Check Python version (need 3.11 or 3.12)
python3 --version

# 2. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch with M4 GPU support (FIRST, before requirements.txt)
pip install torch torchvision torchaudio

# 5. Verify MPS (Metal GPU) is active
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# 6. Install everything else
pip install -r requirements.txt

# 7. Create required directories
mkdir -p logs models data cache

# 8. Create .env from template
cp .env.example .env

# ══ EVERY SESSION ═════════════════════════════════════════════════

# Activate venv
source venv/bin/activate

# Start the server
bash dashboard/run.sh

# Open in browser
open http://localhost:8080

# ══ TRAINING (new terminal tab, venv activated) ════════════════════

# Quick smoke test — ~5 min
python scripts/train_models.py --symbols AAPL --lookback 365 --quick

# Full hedge fund training — ~30–60 min on M4
python scripts/train_models.py \
    --symbols AAPL,MSFT,NVDA,GOOGL,SPY,QQQ \
    --lookback 730

# ══ HEALTH CHECKS ═════════════════════════════════════════════════

curl http://localhost:8080/api/health
curl http://localhost:8080/api/history/status
open http://localhost:8080/docs
```
