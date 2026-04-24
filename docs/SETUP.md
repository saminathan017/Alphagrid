# AlphaGrid Setup Guide

This guide is for running AlphaGrid locally on a Mac with VS Code and Python 3.11.

If you want the shortest version:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
cp .env.example .env

# set a private owner password in .env before first launch
# ALPHAGRID_OWNER_PASSWORD=your-own-strong-password

./Launch\ AlphaGrid.command
```

Then open:

```text
http://localhost:8080/login
```

## Before you start

Make sure you have these installed:

| Tool | How to check | If missing |
|---|---|---|
| Python 3.11 or 3.12 | `python3 --version` | `brew install python@3.11` |
| Homebrew | `brew --version` | install from [brew.sh](https://brew.sh) |
| VS Code | open it | [code.visualstudio.com](https://code.visualstudio.com) |
| VS Code Python extension | check Extensions | install `ms-python.python` |

Use Python `3.11` if you can. It is the safest option for this project.

Avoid Python `3.13` for now.

## 1. Open the project

Open the repository folder in VS Code.

Then open a terminal inside VS Code:

```text
Terminal → New Terminal
```

You can also use `Ctrl+\``.

All commands below should be run from the project root.

## 2. Create a virtual environment

```bash
pwd
python3.11 -m venv venv
source venv/bin/activate
```

After activation, your terminal prompt should show `(venv)`.

If VS Code asks whether to use the virtual environment interpreter, accept it.

If it does not ask:

```text
Cmd+Shift+P → Python: Select Interpreter
```

Then choose the interpreter inside `venv`.

## 3. Install PyTorch first

Install PyTorch before the rest of the requirements.

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

If you are on Apple Silicon and want to confirm MPS support:

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"
```

## 4. Install the project requirements

```bash
pip install -r requirements.txt
```

This can take a few minutes.

If a package fails, these are the most common fixes:

```bash
brew install libomp
pip install lightgbm --no-binary lightgbm
pip install passlib[bcrypt] --no-binary passlib
pip install python-jose[cryptography]
pip install pandas-ta --no-deps
```

## 5. Create local working folders

```bash
mkdir -p logs models data cache
```

## 6. Create the environment file

```bash
cp .env.example .env
```

You can leave most values empty for a basic local run.

Paper trading works without broker keys, and the app can start with free data sources.

Before the first launch, set a private owner password in `.env`.

If you leave `ALPHAGRID_OWNER_PASSWORD` blank, the app generates a local password on first boot and prints it to the server log.

If you want a stable JWT secret for local sessions:

```bash
python3 -c "import secrets; print('ALPHAGRID_JWT_SECRET=' + secrets.token_hex(32))"
```

Paste the output into `.env`.

## 7. Start the app

Use the main launcher:

```bash
./Launch\ AlphaGrid.command
```

That is the recommended local entrypoint.

It should open the app automatically. If it does not, open:

```text
http://localhost:8080/login
```

## 8. Log in

The owner account logs in with the username from `.env`.

By default, that username is:

```text
admin
```

All other people should create their own standard user account from the sign-up form on the login page.

## 9. Train models if needed

You do not need to train models just to open the app.

If you want to run training, open a new terminal tab, activate the virtual environment again, and run one of these:

```bash
source venv/bin/activate

python scripts/train_models.py --symbols AAPL --lookback 365 --quick

python scripts/train_models.py \
    --symbols AAPL,MSFT,NVDA,GOOGL,SPY,QQQ \
    --lookback 730
```

After training, restart the app so it picks up the saved checkpoints.

## Optional VS Code run panel

If the repository has launch configurations in place, you can also use the Run and Debug panel:

```text
Cmd+Shift+D
```

Look for entries such as:

- `Run Server`
- `Train Models`
- `Quick Train`

## Project layout

This is the practical layout you will use most often:

```text
alphagrid/
├── README.md
├── Launch AlphaGrid.command
├── docs/
│   ├── SETUP.md
│   └── CHANGELOG.md
├── deploy/
├── runtime/
├── dashboard/
├── core/
├── data/
├── models/
├── strategies/
├── execution/
├── backtest/
├── risk/
├── scripts/
├── config/
├── requirements.txt
└── .env.example
```

Folders you will probably touch first:

- `dashboard/` for the app and frontend
- `scripts/` for training and operational commands
- `models/` for model logic and evaluation
- `data/` for feeds, history, and news
- `docs/` for setup and project notes

## Troubleshooting

### Virtual environment is not activating

```bash
cd /path/to/alphagrid
python3.11 -m venv venv
source venv/bin/activate
```

### A module is missing

```bash
source venv/bin/activate
pip install <package-name>
```

### Port 8080 is already in use

```bash
lsof -i :8080
kill -9 <PID>
./Launch\ AlphaGrid.command
```

### MPS is not available after installing torch

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

### LightGBM fails to install

```bash
brew install libomp
pip install lightgbm --no-binary lightgbm
```

### Login works but the dashboard loads badly

One common fix is to remove the local auth database and restart:

```bash
rm -f alphagrid_auth.db
./Launch\ AlphaGrid.command
```

### Market data looks empty on first run

That can happen on the first load while caches warm up.

You can check basic status here:

```bash
curl http://localhost:8080/api/history/status
```

## Quick command reference

### First-time setup

```bash
python3 --version
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
mkdir -p logs models data cache
cp .env.example .env
```

### Normal local use

```bash
source venv/bin/activate
./Launch\ AlphaGrid.command
```

### Optional direct shell entrypoint

```bash
bash runtime/run_local.sh
```

### Training

```bash
source venv/bin/activate
python scripts/train_models.py --symbols AAPL --lookback 365 --quick
python scripts/train_models.py --symbols AAPL,MSFT,NVDA,GOOGL,SPY,QQQ --lookback 730
```

### Health checks

```bash
curl http://localhost:8080/api/health
curl http://localhost:8080/api/history/status
open http://localhost:8080/docs
```
