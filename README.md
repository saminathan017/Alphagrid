# AlphaGrid v8

AlphaGrid is a Python-based trading platform that combines market data, signal generation, model evaluation, paper trading, and a web dashboard in one project. It is set up to run locally with free data sources, and it can be extended for live trading if you add your own broker credentials.

This repository includes:

- a FastAPI dashboard with login and role-based access
- market data and news ingestion
- model training and evaluation scripts
- backtesting and paper-trading flows
- deployment files for local and hosted setups

The recommended local launch path is:

```bash
./Launch\ AlphaGrid.command
```

That launcher starts the app on `http://localhost:8080/login`.

Full setup notes are in [docs/SETUP.md](docs/SETUP.md).

## What the project covers

The project is organized around a few main workflows:

- collecting price and news data
- generating trading signals
- monitoring those signals in the dashboard
- evaluating model outputs
- running paper trading by default

The dashboard includes pages for:

- overview
- signals
- chart
- trades
- universe
- models
- evaluation

## Quick start

```bash
git clone https://github.com/saminathan017/Alphagrid.git
cd Alphagrid

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# set a private owner password before first launch
# ALPHAGRID_OWNER_PASSWORD=your-own-strong-password

./Launch\ AlphaGrid.command
```

Then open:

```text
http://localhost:8080/login
```

Before first launch, set a private owner password in `.env`.

Regular users should create their own account from the sign-up flow in the app.

## How it runs locally

The root launcher is the main entrypoint for local use:

```bash
./Launch\ AlphaGrid.command
```

That wrapper forwards to the structured runtime files under `runtime/`, so the repository keeps a clean layout while preserving a simple start command at the root.

If you prefer the shell path for debugging, there is also a secondary runtime script:

```bash
bash runtime/run_local.sh
```

For day-to-day use, stick with the launcher command above.

## Project structure

The repository is grouped so source code, runtime scripts, documentation, and deployment files are easier to find.

```text
Alphagrid/
├── README.md
├── Launch AlphaGrid.command
├── docs/
│   ├── SETUP.md
│   └── CHANGELOG.md
├── deploy/
│   ├── Dockerfile
│   ├── Dockerfile.train
│   ├── docker-compose.yml
│   ├── render.yaml
│   ├── railway.json
│   └── Procfile
├── runtime/
│   ├── Launch AlphaGrid.command
│   └── run_local.sh
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

## Main folders

`dashboard/`

Contains the FastAPI app, frontend pages, API routes, and live dashboard wiring.

`core/`

Shared application pieces such as config loading, auth, database helpers, logging, and internal utilities.

`data/`

Data collection and preprocessing code, including historical prices, live feeds, and news handling.

`models/`

Model-related code for training, scoring, evaluation, and signal logic.

`strategies/`

Trading strategy logic and indicator-driven signal layers.

`execution/`

Paper-trading and broker-facing execution code.

`backtest/`

Backtesting runner and related metrics.

`scripts/`

Operational scripts for training, reporting, and related workflows.

`docs/`

Project documentation. The setup guide in this folder is the best reference for a first local run.

`deploy/`

Deployment files for Docker and hosted platforms.

`runtime/`

Launcher and runtime helper scripts used by the root compatibility entrypoints.

## Data and broker notes

The project is designed to run locally without paid market-data keys for a basic setup. The local experience uses free data sources where available.

Paper trading is the default path.

If you want to route orders through a broker, you can add Alpaca credentials to `.env`. That is optional and not required for local exploration.

## Training and evaluation

The repository includes scripts for retraining and evaluation. A few common entrypoints are:

```bash
python scripts/train_models.py
python scripts/train_models.py --symbols AAPL,MSFT,NVDA
python scripts/train_models.py --symbols AAPL --quick
```

The evaluation surface is available in the dashboard, and the underlying evaluation logic lives in the `models/` and `scripts/` folders.

## Deployment

Deployment-related files are grouped under `deploy/`, with compatibility entrypoints still kept at the repository root for convenience.

Root compatibility files:

- `docker-compose.yml`
- `render.yaml`
- `railway.json`
- `Procfile`

Primary deployment assets:

- [deploy/Dockerfile](deploy/Dockerfile)
- [deploy/docker-compose.yml](deploy/docker-compose.yml)
- [deploy/render.yaml](deploy/render.yaml)
- [deploy/railway.json](deploy/railway.json)

## Recommended reading order

If you are opening the project for the first time, this is the easiest path:

1. Read this file.
2. Follow [docs/SETUP.md](docs/SETUP.md).
3. Start the app with `./Launch\ AlphaGrid.command`.
4. Sign in with your private owner account or create a standard user account.
5. Explore the dashboard pages before changing configs or training models.

## Notes

- Python `3.11` is the safest default for this project.
- Local caches, logs, databases, and model artifacts may be created as you run the app.
- Some features are richer when the local environment already has cached market history.
- The repository does not publish shared login credentials. Keep the owner password in `.env` and out of version control.

## License

MIT. See [LICENSE](LICENSE).
