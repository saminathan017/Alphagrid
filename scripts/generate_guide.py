"""
Generate AlphaGrid Learning Guide PDF
"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"AlphaGrid v6  -  Learning Guide   |   Page {self.page_no()}", align="C")

    def chapter_title(self, text):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 80, 160)
        self.ln(6)
        self.multi_cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(30, 80, 160)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def section_title(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 20, 20)
        self.ln(5)
        self.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def question(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(235, 242, 255)
        self.set_text_color(20, 50, 120)
        self.ln(4)
        self.multi_cell(0, 7, "Q:  " + text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def answer(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        self.multi_cell(0, 6.5, "A:  " + text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def step_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 120, 80)
        self.ln(4)
        self.multi_cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(60, 60, 60)
        self.ln(1)
        self.multi_cell(0, 5.5, text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(0, 0, 0)


pdf = PDF()
pdf.set_margins(20, 20, 20)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── Cover ──────────────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 28)
pdf.set_text_color(30, 80, 160)
pdf.ln(20)
pdf.multi_cell(0, 14, "AlphaGrid v6", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 10, "Complete Learning Guide", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)
pdf.set_font("Helvetica", "I", 11)
pdf.set_text_color(120, 120, 120)
pdf.multi_cell(0, 7, "From Zero to Interview-Ready", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(10)
pdf.set_draw_color(30, 80, 160)
pdf.set_line_width(1.2)
pdf.line(40, pdf.get_y(), pdf.w - 40, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "", 10.5)
pdf.set_text_color(60, 60, 60)
pdf.multi_cell(0, 7,
    "This guide covers AlphaGrid v6  -  a production-grade ML trading system  -  "
    "in two parts. Part 1 explains the project from scratch in plain language. "
    "Part 2 prepares you to discuss it confidently in any technical interview.",
    align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1  -  LEARN FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════════════

pdf.add_page()
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 80, 160)
pdf.multi_cell(0, 12, "PART 1", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 8, "Learn the Project from Scratch", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)
pdf.ln(8)

pdf.step_title("Step 1  -  What problem are we solving?")
pdf.body(
    "Every day, stock prices move up or down. If you could predict that with even 60% accuracy "
    "on high-confidence moments, you make money consistently over time. AlphaGrid tries to do "
    "exactly that  -  predict whether a stock price will go UP or DOWN  -  using machine learning "
    "trained on years of historical price data.\n\n"
    "That is it. Everything else in the codebase is machinery to do that well, reliably, and at scale "
    "across 200 assets simultaneously."
)

pdf.step_title("Step 2  -  Where does data come from?")
pdf.body(
    "yfinance is a free Python library that downloads historical stock prices from Yahoo Finance. "
    "You give it a ticker symbol like AAPL and a date range, and it returns daily "
    "open / high / low / close / volume data. No API key required, no cost.\n\n"
    "The file that handles all data ingestion and processing is: data/feature_engineer.py"
)

pdf.step_title("Step 3  -  What are features?")
pdf.body(
    "Raw price data  -  open, close, volume  -  is too noisy to feed directly into a model. "
    "So we compute 80+ derived signals from it. These are called features.\n\n"
    "Examples:\n"
    "  - RSI: is the stock overbought or oversold?\n"
    "  - MACD: is momentum building or fading?\n"
    "  - ATR: how volatile is the stock right now?\n"
    "  - Hurst exponent: is the price trending or mean-reverting?\n"
    "  - Fourier bands: what are the dominant cycles in the price?\n\n"
    "Think of features as the stats a human analyst would look at before making a trade decision. "
    "The ML model learns which stats matter most and in which combination."
)

pdf.step_title("Step 4  -  What are labels?")
pdf.body(
    "For supervised machine learning, you need to tell the model what the correct answer was "
    "historically. This is called labeling.\n\n"
    "This project uses the triple-barrier method from Marcos Lopez de Prado's book "
    "Advances in Financial Machine Learning:\n\n"
    "  - Set a take-profit target: if price rises 2.5x ATR, label = UP (1)\n"
    "  - Set a stop-loss: if price falls 2.0x ATR first, label = DOWN (0)\n"
    "  - Set a time limit: if neither barrier is hit within N days, label = SKIP (ignored)\n\n"
    "This filters out the boring days where nothing meaningful happened. Only real, measurable "
    "moves get labeled. That is why accuracy on labeled samples jumps from the naive 52% baseline "
    "up to 65-90% on high-confidence predictions."
)

pdf.step_title("Step 5  -  The three ML models")
pdf.body(
    "Once you have features and labels, you train models. AlphaGrid trains three different "
    "model architectures per symbol, because each has different strengths.\n\n"
    "QuantLSTM  (models/lstm_model.py)\n"
    "Takes the last 60 days of features as a time sequence and processes them through a neural "
    "network that understands time order. The architecture combines a TCN front-end for "
    "multi-scale pattern detection with a BiLSTM for sequential memory and multi-head attention "
    "for focusing on the most informative timesteps. It is good at catching patterns that develop "
    "over days or weeks, like momentum building gradually.\n\n"
    "FinancialTransformer  (models/transformer_model.py)\n"
    "Same idea but uses the Transformer architecture  -  the same family as GPT. "
    "It uses Rotary Positional Encoding and Pre-LayerNorm for stable training on small financial "
    "datasets. It is better at long-range dependencies  -  when something that happened 40 bars ago "
    "has a meaningful effect on today.\n\n"
    "LightGBM DART  (models/lgbm_model.py)\n"
    "A gradient boosting tree model. It takes a single snapshot of today's features  -  not a "
    "sequence  -  and makes a prediction. Extremely fast to train and often the most reliable model "
    "on tabular data. It trains three separate versions: one for calm markets, one for normal "
    "conditions, and one for volatile markets. At prediction time it picks the right version "
    "based on current market volatility."
)

pdf.step_title("Step 6  -  The MetaLearner")
pdf.body(
    "Each of the three models outputs a probability between 0 and 1 representing its confidence "
    "that the stock will go UP. The MetaLearner is a fourth model that takes all three predictions "
    "as inputs and learns which model to trust more in which situation.\n\n"
    "Think of it as a committee vote where the chairman  -  the MetaLearner  -  knows each committee "
    "member's historical track record across different market conditions. For example, it might "
    "learn to trust LightGBM more in high-volatility regimes and trust the LSTM more when "
    "momentum is clearly building over multiple weeks.\n\n"
    "When fewer than 100 training samples are available, a full LightGBM meta-model would "
    "overfit. In that case it falls back to AUC-weighted averaging, giving each base model a "
    "weight proportional to its skill above random."
)

pdf.step_title("Step 7  -  The 7-Gate Signal Filter")
pdf.body(
    "Even after the MetaLearner, most predictions are still noise. The 7-gate filter only lets "
    "a signal through if ALL of these conditions are true simultaneously:\n\n"
    "  Gate 1: Model confidence is at or above a dynamic Bayesian threshold (0.55 to 0.75)\n"
    "  Gate 2: The signal direction aligns with the current macro market regime (SPY vol, DXY, credit spreads)\n"
    "  Gate 3: IC-weighted alpha factors confirm the signal direction\n"
    "  Gate 4: The trade has a risk/reward ratio of at least 2.0x (profit target >= 2x stop loss)\n"
    "  Gate 5: The signal is fresh  -  not stale based on the ATR-scaled half-life\n"
    "  Gate 6: Adding this trade does not over-concentrate the portfolio in one sector\n"
    "  Gate 7: The estimated spread is under 10 basis points so trading costs do not eat the edge\n\n"
    "Only 5 to 15 percent of raw signals pass all 7 gates. Those are the ones that consistently "
    "convert to 80-90% accuracy."
)

pdf.step_title("Step 8  -  The Live Dashboard")
pdf.body(
    "dashboard/app.py is a production FastAPI web server. It does the following:\n\n"
    "  - Fetches live prices across 200 symbols every 5 seconds using yfinance\n"
    "  - Recomputes 40+ indicators in real time using Numba-JIT compiled functions for speed\n"
    "  - Runs the trained models and the 7-gate filter to generate live trading signals\n"
    "  - Shows everything in a browser-based dashboard UI\n"
    "  - Supports paper trading: you can open and close positions at live prices with fake money "
    "and track real P&L\n"
    "  - Pushes updates to all connected browser clients via WebSocket every 2 seconds\n"
    "  - Uses JWT authentication with three roles: Admin, Builder, and Trader"
)

pdf.step_title("Step 9  -  The Training Pipeline")
pdf.body(
    "scripts/train_models.py is the orchestrator. It loops through all 150 symbols, downloads "
    "historical data, engineers features, trains all three base models plus the MetaLearner, "
    "evaluates everything on the held-out test set, and saves the trained models to disk. "
    "A full run across 150 symbols takes a few hours.\n\n"
    "scripts/monitor_training.py gives you a live terminal progress bar while training runs. "
    "It shows completed symbols, model metrics, ETA, and highlights any model that achieved "
    "hit@70 above 80%  -  the target threshold for a high-confidence signal."
)

pdf.step_title("Recommended Learning Order")
pdf.body(
    "Start with LightGBM because it has no sequences, no tensors  -  just a table of numbers in "
    "and a prediction out. Once you understand that, the neural models are easier because the "
    "feature engineering is identical."
)
pdf.code_block(
    "1. data/feature_engineer.py        -  understand features and triple-barrier labels\n"
    "2. models/lgbm_model.py            -  simplest model, no sequences, easiest to follow\n"
    "3. models/lstm_model.py            -  deep learning LSTM model\n"
    "4. models/transformer_model.py     -  transformer model\n"
    "5. scripts/train_models.py         -  how everything is orchestrated end to end\n"
    "6. models/signal_filter.py         -  the 7-gate institutional signal filter\n"
    "7. dashboard/app.py                -  the live production system"
)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2  -  INTERVIEW Q&A
# ══════════════════════════════════════════════════════════════════════════════

pdf.add_page()
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 80, 160)
pdf.multi_cell(0, 12, "PART 2", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 8, "Interview Questions and Answers", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)
pdf.ln(8)

# Round 1
pdf.chapter_title("Round 1  -  High Level  (Recruiter / Hiring Manager)")

pdf.question("Tell me about this project in one minute.")
pdf.answer(
    "AlphaGrid is an end-to-end ML trading system. It downloads historical price data for 200 assets, "
    "computes 80+ technical and statistical features, trains an ensemble of three models per symbol  -  "
    "an LSTM, a Transformer, and LightGBM  -  stacks them with a meta-learner, then filters every "
    "prediction through a 7-gate institutional risk framework before surfacing it as a trading signal. "
    "Everything is served through a live FastAPI dashboard with WebSocket price feeds and paper trading."
)

pdf.question("Why did you build this?")
pdf.answer(
    "To understand how a real quant fund operates end to end  -  not just the modeling part, but the "
    "full pipeline: data ingestion, feature engineering, model training, signal filtering, risk sizing, "
    "and live execution. Most tutorials stop at training accuracy. This goes all the way to position "
    "sizing and P&L tracking."
)

pdf.question("What is the business problem you are solving?")
pdf.answer(
    "Predicting whether a stock will go up or down. But the real insight is that you do not need to "
    "be right all the time  -  you need to be right when you are confident. The system filters "
    "aggressively so that only the 5 to 15 percent of signals with genuine edge reach the trader. "
    "Those convert to 80 to 90 percent accuracy."
)

# Round 2
pdf.add_page()
pdf.chapter_title("Round 2  -  ML Engineer")

pdf.question("Why did you use three different model architectures instead of one?")
pdf.answer(
    "Each model has a different inductive bias. The LSTM captures sequential dependencies  -  patterns "
    "that develop over days or weeks. The Transformer handles long-range relationships  -  what happened "
    "40 bars ago affecting today. LightGBM ignores sequence entirely and treats each day as a snapshot, "
    "which makes it more robust on small datasets and less prone to overfitting. No single architecture "
    "dominates across all symbols and market regimes, so stacking them lets the meta-learner learn "
    "when to trust each one."
)

pdf.question("What is the MetaLearner and why is it better than simple averaging?")
pdf.answer(
    "The MetaLearner is a LightGBM model trained on the out-of-fold predictions of the three base "
    "models. Simple averaging assumes each model contributes equally. The meta-learner learns the "
    "conditional relationship  -  for example, trust LightGBM more in high-volatility regimes, trust "
    "the LSTM more when momentum is building. That is stacked generalization, which is how hedge "
    "funds actually combine models in production."
)

pdf.question("What happens when the dataset is too small for the MetaLearner?")
pdf.answer(
    "Below 100 samples, a LightGBM meta-model overfits badly. So it falls back to AUC-weighted "
    "averaging  -  each base model's weight is proportional to its AUC minus 0.5, which represents "
    "its skill above random. A model with AUC 0.6 gets double the weight of a model with AUC 0.55."
)

pdf.question("What is the triple-barrier labeling method and why use it?")
pdf.answer(
    "It comes from Marcos Lopez de Prado's book Advances in Financial Machine Learning. Instead of "
    "labeling each bar with the next-day return, you set three barriers  -  a take-profit at 2.5x ATR, "
    "a stop-loss at 2.0x ATR, and a time limit. Whichever barrier the price hits first determines "
    "the label. Bars where no barrier is hit are dropped entirely. This removes the noise from flat, "
    "indecisive price action and only labels bars where a real measurable move occurred. It is why "
    "accuracy on labeled samples is 65 to 90 percent instead of the naive 52 percent baseline."
)

pdf.question("How do you prevent data leakage?")
pdf.answer(
    "The test set is the last 15 percent of data, held out completely before any training or "
    "hyperparameter tuning. Features are computed only from past data  -  no look-ahead. Normalization "
    "statistics such as mean and standard deviation are computed on the training set and applied to "
    "the test set. The triple-barrier labels use future prices only to generate labels, not as input "
    "features."
)

pdf.question("What is DART boosting and why use it over standard gradient boosting?")
pdf.answer(
    "DART stands for Dropout meets Multiple Additive Regression Trees. Standard gradient boosting "
    "adds trees sequentially where each tree corrects the residual of the previous ones. This causes "
    "the first few trees to dominate  -  they capture the big patterns and later trees just overfit "
    "noise. DART randomly drops trees during training, similar to dropout in neural networks, which "
    "forces all trees to contribute meaningfully and improves generalization. Financial data is small "
    "and noisy, so this regularization matters significantly."
)

pdf.question("Why regime-conditional LightGBM instead of one global model?")
pdf.answer(
    "The signal that predicts direction in a low-volatility trending market is completely different "
    "from the signal that works in a high-volatility mean-reverting market. A moving average "
    "crossover works great when the market trends. It fails badly when the market chops. Training "
    "three separate models  -  one per volatility regime  -  lets each model specialize. At inference, "
    "you check the current volatility ratio and route to the appropriate model."
)

pdf.question("What are monotone constraints and why apply them?")
pdf.answer(
    "They are economic priors enforced directly in the model. For example, higher ADX (trend "
    "strength) should always increase signal confidence, never decrease it. Without constraints, "
    "the model might learn a negative relationship due to noise in training data, which makes no "
    "economic sense. Monotone constraints tell the model this feature must have a monotonically "
    "increasing or decreasing relationship with the prediction. It improves out-of-sample "
    "generalization by baking in domain knowledge."
)

pdf.question("What is hit@70 and why is it the primary metric?")
pdf.answer(
    "Hit@70 is accuracy computed only on predictions where the model's confidence is at or above "
    "70 percent  -  where the absolute value of (probability minus 0.5) times 2 is greater than or "
    "equal to 0.70. Average accuracy across all predictions is near-random in financial ML because "
    "markets are genuinely hard to predict most of the time. But when the model is very confident, "
    "it is right much more often. Hit@70 measures exactly that  -  does the model know when it knows? "
    "A model with 90 percent hit@70 is practically useful even if its overall accuracy is 50 percent."
)

# Round 3
pdf.add_page()
pdf.chapter_title("Round 3  -  System Design")

pdf.question("Walk me through the data pipeline.")
pdf.answer(
    "yfinance downloads OHLCV data for each symbol. feature_engineer.py computes 80+ features "
    "across 10 families  -  returns at multiple horizons, volatility regime, trend and momentum, "
    "mean-reversion, volume and liquidity, market microstructure, multi-timeframe regime, Fourier "
    "spectral features, and fractal and entropy features. Everything is winsorized at the 1st and "
    "99th percentile to handle outliers cleanly. Labels are generated via the triple-barrier method. "
    "The result is a clean feature matrix X and label vector y, ready for model training."
)

pdf.question("How does the live dashboard work?")
pdf.answer(
    "It is a FastAPI server. On startup it loads all trained model files from disk. A background "
    "task polls yfinance every 5 seconds for live prices. Indicators are recomputed in real time "
    "using Numba-JIT compiled functions  -  40+ indicators including RSI, MACD, ATR, Bollinger Bands, "
    "Ichimoku, and SuperTrend. The loaded models score the latest feature vector. Signals that pass "
    "all 7 gates are pushed to connected browser clients via WebSocket every 2 seconds."
)

pdf.question("How does authentication work?")
pdf.answer(
    "JWT-based authentication. Users log in with username and password. Passwords are hashed with "
    "bcrypt so plaintext is never stored. On successful login, the server issues a signed JWT token "
    "with an expiry time. Subsequent API requests include the token in the Authorization header. "
    "The server verifies the signature and expiry on every request. There are three roles  -  Admin, "
    "Builder, and Trader  -  with different endpoint permissions."
)

pdf.question("How does paper trading work?")
pdf.answer(
    "When a signal fires, the dashboard lets you open a position at the current live price. The "
    "system tracks the entry price, current price, position size, and computes unrealized P&L in "
    "real time. On close, it records the completed trade. Performance metrics  -  Sharpe ratio, "
    "Sortino ratio, Calmar ratio, and win rate  -  are all computed from the actual closed trade "
    "history, not simulated or hypothetical data."
)

# Round 4
pdf.add_page()
pdf.chapter_title("Round 4  -  Finance and Quant")

pdf.question("What is the 7-gate signal filter and why does it matter?")
pdf.answer(
    "A model that is right 70 percent of the time is still useless if you trade every single "
    "prediction  -  transaction costs and variance will kill you over time. The 7 gates filter to "
    "only the signals with genuine, durable edge. Gate 1 checks model confidence above a dynamic "
    "Bayesian threshold. Gate 2 checks that direction aligns with the macro market regime. Gate 3 "
    "checks alpha factor confirmation. Gate 4 checks risk/reward ratio above 2x. Gate 5 checks "
    "signal freshness relative to ATR half-life. Gate 6 checks portfolio concentration limits. "
    "Gate 7 checks that the estimated spread is under 10 basis points so trading costs do not "
    "consume the edge. Only 5 to 15 percent of raw signals pass all 7 gates. Those are the ones "
    "that consistently hit 80 to 90 percent accuracy."
)

pdf.question("What is Kelly position sizing?")
pdf.answer(
    "The Kelly criterion is a mathematical formula that tells you what fraction of your capital "
    "to risk on a trade given your historical edge and the odds of winning. Full Kelly maximizes "
    "long-run capital growth mathematically but causes very large drawdowns in practice. This "
    "system uses fractional Kelly  -  typically half-Kelly  -  which reduces position size "
    "proportionally to improve drawdown characteristics. The conviction score from 0 to 100 "
    "produced by the signal filter scales the Kelly fraction further, so a lower-conviction "
    "signal automatically gets a smaller position."
)

pdf.question("What is the Sharpe ratio and how is it computed here?")
pdf.answer(
    "The Sharpe ratio is the average return divided by the standard deviation of returns, "
    "annualized. It measures return per unit of risk taken. A Sharpe above 1.0 is considered "
    "good, above 2.0 is excellent. Here it is computed from the actual paper trade history  -  "
    "each closed trade's return is logged, and Sharpe, Sortino (which only penalizes downside "
    "volatility, not upside), and Calmar (annualized return divided by maximum drawdown) are all "
    "computed from that real trade history."
)

# Round 5
pdf.add_page()
pdf.chapter_title("Round 5  -  Tough Questions")

pdf.question("Your average accuracy is 48 to 50 percent. That is basically random. How is this useful?")
pdf.answer(
    "That is exactly what you expect in efficient markets. The average is near-random because most "
    "predictions are low-confidence and should not be traded. The value is entirely in the "
    "high-confidence tail. When the model says it is confident, it is right 80 to 90 percent of "
    "the time. You do not trade the average  -  you only trade the high-confidence signals that pass "
    "all 7 gates. It is like a doctor who says I am only going to diagnose cancer when I am 90 "
    "percent sure  -  the overall detection rate is low because they pass on most ambiguous cases, "
    "but when they do make a call, they are almost always right."
)

pdf.question("How do you know the results are not just backfit to the training data?")
pdf.answer(
    "Three things. First, the test set is the last 15 percent of data, held out completely  -  never "
    "touched during training, validation, or hyperparameter tuning. Second, the triple-barrier "
    "labels only include bars where something real happened, which reduces the look-ahead bias "
    "from snooping on future returns. Third, the regime-conditional structure and monotone "
    "constraints add economic priors that prevent the model from learning spurious correlations "
    "that only exist in the training period and would not survive out-of-sample."
)

pdf.question("What would you improve if you had more time?")
pdf.answer(
    "Walk-forward cross-validation with an embargo gap between folds to prevent leakage from "
    "autocorrelated financial data. ONNX export for significantly faster inference at serving "
    "time  -  roughly 10x speedup over PyTorch. An online learning component so the model updates "
    "its weights incrementally as new market data arrives rather than going stale between training "
    "runs. And a proper backtesting engine with realistic transaction costs, slippage, and position "
    "limits  -  the current backtest is vectorized and fast but does not model market impact, which "
    "matters at larger position sizes."
)

pdf.question("How would you scale this to a production hedge fund environment?")
pdf.answer(
    "Replace yfinance with a proper market data vendor such as Bloomberg or Refinitiv for "
    "institutional-grade data quality and historical depth. Move model serving to a dedicated "
    "inference cluster with ONNX Runtime for low-latency predictions. Add a proper order "
    "management system and connect to a prime broker API rather than a paper trader. Implement "
    "a model monitoring pipeline that tracks prediction distribution drift and triggers retraining "
    "when the model goes stale. Add a risk management layer with hard position limits, VaR "
    "constraints, and drawdown circuit breakers. Use a feature store to decouple feature "
    "computation from model serving for consistency between training and inference."
)

# Round 6
pdf.add_page()
pdf.chapter_title("Round 6  -  Python and Programming")

pdf.question("How would you compute RSI from scratch in Python without any library?")
pdf.answer(
    "RSI is calculated from average gains and average losses over a window, typically 14 periods.\n\n"
    "  delta = prices.diff()\n"
    "  gain  = delta.clip(lower=0)\n"
    "  loss  = -delta.clip(upper=0)\n"
    "  avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()\n"
    "  avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()\n"
    "  rs  = avg_gain / avg_loss\n"
    "  rsi = 100 - (100 / (1 + rs))\n\n"
    "The EWM (exponential weighted mean) version is Wilder's smoothing method. "
    "RSI above 70 typically signals overbought, below 30 signals oversold. "
    "In this project RSI is computed at 3 different periods to capture mean-reversion at multiple timescales."
)

pdf.question("How does the triple-barrier label function work in code?")
pdf.answer(
    "For each bar, you look forward through future bars and check three conditions:\n\n"
    "  1. Has price risen above entry + (multiplier * ATR)?  -> label = 1 (UP)\n"
    "  2. Has price fallen below entry - (stop_mult * ATR)?  -> label = 0 (DOWN)\n"
    "  3. Have we exceeded the max holding period?           -> label = NaN (dropped)\n\n"
    "In code it looks like this:\n\n"
    "  for i in range(len(prices)):\n"
    "      entry = prices[i]\n"
    "      tp = entry + tp_mult * atr[i]\n"
    "      sl = entry - sl_mult * atr[i]\n"
    "      for j in range(i+1, min(i+max_hold, len(prices))):\n"
    "          if prices[j] >= tp:  labels[i] = 1; break\n"
    "          if prices[j] <= sl:  labels[i] = 0; break\n\n"
    "Bars that never hit either barrier are set to NaN and dropped from training. "
    "This is vectorized in the actual code for speed using NumPy rolling operations."
)

pdf.question("Why use Numba JIT for the indicator functions in the dashboard?")
pdf.answer(
    "Pure Python loops over large arrays are very slow. NumPy is fast for vectorized operations "
    "but some indicators like ATR and SuperTrend require sequential logic where each bar depends "
    "on the previous bar's value. NumPy cannot vectorize sequential dependencies easily.\n\n"
    "Numba compiles Python functions to native machine code the first time they are called. "
    "After that, they run at C speed. The decorator is simply:\n\n"
    "  @numba.jit(nopython=True)\n"
    "  def compute_atr(high, low, close, period):\n"
    "      ...\n\n"
    "This gives a 10 to 50x speedup over pure Python for sequential indicator calculations, "
    "which matters when you are recomputing 40+ indicators on 200 symbols every 5 seconds."
)

pdf.question("What is the difference between a PyTorch Dataset and DataLoader?")
pdf.answer(
    "Dataset defines what the data looks like and how to access a single item. You subclass "
    "torch.utils.data.Dataset and implement two methods:\n\n"
    "  __len__: returns the number of samples\n"
    "  __getitem__: returns one (X, y) pair given an index\n\n"
    "DataLoader wraps the Dataset and handles batching, shuffling, and parallel data loading. "
    "You pass it the Dataset and a batch_size:\n\n"
    "  loader = DataLoader(dataset, batch_size=64, shuffle=False)\n"
    "  for X_batch, y_batch in loader:\n"
    "      ...\n\n"
    "In this project shuffle=False is critical because financial time-series must not be "
    "shuffled. Shuffling would mix future data into training batches and cause severe leakage."
)

pdf.question("How does the LSTM receive input and what shape does it expect?")
pdf.answer(
    "PyTorch LSTM expects input shape (batch_size, sequence_length, input_features). "
    "In this project each sample is 60 days of 80+ features, so the shape is "
    "(batch_size, 60, 83) for example.\n\n"
    "  x = torch.tensor(X)  # shape: (N, 60, 83)\n"
    "  out, (h_n, c_n) = lstm(x)\n"
    "  # out shape: (N, 60, hidden * 2) because BiLSTM doubles hidden size\n"
    "  # h_n shape: (num_layers * 2, N, hidden)\n\n"
    "The BiLSTM processes the sequence both forward and backward, then the attention layer "
    "learns which of the 60 timesteps to weight most for the final prediction."
)

pdf.question("How do you handle class imbalance in the training data?")
pdf.answer(
    "The triple-barrier method naturally produces roughly balanced UP and DOWN labels because "
    "the take-profit and stop-loss barriers are set symmetrically around the entry price. "
    "However, in trending markets one direction can dominate.\n\n"
    "This project handles it two ways:\n\n"
    "  1. LightGBM uses class_weight='balanced', which automatically scales the loss function "
    "to penalize errors on the minority class more heavily.\n\n"
    "  2. The LSTM uses Focal Loss, which down-weights easy examples and focuses training on "
    "the hard, misclassified examples regardless of which class they belong to. The focal "
    "parameter gamma controls how aggressively easy examples are down-weighted."
)

pdf.question("What is the purpose of Stochastic Weight Averaging in the LSTM?")
pdf.answer(
    "Standard SGD and Adam find a single sharp minimum in the loss landscape. Sharp minima "
    "tend to generalize poorly because small perturbations in the weights cause large changes "
    "in predictions.\n\n"
    "Stochastic Weight Averaging (SWA) averages the weights from multiple checkpoints taken "
    "at different points during training. The average typically lands in a flatter, wider "
    "minimum that generalizes better to unseen data.\n\n"
    "In PyTorch it is implemented as:\n\n"
    "  swa_model = torch.optim.swa_utils.AveragedModel(model)\n"
    "  swa_scheduler = SWALR(optimizer, swa_lr=0.01)\n"
    "  # After warmup epochs:\n"
    "  swa_model.update_parameters(model)\n"
    "  swa_scheduler.step()\n\n"
    "The final model used for inference is the averaged SWA model, not the last checkpoint."
)

pdf.question("What is Test-Time Augmentation and how is it applied here?")
pdf.answer(
    "Test-Time Augmentation (TTA) runs multiple slightly different versions of the same input "
    "through the model at inference time and averages the predictions. This reduces prediction "
    "variance and typically improves accuracy by 1 to 3 percent.\n\n"
    "For financial time-series, each of the 8 augmentation passes applies small random noise "
    "to the input features:\n\n"
    "  preds = []\n"
    "  for _ in range(8):\n"
    "      noise = torch.randn_like(x) * 0.01\n"
    "      preds.append(model(x + noise))\n"
    "  final_pred = torch.stack(preds).mean(dim=0)\n\n"
    "The noise magnitude is kept very small (0.01 standard deviations) so it reflects "
    "realistic measurement uncertainty without distorting the signal."
)

pdf.question("How does the WebSocket feed work in the FastAPI dashboard?")
pdf.answer(
    "FastAPI supports WebSockets natively. The server maintains a list of active connections "
    "and a background task pushes updates to all of them every 2 seconds.\n\n"
    "  @app.websocket('/ws')\n"
    "  async def websocket_endpoint(ws: WebSocket):\n"
    "      await ws.accept()\n"
    "      active_connections.append(ws)\n"
    "      try:\n"
    "          while True:\n"
    "              await ws.receive_text()  # keep alive\n"
    "      except WebSocketDisconnect:\n"
    "          active_connections.remove(ws)\n\n"
    "  async def broadcast_loop():\n"
    "      while True:\n"
    "          data = get_latest_prices_and_signals()\n"
    "          for ws in active_connections:\n"
    "              await ws.send_json(data)\n"
    "          await asyncio.sleep(2)\n\n"
    "The browser connects once on page load. All subsequent price and signal updates "
    "are pushed by the server without the browser polling."
)

pdf.question("How does JWT authentication work in code?")
pdf.answer(
    "On login, the server verifies the password hash and creates a JWT token:\n\n"
    "  from jose import jwt\n"
    "  payload = {'sub': username, 'role': role, 'exp': datetime.utcnow() + timedelta(hours=8)}\n"
    "  token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')\n\n"
    "The client stores the token and sends it with every request:\n\n"
    "  Authorization: Bearer <token>\n\n"
    "The server decodes and verifies it on each protected endpoint:\n\n"
    "  payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n"
    "  username = payload['sub']\n"
    "  role = payload['role']\n\n"
    "Passwords are never stored as plaintext. bcrypt hashes them:\n\n"
    "  hashed = bcrypt.hash(plain_password)\n"
    "  bcrypt.verify(plain_password, hashed)  # returns True/False"
)

pdf.question("How is the Kelly fraction computed in code?")
pdf.answer(
    "The Kelly criterion formula is: f = (p * b - q) / b\n"
    "where p = win probability, q = 1 - p, b = win/loss ratio (risk/reward).\n\n"
    "  def kelly_fraction(win_prob, rr_ratio, fraction=0.5):\n"
    "      loss_prob = 1 - win_prob\n"
    "      kelly = (win_prob * rr_ratio - loss_prob) / rr_ratio\n"
    "      kelly = max(0.0, kelly)  # never bet negative\n"
    "      return kelly * fraction  # half-Kelly for safety\n\n"
    "For example, if the model has 70% win rate and a 2:1 risk/reward ratio:\n"
    "  kelly = (0.7 * 2 - 0.3) / 2 = 0.55  -> full Kelly = 55% of capital\n"
    "  half_kelly = 0.275  -> bet 27.5% of capital\n\n"
    "The conviction score from 0 to 100 further scales this down for lower-confidence signals."
)

pdf.question("What does winsorization do and how is it implemented?")
pdf.answer(
    "Winsorization clips extreme values to a percentile boundary rather than removing them. "
    "It prevents outliers from dominating the feature distribution and destabilizing model training.\n\n"
    "  from scipy.stats import mstats\n"
    "  # Clip values below 1st percentile and above 99th percentile\n"
    "  X_winsorized = mstats.winsorize(X, limits=[0.01, 0.01])\n\n"
    "Or with NumPy:\n\n"
    "  lower = np.percentile(X, 1)\n"
    "  upper = np.percentile(X, 99)\n"
    "  X_clipped = np.clip(X, lower, upper)\n\n"
    "Critically, percentiles are computed on the training set only and then applied to both "
    "train and test sets. Computing on the full dataset would cause leakage."
)

# Round 7
pdf.add_page()
pdf.chapter_title("Round 7  -  Debugging and Edge Cases")

pdf.question("Your LSTM loss is NaN from epoch 1. What do you check?")
pdf.answer(
    "This happened in this project and the root cause was a PyTorch MPS (Apple Silicon) bug. "
    "The systematic debugging steps are:\n\n"
    "  1. Check for NaN or Inf in the input features. A single NaN propagates through the "
    "entire batch and produces NaN loss. Fix: assert not np.any(np.isnan(X))\n\n"
    "  2. Check learning rate. If too high, gradients explode immediately. Fix: reduce lr or "
    "add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n\n"
    "  3. Check the loss function on the specific device. In this project, "
    "F.binary_cross_entropy computed on Apple MPS returns garbage values (-28 million, inf). "
    "Fix: move tensors to CPU before computing BCE: loss = criterion(pred.cpu(), y.cpu())\n\n"
    "  4. Check LSTM built-in dropout on MPS. PyTorch's dropout=0.3 argument inside LSTM "
    "layers causes NaN on MPS. Fix: set dropout=0 inside LSTM and add explicit nn.Dropout "
    "layers after the LSTM output.\n\n"
    "  5. Check weight initialization. Very large initial weights cause immediate gradient "
    "explosion. Fix: use conservative gain values in Xavier/Kaiming init."
)

pdf.question("Your model gets 95% training accuracy but 50% test accuracy. What is wrong?")
pdf.answer(
    "This is classic overfitting or data leakage. Check in this order:\n\n"
    "  1. Data leakage: are future values accidentally included as features? Check every "
    "feature computation. Rolling means must use min_periods and closed='left' to avoid "
    "including the current bar in its own feature.\n\n"
    "  2. Label leakage: are labels computed before the train/test split? The triple-barrier "
    "labels look forward in time. If you compute them on the full dataset before splitting, "
    "test-set label information leaks into training.\n\n"
    "  3. Overfitting: 5 million LSTM parameters trained on 300 samples will memorize the "
    "training data perfectly. Fix: add dropout (0.25-0.5), weight decay (1e-3), reduce model "
    "capacity, and use early stopping on validation loss.\n\n"
    "  4. Distribution shift: training covers 2018-2024 but test covers 2024-2025 which had "
    "very different volatility and correlations. This is common in financial ML and is why "
    "the model must be retrained periodically."
)

pdf.question("The LightGBM model always predicts the same class. What is wrong?")
pdf.answer(
    "This is a class collapse problem. Possible causes:\n\n"
    "  1. Labels are severely imbalanced. If 90% of labels are UP, the model learns that "
    "always predicting UP minimizes loss. Fix: use class_weight='balanced' in LGBMClassifier.\n\n"
    "  2. The regime subset has only one class. If you filter for high-volatility regime "
    "bars and all of them happened to be UP in that period, the regime model has nothing to "
    "learn. Fix: check class distribution before training each regime model and skip if only "
    "one class is present.\n\n"
    "  3. The learning rate is too high and the model collapses to the majority class in the "
    "first few boosting rounds. Fix: reduce learning_rate and increase n_estimators to compensate.\n\n"
    "  4. Features are all zero or constant after preprocessing. Check for columns with zero "
    "variance before training."
)

pdf.question("The WebSocket disconnects every 30 seconds. What is causing it?")
pdf.answer(
    "Most load balancers and reverse proxies (Nginx, AWS ALB) close idle WebSocket connections "
    "after a timeout, typically 30 to 60 seconds. The connection looks idle because the server "
    "only sends data but never receives anything from the client.\n\n"
    "Fix: implement a ping/pong heartbeat. The client sends a ping message every 20 seconds "
    "and the server responds with pong. This keeps the connection alive.\n\n"
    "  # Client side (JavaScript)\n"
    "  setInterval(() => ws.send('ping'), 20000)\n\n"
    "  # Server side (Python)\n"
    "  msg = await ws.receive_text()\n"
    "  if msg == 'ping':\n"
    "      await ws.send_text('pong')\n\n"
    "Alternatively, configure the reverse proxy to set proxy_read_timeout to a longer value, "
    "or enable WebSocket-specific keepalive settings."
)

pdf.question("Training is using only 1 CPU core and is very slow. How do you speed it up?")
pdf.answer(
    "Several approaches depending on the bottleneck:\n\n"
    "  1. LightGBM: set n_jobs=-1 to use all CPU cores. This is already done in the BASE_PARAMS "
    "in this project. LightGBM parallelizes both tree building and feature evaluation.\n\n"
    "  2. PyTorch DataLoader: set num_workers=4 or num_workers=8 to load data in parallel "
    "background processes while the GPU is running forward/backward passes.\n\n"
    "  3. Feature engineering: use pandas with vectorized operations, never loop row by row. "
    "For sequential indicators, use Numba JIT as this project does.\n\n"
    "  4. Apple Silicon (MPS): set device='mps' in PyTorch to run on the GPU cores. "
    "This project already does this with automatic fallback to CPU if MPS is unavailable.\n\n"
    "  5. Run symbols in parallel: if training 150 symbols independently, you can run multiple "
    "symbols simultaneously using Python multiprocessing or concurrent.futures.ProcessPoolExecutor."
)

pdf.question("How would you detect if a trained model has gone stale in production?")
pdf.answer(
    "A model goes stale when the market regime changes and its predictions no longer reflect "
    "reality. You detect this by monitoring these signals:\n\n"
    "  1. Prediction distribution drift: track the mean and standard deviation of the model's "
    "output probabilities. If the distribution shifts significantly from its training-time "
    "distribution, the model is seeing inputs it was not trained on.\n\n"
    "  2. Rolling accuracy on paper trades: compute a 30-day rolling win rate. If it drops "
    "significantly below the training hit rate, trigger retraining.\n\n"
    "  3. Feature distribution drift: use statistical tests like the Kolmogorov-Smirnov test "
    "to compare the current feature distribution against the training distribution. A large "
    "KS statistic on key features like volatility or trend strength signals regime change.\n\n"
    "  4. Model confidence collapse: if the model starts outputting probabilities all near 0.5, "
    "it has lost confidence in its predictions. This is tracked in this project by checking "
    "whether the LSTM output standard deviation drops below 0.05."
)

pdf.question("How do you make the FastAPI endpoints non-blocking for slow operations like model inference?")
pdf.answer(
    "FastAPI is built on asyncio. Blocking calls like model inference inside an async endpoint "
    "will block the entire event loop and freeze all other requests.\n\n"
    "Fix: run blocking CPU-bound work in a thread pool using asyncio.run_in_executor:\n\n"
    "  import asyncio\n"
    "  from concurrent.futures import ThreadPoolExecutor\n\n"
    "  executor = ThreadPoolExecutor(max_workers=4)\n\n"
    "  @app.get('/signal/{symbol}')\n"
    "  async def get_signal(symbol: str):\n"
    "      loop = asyncio.get_event_loop()\n"
    "      result = await loop.run_in_executor(executor, run_inference, symbol)\n"
    "      return result\n\n"
    "For truly CPU-intensive model inference at scale, move it to a separate inference "
    "service (a separate process or microservice) and communicate via an async message queue "
    "or HTTP so the FastAPI server itself stays lightweight and non-blocking."
)

pdf.question("How would you write a unit test for the triple-barrier label function?")
pdf.answer(
    "You create synthetic price series with known outcomes and verify the labels are correct:\n\n"
    "  def test_triple_barrier_take_profit():\n"
    "      # Price rises steadily, should hit take-profit\n"
    "      prices = pd.Series([100, 101, 102, 103, 104, 106])\n"
    "      atr = pd.Series([1.0] * 6)\n"
    "      labels = triple_barrier(prices, atr, tp_mult=2.5, sl_mult=2.0, max_hold=5)\n"
    "      assert labels[0] == 1  # UP: price rose 6 points > 2.5 ATR\n\n"
    "  def test_triple_barrier_stop_loss():\n"
    "      # Price drops sharply, should hit stop-loss\n"
    "      prices = pd.Series([100, 99, 98, 97, 95, 94])\n"
    "      atr = pd.Series([1.0] * 6)\n"
    "      labels = triple_barrier(prices, atr, tp_mult=2.5, sl_mult=2.0, max_hold=5)\n"
    "      assert labels[0] == 0  # DOWN: price dropped 6 points > 2.0 ATR\n\n"
    "  def test_triple_barrier_timeout():\n"
    "      # Price barely moves, should time out and be NaN\n"
    "      prices = pd.Series([100, 100.1, 99.9, 100.1, 99.9, 100.0])\n"
    "      atr = pd.Series([2.0] * 6)\n"
    "      labels = triple_barrier(prices, atr, tp_mult=2.5, sl_mult=2.0, max_hold=5)\n"
    "      assert np.isnan(labels[0])  # no barrier hit"
)

# ── Output ──────────────────────────────────────────────────────────────────
output_path = "AlphaGrid_Learning_Guide.pdf"
pdf.output(output_path)
print(f"PDF saved: {output_path}")
