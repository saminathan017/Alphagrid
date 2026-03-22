"""
core/ticker_universe.py
Top 150 US equities + 50 Forex pairs.
All fetched via yfinance (completely free, no API key required).
Symbols pre-classified by sector, market cap tier, and volatility bucket.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Sector(str, Enum):
    TECH       = "Technology"
    FINANCE    = "Financial"
    HEALTH     = "Healthcare"
    CONSUMER   = "Consumer"
    ENERGY     = "Energy"
    INDUSTRIAL = "Industrial"
    COMMS      = "Communication"
    REALESTATE = "Real Estate"
    UTILITIES  = "Utilities"
    GROWTH     = "Growth"
    ETF        = "ETF"
    FOREX      = "Forex"


class CapTier(str, Enum):
    MEGA  = "mega"    # >$500B
    LARGE = "large"   # $50B–$500B
    MID   = "mid"     # $10B–$50B
    SMALL = "small"   # <$10B


@dataclass
class TickerMeta:
    symbol:   str
    name:     str
    sector:   Sector
    cap_tier: CapTier
    avg_volume_m: float   # avg daily volume in millions
    adr_pct:  float       # avg daily range %
    day_tradeable: bool   # high enough volume/volatility for day trading
    swing_tradeable: bool


# ── 150 US Equities ───────────────────────────────────────────────────────────
US_EQUITIES: list[TickerMeta] = [
    # Mega-cap Tech
    TickerMeta("AAPL",  "Apple",          Sector.TECH,    CapTier.MEGA,  80.0, 1.5, True,  True),
    TickerMeta("MSFT",  "Microsoft",      Sector.TECH,    CapTier.MEGA,  25.0, 1.3, True,  True),
    TickerMeta("NVDA",  "NVIDIA",         Sector.TECH,    CapTier.MEGA,  50.0, 3.5, True,  True),
    TickerMeta("GOOGL", "Alphabet A",     Sector.TECH,    CapTier.MEGA,  25.0, 1.4, True,  True),
    TickerMeta("META",  "Meta",           Sector.TECH,    CapTier.MEGA,  18.0, 2.1, True,  True),
    TickerMeta("AMZN",  "Amazon",         Sector.CONSUMER,CapTier.MEGA,  40.0, 1.8, True,  True),
    TickerMeta("TSLA",  "Tesla",          Sector.CONSUMER,CapTier.MEGA, 120.0, 4.5, True,  True),
    TickerMeta("AVGO",  "Broadcom",       Sector.TECH,    CapTier.MEGA,   8.0, 2.0, True,  True),
    TickerMeta("ORCL",  "Oracle",         Sector.TECH,    CapTier.LARGE,  8.0, 1.6, True,  True),
    TickerMeta("AMD",   "AMD",            Sector.TECH,    CapTier.LARGE, 55.0, 3.8, True,  True),
    # Semiconductor
    TickerMeta("INTC",  "Intel",          Sector.TECH,    CapTier.LARGE, 35.0, 2.5, True,  True),
    TickerMeta("QCOM",  "Qualcomm",       Sector.TECH,    CapTier.LARGE, 10.0, 2.0, True,  True),
    TickerMeta("TXN",   "Texas Instr.",   Sector.TECH,    CapTier.LARGE,  5.0, 1.4, True,  True),
    TickerMeta("MU",    "Micron",         Sector.TECH,    CapTier.LARGE, 25.0, 3.2, True,  True),
    TickerMeta("AMAT",  "Appl. Materials",Sector.TECH,    CapTier.LARGE,  8.0, 2.5, True,  True),
    TickerMeta("LRCX",  "Lam Research",   Sector.TECH,    CapTier.LARGE,  3.0, 2.3, True,  True),
    TickerMeta("KLAC",  "KLA Corp",       Sector.TECH,    CapTier.LARGE,  2.0, 2.2, True,  True),
    TickerMeta("MRVL",  "Marvell Tech",   Sector.TECH,    CapTier.LARGE, 15.0, 3.0, True,  True),
    TickerMeta("ARM",   "ARM Holdings",   Sector.TECH,    CapTier.LARGE, 20.0, 4.0, True,  True),
    TickerMeta("SMCI",  "Super Micro",    Sector.TECH,    CapTier.MID,   30.0, 6.0, True,  True),
    # Financial
    TickerMeta("JPM",   "JPMorgan",       Sector.FINANCE, CapTier.MEGA,  10.0, 1.2, True,  True),
    TickerMeta("BAC",   "Bank of America",Sector.FINANCE, CapTier.LARGE, 45.0, 1.5, True,  True),
    TickerMeta("WFC",   "Wells Fargo",    Sector.FINANCE, CapTier.LARGE, 18.0, 1.4, True,  True),
    TickerMeta("GS",    "Goldman Sachs",  Sector.FINANCE, CapTier.LARGE,  3.0, 1.5, True,  True),
    TickerMeta("MS",    "Morgan Stanley", Sector.FINANCE, CapTier.LARGE,  8.0, 1.6, True,  True),
    TickerMeta("BLK",   "BlackRock",      Sector.FINANCE, CapTier.LARGE,  1.0, 1.3, False, True),
    TickerMeta("V",     "Visa",           Sector.FINANCE, CapTier.MEGA,   5.0, 1.1, True,  True),
    TickerMeta("MA",    "Mastercard",     Sector.FINANCE, CapTier.MEGA,   3.0, 1.2, True,  True),
    TickerMeta("PYPL",  "PayPal",         Sector.FINANCE, CapTier.LARGE, 12.0, 2.5, True,  True),
    TickerMeta("SCHW",  "Schwab",         Sector.FINANCE, CapTier.LARGE, 12.0, 1.8, True,  True),
    # Healthcare
    TickerMeta("LLY",   "Eli Lilly",      Sector.HEALTH,  CapTier.MEGA,   4.0, 2.0, True,  True),
    TickerMeta("UNH",   "UnitedHealth",   Sector.HEALTH,  CapTier.MEGA,   3.0, 1.5, True,  True),
    TickerMeta("JNJ",   "Johnson & J.",   Sector.HEALTH,  CapTier.MEGA,   7.0, 0.9, False, True),
    TickerMeta("ABBV",  "AbbVie",         Sector.HEALTH,  CapTier.LARGE,  6.0, 1.2, True,  True),
    TickerMeta("MRK",   "Merck",          Sector.HEALTH,  CapTier.LARGE,  8.0, 1.2, True,  True),
    TickerMeta("TMO",   "Thermo Fisher",  Sector.HEALTH,  CapTier.LARGE,  2.0, 1.4, False, True),
    TickerMeta("ABT",   "Abbott Labs",    Sector.HEALTH,  CapTier.LARGE,  5.0, 1.1, True,  True),
    TickerMeta("PFE",   "Pfizer",         Sector.HEALTH,  CapTier.LARGE, 40.0, 1.8, True,  True),
    TickerMeta("AMGN",  "Amgen",          Sector.HEALTH,  CapTier.LARGE,  3.0, 1.3, True,  True),
    TickerMeta("GILD",  "Gilead",         Sector.HEALTH,  CapTier.LARGE,  8.0, 1.4, True,  True),
    # Consumer
    TickerMeta("HD",    "Home Depot",     Sector.CONSUMER,CapTier.LARGE,  4.0, 1.2, True,  True),
    TickerMeta("MCD",   "McDonald's",     Sector.CONSUMER,CapTier.LARGE,  3.0, 0.9, False, True),
    TickerMeta("NKE",   "Nike",           Sector.CONSUMER,CapTier.LARGE,  8.0, 1.8, True,  True),
    TickerMeta("SBUX",  "Starbucks",      Sector.CONSUMER,CapTier.LARGE,  8.0, 1.5, True,  True),
    TickerMeta("TGT",   "Target",         Sector.CONSUMER,CapTier.LARGE,  5.0, 2.0, True,  True),
    TickerMeta("COST",  "Costco",         Sector.CONSUMER,CapTier.LARGE,  3.0, 1.1, False, True),
    TickerMeta("WMT",   "Walmart",        Sector.CONSUMER,CapTier.MEGA,   8.0, 1.0, True,  True),
    TickerMeta("BKNG",  "Booking",        Sector.CONSUMER,CapTier.LARGE,  1.0, 1.8, False, True),
    TickerMeta("UBER",  "Uber",           Sector.CONSUMER,CapTier.LARGE, 30.0, 2.8, True,  True),
    TickerMeta("LYFT",  "Lyft",           Sector.CONSUMER,CapTier.MID,   15.0, 4.0, True,  True),
    # Energy
    TickerMeta("XOM",   "ExxonMobil",     Sector.ENERGY,  CapTier.MEGA,  15.0, 1.4, True,  True),
    TickerMeta("CVX",   "Chevron",        Sector.ENERGY,  CapTier.MEGA,   8.0, 1.3, True,  True),
    TickerMeta("COP",   "ConocoPhillips", Sector.ENERGY,  CapTier.LARGE,  8.0, 1.8, True,  True),
    TickerMeta("SLB",   "SLB",            Sector.ENERGY,  CapTier.LARGE, 12.0, 2.2, True,  True),
    TickerMeta("EOG",   "EOG Resources",  Sector.ENERGY,  CapTier.LARGE,  5.0, 1.9, True,  True),
    # Industrial
    TickerMeta("CAT",   "Caterpillar",    Sector.INDUSTRIAL,CapTier.LARGE,3.0, 1.5, True,  True),
    TickerMeta("DE",    "Deere & Co.",    Sector.INDUSTRIAL,CapTier.LARGE,2.0, 1.5, False, True),
    TickerMeta("HON",   "Honeywell",      Sector.INDUSTRIAL,CapTier.LARGE,3.0, 1.1, False, True),
    TickerMeta("UPS",   "UPS",            Sector.INDUSTRIAL,CapTier.LARGE,4.0, 1.3, True,  True),
    TickerMeta("FDX",   "FedEx",          Sector.INDUSTRIAL,CapTier.LARGE,3.0, 1.8, True,  True),
    TickerMeta("BA",    "Boeing",         Sector.INDUSTRIAL,CapTier.LARGE,8.0, 2.0, True,  True),
    TickerMeta("GE",    "GE Aerospace",   Sector.INDUSTRIAL,CapTier.LARGE,12.0,1.8, True,  True),
    # Communication
    TickerMeta("NFLX",  "Netflix",        Sector.COMMS,   CapTier.LARGE,  8.0, 2.5, True,  True),
    TickerMeta("DIS",   "Disney",         Sector.COMMS,   CapTier.LARGE, 12.0, 1.6, True,  True),
    TickerMeta("CMCSA", "Comcast",        Sector.COMMS,   CapTier.LARGE, 20.0, 1.2, True,  True),
    TickerMeta("T",     "AT&T",           Sector.COMMS,   CapTier.LARGE, 50.0, 1.3, True,  True),
    TickerMeta("VZ",    "Verizon",        Sector.COMMS,   CapTier.LARGE, 25.0, 1.0, False, True),
    # Growth / High-Beta
    TickerMeta("CRWD",  "CrowdStrike",    Sector.GROWTH,  CapTier.LARGE, 10.0, 3.5, True,  True),
    TickerMeta("SNOW",  "Snowflake",      Sector.GROWTH,  CapTier.LARGE,  8.0, 4.0, True,  True),
    TickerMeta("DDOG",  "Datadog",        Sector.GROWTH,  CapTier.LARGE,  5.0, 3.8, True,  True),
    TickerMeta("NET",   "Cloudflare",     Sector.GROWTH,  CapTier.LARGE,  8.0, 3.5, True,  True),
    TickerMeta("PLTR",  "Palantir",       Sector.GROWTH,  CapTier.LARGE, 60.0, 4.5, True,  True),
    TickerMeta("COIN",  "Coinbase",       Sector.GROWTH,  CapTier.MID,   20.0, 7.0, True,  True),
    TickerMeta("MSTR",  "MicroStrategy",  Sector.GROWTH,  CapTier.MID,   10.0,10.0, True,  True),
    TickerMeta("RBLX",  "Roblox",         Sector.GROWTH,  CapTier.MID,   15.0, 4.0, True,  True),
    TickerMeta("SHOP",  "Shopify",        Sector.GROWTH,  CapTier.LARGE, 10.0, 3.5, True,  True),
    TickerMeta("ZS",    "Zscaler",        Sector.GROWTH,  CapTier.LARGE,  5.0, 4.0, True,  True),
    # Industrial (expanded)
    TickerMeta("RTX",   "RTX Corp",       Sector.INDUSTRIAL,CapTier.LARGE,5.0, 1.3, True,  True),
    TickerMeta("LMT",   "Lockheed Martin",Sector.INDUSTRIAL,CapTier.LARGE,1.5, 1.2, False, True),
    # Consumer Staples
    TickerMeta("PG",    "Procter & Gamble",Sector.CONSUMER,CapTier.MEGA,  6.0, 0.8, False, True),
    TickerMeta("KO",    "Coca-Cola",      Sector.CONSUMER,CapTier.MEGA,  14.0, 0.8, False, True),
    TickerMeta("PEP",   "PepsiCo",        Sector.CONSUMER,CapTier.MEGA,   6.0, 0.9, False, True),
    # Financial (expanded)
    TickerMeta("C",     "Citigroup",      Sector.FINANCE, CapTier.LARGE, 15.0, 1.6, True,  True),
    TickerMeta("HOOD",  "Robinhood",      Sector.FINANCE, CapTier.MID,   20.0, 5.0, True,  True),
    TickerMeta("SOFI",  "SoFi Tech",      Sector.FINANCE, CapTier.MID,   30.0, 4.5, True,  True),
    # Semiconductor (expanded)
    TickerMeta("ADI",   "Analog Devices", Sector.TECH,    CapTier.LARGE,  4.0, 1.8, True,  True),
    TickerMeta("ASML",  "ASML Holding",   Sector.TECH,    CapTier.MEGA,   2.5, 2.2, True,  True),
    TickerMeta("TSM",   "Taiwan Semi",    Sector.TECH,    CapTier.MEGA,  15.0, 2.0, True,  True),
    # Growth (expanded)
    TickerMeta("AFRM",  "Affirm",         Sector.GROWTH,  CapTier.MID,   25.0, 6.0, True,  True),
    TickerMeta("HIMS",  "Hims & Hers",    Sector.GROWTH,  CapTier.MID,   18.0, 6.5, True,  True),
    # ETFs
    TickerMeta("SPY",   "S&P 500 ETF",    Sector.ETF,     CapTier.MEGA,  80.0, 0.8, True,  True),
    TickerMeta("QQQ",   "Nasdaq 100 ETF", Sector.ETF,     CapTier.MEGA,  50.0, 1.2, True,  True),
    TickerMeta("IWM",   "Russell 2000",   Sector.ETF,     CapTier.LARGE, 30.0, 1.4, True,  True),
    TickerMeta("GLD",   "Gold ETF",       Sector.ETF,     CapTier.LARGE,  8.0, 0.7, True,  True),
    TickerMeta("TLT",   "20Y Treasury",   Sector.ETF,     CapTier.LARGE, 20.0, 1.0, True,  True),
    TickerMeta("XLK",   "Tech Sector",    Sector.ETF,     CapTier.LARGE,  8.0, 1.2, True,  True),
    TickerMeta("XLF",   "Financial Sect.",Sector.ETF,     CapTier.LARGE, 40.0, 1.0, True,  True),
    TickerMeta("SOXS",  "Semi Bear 3x",   Sector.ETF,     CapTier.MID,   20.0, 8.0, True,  False),
    TickerMeta("SOXL",  "Semi Bull 3x",   Sector.ETF,     CapTier.MID,   25.0, 8.0, True,  False),
    TickerMeta("TQQQ",  "Nasdaq Bull 3x", Sector.ETF,     CapTier.MID,   60.0, 3.5, True,  False),
]

# ── 50 Forex Pairs (yfinance =X suffix) ───────────────────────────────────────
FOREX_PAIRS: list[TickerMeta] = [
    # 8 Majors
    TickerMeta("EURUSD=X","EUR/USD",       Sector.FOREX, CapTier.MEGA,  0,1.0, True, True),
    TickerMeta("GBPUSD=X","GBP/USD",       Sector.FOREX, CapTier.MEGA,  0,1.2, True, True),
    TickerMeta("USDJPY=X","USD/JPY",       Sector.FOREX, CapTier.MEGA,  0,0.7, True, True),
    TickerMeta("USDCHF=X","USD/CHF",       Sector.FOREX, CapTier.LARGE, 0,0.8, True, True),
    TickerMeta("AUDUSD=X","AUD/USD",       Sector.FOREX, CapTier.LARGE, 0,0.8, True, True),
    TickerMeta("USDCAD=X","USD/CAD",       Sector.FOREX, CapTier.LARGE, 0,0.7, True, True),
    TickerMeta("NZDUSD=X","NZD/USD",       Sector.FOREX, CapTier.LARGE, 0,0.9, True, True),
    TickerMeta("USDMXN=X","USD/MXN",       Sector.FOREX, CapTier.LARGE, 0,1.5, True, True),
    # 14 Minors
    TickerMeta("EURGBP=X","EUR/GBP",       Sector.FOREX, CapTier.LARGE, 0,0.6, True, True),
    TickerMeta("EURJPY=X","EUR/JPY",       Sector.FOREX, CapTier.LARGE, 0,1.0, True, True),
    TickerMeta("GBPJPY=X","GBP/JPY",       Sector.FOREX, CapTier.LARGE, 0,1.3, True, True),
    TickerMeta("EURCHF=X","EUR/CHF",       Sector.FOREX, CapTier.MID,   0,0.5, True, True),
    TickerMeta("EURAUD=X","EUR/AUD",       Sector.FOREX, CapTier.MID,   0,1.0, False,True),
    TickerMeta("EURCAD=X","EUR/CAD",       Sector.FOREX, CapTier.MID,   0,0.8, False,True),
    TickerMeta("GBPCHF=X","GBP/CHF",       Sector.FOREX, CapTier.MID,   0,1.0, False,True),
    TickerMeta("GBPAUD=X","GBP/AUD",       Sector.FOREX, CapTier.MID,   0,1.2, False,True),
    TickerMeta("AUDJPY=X","AUD/JPY",       Sector.FOREX, CapTier.MID,   0,1.0, True, True),
    TickerMeta("AUDCHF=X","AUD/CHF",       Sector.FOREX, CapTier.SMALL, 0,0.8, False,True),
    TickerMeta("NZDJPY=X","NZD/JPY",       Sector.FOREX, CapTier.SMALL, 0,1.0, False,True),
    TickerMeta("CADJPY=X","CAD/JPY",       Sector.FOREX, CapTier.SMALL, 0,0.9, False,True),
    TickerMeta("CHFJPY=X","CHF/JPY",       Sector.FOREX, CapTier.SMALL, 0,0.9, False,True),
    TickerMeta("NZDCAD=X","NZD/CAD",       Sector.FOREX, CapTier.SMALL, 0,0.7, False,True),
    # Metals
    TickerMeta("XAUUSD=X","Gold/USD",      Sector.FOREX, CapTier.LARGE, 0,1.0, True, True),
    TickerMeta("XAGUSD=X","Silver/USD",    Sector.FOREX, CapTier.MID,   0,2.0, True, True),
    # More Minors
    TickerMeta("GBPCAD=X","GBP/CAD",       Sector.FOREX, CapTier.MID,   0,1.1, False,True),
    TickerMeta("AUDCAD=X","AUD/CAD",       Sector.FOREX, CapTier.SMALL, 0,0.8, False,True),
    TickerMeta("AUDNZD=X","AUD/NZD",       Sector.FOREX, CapTier.SMALL, 0,0.7, False,True),
    TickerMeta("EURNZD=X","EUR/NZD",       Sector.FOREX, CapTier.SMALL, 0,1.1, False,True),
    TickerMeta("NZDCHF=X","NZD/CHF",       Sector.FOREX, CapTier.SMALL, 0,0.8, False,True),
    TickerMeta("CADCHF=X","CAD/CHF",       Sector.FOREX, CapTier.SMALL, 0,0.7, False,True),
    # Precious Metals
    TickerMeta("XPTUSD=X","Platinum/USD",  Sector.FOREX, CapTier.SMALL, 0,1.5, False,True),
    TickerMeta("XPDUSD=X","Palladium/USD", Sector.FOREX, CapTier.SMALL, 0,2.0, False,True),
    # EM / Exotics
    TickerMeta("USDZAR=X","USD/ZAR",       Sector.FOREX, CapTier.MID,   0,2.0, False,True),
    TickerMeta("USDTRY=X","USD/TRY",       Sector.FOREX, CapTier.MID,   0,3.0, False,True),
    TickerMeta("USDBRL=X","USD/BRL",       Sector.FOREX, CapTier.MID,   0,1.5, False,True),
    TickerMeta("USDSGD=X","USD/SGD",       Sector.FOREX, CapTier.MID,   0,0.5, False,True),
    TickerMeta("USDINR=X","USD/INR",       Sector.FOREX, CapTier.MID,   0,0.4, False,True),
    TickerMeta("USDNOK=X","USD/NOK",       Sector.FOREX, CapTier.MID,   0,1.2, False,True),
    TickerMeta("USDSEK=X","USD/SEK",       Sector.FOREX, CapTier.MID,   0,1.0, False,True),
    TickerMeta("USDPLN=X","USD/PLN",       Sector.FOREX, CapTier.SMALL, 0,1.5, False,True),
    TickerMeta("USDTHB=X","USD/THB",       Sector.FOREX, CapTier.SMALL, 0,0.5, False,True),
    TickerMeta("USDHKD=X","USD/HKD",       Sector.FOREX, CapTier.SMALL, 0,0.1, False,False),
    TickerMeta("USDDKK=X","USD/DKK",       Sector.FOREX, CapTier.SMALL, 0,0.7, False,True),
    TickerMeta("USDHUF=X","USD/HUF",       Sector.FOREX, CapTier.SMALL, 0,1.3, False,True),
    TickerMeta("USDCZK=X","USD/CZK",       Sector.FOREX, CapTier.SMALL, 0,1.2, False,True),
    TickerMeta("USDILS=X","USD/ILS",       Sector.FOREX, CapTier.SMALL, 0,0.8, False,True),
    # Asian JPY crosses
    TickerMeta("NOKJPY=X","NOK/JPY",       Sector.FOREX, CapTier.SMALL, 0,1.0, False,True),
    TickerMeta("SEKJPY=X","SEK/JPY",       Sector.FOREX, CapTier.SMALL, 0,1.0, False,True),
    TickerMeta("ZARJPY=X","ZAR/JPY",       Sector.FOREX, CapTier.SMALL, 0,2.0, False,True),
    TickerMeta("DKKJPY=X","DKK/JPY",       Sector.FOREX, CapTier.SMALL, 0,0.9, False,True),
]

# ── Convenience accessors ─────────────────────────────────────────────────────
ALL_TICKERS       = US_EQUITIES + FOREX_PAIRS
US_SYMBOLS        = [t.symbol for t in US_EQUITIES]
FOREX_SYMBOLS     = [t.symbol for t in FOREX_PAIRS]
ALL_SYMBOLS       = US_SYMBOLS + FOREX_SYMBOLS
DAY_TRADE_SYMBOLS = [t.symbol for t in ALL_TICKERS if t.day_tradeable]
SWING_SYMBOLS     = [t.symbol for t in ALL_TICKERS if t.swing_tradeable]
TICKER_MAP        = {t.symbol: t for t in ALL_TICKERS}

# Sector buckets for rotation strategies
SECTOR_MAP: dict[str, list[str]] = {}
for t in US_EQUITIES:
    SECTOR_MAP.setdefault(t.sector.value, []).append(t.symbol)

HIGH_BETA = [t.symbol for t in US_EQUITIES if t.adr_pct >= 3.0]

print(f"Universe: {len(US_EQUITIES)} equities + {len(FOREX_PAIRS)} forex = {len(ALL_TICKERS)} total")
