import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# Optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(
    page_title="IDX Pro Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Configuration
# =========================
IDX_TICKERS = [
    "BBCA", "TLKM", "ASII", "BBRI", "BMRI", "ICBP", "UNVR", "INDF",
    "ANTM", "MDKA", "ADRO", "GOTO", "PGAS", "CPIN", "KLBF", "ACES",
    "AMRT", "BRIS", "EXCL", "MAPI", "AKRA", "ITMG", "TOWR", "SMGR",
]

WATCHLIST_DEFAULT = ["BBCA", "TLKM", "ASII", "BBRI", "BMRI", "ANTM"]
INDEX_MAP = {
    "IHSG": "^JKSE",
    "LQ45": "^JKLQ45",  # may not always resolve on Yahoo; handled gracefully
    "DJI": "^DJI",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
}

COMPANY_META = {
    "BBCA": {"name": "Bank Central Asia Tbk.", "sector": "Perbankan"},
    "TLKM": {"name": "Telkom Indonesia (Persero) Tbk.", "sector": "Telekomunikasi"},
    "ASII": {"name": "Astra International Tbk.", "sector": "Otomotif"},
    "BBRI": {"name": "Bank Rakyat Indonesia Tbk.", "sector": "Perbankan"},
    "BMRI": {"name": "Bank Mandiri (Persero) Tbk.", "sector": "Perbankan"},
    "ICBP": {"name": "Indofood CBP Sukses Makmur Tbk.", "sector": "Konsumsi"},
    "UNVR": {"name": "Unilever Indonesia Tbk.", "sector": "Konsumsi"},
    "INDF": {"name": "Indofood Sukses Makmur Tbk.", "sector": "Konsumsi"},
    "ANTM": {"name": "Aneka Tambang Tbk.", "sector": "Komoditas"},
    "MDKA": {"name": "Merdeka Copper Gold Tbk.", "sector": "Komoditas"},
    "ADRO": {"name": "Adaro Energy Indonesia Tbk.", "sector": "Energi"},
    "GOTO": {"name": "GoTo Gojek Tokopedia Tbk.", "sector": "Teknologi"},
    "PGAS": {"name": "Perusahaan Gas Negara Tbk.", "sector": "Energi"},
    "CPIN": {"name": "Charoen Pokphand Indonesia Tbk.", "sector": "Konsumsi"},
    "KLBF": {"name": "Kalbe Farma Tbk.", "sector": "Farmasi"},
    "ACES": {"name": "Aspirasi Hidup Indonesia Tbk.", "sector": "Ritel"},
    "AMRT": {"name": "Sumber Alfaria Trijaya Tbk.", "sector": "Ritel"},
    "BRIS": {"name": "Bank Syariah Indonesia Tbk.", "sector": "Perbankan"},
    "EXCL": {"name": "XL Axiata Tbk.", "sector": "Telekomunikasi"},
    "MAPI": {"name": "Mitra Adiperkasa Tbk.", "sector": "Ritel"},
    "AKRA": {"name": "AKR Corporindo Tbk.", "sector": "Distribusi"},
    "ITMG": {"name": "Indo Tambangraya Megah Tbk.", "sector": "Energi"},
    "TOWR": {"name": "Sarana Menara Nusantara Tbk.", "sector": "Infrastruktur"},
    "SMGR": {"name": "Semen Indonesia Tbk.", "sector": "Material Dasar"},
}

# =========================
# CSS
# =========================
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #0f1a2e;
        --panel-2: #101b31;
        --border: rgba(112, 160, 255, 0.18);
        --text: #e8eefc;
        --muted: #94a3b8;
        --green: #00ff9c;
        --red: #ff4d4f;
        --blue: #4ea1ff;
        --purple: #9b6dff;
        --orange: #ffad33;
    }
    .stApp {
        background: linear-gradient(180deg, #09111d 0%, #0b1220 100%);
        color: var(--text);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1rem;
        max-width: 1800px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07101c 0%, #09111d 100%);
        border-right: 1px solid var(--border);
    }
    .panel {
        background: linear-gradient(180deg, rgba(15,26,46,0.98) 0%, rgba(10,20,36,0.98) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 0 0 1px rgba(78,161,255,0.03), 0 10px 28px rgba(0,0,0,0.22);
    }
    .mini-panel {
        background: linear-gradient(180deg, rgba(12,20,34,0.95) 0%, rgba(10,16,28,0.95) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    }
    .metric-title { color: var(--muted); font-size: 0.78rem; }
    .metric-value { color: var(--text); font-size: 1.35rem; font-weight: 700; }
    .metric-change-up { color: var(--green); font-size: 0.86rem; font-weight: 600; }
    .metric-change-down { color: var(--red); font-size: 0.86rem; font-weight: 600; }
    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        color: var(--text);
    }
    .section-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
    }
    .selected-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(78, 161, 255, 0.08);
        color: #cfe0ff;
        font-size: 0.84rem;
        font-weight: 700;
        margin-top: 0.2rem;
        margin-bottom: 0.6rem;
    }
    .score-box {
        border: 1px solid rgba(0,255,156,0.35);
        background: rgba(0,255,156,0.06);
        border-radius: 14px;
        padding: 10px 12px;
        text-align: center;
    }
    .score-num { font-size: 1.8rem; font-weight: 800; color: var(--green); }
    .score-label { font-size: 0.78rem; color: #a5f3c7; }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .kpi-cell {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(112,160,255,0.12);
        border-radius: 12px;
        padding: 8px 10px;
    }
    .status-open, .status-closed {
        display:inline-block; border-radius:999px; padding:0.28rem 0.68rem; font-size:0.78rem; font-weight:700;
    }
    .status-open { background: rgba(0,255,156,0.12); color: var(--green); border:1px solid rgba(0,255,156,0.24); }
    .status-closed { background: rgba(255,77,79,0.12); color: var(--red); border:1px solid rgba(255,77,79,0.24); }
    .small-note { color: var(--muted); font-size: 0.78rem; }
    .news-item { padding: 0.45rem 0; border-bottom: 1px solid rgba(112,160,255,0.08); }
    .buy-chip, .wait-chip, .hold-chip, .sell-chip {
        display:inline-block; border-radius:10px; padding:0.35rem 0.7rem; font-size:0.83rem; font-weight:700;
    }
    .buy-chip { background: rgba(0,255,156,0.14); color: var(--green); }
    .wait-chip { background: rgba(255,173,51,0.12); color: var(--orange); }
    .hold-chip { background: rgba(78,161,255,0.12); color: var(--blue); }
    .sell-chip { background: rgba(255,77,79,0.12); color: var(--red); }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(112,160,255,0.18);
        background: linear-gradient(180deg, #11213b 0%, #0c1730 100%);
        color: #e8eefc;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover {
        border: 1px solid rgba(78,161,255,0.45);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
def normalize_ticker(ticker: str) -> str:
    return ticker if ticker.startswith("^") or ticker.endswith(".JK") else f"{ticker}.JK"


def fmt_num(x, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    try:
        return f"{x:,.{digits}f}"
    except Exception:
        return str(x)


def fmt_short(n):
    if n is None or pd.isna(n):
        return "-"
    n = float(n)
    abs_n = abs(n)
    if abs_n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.2f}T"
    if abs_n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if abs_n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs_n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.2f}"


def market_open_now() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    # Approximate Jakarta session window
    return 9 <= now.hour < 16


# =========================
# Data loading
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def load_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    yf_ticker = normalize_ticker(ticker)
    df = yf.download(yf_ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df = df[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]].copy()
    df.dropna(how="all", inplace=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_company_snapshot(ticker: str) -> dict:
    meta = COMPANY_META.get(ticker, {}).copy()
    try:
        tk = yf.Ticker(normalize_ticker(ticker))
        info = tk.fast_info if hasattr(tk, "fast_info") else {}
        meta["market_cap"] = info.get("market_cap")
        meta["currency"] = info.get("currency", "IDR")
        meta["last_price"] = info.get("last_price")
    except Exception:
        meta.setdefault("market_cap", np.nan)
        meta.setdefault("currency", "IDR")
    return meta


@st.cache_data(ttl=60, show_spinner=False)
def load_index_snapshot(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period="5d", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.rename(columns=str.title)
    except Exception:
        return pd.DataFrame()


# =========================
# Indicators
# =========================
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    vol_ma = df["Volume"].rolling(window).mean()
    return (df["Volume"] / vol_ma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).fillna(0).cumsum()


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / denom
    mfv = mfm.fillna(0) * df["Volume"]
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)


def calculate_ad_line(df: pd.DataFrame) -> pd.Series:
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / denom
    ad = (clv.fillna(0) * df["Volume"]).cumsum()
    return ad


def calculate_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0):
    ma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def calculate_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    lowest = df["Low"].rolling(period).min()
    highest = df["High"].rolling(period).max()
    stoch = 100 * (df["Close"] - lowest) / (highest - lowest).replace(0, np.nan)
    return stoch.fillna(50)


def calculate_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    return ((df["Close"] / df["Close"].shift(period)) - 1) * 100


def detect_accumulation(df: pd.DataFrame) -> str:
    if len(df) < 25:
        return "Neutral"
    cmf = calculate_cmf(df).iloc[-1]
    obv = calculate_obv(df)
    ad = calculate_ad_line(df)
    obv_up = obv.iloc[-1] > obv.iloc[-6] if len(obv) > 6 else False
    ad_up = ad.iloc[-1] > ad.iloc[-6] if len(ad) > 6 else False
    if cmf > 0.08 and obv_up and ad_up:
        return "Accumulation"
    if cmf < -0.08 and not obv_up and not ad_up:
        return "Distribution"
    return "Neutral"


def score_stock(df: pd.DataFrame) -> tuple[int, dict]:
    if len(df) < 50:
        return 40, {
            "rsi": 50, "macd_bull": False, "rvol": 1.0, "cmf": 0.0,
            "trend": "Weak", "accumulation": "Neutral", "signal": "WAIT"
        }

    close = df["Close"]
    rsi = calculate_rsi(df)
    macd, signal_line, hist = calculate_macd(df)
    rvol = calculate_rvol(df)
    obv = calculate_obv(df)
    cmf = calculate_cmf(df)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    bb_mid, bb_upper, bb_lower = calculate_bollinger(df)

    score = 0
    last_close = close.iloc[-1]
    last_rsi = float(rsi.iloc[-1])
    last_macd = float(macd.iloc[-1])
    last_signal = float(signal_line.iloc[-1])
    last_hist = float(hist.iloc[-1])
    last_rvol = float(rvol.iloc[-1])
    last_cmf = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
    accumulation = detect_accumulation(df)

    # RSI
    if 52 <= last_rsi <= 68:
        score += 14
    elif 45 <= last_rsi < 52 or 68 < last_rsi <= 75:
        score += 8
    else:
        score += 3

    # MACD
    if last_macd > last_signal and last_hist > 0:
        score += 16
    elif last_macd > last_signal:
        score += 10
    else:
        score += 3

    # Moving average alignment
    if not pd.isna(ma20.iloc[-1]) and last_close > ma20.iloc[-1]:
        score += 10
    if not pd.isna(ma50.iloc[-1]) and last_close > ma50.iloc[-1]:
        score += 10
    if not pd.isna(ma200.iloc[-1]) and last_close > ma200.iloc[-1]:
        score += 8
    if not pd.isna(ma20.iloc[-1]) and not pd.isna(ma50.iloc[-1]) and ma20.iloc[-1] > ma50.iloc[-1]:
        score += 6

    # Bollinger position
    if not pd.isna(bb_mid.iloc[-1]) and not pd.isna(bb_upper.iloc[-1]):
        if bb_mid.iloc[-1] <= last_close <= bb_upper.iloc[-1]:
            score += 8
        elif last_close > bb_upper.iloc[-1]:
            score += 4
        else:
            score += 2

    # RVOL
    if last_rvol >= 1.8:
        score += 12
    elif last_rvol >= 1.2:
        score += 8
    elif last_rvol >= 1.0:
        score += 5
    else:
        score += 2

    # OBV trend
    obv_up = obv.iloc[-1] > obv.iloc[-5] if len(obv) > 5 else False
    score += 8 if obv_up else 2

    # CMF
    if last_cmf > 0.1:
        score += 8
    elif last_cmf > 0:
        score += 5
    else:
        score += 1

    # Breakout-ish
    rolling_high = df["High"].rolling(20).max().shift(1)
    if len(rolling_high.dropna()) > 0 and last_close >= rolling_high.iloc[-1]:
        score += 8

    score = int(max(0, min(100, score)))

    if score >= 85:
        label = "Sangat Kuat"
        signal = "BUY"
        trend = "Uptrend"
    elif score >= 70:
        label = "Kuat"
        signal = "BUY"
        trend = "Uptrend"
    elif score >= 55:
        label = "Netral Positif"
        signal = "HOLD"
        trend = "Mixed"
    elif score >= 40:
        label = "Lemah"
        signal = "WAIT"
        trend = "Weak"
    else:
        label = "Berisiko"
        signal = "SELL"
        trend = "Downtrend"

    details = {
        "rsi": last_rsi,
        "macd_bull": last_macd > last_signal,
        "rvol": last_rvol,
        "cmf": last_cmf,
        "trend": trend,
        "accumulation": accumulation,
        "signal": signal,
        "label": label,
    }
    return score, details


# =========================
# Screener builder
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def build_top_screener(idx_tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in idx_tickers:
        try:
            df = load_stock_data(ticker, period="6mo", interval="1d")
            if df.empty or len(df) < 50:
                continue
            meta = load_company_snapshot(ticker)
            score, details = score_stock(df)
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            chg = last - prev
            pct = (chg / prev * 100) if prev else 0
            rsi = calculate_rsi(df).iloc[-1]
            macd, signal_line, hist = calculate_macd(df)
            ad = calculate_ad_line(df)
            value_traded = float(last * df["Volume"].iloc[-1])

            rows.append({
                "Ticker": ticker,
                "Name": meta.get("name", ticker),
                "Sector": meta.get("sector", "-") ,
                "Price": last,
                "Change": chg,
                "Pct": pct,
                "Score": score,
                "ScoreLabel": details["label"],
                "Volume": float(df["Volume"].iloc[-1]),
                "ValueTraded": value_traded,
                "MarketCap": meta.get("market_cap", np.nan),
                "RSI": float(rsi),
                "MACD": float(macd.iloc[-1]),
                "Signal": float(signal_line.iloc[-1]),
                "Hist": float(hist.iloc[-1]),
                "CMF": float(calculate_cmf(df).iloc[-1]) if len(df) >= 20 else 0,
                "OBVUp": calculate_obv(df).iloc[-1] > calculate_obv(df).iloc[-5],
                "AccStatus": details["accumulation"],
                "Trend": details["trend"],
                "SignalRec": details["signal"],
                "DF": df.tail(120),
                "ADTail": ad.tail(60),
            })
        except Exception:
            continue

    screener = pd.DataFrame(rows)
    if screener.empty:
        return screener
    screener = screener.sort_values(["Score", "Pct"], ascending=[False, False]).reset_index(drop=True)
    screener["Rank"] = np.arange(1, len(screener) + 1)
    return screener


# =========================
# Plot helpers
# =========================
def mini_line_figure(series: pd.Series, color: str, height: int = 90) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def create_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    data = df.copy()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    _, bb_upper, bb_lower = calculate_bollinger(data)
    data["BBU"] = bb_upper
    data["BBL"] = bb_lower

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name=ticker,
        increasing_line_color="#00ff9c",
        decreasing_line_color="#ff4d4f",
        increasing_fillcolor="#00ff9c",
        decreasing_fillcolor="#ff4d4f",
    ))
    for col, color, width in [("MA20", "#ffad33", 1.8), ("MA50", "#4ea1ff", 1.8), ("MA200", "#9b6dff", 1.8)]:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines", name=col, line=dict(color=color, width=width)))
    fig.add_trace(go.Scatter(x=data.index, y=data["BBU"], mode="lines", name="BB Upper", line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data["BBL"], mode="lines", name="BB Lower", line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.05)"))
    fig.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        font=dict(color="#e8eefc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
    )
    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    macd, signal_line, hist = calculate_macd(df)
    colors = np.where(hist >= 0, "rgba(0,255,156,0.75)", "rgba(255,77,79,0.75)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color=colors))
    fig.add_trace(go.Scatter(x=df.index, y=macd, mode="lines", name="MACD", line=dict(color="#4ea1ff", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode="lines", name="Signal", line=dict(color="#ffad33", width=2)))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eefc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
    )
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    rsi = calculate_rsi(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", name="RSI 14", line=dict(color="#9b6dff", width=2)))
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,77,79,0.7)")
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,156,0.7)")
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eefc"),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        showlegend=False,
    )
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    vol_ma = df["Volume"].rolling(20).mean()
    colors = np.where(df["Close"].diff().fillna(0) >= 0, "rgba(0,255,156,0.75)", "rgba(255,77,79,0.75)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors))
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, mode="lines", name="Vol MA20", line=dict(color="#4ea1ff", width=2)))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eefc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
    )
    return fig


def create_ad_chart(df: pd.DataFrame) -> go.Figure:
    ad = calculate_ad_line(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ad, mode="lines", line=dict(color="#4ade80", width=2), name="A/D Line"))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eefc"),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
        showlegend=False,
    )
    return fig


# =========================
# Renderers
# =========================
def render_sidebar():
    with st.sidebar:
        st.markdown("## STREAMLIS")
        menu = [
            "Dashboard", "Watchlist", "Market", "Stock Screener", "Chart",
            "News", "Portfolio", "Orders", "Alerts", "Settings",
        ]
        active = "Stock Screener"
        for item in menu:
            prefix = "▣" if item == active else "◻"
            st.markdown(f"**{prefix} {item}**" if item == active else f"{item}")
        st.markdown("---")
        st.caption("Theme")
        st.write("🌙  ☀️")
        st.markdown("---")
        st.caption("Data: yfinance prototype")


def render_top_market_strip():
    cols = st.columns(len(INDEX_MAP) + 1)
    for i, (name, symbol) in enumerate(INDEX_MAP.items()):
        with cols[i]:
            df = load_index_snapshot(symbol)
            if df.empty or "Close" not in df.columns or len(df) < 2:
                st.markdown(f'<div class="mini-panel"><div class="metric-title">{name}</div><div class="metric-value">-</div><div class="small-note">Unavailable</div></div>', unsafe_allow_html=True)
                continue
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            pct = (last - prev) / prev * 100 if prev else 0
            cls = "metric-change-up" if pct >= 0 else "metric-change-down"
            st.markdown(
                f'<div class="mini-panel"><div class="metric-title">{name}</div><div class="metric-value">{fmt_num(last, 2)}</div><div class="{cls}">{pct:+.2f}%</div></div>',
                unsafe_allow_html=True,
            )
        with cols[i]:
            spark = mini_line_figure(df["Close"].tail(30), "#00ff9c" if pct >= 0 else "#ff4d4f", height=55)
            st.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False})
    with cols[-1]:
        now = datetime.now()
        st.markdown(
            f'<div class="mini-panel"><div class="metric-title">Waktu</div><div class="metric-value">{now.strftime("%H:%M:%S")}</div><div class="small-note">{now.strftime("%d %b %Y")}</div></div>',
            unsafe_allow_html=True,
        )


def render_top_controls(screener: pd.DataFrame) -> pd.DataFrame:
    st.markdown('<div class="section-title">Screener — Top Emiten</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Saham dengan kondisi teknikal terbaik hari ini</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1.1, 1])
    sectors = ["Semua Sektor"] + sorted({COMPANY_META.get(t, {}).get("sector", "-") for t in IDX_TICKERS})
    with c1:
        sector_filter = st.selectbox("Sektor", sectors)
    with c2:
        score_filter = st.selectbox("Filter", ["Semua Score", ">= 80", ">= 70", ">= 55"])
    with c3:
        refresh = st.button("⟳ Refresh Screener", use_container_width=True)
        if refresh:
            st.cache_data.clear()
            st.rerun()

    filtered = screener.copy()
    if sector_filter != "Semua Sektor":
        filtered = filtered[filtered["Sector"] == sector_filter]
    if score_filter == ">= 80":
        filtered = filtered[filtered["Score"] >= 80]
    elif score_filter == ">= 70":
        filtered = filtered[filtered["Score"] >= 70]
    elif score_filter == ">= 55":
        filtered = filtered[filtered["Score"] >= 55]
    return filtered.reset_index(drop=True)


def render_top_cards(top_df: pd.DataFrame):
    if top_df.empty:
        st.warning("Tidak ada data screener yang tersedia.")
        return

    cols = st.columns(3)

    for idx, (_, row) in enumerate(top_df.head(3).iterrows()):
        with cols[idx]:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            header_cols = st.columns([0.6, 2.7, 1.25])

            with header_cols[0]:
                st.markdown(f"### {int(row['Rank'])}")

            with header_cols[1]:
                if st.button(row["Ticker"], key=f"card_ticker_{row['Ticker']}"):
                    st.session_state["selected_ticker"] = row["Ticker"]
                    st.rerun()
                st.caption(row["Name"])

            with header_cols[2]:
                st.markdown(
                    f"""
                    <div class="score-box">
                        <div class="score-num">{int(row["Score"])}</div>
                        <div class="score-label">{row["ScoreLabel"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            color_cls = "metric-change-up" if row["Pct"] >= 0 else "metric-change-down"
            st.markdown(
                f"""
                <div class="metric-value">{fmt_num(row["Price"], 0)}</div>
                <div class="{color_cls}">{row["Change"]:+.0f} ({row["Pct"]:+.2f}%)</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi-cell">
                        <div class="metric-title">Sektor</div>
                        <div>{row['Sector']}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="metric-title">Market Cap</div>
                        <div>{fmt_short(row['MarketCap'])}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="metric-title">Volume</div>
                        <div>{fmt_short(row['Volume'])}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="metric-title">Value</div>
                        <div>{fmt_short(row['ValueTraded'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            small_df = row["DF"].copy()
            rsi_series = calculate_rsi(small_df).tail(60)
            macd, signal_line, _ = calculate_macd(small_df)
            ad = calculate_ad_line(small_df)

            st.caption(f"RSI (14)  {rsi_series.iloc[-1]:.2f}")
            st.plotly_chart(
                mini_line_figure(rsi_series, "#9b6dff", 90),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.caption(f"MACD (12,26,9)  {macd.iloc[-1]:.2f} | Signal {signal_line.iloc[-1]:.2f}")
            st.plotly_chart(
                mini_line_figure(macd.tail(60), "#4ea1ff", 90),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.caption(f"Akumulasi / Distribusi  {row['AccStatus']}")
            st.plotly_chart(
                mini_line_figure(ad.tail(60), "#4ade80", 90),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            cmf_value = row["CMF"]
            obv_trend = "Naik" if row["OBVUp"] else "Turun"
            volume_trend = (
                "Akumulasi"
                if row["AccStatus"] == "Accumulation"
                else ("Distribusi" if row["AccStatus"] == "Distribution" else "Netral")
            )

            info_df = pd.DataFrame(
                {
                    "Indikator": [
                        "Chaikin Money Flow (20)",
                        "On Balance Volume (OBV)",
                        "Volume Trend",
                    ],
                    "Nilai": [
                        f"{cmf_value:.2f}",
                        obv_trend,
                        volume_trend,
                    ],
                    "Sinyal": [
                        row["AccStatus"],
                        row["AccStatus"],
                        row["SignalRec"],
                    ],
                }
            )
            st.dataframe(info_df, use_container_width=True, hide_index=True)

            bottom_cols = st.columns(3)

            with bottom_cols[0]:
                st.caption("Trend")
                st.write(row["Trend"])

            with bottom_cols[1]:
                st.caption("Volatilitas")
                st.write("Sedang" if abs(row["Pct"]) > 1 else "Rendah")

            with bottom_cols[2]:
                label = row["SignalRec"]
                cls = (
                    "buy-chip"
                    if label == "BUY"
                    else ("hold-chip" if label == "HOLD" else ("sell-chip" if label == "SELL" else "wait-chip"))
                )
                st.caption("Rekomendasi")
                st.markdown(f'<span class="{cls}">{label}</span>', unsafe_allow_html=True)

            if st.button(f"Detail {row['Ticker']}", key=f"detail_{row['Ticker']}"):
                st.session_state["selected_ticker"] = row["Ticker"]
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


def render_watchlist(screener: pd.DataFrame):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("WATCHLIST")
    show = screener[screener["Ticker"].isin(WATCHLIST_DEFAULT)].copy()
    if show.empty:
        show = screener.head(6).copy()
    for _, row in show.iterrows():
        cols = st.columns([1.2, 1, 1, 1.2])
        with cols[0]:
            if st.button(row["Ticker"], key=f"watch_{row['Ticker']}"):
                st.session_state["selected_ticker"] = row["Ticker"]
                st.rerun()
        with cols[1]:
            st.write(fmt_num(row["Price"], 0))
        with cols[2]:
            st.write(f"{row['Change']:+.0f}")
        with cols[3]:
            cls = "metric-change-up" if row["Pct"] >= 0 else "metric-change-down"
            st.markdown(f'<div class="{cls}">{row["Pct"]:+.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_market_summary():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("MARKET SUMMARY")
    rows = []
    for name, symbol in INDEX_MAP.items():
        df = load_index_snapshot(symbol)
        if df.empty or len(df) < 2:
            rows.append((name, np.nan, np.nan))
        else:
            last = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2]
            pct = (last - prev) / prev * 100 if prev else 0
            rows.append((name, last, pct))
    summary_df = pd.DataFrame(rows, columns=["Index", "Last", "Chg%"])
    st.dataframe(summary_df.style.format({"Last": "{:,.2f}", "Chg%": "{:+.2f}%"}), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_news_panel(selected_ticker: str):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("NEWS & RESEARCH")
    sample = [
        ("10:05", f"{selected_ticker} momentum menguat di tengah volume tinggi"),
        ("09:58", "IHSG cenderung stabil, sektor perbankan menopang indeks"),
        ("09:45", f"Analisa teknikal {selected_ticker}: RSI dan MACD mengarah positif"),
        ("09:30", "Sentimen global mixed, pasar menanti arah suku bunga"),
    ]
    for tm, title in sample:
        st.markdown(f'<div class="news-item"><span class="small-note">{tm}</span><br>{title}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_signal_summary(df: pd.DataFrame, score: int, details: dict):
    rsi = calculate_rsi(df).iloc[-1]
    rvol = calculate_rvol(df).iloc[-1]
    stoch = calculate_stochastic(df).iloc[-1]
    roc = calculate_roc(df).iloc[-1]
    trend = details["trend"]
    accumulation = details["accumulation"]
    signal = details["signal"]
    momentum = "Strong" if details["macd_bull"] and rsi > 55 and roc > 0 else "Weak"
    volume = "Confirm" if rvol > 1.2 else "Weak"
    risk = "Low" if score >= 75 else ("Medium" if score >= 55 else "High")
    cls = "buy-chip" if signal == "BUY" else ("hold-chip" if signal == "HOLD" else ("sell-chip" if signal == "SELL" else "wait-chip"))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("SIGNAL SUMMARY")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Trend", trend), ("Momentum", momentum), ("Volume", volume), ("Accumulation", accumulation),
        ("Stochastic", f"{stoch:.1f}"), ("ROC", f"{roc:.2f}%"), ("Risk", risk), ("Action", signal),
    ]
    cards = [c1, c2, c3, c4, c1, c2, c3, c4]
    for col, (title, value) in zip(cards, metrics):
        with col:
            st.markdown(f'<div class="mini-panel"><div class="metric-title">{title}</div><div>{value}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<br><span class="{cls}">{signal}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_stock_detail(selected_ticker: str):
    tf_map = {
        "1M": ("1mo", "1d"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1wk"),
        "5Y": ("5y", "1wk"),
    }
    timeframe = st.radio("Timeframe", list(tf_map.keys()), horizontal=True, index=2)
    period, interval = tf_map[timeframe]
    df = load_stock_data(selected_ticker, period=period, interval=interval)
    if df.empty:
        st.error(f"Data untuk {selected_ticker} tidak tersedia.")
        return

    meta = load_company_snapshot(selected_ticker)
    score, details = score_stock(load_stock_data(selected_ticker, period="1y", interval="1d"))
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else last
    chg = last - prev
    pct = (chg / prev * 100) if prev else 0
    color_cls = "metric-change-up" if pct >= 0 else "metric-change-down"
    open_status_cls = "status-open" if market_open_now() else "status-closed"
    open_status_text = "MARKET OPEN" if market_open_now() else "MARKET CLOSED"

    st.markdown(f'<div class="selected-pill">Selected Ticker: {selected_ticker}</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    h1, h2, h3 = st.columns([2.2, 1.2, 1])
    with h1:
        st.markdown(f"## {selected_ticker}")
        st.caption(meta.get("name", selected_ticker))
        st.markdown(f'<div class="metric-value">{fmt_num(last, 0)}</div><div class="{color_cls}">{chg:+.0f} ({pct:+.2f}%)</div>', unsafe_allow_html=True)
        st.write(meta.get("sector", "-"))
    with h2:
        st.markdown(f'<span class="{open_status_cls}">{open_status_text}</span>', unsafe_allow_html=True)
        st.markdown(f"<div class='small-note'>Market Cap</div><div>{fmt_short(meta.get('market_cap'))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-note'>Score</div><div class='score-num' style='font-size:1.6rem'>{score}</div>", unsafe_allow_html=True)
    with h3:
        st.markdown(f"<div class='small-note'>Signal</div>", unsafe_allow_html=True)
        rec = details["signal"]
        cls = "buy-chip" if rec == "BUY" else ("hold-chip" if rec == "HOLD" else ("sell-chip" if rec == "SELL" else "wait-chip"))
        st.markdown(f'<span class="{cls}">{rec}</span>', unsafe_allow_html=True)
        st.markdown(f"<div class='small-note' style='margin-top:10px'>Acc/Dist</div><div>{details['accumulation']}</div>", unsafe_allow_html=True)
    st.plotly_chart(create_price_chart(df, selected_ticker), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("RSI")
        st.plotly_chart(create_rsi_chart(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("MACD")
        st.plotly_chart(create_macd_chart(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    r3, r4 = st.columns(2)
    with r3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("VOLUME")
        st.plotly_chart(create_volume_chart(df), use_container_width=True, config={"displayModeBar": False})
        rvol = calculate_rvol(df).iloc[-1]
        vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
        val_traded = last * df["Volume"].iloc[-1]
        spike = "Yes" if rvol > 1.8 else "No"
        vol_info = pd.DataFrame(
            {
                "Metric": ["Daily Volume", "Vol MA20", "RVOL", "Volume Spike", "Value Traded"],
                "Value": [fmt_short(df["Volume"].iloc[-1]), fmt_short(vol_ma20), f"{rvol:.2f}", spike, fmt_short(val_traded)],
            }
        )
        st.dataframe(vol_info, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("ACCUMULATION / DISTRIBUTION")
        st.plotly_chart(create_ad_chart(df), use_container_width=True, config={"displayModeBar": False})
        ad_info = pd.DataFrame(
            {
                "Metric": ["CMF (20)", "OBV Trend", "A/D Status"],
                "Value": [
                    f"{calculate_cmf(df).iloc[-1]:.2f}",
                    "Up" if calculate_obv(df).iloc[-1] > calculate_obv(df).iloc[-5] else "Down",
                    detect_accumulation(df),
                ],
            }
        )
        st.dataframe(ad_info, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    render_signal_summary(df, score, details)


def ensure_selected_ticker(screener: pd.DataFrame):
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "BBCA"
    if screener.empty:
        return
    if st.session_state["selected_ticker"] not in set(screener["Ticker"]):
        st.session_state["selected_ticker"] = screener.iloc[0]["Ticker"]


# =========================
# Main app
# =========================
def main():
    if st_autorefresh is not None:
        st_autorefresh(interval=60_000, key="dashboard_refresh")

    render_sidebar()
    render_top_market_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Memuat screener IDX..."):
        screener = build_top_screener(IDX_TICKERS)

    ensure_selected_ticker(screener)

    left, right = st.columns([4.8, 1.5], gap="large")
    with left:
        filtered = render_top_controls(screener)
        render_top_cards(filtered if not filtered.empty else screener)
        st.markdown("<br>", unsafe_allow_html=True)
        render_stock_detail(st.session_state["selected_ticker"])
    with right:
        render_watchlist(screener)
        st.markdown("<br>", unsafe_allow_html=True)
        render_market_summary()
        st.markdown("<br>", unsafe_allow_html=True)
        render_news_panel(st.session_state["selected_ticker"])

    st.markdown("<br>", unsafe_allow_html=True)
    status = "MARKET OPEN" if market_open_now() else "MARKET CLOSED"
    cls = "status-open" if market_open_now() else "status-closed"
    st.markdown(
        f'<div class="panel"><span class="{cls}">{status}</span> <span class="small-note">&nbsp;&nbsp;Data prototype via yfinance | Auto refresh 60s jika streamlit-autorefresh terpasang</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
