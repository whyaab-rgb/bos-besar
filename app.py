import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Auto Scan Oversold Rebound Ketat", layout="wide")

# =========================================================
# WATCHLIST MASTER IDX
# =========================================================
SYMBOLS = [
    "AALI.JK","ACES.JK","ADRO.JK","AKRA.JK","AMRT.JK","ANTM.JK","ARTO.JK","ASII.JK","BBCA.JK","BBNI.JK",
    "BBRI.JK","BBTN.JK","BMRI.JK","BRIS.JK","BRMS.JK","BRPT.JK","BUKA.JK","CPIN.JK","CTRA.JK","ERAA.JK",
    "ESSA.JK","EXCL.JK","GOTO.JK","HRUM.JK","ICBP.JK","INCO.JK","INDF.JK","INKP.JK","INTP.JK","ISAT.JK",
    "ITMG.JK","JPFA.JK","JSMR.JK","KLBF.JK","MAPI.JK","MDKA.JK","MEDC.JK","MIKA.JK","MTDL.JK","MYOR.JK",
    "PGAS.JK","PTBA.JK","PWON.JK","RAJA.JK","SIDO.JK","SMGR.JK","SMRA.JK","TBIG.JK","TLKM.JK","TOWR.JK",
    "TPIA.JK","UNTR.JK","UNVR.JK","BBYB.JK","HEAL.JK","MAPA.JK","NIKL.JK","PGEO.JK","SCMA.JK","SILO.JK",
    "SSIA.JK","TKIM.JK","WIKA.JK","WSKT.JK","GJTL.JK","ENRG.JK","DOID.JK","ELSA.JK","INDY.JK","ADMR.JK",
    "AVIA.JK","BMHS.JK","BTPS.JK","CARS.JK","CMRY.JK","DAYA.JK","DMAS.JK","EMTK.JK","FAST.JK","FILM.JK",
    "HMSP.JK","HOKI.JK","IPTV.JK","KAEF.JK","LSIP.JK","MNCN.JK","MPMX.JK","MTLA.JK","PNLF.JK","PTPP.JK",
    "SMDR.JK","SRTG.JK","TINS.JK","TMAS.JK","WEGE.JK","WTON.JK","ASRI.JK","BEST.JK","CLEO.JK","DGIK.JK"
]

TOP_N = 20
MAX_PRICE = 1000

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #081018;
    color: white;
}
.block-container {
    max-width: 99%;
    padding-top: 0.8rem;
    padding-bottom: 1rem;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #071019 0%, #0a1320 100%);
}
[data-testid="stSidebar"] {
    background-color: #09111d;
}
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #e8f0ff !important;
}
.small-note {
    font-size: 12px;
    color: #9db1cc !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def latest(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan

def fmt_price(v):
    if pd.isna(v):
        return "-"
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:,.2f}"

def fmt_pct(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}%"

def rsi_text(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}"

def human_value(v):
    if pd.isna(v):
        return "-"
    if v >= 1_000_000_000_000:
        return f"{v / 1_000_000_000_000:.1f}T"
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    return f"{v:,.0f}"

# =========================================================
# DATA SOURCE
# =========================================================
@st.cache_data(ttl=600)
def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return pd.DataFrame()
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
        return df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    except Exception:
        return pd.DataFrame()

# =========================================================
# INDICATORS
# =========================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA50"] = x["Close"].rolling(50).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    x["RSI"] = 100 - (100 / (1 + rs))

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    x["VOL_MA5"] = x["Volume"].rolling(5).mean()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()

    x["SUPPORT20"] = x["Low"].rolling(20).min()
    x["RESIST20"] = x["High"].rolling(20).max()

    high_low = x["High"] - x["Low"]
    high_close = np.abs(x["High"] - x["Close"].shift())
    low_close = np.abs(x["Low"] - x["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    body = (x["Close"] - x["Open"]).abs()
    upper_wick = x["High"] - x[["Open", "Close"]].max(axis=1)
    lower_wick = x[["Open", "Close"]].min(axis=1) - x["Low"]
    candle_range = (x["High"] - x["Low"]).replace(0, np.nan)

    x["BODY"] = body
    x["UPPER_WICK"] = upper_wick.clip(lower=0)
    x["LOWER_WICK"] = lower_wick.clip(lower=0)
    x["WICK_PCT"] = ((x["UPPER_WICK"] + x["LOWER_WICK"]) / candle_range) * 100

    return x

# =========================================================
# OVERSOLD REBOUND ENGINE (STRICT)
# =========================================================
def get_rebound_signal(
    close_, prev_close, rsi, macd, macd_signal, macd_hist, prev_macd_hist,
    vol, vol_ma5, vol_ma20, support, bb_lower_proxy, wick, ema9
):
    if any(pd.isna(v) for v in [
        close_, prev_close, rsi, macd, macd_signal, macd_hist,
        prev_macd_hist, vol, vol_ma5, wick
    ]):
        return "TUNGGU"

    # hard oversold focus
    oversold_strict = rsi <= 30
    oversold_soft = rsi <= 35

    price_up = close_ > prev_close
    macd_improve = macd_hist > prev_macd_hist
    macd_cross_up = macd > macd_signal
    vol_active_5 = vol > vol_ma5 if not pd.isna(vol_ma5) else False
    vol_active_20 = vol > vol_ma20 if not pd.isna(vol_ma20) else False
    wick_ok = wick < 35

    near_support = False if pd.isna(support) else close_ <= support * 1.08
    near_lower_band = False if pd.isna(bb_lower_proxy) else close_ <= bb_lower_proxy * 1.04
    near_ema9 = False if pd.isna(ema9) else close_ >= ema9 * 0.985

    # strongest rebound: really oversold + early upward reaction
    if (
        oversold_strict
        and price_up
        and macd_improve
        and (vol_active_5 or vol_active_20)
        and wick_ok
        and (near_support or near_lower_band)
        and near_ema9
    ):
        return "REBOUND KUAT"

    # good early rebound but still oversold
    if (
        oversold_soft
        and price_up
        and macd_improve
        and wick_ok
        and (near_support or near_lower_band)
    ):
        return "REBOUND SIAP"

    # only show if still oversold
    if oversold_soft and (macd_improve or macd_cross_up):
        return "PANTAU REBOUND"

    return "TUNGGU"

def get_rebound_action(signal_label, close_, entry):
    if signal_label == "REBOUND KUAT":
        if not pd.isna(entry) and close_ <= entry * 1.02:
            return "BELI BERTAHAP"
        return "PANTAU DEKAT"
    if signal_label == "REBOUND SIAP":
        return "TUNGGU KONFIRMASI"
    if signal_label == "PANTAU REBOUND":
        return "PANTAU"
    return "TUNGGU"

def compute_rebound_score(
    close_, prev_close, rsi, macd, macd_signal, macd_hist, prev_macd_hist,
    vol, vol_ma5, vol_ma20, support, wick, ema9
):
    score = 0

    # oversold gets biggest weight
    if not pd.isna(rsi):
        if rsi <= 26:
            score += 38
        elif rsi <= 30:
            score += 32
        elif rsi <= 33:
            score += 26
        elif rsi <= 35:
            score += 18
        else:
            score -= 25  # punish non-oversold strongly

    if not pd.isna(close_) and not pd.isna(prev_close) and close_ > prev_close:
        score += 12

    if not pd.isna(macd_hist) and not pd.isna(prev_macd_hist) and macd_hist > prev_macd_hist:
        score += 16

    if not pd.isna(macd) and not pd.isna(macd_signal) and macd > macd_signal:
        score += 8

    if not pd.isna(vol) and not pd.isna(vol_ma5) and vol > vol_ma5:
        score += 10

    if not pd.isna(vol) and not pd.isna(vol_ma20) and vol > vol_ma20:
        score += 8

    if not pd.isna(support) and close_ <= support * 1.08:
        score += 8

    if not pd.isna(ema9) and close_ >= ema9 * 0.985:
        score += 8

    if not pd.isna(wick):
        if wick < 20:
            score += 8
        elif wick < 35:
            score += 4
        elif wick >= 45:
            score -= 10

    return max(min(score, 100), 0)

# =========================================================
# ROW BUILDER
# =========================================================
def build_row(symbol: str, daily_df: pd.DataFrame):
    df = calc_indicators(daily_df)
    if len(df) < 40:
        return None

    close_ = latest(df["Close"])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else close_
    gain = ((close_ - prev_close) / prev_close * 100) if prev_close else 0.0

    rsi = latest(df["RSI"])
    macd = latest(df["MACD"])
    macd_signal = latest(df["MACD_SIGNAL"])
    macd_hist = latest(df["MACD_HIST"])
    prev_macd_hist = float(df["MACD_HIST"].iloc[-2]) if len(df) > 1 and not pd.isna(df["MACD_HIST"].iloc[-2]) else np.nan

    vol = latest(df["Volume"])
    vol_ma5 = latest(df["VOL_MA5"])
    vol_ma20 = latest(df["VOL_MA20"])

    support = latest(df["SUPPORT20"])
    resistance = latest(df["RESIST20"])
    ema9 = latest(df["EMA9"])
    ma20 = latest(df["MA20"])
    ma50 = latest(df["MA50"])
    atr = latest(df["ATR14"])
    wick = latest(df["WICK_PCT"])

    std20 = latest(df["Close"].rolling(20).std())
    bb_lower_proxy = ma20 - (2 * std20) if not pd.isna(ma20) and not pd.isna(std20) else np.nan

    rvol = (vol / vol_ma20 * 100) if not pd.isna(vol_ma20) and vol_ma20 > 0 else np.nan

    entry = round(max(close_, ema9)) if not pd.isna(ema9) else round(close_)
    tp = round(close_ + (atr * 1.2)) if not pd.isna(atr) else round(close_ * 1.04)
    sl = round(close_ - (atr * 0.8)) if not pd.isna(atr) else round(close_ * 0.97)

    profit = ((close_ - entry) / entry * 100) if entry else 0.0
    to_tp = ((tp - close_) / close_ * 100) if close_ else 0.0
    val = close_ * vol if not pd.isna(close_) and not pd.isna(vol) else np.nan

    signal_label = get_rebound_signal(
        close_, prev_close, rsi, macd, macd_signal, macd_hist, prev_macd_hist,
        vol, vol_ma5, vol_ma20, support, bb_lower_proxy, wick, ema9
    )

    action_label = get_rebound_action(signal_label, close_, entry)

    rebound_score = compute_rebound_score(
        close_, prev_close, rsi, macd, macd_signal, macd_hist, prev_macd_hist,
        vol, vol_ma5, vol_ma20, support, wick, ema9
    )

    trend = (
        "NAIK" if not pd.isna(close_) and not pd.isna(ma20) and not pd.isna(ma50) and close_ > ma20 > ma50
        else "TURUN" if not pd.isna(close_) and not pd.isna(ma20) and not pd.isna(ma50) and close_ < ma20 < ma50
        else "NETRAL"
    )

    return {
        "symbol": symbol.replace(".JK", ""),
        "full_symbol": symbol,
        "harga": close_,
        "gain": gain,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "rvol": rvol,
        "wick": wick,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "profit": profit,
        "to_tp": to_tp,
        "value": val,
        "support": support,
        "resistance": resistance,
        "ema9": ema9,
        "trend": trend,
        "aksi": action_label,
        "sinyal": signal_label,
        "score_rebound": rebound_score,
        "daily_df": df
    }

@st.cache_data(ttl=300)
def run_oversold_scanner(symbols, period, interval, max_price):
    rows = []

    for symbol in symbols:
        try:
            daily = get_ohlcv(symbol, period=period, interval=interval)
            if daily.empty:
                continue

            row = build_row(symbol, daily)
            if row is None:
                continue

            # HARD FILTER:
            # only under max price, only oversold zone, only rebound labels
            if (
                not pd.isna(row["harga"])
                and row["harga"] <= max_price
                and not pd.isna(row["rsi"])
                and row["rsi"] <= 35
                and row["sinyal"] in ["REBOUND KUAT", "REBOUND SIAP", "PANTAU REBOUND"]
            ):
                rows.append(row)

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    # prioritize strongest oversold first:
    # higher score, lower RSI, better volume, then positive gain
    return pd.DataFrame(rows).sort_values(
        ["score_rebound", "rsi", "rvol", "gain"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)

# =========================================================
# CELL COLORS
# =========================================================
def bg_score(v):
    if pd.isna(v):
        return "#243244"
    if v >= 80:
        return "#9333ea"
    if v >= 65:
        return "#16a34a"
    if v >= 50:
        return "#2563eb"
    return "#374151"

def bg_signal(v):
    mapping = {
        "REBOUND KUAT": "#7e22ce",
        "REBOUND SIAP": "#16a34a",
        "PANTAU REBOUND": "#2563eb",
        "TUNGGU": "#111827"
    }
    return mapping.get(v, "#334155")

def bg_action(v):
    mapping = {
        "BELI BERTAHAP": "#7c3aed",
        "PANTAU DEKAT": "#1d4ed8",
        "TUNGGU KONFIRMASI": "#b45309",
        "PANTAU": "#334155",
        "TUNGGU": "#111827"
    }
    return mapping.get(v, "#334155")

def bg_rsi(v):
    if pd.isna(v):
        return "#243244"
    if v <= 28:
        return "#9333ea"
    if v <= 32:
        return "#16a34a"
    if v <= 35:
        return "#2563eb"
    return "#374151"

def bg_gain(v):
    if pd.isna(v):
        return "#243244"
    if v > 2:
        return "#10b981"
    if v > 0:
        return "#15803d"
    if v > -2:
        return "#dc2626"
    return "#991b1b"

def bg_rvol(v):
    if pd.isna(v):
        return "#243244"
    if v >= 200:
        return "#9333ea"
    if v >= 130:
        return "#f97316"
    if v >= 100:
        return "#2563eb"
    return "#374151"

def bg_trend(v):
    mapping = {"NAIK": "#16a34a", "TURUN": "#dc2626", "NETRAL": "#6b7280"}
    return mapping.get(v, "#334155")

# =========================================================
# HTML TABLE
# =========================================================
def make_html_table(df: pd.DataFrame, title: str, sub: str):
    html = textwrap.dedent(f"""
    <html>
    <head>
    <style>
    body {{
        margin: 0;
        background: #07111b;
        color: white;
        font-family: Arial, Helvetica, sans-serif;
    }}
    .screen-box {{
        border: 1px solid #17324d;
        border-radius: 10px;
        padding: 8px;
        background: #07111b;
        box-sizing: border-box;
        width: 100%;
    }}
    .screener-title {{
        text-align: center;
        font-weight: 800;
        font-size: 13px;
        color: #eaf2ff;
        margin-bottom: 4px;
        letter-spacing: 0.3px;
    }}
    .screener-sub {{
        text-align: center;
        color: #9fb5d1;
        font-size: 10px;
        margin-bottom: 6px;
    }}
    .table-wrap {{
        width: 100%;
        overflow-x: auto;
    }}
    .custom-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        min-width: 1450px;
    }}
    .custom-table th {{
        background: #184574;
        color: #ffffff;
        border: 1px solid #2a527b;
        padding: 5px 3px;
        text-align: center;
        white-space: nowrap;
        font-weight: 800;
    }}
    .custom-table td {{
        border: 1px solid #20364e;
        padding: 4px 3px;
        text-align: center;
        white-space: nowrap;
        font-weight: 700;
    }}
    .footer-line {{
        margin-top: 6px;
        text-align: center;
        color: #ffd451;
        font-size: 10px;
        font-weight: 700;
    }}
    </style>
    </head>
    <body>
    <div class="screen-box">
      <div class="screener-title">{title}</div>
      <div class="screener-sub">{sub}</div>
      <div class="table-wrap">
      <table class="custom-table">
        <thead>
          <tr>
            <th>RANK</th>
            <th>EMITEN</th>
            <th>SKOR REBOUND</th>
            <th>SINYAL</th>
            <th>AKSI</th>
            <th>HARGA</th>
            <th>KENAIKAN</th>
            <th>RSI</th>
            <th>RVOL</th>
            <th>WICK</th>
            <th>AREA BELI</th>
            <th>TP</th>
            <th>BATAS RUGI</th>
            <th>% KE TP</th>
            <th>NILAI</th>
            <th>TREND</th>
          </tr>
        </thead>
        <tbody>
    """)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        html += f"""
        <tr>
            <td style="background:#0f172a;color:#fff;">{i}</td>
            <td style="background:#1d4ed8;color:#fff;">{row['symbol']}</td>
            <td style="background:{bg_score(row['score_rebound'])};color:#fff;">{int(row['score_rebound'])}</td>
            <td style="background:{bg_signal(row['sinyal'])};color:#fff;">{row['sinyal']}</td>
            <td style="background:{bg_action(row['aksi'])};color:#fff;">{row['aksi']}</td>
            <td style="background:#2563eb;color:#fff;">{fmt_price(row['harga'])}</td>
            <td style="background:{bg_gain(row['gain'])};color:#fff;">{fmt_pct(row['gain'])}</td>
            <td style="background:{bg_rsi(row['rsi'])};color:#fff;">{rsi_text(row['rsi'])}</td>
            <td style="background:{bg_rvol(row['rvol'])};color:#fff;">{fmt_pct(row['rvol'])}</td>
            <td style="background:#334155;color:#fff;">{fmt_pct(row['wick'])}</td>
            <td style="background:#1d4ed8;color:#fff;">{fmt_price(row['entry'])}</td>
            <td style="background:#16a34a;color:#fff;">{fmt_price(row['tp'])}</td>
            <td style="background:#b91c1c;color:#fff;">{fmt_price(row['sl'])}</td>
            <td style="background:#0f766e;color:#fff;">{fmt_pct(row['to_tp'])}</td>
            <td style="background:#183b69;color:#fff;">{human_value(row['value'])}</td>
            <td style="background:{bg_trend(row['trend'])};color:#fff;">{row['trend']}</td>
        </tr>
        """

    html += """
        </tbody>
      </table>
      </div>
      <div class="footer-line">AUTO SCAN OVERSOLD REBOUND KETAT | hanya RSI ≤ 35 | prioritaskan oversold valid, bukan overbought</div>
    </div>
    </body>
    </html>
    """
    return html

# =========================================================
# CHART DETAIL
# =========================================================
def show_detail_chart(df: pd.DataFrame, symbol_name: str):
    st.subheader(f"Chart Detail: {symbol_name}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA9"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.update_layout(
        height=520,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# HEADER
# =========================================================
st.title("AUTO SCAN OVERSOLD REBOUND KETAT")
st.markdown(
    '<div class="small-note">hanya menampilkan saham oversold yang mulai naik | harga ≤ 1000 | prioritaskan oversold valid dibanding saham yang sudah panas</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Pengaturan Scan")

    preset = st.selectbox("Universe Scan", ["IDX 50", "IDX 100"], index=1)
    symbols = SYMBOLS[:50] if preset == "IDX 50" else SYMBOLS[:100]

    period = st.selectbox("Periode", ["3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    max_price = 1000
    st.info("Filter aktif: hanya saham dengan harga ≤ 1000")
    st.info("Filter aktif: hanya saham oversold dengan RSI ≤ 35")

    auto_refresh = st.checkbox("Auto Refresh", value=False)
    refresh_sec = st.selectbox("Refresh tiap", [60, 120, 300, 600], index=1)

    run_btn = st.button("Jalankan Scanner", use_container_width=True)

# =========================================================
# RUN
# =========================================================
if run_btn or "scanner_df" not in st.session_state:
    with st.spinner("Scanning saham oversold rebound..."):
        st.session_state["scanner_df"] = run_oversold_scanner(symbols, period, interval, max_price)
        st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

scanner_df = st.session_state.get("scanner_df", pd.DataFrame())

if scanner_df.empty:
    st.error("Belum ada saham oversold yang mulai naik dan lolos kriteria ketat.")
    st.stop()

display_df = scanner_df.head(TOP_N).reset_index(drop=True)
last_run = st.session_state.get("last_run", "-")

# =========================================================
# METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("TOP PICK", display_df.iloc[0]["symbol"])
m2.metric("SKOR TERATAS", int(display_df.iloc[0]["score_rebound"]))
m3.metric("SINYAL TERATAS", display_df.iloc[0]["sinyal"])
m4.metric("SCAN TERAKHIR", last_run)

# =========================================================
# TABLE
# =========================================================
st.subheader("Top Oversold Rebound")
components.html(
    make_html_table(
        display_df,
        "AUTO SCAN OVERSOLD REBOUND KETAT",
        f"Update: {last_run} | Universe: {preset}"
    ),
    height=560,
    scrolling=True
)

# =========================================================
# RANKING
# =========================================================
st.subheader("Ranking Kandidat Rebound")
rank_df = display_df[[
    "symbol", "harga", "gain", "rsi", "rvol", "wick",
    "trend", "sinyal", "aksi", "score_rebound"
]].copy()
rank_df.columns = [
    "EMITEN", "HARGA", "KENAIKAN", "RSI", "RVOL", "WICK",
    "TREND", "SINYAL", "AKSI", "SKOR REBOUND"
]
rank_df["HARGA"] = rank_df["HARGA"].apply(fmt_price)
rank_df["KENAIKAN"] = rank_df["KENAIKAN"].apply(fmt_pct)
rank_df["RVOL"] = rank_df["RVOL"].apply(fmt_pct)
rank_df["RSI"] = rank_df["RSI"].apply(rsi_text)
rank_df["WICK"] = rank_df["WICK"].apply(fmt_pct)
st.dataframe(rank_df, use_container_width=True, height=360)

# =========================================================
# DETAIL
# =========================================================
selected_symbol = st.selectbox("Pilih saham untuk detail", display_df["full_symbol"].tolist())
selected_row = display_df[display_df["full_symbol"] == selected_symbol].iloc[0]
selected_df = selected_row["daily_df"]

d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("EMITEN", selected_row["symbol"])
d2.metric("HARGA", fmt_price(selected_row["harga"]))
d3.metric("KENAIKAN", fmt_pct(selected_row["gain"]))
d4.metric("RSI", rsi_text(selected_row["rsi"]))
d5.metric("RVOL", fmt_pct(selected_row["rvol"]))
d6.metric("SKOR REBOUND", int(selected_row["score_rebound"]))

show_detail_chart(selected_df, selected_row["symbol"])

st.subheader("Analisa Rebound")
c1, c2, c3 = st.columns(3)
with c1:
    st.write(f"**Sinyal:** {selected_row['sinyal']}")
    st.write(f"**Aksi:** {selected_row['aksi']}")
    st.write(f"**Trend:** {selected_row['trend']}")
with c2:
    st.write(f"**Area Beli:** {fmt_price(selected_row['entry'])}")
    st.write(f"**Target:** {fmt_price(selected_row['tp'])}")
    st.write(f"**Batas Rugi:** {fmt_price(selected_row['sl'])}")
with c3:
    st.write(f"**RSI:** {rsi_text(selected_row['rsi'])}")
    st.write(f"**RVOL:** {fmt_pct(selected_row['rvol'])}")
    st.write(f"**Nilai:** {human_value(selected_row['value'])}")

# =========================================================
# AUTO REFRESH
# =========================================================
if auto_refresh:
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {refresh_sec * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )
