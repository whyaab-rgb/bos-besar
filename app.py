from datetime import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(page_title="IDX Pro Dashboard V2", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# =========================
# CONFIG
# =========================
ALL_IDX_TICKERS = sorted(list(set([
    "AADI","AALI","ABBA","ABDA","ABMM","ACES","ACST","ADCP","ADES","ADHI","ADMF","ADMR","ADRO",
    "AGAR","AGII","AGRO","AGRS","AHAP","AIMS","AISA","AKKU","AKPI","AKRA","AKSI","ALDO","ALKA",
    "ALMI","ALTO","AMAG","AMFG","AMIN","AMMN","AMRT","ANDI","ANJT","ANTM","APEX","APIC","APII",
    "APLI","APLN","ARCI","ARGO","ARII","ARKA","ARMY","ARTO","ASBI","ASDM","ASGR","ASII","ASJT",
    "ASMI","ASRI","ASUR","AUTO","AVIA","AWAN","AXIO","BACA","BAJA","BALI","BANK","BAPA","BATA",
    "BAUT","BBCA","BBHI","BBKP","BBLD","BBMD","BBNI","BBRI","BBTN","BBYB","BCAP","BCIC","BDMN",
    "BEBS","BELL","BESS","BEST","BFIN","BGTG","BHAT","BIKA","BIMA","BINA","BIPI","BJBR","BJTM",
    "BKDP","BKSL","BLTA","BLTZ","BMAS","BMHS","BMRI","BMSR","BMTR","BNBA","BNBR","BNGA","BNII",
    "BNLI","BOBA","BOLA","BOSS","BRAM","BRIS","BRMS","BRNA","BROT","BSDE","BSIM","BSSR","BSWD",
    "BTEK","BTPS","BUKA","BUKK","BULL","BUMI","BUVA","BVIC","BWPT","BYAN","CBMF","CCSI","CEKA",
    "CENT","CFIN","CINT","CITY","CLAY","CMNP","CMRY","CMPP","CNKO","CODE","CPIN","CPRO","CSAP",
    "CSIS","CTBN","CTRA","CUAN","DAYA","DEFI","DEPO","DGIK","DILD","DKFT","DLTA","DMAS","DNAR",
    "DOID","DSFI","DSNG","DSSA","DUTI","DYAN","EAST","ECII","ELSA","EMDE","EMTK","ENRG","ERAA",
    "ESSA","ESTA","EXCL","FAPA","FILM","FINN","FISH","FORU","FPNI","FREN","GAMA","GDST","GEMS",
    "GGRM","GIAA","GJTL","GLVA","GMFI","GOLD","GOOD","GOTO","GPRA","GSMF","GTBO","GWSA","HEAL",
    "HELI","HERO","HITS","HKMU","HMSP","HOME","HRTA","HRUM","IATA","IBST","ICBP","INAF","INAI",
    "INCO","INDF","INDR","INKP","INOV","INTP","IPCC","ISAT","ITMG","JARR","JASS","JAWA","JECC",
    "JGLE","JPFA","JRPT","JSMR","JTPE","KEEN","KIAS","KINO","KLBF","KMDS","KMTR","KOBX","KONI",
    "KOPI","KPAS","KRAS","LABA","LAPD","LCGP","LIFE","LINK","LMAS","LPCK","LPIN","LPKR","LPPF",
    "LRNA","LSIP","MAIN","MAPA","MAPI","MARK","MASA","MAYA","MBAP","MBMA","MCAS","MDKA","MEDC",
    "MFIN","MIDI","MIKA","MLBI","MLIA","MLPL","MMIX","MNCN","MPMX","MREI","MSIN","MTDL","MTEL",
    "MYOH","MYOR","NCKL","NELY","NICL","NIKL","NISP","PACK","PANR","PANI","PANS","PBID","PCAR",
    "PGAS","PGEO","PGLI","PICO","PJAA","PKPK","PLAN","PLIN","PMJS","PNBN","PNGO","PNIN","PNLF",
    "POLL","PORT","PPRE","PPRO","PRDA","PTBA","PTMP","PTPP","PURA","PWON","RAJA","RALS","RAMS",
    "RBMS","RICY","RMKE","ROTI","SCMA","SDMU","SDPC","SIDO","SILO","SIMA","SIMP","SKBM","SKLT",
    "SMAR","SMBR","SMDR","SMGR","SMIL","SMMT","SMRA","SMSM","SOHO","SPMA","SRIL","SSIA","SSMS",
    "STAR","SURE","TAPG","TBIG","TBLA","TBMS","TBUO","TCID","TECH","TELE","TFAS","TGKA","TINS",
    "TKIM","TLKM","TMAS","TOBA","TOOL","TOPS","TOWR","TPIA","TRAM","TRIM","TRIS","TRJA","TSPC",
    "ULTJ","UNIQ","UNTR","UNVR","WICO","WIKA","WINS","WOOD","WSBP","WSKT","WTON","ZINC"
])))

COMPANY_META = {
    "BBCA": {"name": "Bank Central Asia Tbk.", "sector": "Perbankan"},
    "BBRI": {"name": "Bank Rakyat Indonesia Tbk.", "sector": "Perbankan"},
    "BMRI": {"name": "Bank Mandiri (Persero) Tbk.", "sector": "Perbankan"},
    "BBNI": {"name": "Bank Negara Indonesia Tbk.", "sector": "Perbankan"},
    "TLKM": {"name": "Telkom Indonesia Tbk.", "sector": "Telekomunikasi"},
    "ASII": {"name": "Astra International Tbk.", "sector": "Otomotif"},
    "ANTM": {"name": "Aneka Tambang Tbk.", "sector": "Komoditas"},
    "ADRO": {"name": "Adaro Energy Indonesia Tbk.", "sector": "Energi"},
    "AKRA": {"name": "AKR Corporindo Tbk.", "sector": "Distribusi"},
    "AMRT": {"name": "Sumber Alfaria Trijaya Tbk.", "sector": "Ritel"},
    "BRIS": {"name": "Bank Syariah Indonesia Tbk.", "sector": "Perbankan"},
    "CPIN": {"name": "Charoen Pokphand Indonesia Tbk.", "sector": "Konsumsi"},
    "GOTO": {"name": "GoTo Gojek Tokopedia Tbk.", "sector": "Teknologi"},
    "ICBP": {"name": "Indofood CBP Sukses Makmur Tbk.", "sector": "Konsumsi"},
    "INDF": {"name": "Indofood Sukses Makmur Tbk.", "sector": "Konsumsi"},
    "ITMG": {"name": "Indo Tambangraya Megah Tbk.", "sector": "Energi"},
    "KLBF": {"name": "Kalbe Farma Tbk.", "sector": "Farmasi"},
    "MAPI": {"name": "Mitra Adiperkasa Tbk.", "sector": "Ritel"},
    "MDKA": {"name": "Merdeka Copper Gold Tbk.", "sector": "Komoditas"},
    "MEDC": {"name": "Medco Energi Internasional Tbk.", "sector": "Energi"},
    "PGAS": {"name": "Perusahaan Gas Negara Tbk.", "sector": "Energi"},
    "PTBA": {"name": "Bukit Asam Tbk.", "sector": "Energi"},
    "SIDO": {"name": "Industri Jamu Dan Farmasi Sido Muncul Tbk.", "sector": "Farmasi"},
    "SMGR": {"name": "Semen Indonesia Tbk.", "sector": "Material Dasar"},
    "TLKM": {"name": "Telkom Indonesia (Persero) Tbk.", "sector": "Telekomunikasi"},
    "TOWR": {"name": "Sarana Menara Nusantara Tbk.", "sector": "Infrastruktur"},
    "UNTR": {"name": "United Tractors Tbk.", "sector": "Industri"},
    "UNVR": {"name": "Unilever Indonesia Tbk.", "sector": "Konsumsi"},
}

MARKET_SYMBOLS = {
    "IHSG": "^JKSE",
    "DJI": "^DJI",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
}

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #0f1a2e;
        --panel2: #121c31;
        --line: rgba(95, 146, 255, 0.16);
        --text: #e6eefc;
        --muted: #93a4bf;
        --green: #00ff9c;
        --red: #ff4d4f;
        --blue: #4ea1ff;
        --orange: #ffb347;
        --purple: #9b6dff;
    }
    .stApp { background: linear-gradient(180deg, #09111c 0%, #0b1220 100%); color: var(--text); }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1800px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #07101b 0%, #0a111d 100%); border-right: 1px solid var(--line); }
    .panel {
        background: linear-gradient(180deg, rgba(15,26,46,0.98) 0%, rgba(10,18,34,0.98) 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }
    .mini-panel {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 10px 12px;
    }
    .title-main { font-size: 1.6rem; font-weight: 800; margin-bottom: .2rem; }
    .subtitle-main { color: var(--muted); margin-bottom: .8rem; }
    .metric-title { color: var(--muted); font-size: .78rem; }
    .metric-value { color: var(--text); font-size: 1.35rem; font-weight: 800; }
    .up { color: var(--green); font-weight: 700; }
    .down { color: var(--red); font-weight: 700; }
    .score-box {
        border: 1px solid rgba(0,255,156,.28);
        background: rgba(0,255,156,.07);
        border-radius: 14px;
        padding: 10px 12px;
        text-align: center;
    }
    .score-num { font-size: 1.8rem; font-weight: 800; color: var(--green); }
    .score-label { font-size: .78rem; color: #b7ffd8; }
    .pill {
        display: inline-block; padding: .32rem .7rem; border-radius: 999px;
        background: rgba(78,161,255,.09); border: 1px solid var(--line); color: #cfe1ff; font-size: .82rem; font-weight: 700;
    }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 10px 0; }
    .kpi-cell { background: rgba(255,255,255,.02); border: 1px solid rgba(95,146,255,.1); border-radius: 12px; padding: 8px 10px; }
    .buy-chip,.hold-chip,.wait-chip,.sell-chip {
        display:inline-block; border-radius:10px; padding:.34rem .7rem; font-size:.84rem; font-weight:800;
    }
    .buy-chip { background: rgba(0,255,156,.14); color: var(--green); }
    .hold-chip { background: rgba(78,161,255,.14); color: var(--blue); }
    .wait-chip { background: rgba(255,179,71,.14); color: var(--orange); }
    .sell-chip { background: rgba(255,77,79,.14); color: var(--red); }
    .status-open,.status-closed {
        display:inline-block; border-radius:999px; padding:.28rem .68rem; font-size:.76rem; font-weight:800;
    }
    .status-open { background: rgba(0,255,156,.12); color: var(--green); border: 1px solid rgba(0,255,156,.24); }
    .status-closed { background: rgba(255,77,79,.12); color: var(--red); border: 1px solid rgba(255,77,79,.24); }
    .small-note { color: var(--muted); font-size: .78rem; }
    .stButton > button {
        width: 100%; border-radius: 12px; font-weight: 700;
        border: 1px solid rgba(95,146,255,.18);
        background: linear-gradient(180deg, #11203a 0%, #0c162d 100%); color: #eef4ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# BASIC HELPERS
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def build_master_search_options() -> pd.DataFrame:
    rows = []
    for ticker in ALL_IDX_TICKERS:
        meta = COMPANY_META.get(ticker, {})
        name = meta.get("name", ticker)
        sector = meta.get("sector", "-")
        rows.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                "Label": f"{ticker} — {name}",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
def normalize_ticker(ticker: str) -> str:
    return ticker if ticker.startswith("^") or ticker.endswith(".JK") else f"{ticker}.JK"


def fmt_num(x, digits=2):
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.{digits}f}"


def fmt_short(x):
    if x is None or pd.isna(x):
        return "-"
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.2f}T"
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:.0f}"


def market_open_now() -> bool:
    now = datetime.now()
    return now.weekday() < 5 and 9 <= now.hour < 16


# =========================
# DATA
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def load_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(normalize_ticker(ticker), period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.dropna(how="all", inplace=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_fast_info(ticker: str) -> dict:
    meta = COMPANY_META.get(ticker, {}).copy()
    try:
        tk = yf.Ticker(normalize_ticker(ticker))
        fi = tk.fast_info if hasattr(tk, "fast_info") else {}
        meta["market_cap"] = fi.get("market_cap")
        meta["last_price"] = fi.get("last_price")
    except Exception:
        meta.setdefault("market_cap", np.nan)
        meta.setdefault("last_price", np.nan)
    return meta


@st.cache_data(ttl=300, show_spinner=False)
def load_index_data(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.title)


# =========================
# INDICATORS
# =========================
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return (df["Volume"] / df["Volume"].rolling(window).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / denom
    mfv = mfm.fillna(0) * df["Volume"]
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)


def calculate_ad_line(df: pd.DataFrame) -> pd.Series:
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / denom
    return (clv.fillna(0) * df["Volume"]).cumsum()


def calculate_bollinger(df: pd.DataFrame, period: int = 20, stdv: float = 2.0):
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    return mid, mid + stdv * std, mid - stdv * std


def detect_accumulation(df: pd.DataFrame) -> str:
    cmf = calculate_cmf(df).iloc[-1]
    obv = calculate_obv(df)
    ad = calculate_ad_line(df)
    obv_up = obv.iloc[-1] > obv.iloc[-5] if len(obv) > 5 else False
    ad_up = ad.iloc[-1] > ad.iloc[-5] if len(ad) > 5 else False
    if cmf > 0.08 and obv_up and ad_up:
        return "Accumulation"
    if cmf < -0.08 and not obv_up and not ad_up:
        return "Distribution"
    return "Neutral"


def score_stock(df: pd.DataFrame):
    if df.empty or len(df) < 50:
        return 0, {}
    close = df["Close"]
    rsi = calculate_rsi(df)
    macd, sig, hist = calculate_macd(df)
    rvol = calculate_rvol(df)
    obv = calculate_obv(df)
    cmf = calculate_cmf(df)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    bb_mid, bb_up, bb_low = calculate_bollinger(df)

    last_close = float(close.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_rvol = float(rvol.iloc[-1])
    last_cmf = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
    last_macd = float(macd.iloc[-1])
    last_sig = float(sig.iloc[-1])
    hist_up = float(hist.iloc[-1]) > 0
    obv_up = obv.iloc[-1] > obv.iloc[-5] if len(obv) > 5 else False

    score = 0
    score += 14 if 52 <= last_rsi <= 68 else 8 if 45 <= last_rsi <= 75 else 3
    score += 16 if last_macd > last_sig and hist_up else 8 if last_macd > last_sig else 2
    score += 10 if last_close > ma20.iloc[-1] else 0
    score += 10 if last_close > ma50.iloc[-1] else 0
    score += 8 if (not pd.isna(ma200.iloc[-1]) and last_close > ma200.iloc[-1]) else 0
    score += 6 if (not pd.isna(ma20.iloc[-1]) and not pd.isna(ma50.iloc[-1]) and ma20.iloc[-1] > ma50.iloc[-1]) else 0
    if not pd.isna(bb_mid.iloc[-1]) and not pd.isna(bb_up.iloc[-1]):
        score += 8 if bb_mid.iloc[-1] <= last_close <= bb_up.iloc[-1] else 3
    score += 12 if last_rvol >= 1.8 else 8 if last_rvol >= 1.2 else 4 if last_rvol >= 1 else 1
    score += 8 if obv_up else 2
    score += 8 if last_cmf > 0.1 else 5 if last_cmf > 0 else 1

    rolling_high = df["High"].rolling(20).max().shift(1)
    score += 8 if (not pd.isna(rolling_high.iloc[-1]) and last_close >= rolling_high.iloc[-1]) else 0
    score = int(max(0, min(100, score)))

    label = "Sangat Kuat" if score >= 85 else "Kuat" if score >= 70 else "Netral Positif" if score >= 55 else "Lemah" if score >= 40 else "Berisiko"
    final_signal = "BUY" if score >= 70 else "HOLD" if score >= 55 else "WAIT" if score >= 40 else "SELL"
    trend = "Uptrend" if (last_close > ma20.iloc[-1] and last_close > ma50.iloc[-1]) else "Downtrend"
    return score, {
        "rsi": last_rsi,
        "macd_bull": last_macd > last_sig,
        "rvol": last_rvol,
        "cmf": last_cmf,
        "obv_up": obv_up,
        "trend": trend,
        "accumulation": detect_accumulation(df),
        "signal": final_signal,
        "label": label,
    }


# =========================
# SPECIAL SCREENERS
# =========================
def compute_screener_logic(df: pd.DataFrame, base_score: int, details: dict) -> dict:
    close = df["Close"]
    rsi = calculate_rsi(df)
    macd, sig, hist = calculate_macd(df)
    rvol = calculate_rvol(df)
    obv = calculate_obv(df)
    cmf = calculate_cmf(df)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    chg_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0
    rsi_now = float(rsi.iloc[-1])
    macd_now = float(macd.iloc[-1])
    sig_now = float(sig.iloc[-1])
    hist_now = float(hist.iloc[-1])
    rvol_now = float(rvol.iloc[-1])
    cmf_now = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
    obv_up = obv.iloc[-1] > obv.iloc[-5] if len(obv) > 5 else False
    above_ma20 = last_close > ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else False
    above_ma50 = last_close > ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else False
    acc_status = details.get("accumulation", "Neutral")

    # Assumption: BSJP = breakout setup jangka pendek
    bsjp = 0
    bsjp += 25 if 55 <= rsi_now <= 72 else 10
    bsjp += 20 if macd_now > sig_now and hist_now > 0 else 6
    bsjp += 18 if rvol_now > 1.4 else 8 if rvol_now > 1.0 else 0
    bsjp += 18 if above_ma20 and above_ma50 else 6
    bsjp += 19 if acc_status == "Accumulation" else 5

    swing = 0
    swing += 20 if above_ma20 and above_ma50 else 6
    swing += 20 if 50 <= rsi_now <= 68 else 8
    swing += 20 if macd_now > sig_now else 5
    swing += 15 if cmf_now > 0 else 4
    swing += 25 if base_score >= 70 else 10 if base_score >= 55 else 0

    day = 0
    day += 30 if rvol_now >= 1.8 else 15 if rvol_now >= 1.2 else 0
    day += 20 if abs(chg_pct) >= 2 else 8 if abs(chg_pct) >= 1 else 0
    day += 20 if macd_now > sig_now and hist_now > 0 else 5
    day += 15 if 52 <= rsi_now <= 75 else 6
    day += 15 if above_ma20 else 4

    bandar = 0
    bandar += 30 if acc_status == "Accumulation" else 0
    bandar += 20 if cmf_now > 0.08 else 10 if cmf_now > 0 else 0
    bandar += 20 if obv_up else 5
    bandar += 15 if rvol_now > 1.2 else 5
    bandar += 15 if above_ma20 else 5

    ara = 0
    ara += 28 if chg_pct >= 4 else 10 if chg_pct >= 2 else 0
    ara += 22 if rvol_now >= 2 else 10 if rvol_now >= 1.4 else 0
    ara += 15 if rsi_now >= 60 else 5
    ara += 15 if macd_now > sig_now and hist_now > 0 else 4
    ara += 20 if acc_status == "Accumulation" else 5

    return {
        "BSJP Score": int(min(100, bsjp)),
        "Swing Score": int(min(100, swing)),
        "Day Score": int(min(100, day)),
        "Bandar Score": int(min(100, bandar)),
        "ARA Score": int(min(100, ara)),
    }


@st.cache_data(ttl=60, show_spinner=False)
def build_top_screener(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            df = load_stock_data(ticker, period="1y", interval="1d")
            if df.empty or len(df) < 80:
                continue
            meta = load_fast_info(ticker)
            score, details = score_stock(df)
            if not details:
                continue
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            change = last - prev
            pct = (change / prev * 100) if prev else 0.0
            extra = compute_screener_logic(df, score, details)
            rows.append({
                "Ticker": ticker,
                "Name": meta.get("name", ticker),
                "Sector": meta.get("sector", "-"),
                "Price": last,
                "Change": change,
                "Pct": pct,
                "Score": score,
                "ScoreLabel": details["label"],
                "Volume": float(df["Volume"].iloc[-1]),
                "ValueTraded": float(last * df["Volume"].iloc[-1]),
                "MarketCap": meta.get("market_cap", np.nan),
                "RSI": details["rsi"],
                "RVOL": details["rvol"],
                "CMF": details["cmf"],
                "Trend": details["trend"],
                "AccStatus": details["accumulation"],
                "SignalRec": details["signal"],
                **extra,
            })
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Score", "Pct"], ascending=[False, False]).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


# =========================
# PLOTS
# =========================
def mini_line(series: pd.Series, color: str, height: int = 85) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig


def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    bb_mid, bb_up, bb_low = calculate_bollinger(df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker, increasing_line_color="#00ff9c", decreasing_line_color="#ff4d4f"))
    fig.add_trace(go.Scatter(x=df.index, y=ma20, name="MA20", line=dict(color="#ffb347", width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=ma50, name="MA50", line=dict(color="#4ea1ff", width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name="MA200", line=dict(color="#9b6dff", width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=bb_up, name="BB Upper", line=dict(color="rgba(147,164,191,.7)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=bb_low, name="BB Lower", line=dict(color="rgba(147,164,191,.7)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(147,164,191,.05)"))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6eefc"), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    rsi = calculate_rsi(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", line=dict(color="#9b6dff", width=2), name="RSI"))
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,77,79,.7)")
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,156,.7)")
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0,100]), font=dict(color="#e6eefc"), showlegend=False)
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    macd, sig, hist = calculate_macd(df)
    colors = np.where(hist >= 0, "rgba(0,255,156,.75)", "rgba(255,77,79,.75)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=colors, name="Hist"))
    fig.add_trace(go.Scatter(x=df.index, y=macd, mode="lines", line=dict(color="#4ea1ff", width=2), name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=sig, mode="lines", line=dict(color="#ffb347", width=2), name="Signal"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6eefc"))
    return fig


def volume_chart(df: pd.DataFrame) -> go.Figure:
    ma = df["Volume"].rolling(20).mean()
    colors = np.where(df["Close"].diff().fillna(0) >= 0, "rgba(0,255,156,.75)", "rgba(255,77,79,.75)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"))
    fig.add_trace(go.Scatter(x=df.index, y=ma, mode="lines", line=dict(color="#4ea1ff", width=2), name="Vol MA20"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6eefc"))
    return fig


def ad_chart(df: pd.DataFrame) -> go.Figure:
    ad = calculate_ad_line(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ad, mode="lines", line=dict(color="#4ade80", width=2), name="A/D Line"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6eefc"), showlegend=False)
    return fig


# =========================
# UI HELPERS
# =========================
def chip_class(signal: str) -> str:
    return "buy-chip" if signal == "BUY" else "hold-chip" if signal == "HOLD" else "sell-chip" if signal == "SELL" else "wait-chip"


def render_sidebar(screener: pd.DataFrame):
    with st.sidebar:
        st.markdown("## STREAMLIS PRO")
        ihsg = load_index_data(MARKET_SYMBOLS["IHSG"])
        if not ihsg.empty and len(ihsg) >= 2:
            last = float(ihsg["Close"].iloc[-1])
            prev = float(ihsg["Close"].iloc[-2])
            pct = (last - prev) / prev * 100 if prev else 0
            cls = "up" if pct >= 0 else "down"
            st.markdown(
                f'<div class="panel"><div class="metric-title">Market IHSG</div><div class="metric-value">{fmt_num(last, 2)}</div><div class="{cls}">{pct:+.2f}%</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        menu_options = [
            "Dashboard",
            "Watchlist > 50",
            "BSJP Screener",
            "Swing Trade Screener",
            "Day Trade Screener",
            "Bandarmology Screener",
            "ARA Screener",
        ]

        active_menu = st.radio(
            "Menu",
            menu_options,
            index=menu_options.index(st.session_state.get("active_menu", "Dashboard")),
            key="sidebar_menu_radio",
        )
        st.session_state["active_menu"] = active_menu

        st.markdown("---")
        st.caption("Watchlist score > 50")

        wl = screener[screener["Score"] > 50][["Ticker", "Score"]].sort_values("Score", ascending=False)
        for _, row in wl.head(18).iterrows():
            if st.button(f'{row["Ticker"]}  |  {int(row["Score"])}', key=f'sb_{row["Ticker"]}'):
                st.session_state["selected_ticker"] = row["Ticker"]
                st.rerun()


def render_top_market_bar():
    cols = st.columns(len(MARKET_SYMBOLS) + 1)
    for i, (name, symbol) in enumerate(MARKET_SYMBOLS.items()):
        with cols[i]:
            df = load_index_data(symbol)
            if df.empty or len(df) < 2:
                st.markdown(f'<div class="mini-panel"><div class="metric-title">{name}</div><div class="metric-value">-</div></div>', unsafe_allow_html=True)
                continue
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            pct = (last - prev) / prev * 100 if prev else 0.0
            cls = "up" if pct >= 0 else "down"
            st.markdown(f'<div class="mini-panel"><div class="metric-title">{name}</div><div class="metric-value">{fmt_num(last,2)}</div><div class="{cls}">{pct:+.2f}%</div></div>', unsafe_allow_html=True)
            st.plotly_chart(mini_line(df["Close"].tail(30), "#00ff9c" if pct >= 0 else "#ff4d4f", 55), use_container_width=True, config={"displayModeBar": False})
    with cols[-1]:
        now = datetime.now()
        st.markdown(f'<div class="mini-panel"><div class="metric-title">Waktu</div><div class="metric-value">{now.strftime("%H:%M:%S")}</div><div class="small-note">{now.strftime("%d %b %Y")}</div></div>', unsafe_allow_html=True)


def render_ticker_search_combined(screener: pd.DataFrame):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("SEARCH EMITEN")

    master_options = build_master_search_options()
    current = st.session_state.get("selected_ticker", "BBCA")

    default_idx = 0
    matched_idx = master_options.index[master_options["Ticker"] == current].tolist()
    if matched_idx:
        default_idx = matched_idx[0]

    selected_label = st.selectbox(
        "Cari dari semua ticker IDX",
        master_options["Label"].tolist(),
        index=default_idx if default_idx < len(master_options) else 0,
    )

    c1, c2 = st.columns([3, 1.1])
    with c1:
        manual = st.text_input(
            "Atau ketik ticker / nama emiten",
            value=current,
            placeholder="Contoh: BBCA, BMRI, TLKM, PGEO, Bank Mandiri, Telkom",
        )
    with c2:
        open_btn = st.button("Open Ticker", use_container_width=True)

    selected_from_box = master_options.loc[
        master_options["Label"] == selected_label, "Ticker"
    ].iloc[0]

    if selected_from_box != current:
        st.session_state["selected_ticker"] = selected_from_box
        st.rerun()

    if open_btn:
        typed = manual.strip().upper().replace(".JK", "")

        alias_map = {
            "MANDIRI": "BMRI",
            "BANK MANDIRI": "BMRI",
            "BCA": "BBCA",
            "BANK CENTRAL ASIA": "BBCA",
            "BRI": "BBRI",
            "BANK RAKYAT INDONESIA": "BBRI",
            "BNI": "BBNI",
            "BANK NEGARA INDONESIA": "BBNI",
            "TELKOM": "TLKM",
            "TELKOM INDONESIA": "TLKM",
            "ASTRA": "ASII",
            "GOTO": "GOTO",
        }

        if typed in alias_map:
            typed = alias_map[typed]

        if typed in ALL_IDX_TICKERS:
            st.session_state["selected_ticker"] = typed
            st.rerun()

        name_match = master_options[
            master_options["Ticker"].str.upper().str.contains(typed, na=False)
            | master_options["Name"].str.upper().str.contains(typed, na=False)
        ]

        if not name_match.empty:
            st.session_state["selected_ticker"] = name_match.iloc[0]["Ticker"]
            st.rerun()

        st.warning(f"Ticker / nama emiten '{manual}' tidak ditemukan di master IDX.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_top_cards(top_df: pd.DataFrame):
    if top_df.empty:
        st.warning("Tidak ada data screener.")
        return
    st.markdown('<div class="title-main">Top Emiten</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-main">Klik ticker untuk memunculkan detail saham, score, dan screener rekomendasinya</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_df.head(3).iterrows()):
        with cols[idx]:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([0.6, 2.5, 1.2])
            with c1:
                st.markdown(f"### {int(row['Rank'])}")
            with c2:
                if st.button(row["Ticker"], key=f'card_{row["Ticker"]}'):
                    st.session_state["selected_ticker"] = row["Ticker"]
                    st.rerun()
                st.caption(row["Name"])
            with c3:
                st.markdown(f'''<div class="score-box"><div class="score-num">{int(row["Score"])} </div><div class="score-label">{row["ScoreLabel"]}</div></div>''', unsafe_allow_html=True)
            cls = "up" if row["Pct"] >= 0 else "down"
            st.markdown(f'<div class="metric-value">{fmt_num(row["Price"],0)}</div><div class="{cls}">{row["Change"]:+.0f} ({row["Pct"]:+.2f}%)</div>', unsafe_allow_html=True)
            st.markdown(f'''
                <div class="kpi-grid">
                    <div class="kpi-cell"><div class="metric-title">Sektor</div><div>{row['Sector']}</div></div>
                    <div class="kpi-cell"><div class="metric-title">MCap</div><div>{fmt_short(row['MarketCap'])}</div></div>
                    <div class="kpi-cell"><div class="metric-title">Volume</div><div>{fmt_short(row['Volume'])}</div></div>
                    <div class="kpi-cell"><div class="metric-title">Value</div><div>{fmt_short(row['ValueTraded'])}</div></div>
                </div>
            ''', unsafe_allow_html=True)
            df = load_stock_data(row["Ticker"], period="6mo", interval="1d")
            st.caption(f'RSI {row["RSI"]:.2f}')
            st.plotly_chart(mini_line(calculate_rsi(df).tail(60), "#9b6dff"), use_container_width=True, config={"displayModeBar": False})
            macd, _, _ = calculate_macd(df)
            st.caption(f'MACD {macd.iloc[-1]:.2f}')
            st.plotly_chart(mini_line(macd.tail(60), "#4ea1ff"), use_container_width=True, config={"displayModeBar": False})
            st.caption(f'Akumulasi / Distribusi {row["AccStatus"]}')
            st.plotly_chart(mini_line(calculate_ad_line(df).tail(60), "#4ade80"), use_container_width=True, config={"displayModeBar": False})
            if st.button(f'Detail {row["Ticker"]}', key=f'detail_{row["Ticker"]}'):
                st.session_state["selected_ticker"] = row["Ticker"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_rank_table(title: str, df: pd.DataFrame, score_col: str, key_prefix: str):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(title)
    show = df.sort_values(score_col, ascending=False).head(12).copy()
    if show.empty:
        st.write("Tidak ada data.")
    else:
        for _, row in show.iterrows():
            cols = st.columns([1.4, 1, 1.2])
            with cols[0]:
                if st.button(row["Ticker"], key=f'{key_prefix}_{row["Ticker"]}'):
                    st.session_state["selected_ticker"] = row["Ticker"]
                    st.rerun()
            with cols[1]:
                st.write(f'{int(row[score_col])}')
            with cols[2]:
                st.markdown(f'<span class="{chip_class(row["SignalRec"])}">{row["SignalRec"]}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_stock_detail(selected_ticker: str, screener: pd.DataFrame):
    tf_map = {
        "1M": ("1mo", "1d"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1wk"),
    }
    timeframe = st.radio("Timeframe", list(tf_map.keys()), horizontal=True, index=2)
    period, interval = tf_map[timeframe]

    df = load_stock_data(selected_ticker, period=period, interval=interval)
    if df.empty:
        st.error(f"Data {selected_ticker} tidak tersedia.")
        return

    row = screener[screener["Ticker"] == selected_ticker]

    if row.empty:
        base_df = load_stock_data(selected_ticker, period="1y", interval="1d")
        if base_df.empty:
            st.error(f"Ticker {selected_ticker} tidak ditemukan atau data tidak tersedia.")
            return

        score, details = score_stock(base_df)
        logic = compute_screener_logic(base_df, score, details)

        extra = {
            "BSJP Score": logic["BSJP Score"],
            "Swing Score": logic["Swing Score"],
            "Day Score": logic["Day Score"],
            "Bandar Score": logic["Bandar Score"],
            "ARA Score": logic["ARA Score"],
            "SignalRec": details.get("signal", "WAIT"),
            "AccStatus": details.get("accumulation", "Neutral"),
            "Trend": details.get("trend", "Downtrend"),
        }
    else:
        row = row.iloc[0]
        score = int(row["Score"])
        extra = {
            "BSJP Score": int(row["BSJP Score"]),
            "Swing Score": int(row["Swing Score"]),
            "Day Score": int(row["Day Score"]),
            "Bandar Score": int(row["Bandar Score"]),
            "ARA Score": int(row["ARA Score"]),
            "SignalRec": row["SignalRec"],
            "AccStatus": row["AccStatus"],
            "Trend": row["Trend"],
        }

    meta = load_fast_info(selected_ticker)
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    pct = (last - prev) / prev * 100 if prev else 0.0
    cls = "up" if pct >= 0 else "down"

    st.markdown(f'<span class="pill">Selected Ticker: {selected_ticker}</span>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    h1, h2, h3 = st.columns([2.2, 1.2, 1.8])
    with h1:
        st.markdown(f'## {selected_ticker}')
        st.caption(meta.get("name", selected_ticker))
        st.markdown(f'<div class="metric-value">{fmt_num(last,0)}</div><div class="{cls}">{last - prev:+.0f} ({pct:+.2f}%)</div>', unsafe_allow_html=True)
        st.write(meta.get("sector", "-"))
    with h2:
        status_cls = "status-open" if market_open_now() else "status-closed"
        status_text = "MARKET OPEN" if market_open_now() else "MARKET CLOSED"
        st.markdown(f'<span class="{status_cls}">{status_text}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-note">Main Score</div><div class="score-num" style="font-size:1.5rem">{score}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-note">Signal</div><span class="{chip_class(extra["SignalRec"])}">{extra["SignalRec"]}</span>', unsafe_allow_html=True)
    with h3:
        sc_df = pd.DataFrame({
            "Screener": ["BSJP", "Swing", "Day Trade", "Bandarmology", "ARA"],
            "Point": [extra["BSJP Score"], extra["Swing Score"], extra["Day Score"], extra["Bandar Score"], extra["ARA Score"]],
        })
        st.dataframe(sc_df, use_container_width=True, hide_index=True)
    st.plotly_chart(price_chart(df, selected_ticker), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("RSI")
        st.plotly_chart(rsi_chart(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("MACD")
        st.plotly_chart(macd_chart(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    c, d = st.columns(2)
    with c:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("VOLUME")
        st.plotly_chart(volume_chart(df), use_container_width=True, config={"displayModeBar": False})
        vol_df = pd.DataFrame({
            "Metric": ["Daily Volume", "Vol MA20", "RVOL", "Value Traded"],
            "Value": [fmt_short(df["Volume"].iloc[-1]), fmt_short(df["Volume"].rolling(20).mean().iloc[-1]), f'{calculate_rvol(df).iloc[-1]:.2f}', fmt_short(last * df["Volume"].iloc[-1])],
        })
        st.dataframe(vol_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with d:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("ACCUMULATION / DISTRIBUTION")
        st.plotly_chart(ad_chart(df), use_container_width=True, config={"displayModeBar": False})
        acc_df = pd.DataFrame({
            "Metric": ["CMF", "OBV Trend", "Status"],
            "Value": [f'{calculate_cmf(df).iloc[-1]:.2f}', 'Up' if calculate_obv(df).iloc[-1] > calculate_obv(df).iloc[-5] else 'Down', detect_accumulation(df)],
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


def ensure_state(screener: pd.DataFrame):
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "BBCA"


def ensure_menu_state():
    if "active_menu" not in st.session_state:
        st.session_state["active_menu"] = "Dashboard"


# =========================
# MAIN
# =========================
def main():
    if st_autorefresh is not None:
        st_autorefresh(interval=60_000, key="auto")

    with st.spinner("Memuat screener IDX..."):
        screener = build_top_screener(IDX_TICKERS)

    ensure_state(screener)
    ensure_menu_state()

    render_sidebar(screener)
    render_top_market_bar()
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([4.9, 1.7], gap="large")

    with left:
        render_top_cards(screener)
        st.markdown("<br>", unsafe_allow_html=True)
        render_stock_detail(st.session_state["selected_ticker"], screener)

    with right:
        render_ticker_search_combined(screener)
        st.markdown("<br>", unsafe_allow_html=True)

        active_menu = st.session_state.get("active_menu", "Dashboard")

        if active_menu == "Dashboard":
            render_rank_table("BSJP Screener", screener, "BSJP Score", "bsjp")
            st.markdown("<br>", unsafe_allow_html=True)
            render_rank_table("Swing Trade Screener", screener, "Swing Score", "swing")
            st.markdown("<br>", unsafe_allow_html=True)
            render_rank_table("Day Trade Screener", screener, "Day Score", "day")

        elif active_menu == "Watchlist > 50":
            render_rank_table("Watchlist Score > 50", screener[screener["Score"] > 50], "Score", "watch50")

        elif active_menu == "BSJP Screener":
            render_rank_table("BSJP Screener", screener, "BSJP Score", "bsjp")

        elif active_menu == "Swing Trade Screener":
            render_rank_table("Swing Trade Screener", screener, "Swing Score", "swing")

        elif active_menu == "Day Trade Screener":
            render_rank_table("Day Trade Screener", screener, "Day Score", "day")

        elif active_menu == "Bandarmology Screener":
            render_rank_table("Bandarmology Screener", screener, "Bandar Score", "bandar")

        elif active_menu == "ARA Screener":
            render_rank_table("ARA Screener", screener, "ARA Score", "ara")

    st.markdown("<br>", unsafe_allow_html=True)
    status = "MARKET OPEN" if market_open_now() else "MARKET CLOSED"
    cls = "status-open" if market_open_now() else "status-closed"
    st.markdown(
        f'<div class="panel"><span class="{cls}">{status}</span> <span class="small-note">&nbsp;&nbsp;Sidebar menampilkan IHSG dan watchlist score > 50 | BSJP diasumsikan sebagai breakout setup jangka pendek</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
