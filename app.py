import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pytz
from datetime import datetime
import random
import warnings
import requests
import io
import json
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI", page_icon="🏛️", layout="wide")

# ── UPTIMEROBOT PING — responde rápido sin cargar la app ─────────
# En UptimeRobot apunta a: https://tu-app.streamlit.app/?ping=1
if st.query_params.get("ping") == "1":
    st.write("ok")
    st.stop()

# ── SECRETS ──────────────────────────────────────────────────────
try:
    TG_TOKEN        = st.secrets["TG_TOKEN"]
    TG_CHAT_ID      = st.secrets["TG_CHAT_ID"]
    GH_TOKEN        = st.secrets["GITHUB_TOKEN"]
    GH_REPO         = st.secrets["GITHUB_REPO"]
    TWELVEDATA_KEY  = st.secrets.get("TWELVEDATA_KEY", "")
    NEWS_API_KEY    = st.secrets.get("NEWS_API_KEY", "")
    FINNHUB_KEY     = st.secrets.get("FINNHUB_KEY", "")
    ANTHROPIC_KEY   = st.secrets.get("ANTHROPIC_KEY", "")
except:
    TG_TOKEN = TG_CHAT_ID = GH_TOKEN = GH_REPO = TWELVEDATA_KEY = ''
    NEWS_API_KEY = FINNHUB_KEY = ANTHROPIC_KEY = ''

# Import signal generator
from signal_generator import build_signal, save_signal_github

# ════════════════════════════════════════════════════════════════
#  PRECIO INTERNO — TwelveData primero, yfinance como emergencia
#  El precio visual en pantalla lo maneja el widget JS
# ════════════════════════════════════════════════════════════════
def get_precio_fallback():
    """Precio para lógica interna (paper trades, SL/TP, Telegram)
    Orden: Finnhub → TwelveData → yfinance emergencia
    """
    # 1. Finnhub — 60 calls/min, sin límite práctico
    if FINNHUB_KEY:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol":"OANDA:XAU_USD","token":FINNHUB_KEY},
                timeout=4)
            if r.status_code == 200:
                p = float(r.json().get("c", 0))
                if p > 100: return p
        except: pass
    # 2. TwelveData fallback
    if TWELVEDATA_KEY:
        try:
            r = requests.get(
                "https://api.twelvedata.com/price",
                params={"symbol":"XAU/USD","apikey":TWELVEDATA_KEY},
                timeout=4)
            if r.status_code == 200:
                p = float(r.json().get("price", 0))
                if p > 100: return p
        except: pass
    # 3. Stooq diario — precio de cierre más reciente
    try:
        from pandas_datareader import data as pdr
        df_s = pdr.DataReader("GC.F", "stooq")
        if df_s is not None and len(df_s) > 0:
            return float(df_s['Close'].iloc[0])
    except: pass
    return None

# ════════════════════════════════════════════════════════════════
#  TWELVEDATA — Fuente principal para TODO
#  yfinance solo como fallback de emergencia
# ════════════════════════════════════════════════════════════════
# Mapeo de intervalos
TD_INTERVAL_MAP   = {"5m":"5min","15m":"15min","30m":"30min","1h":"1h","4h":"4h","1d":"1day"}
TD_OUTPUTSIZE_MAP = {"5m":500,"15m":500,"30m":500,"1h":700,"4h":700,"1d":750}
FH_INTERVAL_MAP   = {"5m":"5","15m":"15","30m":"30","1h":"60","4h":"D","1d":"D"}

def _df_from_values(values_list):
    """Convierte lista de dicts OHLCV a DataFrame limpio."""
    df = pd.DataFrame(values_list)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns: df["Volume"] = 0
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    return df if len(df) >= 50 else None

def _fetch_finnhub_series(interval_yf, outputsize=500):
    """Descarga velas desde Finnhub — 60 calls/min gratis."""
    if not FINNHUB_KEY: return None
    # Finnhub candles usa timestamps Unix
    import time as _time
    resolution = FH_INTERVAL_MAP.get(interval_yf, "D")
    # Calcular rango de tiempo según outputsize e intervalo
    mins_map = {"5":5,"15":15,"30":30,"60":60,"D":1440}
    mins     = mins_map.get(resolution, 1440)
    t_to     = int(_time.time())
    t_from   = t_to - (outputsize * mins * 60)
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/forex/candle",
            params={"symbol":"OANDA:XAU_USD","resolution":resolution,
                    "from":t_from,"to":t_to,"token":FINNHUB_KEY},
            timeout=10)
        if r.status_code != 200: return None
        data = r.json()
        if data.get("s") != "ok" or "c" not in data: return None
        n = len(data["c"])
        rows = []
        for i in range(n):
            rows.append({
                "datetime": pd.to_datetime(data["t"][i], unit="s"),
                "Open":  float(data["o"][i]),
                "High":  float(data["h"][i]),
                "Low":   float(data["l"][i]),
                "Close": float(data["c"][i]),
                "Volume": float(data.get("v",[0]*n)[i]) if "v" in data else 0
            })
        if not rows: return None
        df = pd.DataFrame(rows)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        df.dropna(subset=["Open","High","Low","Close"], inplace=True)
        return df if len(df) >= 50 else None
    except: return None

def _fetch_td_series(interval_yf, outputsize=500):
    """Descarga velas desde TwelveData — fallback de Finnhub."""
    if not TWELVEDATA_KEY: return None
    td_iv = TD_INTERVAL_MAP.get(interval_yf, "1day")
    try:
        r = requests.get("https://api.twelvedata.com/time_series",
            params={"symbol":"XAU/USD","interval":td_iv,"outputsize":outputsize,
                    "apikey":TWELVEDATA_KEY,"format":"JSON","order":"ASC"}, timeout=12)
        if r.status_code != 200: return None
        data = r.json()
        if data.get("status") != "ok" or "values" not in data: return None
        return _df_from_values(data["values"])
    except: return None

def _fetch_yf_fallback(interval, period):
    """Stooq como fallback — gratis, sin key, sin límites de rate."""
    import datetime as _dt
    try:
        # Stooq via pandas_datareader
        from pandas_datareader import data as pdr
        # Stooq solo soporta diario para metales
        df = pdr.DataReader("GC.F", "stooq")
        if df is not None and len(df) >= 50:
            df = df.sort_index()
            # Asegurar columnas correctas
            df.columns = [c.capitalize() if c.lower() in ['open','high','low','close','volume'] 
                         else c for c in df.columns]
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            df.dropna(subset=['Open','High','Low','Close'], inplace=True)
            return df if len(df) >= 50 else None
    except: pass
    # Ultimo recurso: yfinance con GC=F
    try:
        import yfinance as _yf
        df = _yf.download("GC=F", period=period, interval=interval, progress=False)
        if df is not None and len(df) > 1:
            df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            return df if len(df) >= 50 else None
    except: pass
    return None

@st.cache_data(ttl=300)
def get_data(interval="1d", period="2y"):
    """Datos OHLCV — Finnhub primero, TwelveData segundo, yfinance emergencia."""
    outputsize = TD_OUTPUTSIZE_MAP.get(interval, 500)
    # 1. Finnhub
    df = _fetch_finnhub_series(interval, outputsize)
    if df is not None: return df
    # 2. TwelveData
    df = _fetch_td_series(interval, outputsize)
    if df is not None: return df
    # 3. yfinance emergencia
    df = _fetch_yf_fallback(interval, period)
    if df is not None: return df
    return _fetch_yf_fallback("1d", "2y")

@st.cache_data(ttl=300)  # 5 minutos
def add_ind(df_json):
    df = pd.read_json(io.StringIO(df_json), orient='split')

    # Asegurar que Volume existe y no tiene ceros
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        df['Volume'] = 1.0
    df['Volume'] = df['Volume'].replace(0, 1.0)

    # ── EMAs ─────────────────────────────────────────────────────
    df['EMA_9']   = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_20']  = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50']  = ta.trend.ema_indicator(df['Close'], window=50)

    # ── Momentum ─────────────────────────────────────────────────
    df['RSI']             = ta.momentum.rsi(df['Close'], window=14)
    df['RSI_fast']        = ta.momentum.rsi(df['Close'], window=7)
    df['RSI_delta']       = df['RSI'].diff(3)

    # ── MACD ─────────────────────────────────────────────────────
    df['MACD_hist']       = ta.trend.macd_diff(df['Close'])
    df['MACD_hist_delta'] = df['MACD_hist'].diff(2)

    # ── Volatilidad ───────────────────────────────────────────────
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
    df['ATR']      = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    atr_avg        = df['ATR'].rolling(20).mean()
    df['ATR_avg']  = atr_avg
    df['ATR_rel']  = (df['ATR'] / atr_avg).fillna(1.0)

    # ── Volumen — seguro ante datos sin volumen real ──────────────
    df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['OBV_delta'] = df['OBV'].pct_change(3).fillna(0)
    vol_avg         = df['Volume'].rolling(20).mean().replace(0, 1)
    df['Vol_ratio'] = (df['Volume'] / vol_avg).fillna(1.0)

    # ── Distancia EMAs ────────────────────────────────────────────
    df['Dist_EMA20'] = (df['Close'] - df['EMA_20']) / df['Close'] * 100
    df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['Close'] * 100
    df['EMA_cross']  = df['EMA_9'] - df['EMA_20']

    # ── Price action ──────────────────────────────────────────────
    df['Return_1d']  = df['Close'].pct_change(1).fillna(0)
    df['High_Low']   = (df['High'] - df['Low']) / df['Close']
    hl_range         = (df['High'] - df['Low']).replace(0, 1e-9)
    df['Body_ratio'] = (abs(df['Close'] - df['Open']) / hl_range).fillna(0.5)

    # Rellenar cualquier NaN restante con 0 antes de dropna
    feat_cols = ['EMA_9','EMA_20','EMA_50','RSI','RSI_fast','RSI_delta',
                 'MACD_hist','MACD_hist_delta','BB_width','ATR','ATR_rel',
                 'OBV_delta','Vol_ratio','Dist_EMA20','Dist_EMA50',
                 'EMA_cross','Return_1d','High_Low','Body_ratio']
    for c in feat_cols:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)

    df.dropna(subset=['Close','High','Low','Open'], inplace=True)
    return df

@st.cache_data(ttl=300)  # 5 minutos
def train_model(df_json, umbral):
    """
    Modelo optimizado:
    - Features seleccionados por importancia real para XAU/USD
    - Filtro de volatilidad mínima (elimina señales en mercado plano)
    - Horizonte ajustado por estilo de trading
    """
    df = pd.read_json(io.StringIO(df_json), orient='split')

    # Target — horizonte según estilo de trading
    _horizonte = SC.get('horizonte', 5) if 'SC' in dir() else 5
    df['Future_Return'] = df['Close'].pct_change(_horizonte).shift(-_horizonte)
    df['Target'] = 0
    df.loc[df['Future_Return'] >  umbral, 'Target'] =  1
    df.loc[df['Future_Return'] < -umbral, 'Target'] = -1

    # ── FILTRO DE VOLATILIDAD — elimina velas en mercado plano ───
    # Solo entrenar con velas donde el mercado estaba activo
    # ATR_rel > 0.8 = mercado con movimiento suficiente
    if 'ATR_rel' in df.columns:
        df = df[df['ATR_rel'] > 0.8]

    df.dropna(inplace=True)

    # ── FEATURES OPTIMIZADOS ─────────────────────────────────────
    # Orden por importancia estimada para XAU/USD:
    feats = [
        # Alta importancia — precio relativo y momentum
        'RSI', 'RSI_fast', 'RSI_delta',
        'MACD_hist', 'MACD_hist_delta',
        'Dist_EMA20', 'Dist_EMA50', 'EMA_cross',
        # Media importancia — volatilidad y estructura
        'ATR', 'ATR_rel', 'BB_width',
        'Body_ratio', 'High_Low',
        # Complementario — volumen y precio
        'OBV_delta', 'Vol_ratio',
        'Return_1d',
    ]
    feats = [f for f in feats if f in df.columns]

    X, y = df[feats], df['Target']
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.2, random_state=42, shuffle=False)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,   # max_depth reducido para evitar overfitting
        min_samples_leaf=5,              # mínimo 5 muestras por hoja
        random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)

    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4,
        min_samples_leaf=5,
        random_state=42)
    gb.fit(Xtr, ytr)

    m = rf if accuracy_score(yte, rf.predict(Xte)) >= accuracy_score(yte, gb.predict(Xte)) else gb
    return m, sc, feats, df

@st.cache_data(ttl=300)
def mtf_conf():
    """Multi-timeframe — TwelveData para todos los TFs"""
    sigs = {}
    tf_list = [("D1","1d",700),("H4","4h",600),("H1","1h",600),("M15","15m",500)]
    for name, iv, osz in tf_list:
        try:
            # Finnhub primero, luego TwelveData, luego yfinance
            df = _fetch_finnhub_series(iv, osz)
            if df is None: df = _fetch_td_series(iv, osz)
            # Fallback yfinance
            if df is None:
                period_map = {"1d":"2y","4h":"180d","1h":"60d","15m":"10d"}
                df = _fetch_yf_fallback(iv, period_map.get(iv,"60d"))
            if df is None or len(df) < 30: continue
            e20  = ta.trend.ema_indicator(df['Close'], window=20)
            e50  = ta.trend.ema_indicator(df['Close'], window=50)
            rsi  = ta.momentum.rsi(df['Close'], window=14)
            macd = ta.trend.macd(df['Close'])
            md   = ta.trend.macd_diff(df['Close'])
            p    = float(df['Close'].iloc[-1])
            em20 = float(e20.iloc[-1]); em50 = float(e50.iloc[-1])
            r    = float(rsi.iloc[-1])
            m    = float(macd.iloc[-1]) if macd is not None else 0
            mh   = float(md.iloc[-1])   if md   is not None else 0
            score = sum([p>em20, p>em50, em20>em50, r>50, m>0 and mh>0])
            sigs[name] = {'score':score,'bias':'LONG' if score>=3 else 'SHORT' if score<=1 else 'NEUTRAL','rsi':r,'precio':p}
        except: pass
    if not sigs: return sigs, 'NEUTRAL', 50
    total = sum(s['score'] for s in sigs.values())
    pct   = total / (len(sigs) * 5) * 100
    return sigs, 'LONG' if pct>=60 else 'SHORT' if pct<=40 else 'NEUTRAL', pct

# ════════════════════════════════════════════════════════════════
#  WYCKOFF — AMD (Accumulation · Manipulation · Distribution)
# ════════════════════════════════════════════════════════════════
def detect_swing_points(df, strength=3):
    highs, lows = [], []
    H = df['High'].values; L = df['Low'].values
    for i in range(strength, len(df) - strength):
        if all(H[i] > H[i-j] for j in range(1, strength+1)) and \
           all(H[i] > H[i+j] for j in range(1, strength+1)):
            highs.append({'idx': i, 'price': H[i], 'fecha': df.index[i]})
        if all(L[i] < L[i-j] for j in range(1, strength+1)) and \
           all(L[i] < L[i+j] for j in range(1, strength+1)):
            lows.append({'idx': i, 'price': L[i], 'fecha': df.index[i]})
    return highs, lows

def classify_market_structure(highs, lows):
    """Clasifica HH/HL/LH/LL y determina tendencia con BOS"""
    if len(highs) < 2 or len(lows) < 2:
        return 'INDETERMINADO', []
    labels = []
    trend  = 'NEUTRAL'
    # Últimos 4 swings
    for i in range(1, min(4, len(highs))):
        if highs[i]['price'] > highs[i-1]['price']:
            labels.append({'tipo': 'HH', 'nivel': highs[i]['price'], 'idx': highs[i]['idx']})
        else:
            labels.append({'tipo': 'LH', 'nivel': highs[i]['price'], 'idx': highs[i]['idx']})
    for i in range(1, min(4, len(lows))):
        if lows[i]['price'] > lows[i-1]['price']:
            labels.append({'tipo': 'HL', 'nivel': lows[i]['price'], 'idx': lows[i]['idx']})
        else:
            labels.append({'tipo': 'LL', 'nivel': lows[i]['price'], 'idx': lows[i]['idx']})
    # Determinar tendencia
    hh = sum(1 for l in labels if l['tipo'] == 'HH')
    hl = sum(1 for l in labels if l['tipo'] == 'HL')
    lh = sum(1 for l in labels if l['tipo'] == 'LH')
    ll = sum(1 for l in labels if l['tipo'] == 'LL')
    if hh >= 2 and hl >= 1:   trend = 'ALCISTA'
    elif ll >= 2 and lh >= 1: trend = 'BAJISTA'
    else:                     trend = 'RANGO'
    return trend, sorted(labels, key=lambda x: x['idx'])

def detect_accumulation_wyckoff(df, window=20):
    """Detecta zonas de acumulación Wyckoff — rango lateral + baja volatilidad"""
    zones = []
    for i in range(window, len(df) - window):
        segment   = df.iloc[i-window:i]
        high_zone = float(segment['High'].max())
        low_zone  = float(segment['Low'].min())
        rango     = (high_zone - low_zone) / float(df['Close'].iloc[i])
        # Zona válida: rango <3%, volumen bajo, precio no en extremos
        if rango < 0.03:
            avg_vol = float(segment['Volume'].mean()) if 'Volume' in df.columns else 1
            cur_vol = float(df['Volume'].iloc[i]) if 'Volume' in df.columns else 1
            vol_ok  = cur_vol <= avg_vol * 1.2
            zones.append({
                'idx':        i,
                'support':    round(low_zone, 2),
                'resistance': round(high_zone, 2),
                'mid':        round((high_zone + low_zone) / 2, 2),
                'rango_pct':  round(rango * 100, 2),
                'vol_baja':   vol_ok
            })
    # Solo últimas 3 zonas relevantes
    return zones[-3:] if zones else []

def detect_manipulation_wyckoff(df, acc_zones):
    """Detecta Liquidity Grabs — mechas largas que barren stops y regresan"""
    manipulations = []
    for zone in acc_zones:
        i = zone['idx']
        if i >= len(df): continue
        c = float(df['Close'].iloc[i]); h = float(df['High'].iloc[i])
        l = float(df['Low'].iloc[i]);   o = float(df['Open'].iloc[i])
        body        = abs(c - o)
        upper_wick  = h - max(c, o)
        lower_wick  = min(c, o) - l
        vol_filter  = abs(h - l) > c * 0.0015  # Solo velas con movimiento real
        if body == 0: continue
        # BULLISH GRAB — mecha larga abajo, cierra arriba del soporte
        if lower_wick > body * 2 and l < zone['support'] and c > zone['support'] and vol_filter:
            manipulations.append({
                'idx': i, 'tipo': 'BULL_GRAB',
                'nivel': zone['support'], 'zona': zone,
                'confianza': 'ALTA' if lower_wick > body * 3 else 'MEDIA'
            })
        # BEARISH GRAB — mecha larga arriba, cierra abajo de la resistencia
        elif upper_wick > body * 2 and h > zone['resistance'] and c < zone['resistance'] and vol_filter:
            manipulations.append({
                'idx': i, 'tipo': 'BEAR_GRAB',
                'nivel': zone['resistance'], 'zona': zone,
                'confianza': 'ALTA' if upper_wick > body * 3 else 'MEDIA'
            })
    return manipulations

def detect_distribution_wyckoff(df, manipulations):
    """Detecta breakout real después de manipulación — Distribución/Ruptura verdadera"""
    signals = []
    if 'ATR' not in df.columns: return signals
    macd_hist = ta.trend.macd_diff(df['Close'])
    rsi_s     = ta.momentum.rsi(df['Close'])
    ema20     = ta.trend.ema_indicator(df['Close'], window=20)
    for manip in manipulations:
        i = manip['idx']
        if i + 3 >= len(df): continue
        for j in range(i + 1, min(i + 4, len(df))):
            c    = float(df['Close'].iloc[j])
            macd = float(macd_hist.iloc[j]) if not pd.isna(macd_hist.iloc[j]) else 0
            rsi  = float(rsi_s.iloc[j])     if not pd.isna(rsi_s.iloc[j])     else 50
            em20 = float(ema20.iloc[j])     if not pd.isna(ema20.iloc[j])     else c
            atr  = float(df['ATR'].iloc[j]) if not pd.isna(df['ATR'].iloc[j]) else 50
            # Filtro rango muerto — no operar en consolidación extrema
            recent_range = float(df['High'].iloc[j-5:j].max()) - float(df['Low'].iloc[j-5:j].min())
            if recent_range / c < 0.002: continue
            # LONG — ruptura real con momentum
            if manip['tipo'] == 'BULL_GRAB':
                if c > manip['zona']['resistance'] and macd > 0 and rsi > 50 and c > em20:
                    sl = round(min(manip['nivel'], manip['zona']['support']) - atr * 1.2, 2)
                    tp = round(c + (c - sl) * 2.5, 2)
                    signals.append({'tipo':'LONG','entrada':round(c,2),'sl':sl,'tp':tp,
                                    'rr':2.5,'confianza':manip['confianza'],'idx':j})
                    break
            # SHORT — ruptura real hacia abajo
            elif manip['tipo'] == 'BEAR_GRAB':
                if c < manip['zona']['support'] and macd < 0 and rsi < 50 and c < em20:
                    sl = round(max(manip['nivel'], manip['zona']['resistance']) + atr * 1.2, 2)
                    tp = round(c - (sl - c) * 2.5, 2)
                    signals.append({'tipo':'SHORT','entrada':round(c,2),'sl':sl,'tp':tp,
                                    'rr':2.5,'confianza':manip['confianza'],'idx':j})
                    break
    return signals[-3:] if signals else []

def run_wyckoff(df):
    """Corre análisis Wyckoff completo y devuelve resumen"""
    highs, lows   = detect_swing_points(df, strength=3)
    trend, struct = classify_market_structure(highs, lows)
    acc_zones     = detect_accumulation_wyckoff(df)
    manips        = detect_manipulation_wyckoff(df, acc_zones)
    signals       = detect_distribution_wyckoff(df, manips)
    # Señal activa — la más reciente si existe
    active = signals[-1] if signals else None
    return {
        'trend':      trend,
        'structure':  struct[-6:],
        'acc_zones':  acc_zones,
        'manips':     manips[-3:],
        'signals':    signals,
        'active':     active,
        'highs':      highs[-5:],
        'lows':       lows[-5:]
    }

# ════════════════════════════════════════════════════════════════
#  SMC AVANZADO
# ════════════════════════════════════════════════════════════════
def detect_smc_advanced(df, lookback=50):
    res = {
        'bos':[], 'msb':[], 'order_blocks':[], 'fvg':[],
        'eqh':[], 'eql':[], 'liquidity_swings':[],
        'bias':'NEUTRAL', 'bias_score':0, 'gladiador_entry':None
    }
    if len(df) < lookback + 5: return res
    H = df['High'].values; L = df['Low'].values
    C = df['Close'].values; O = df['Open'].values
    V = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))

    # Swing points
    swing_highs = []; swing_lows = []
    for i in range(3, len(df)-3):
        if all(H[i]>H[i-j] for j in range(1,4)) and all(H[i]>H[i+j] for j in range(1,4)):
            swing_highs.append((i, H[i]))
        if all(L[i]<L[i-j] for j in range(1,4)) and all(L[i]<L[i+j] for j in range(1,4)):
            swing_lows.append((i, L[i]))
    swing_highs = swing_highs[-8:]; swing_lows = swing_lows[-8:]
    for sh in swing_highs[-4:]:
        res['liquidity_swings'].append({'tipo':'SWING HIGH','nivel':round(sh[1],2),'idx':sh[0]})
    for sl in swing_lows[-4:]:
        res['liquidity_swings'].append({'tipo':'SWING LOW','nivel':round(sl[1],2),'idx':sl[0]})

    precio_actual = C[-1]
    # BOS / MSB
    if len(swing_highs)>=2:
        prev_sh = swing_highs[-2][1]; last_sh = swing_highs[-1][1]
        if last_sh > prev_sh:
            res['bos'].append({'tipo':'BOS ALCISTA','nivel':round(last_sh,2),'fuerza':'FUERTE' if last_sh>prev_sh*1.002 else 'DÉBIL'})
            res['bias_score']+=2
        else:
            res['msb'].append({'tipo':'MSB BAJISTA','nivel':round(last_sh,2),'fuerza':'FUERTE' if last_sh<prev_sh*0.998 else 'DÉBIL'})
            res['bias_score']-=2
    if len(swing_lows)>=2:
        prev_sl = swing_lows[-2][1]; last_sl = swing_lows[-1][1]
        if last_sl < prev_sl:
            res['bos'].append({'tipo':'BOS BAJISTA','nivel':round(last_sl,2),'fuerza':'FUERTE' if last_sl<prev_sl*0.998 else 'DÉBIL'})
            res['bias_score']-=2
        else:
            res['msb'].append({'tipo':'MSB ALCISTA','nivel':round(last_sl,2),'fuerza':'FUERTE' if last_sl>prev_sl*1.002 else 'DÉBIL'})
            res['bias_score']+=2

    # Order Blocks
    avg_body = np.mean([abs(C[j]-O[j]) for j in range(-lookback,-1)])
    avg_vol  = np.mean(V[-lookback:]) if V is not None else 1
    for i in range(-lookback+2, -2):
        body  = abs(C[i]-O[i]); vol_i = V[i] if V is not None else 1
        next_move = C[i+2]-C[i]
        if body > avg_body*1.2:
            strength = 'FUERTE' if body>avg_body*2 or vol_i>avg_vol*1.5 else 'NORMAL'
            if C[i]<O[i] and next_move>avg_body*1.5:
                res['order_blocks'].append({'tipo':'OB ALCISTA','top':round(O[i],2),'bottom':round(C[i],2),'mid':round((O[i]+C[i])/2,2),'fuerza':strength})
                res['bias_score']+=1
            elif C[i]>O[i] and next_move<-avg_body*1.5:
                res['order_blocks'].append({'tipo':'OB BAJISTA','top':round(C[i],2),'bottom':round(O[i],2),'mid':round((O[i]+C[i])/2,2),'fuerza':strength})
                res['bias_score']-=1
    res['order_blocks'] = res['order_blocks'][-5:]

    # EQH / EQL
    tol = precio_actual*0.0015
    for i in range(len(swing_highs)):
        for j in range(i+1,len(swing_highs)):
            if abs(swing_highs[i][1]-swing_highs[j][1])<tol:
                lvl = round((swing_highs[i][1]+swing_highs[j][1])/2,2)
                res['eqh'].append({'nivel':lvl,'tipo':'EQH — Liquidez Alcista'})
    for i in range(len(swing_lows)):
        for j in range(i+1,len(swing_lows)):
            if abs(swing_lows[i][1]-swing_lows[j][1])<tol:
                lvl = round((swing_lows[i][1]+swing_lows[j][1])/2,2)
                res['eql'].append({'nivel':lvl,'tipo':'EQL — Liquidez Bajista'})

    # FVG
    for i in range(-lookback,-2):
        gap = abs(L[i+2]-H[i])
        if gap > precio_actual*0.001:
            if L[i+2]>H[i]:
                res['fvg'].append({'tipo':'FVG ALCISTA','top':round(L[i+2],2),'bottom':round(H[i],2),'size':round(gap,2)})
            elif H[i+2]<L[i]:
                g2=abs(L[i]-H[i+2])
                res['fvg'].append({'tipo':'FVG BAJISTA','top':round(L[i],2),'bottom':round(H[i+2],2),'size':round(g2,2)})
    res['fvg'] = res['fvg'][-4:]

    if res['bias_score']>=2:    res['bias']='ALCISTA'
    elif res['bias_score']<=-2: res['bias']='BAJISTA'
    else:                       res['bias']='NEUTRAL'

    # Gladiador micro entry
    near_ob_bull  = any(ob['bottom']<=precio_actual<=ob['top']*1.001 for ob in res['order_blocks'] if 'ALCISTA' in ob['tipo'])
    near_ob_bear  = any(ob['bottom']*0.999<=precio_actual<=ob['top'] for ob in res['order_blocks'] if 'BAJISTA' in ob['tipo'])
    near_eqh      = any(abs(e['nivel']-precio_actual)/precio_actual<0.002 for e in res['eqh'])
    near_eql      = any(abs(e['nivel']-precio_actual)/precio_actual<0.002 for e in res['eql'])
    near_fvg_bull = any(f['bottom']<=precio_actual<=f['top'] for f in res['fvg'] if 'ALCISTA' in f['tipo'])
    near_fvg_bear = any(f['bottom']<=precio_actual<=f['top'] for f in res['fvg'] if 'BAJISTA' in f['tipo'])
    if near_ob_bull or near_fvg_bull:   res['gladiador_entry']='LONG_REBOTE'
    elif near_ob_bear or near_fvg_bear: res['gladiador_entry']='SHORT_REBOTE'
    elif near_eqh:                      res['gladiador_entry']='SHORT_LIQUIDEZ'
    elif near_eql:                      res['gladiador_entry']='LONG_LIQUIDEZ'
    return res

# ════════════════════════════════════════════════════════════════
#  MODOS DE SEÑAL
# ════════════════════════════════════════════════════════════════
MODO_CONFIG = {
    # ORÁCULO — calidad sobre cantidad
    # Requiere estructura completa: BOS + sesión + OB o RSI divergencia
    # Opera SOLO en nivel FUERTE (≥6.0) o MEDIA (≥4.5)
    # SL/TP amplios — trades de mayor duración
    'Oráculo 🏛️': {
        'umbral':    0.004,   # movimiento mínimo 0.4% para generar señal
        'atr_sl':    1.8,     # SL amplio — aguanta ruido del mercado
        'atr_tp':    3.5,     # TP 3.5×ATR — R:R 1:1.9
        'min_conf':  58,      # confianza mínima ML alta
        'min_score': 4.5,     # score mínimo alto
        'req_bos':   True,    # BOS obligatorio siempre
        'req_sesion':True,    # solo en ventanas activas
        'desc': 'Calidad máxima. BOS obligatorio. Score ≥4.5. Solo ventanas activas.'
    },
    # GLADIADOR — frecuencia y agilidad
    # Opera con estructura parcial: solo OB, FVG o EQH/EQL
    # No requiere BOS — detecta micro movimientos
    # SL/TP ajustados — trades rápidos de scalping
    'Gladiador ⚔️': {
        'umbral':    0.001,   # movimiento mínimo 0.1% — mucho más sensible
        'atr_sl':    0.5,     # SL muy ajustado
        'atr_tp':    1.0,     # TP 1×ATR — salida rápida
        'min_conf':  38,      # confianza mínima baja — más señales
        'min_score': 2.5,     # score mínimo bajo — opera con confluencia parcial
        'req_bos':   False,   # NO requiere BOS
        'req_sesion':False,   # puede operar fuera de ventana principal
        'desc': 'Alta frecuencia. Sin BOS requerido. Score ≥2.5. Micro entradas SMC.'
    },
}
STYLE_CONFIG = {
    # ── SCALPING M5 ───────────────────────────────────────────────
    # Velas de 5 minutos. Trades de 5-30 minutos.
    # SL muy pequeño ($3-8), TP pequeño ($5-15)
    # Señales frecuentes, alta concentración requerida
    "Scalping": {
        "interval": "5m",    # datos en velas de 5 min
        "period":   "5d",
        "label":    "M5",
        "atr_sl":   0.4,     # SL = 0.4×ATR (~$3-6)
        "atr_tp":   0.8,     # TP = 0.8×ATR (~$6-12) R:R 1:2
        "umbral":   0.0006,  # detecta movimientos desde 0.06%
        "min_score":2.5,     # umbral bajo → más señales
        "horizonte":2,       # predice 2 velas adelante (10 min)
        "desc": "M5 · Trades 5-30min · SL $3-8 · Solo London+NY Open"
    },
    # ── DAY TRADING M15 ───────────────────────────────────────────
    # Velas de 15 minutos. Trades de 1-4 horas.
    # SL medio ($15-30), TP medio ($30-60)
    # Balance entre frecuencia y calidad
    "Day Trading": {
        "interval": "15m",   # datos en velas de 15 min
        "period":   "10d",
        "label":    "M15",
        "atr_sl":   1.2,     # SL = 1.2×ATR (~$15-25)
        "atr_tp":   2.4,     # TP = 2.4×ATR (~$30-50) R:R 1:2
        "umbral":   0.0025,  # detecta movimientos desde 0.25%
        "min_score":4.0,     # umbral medio → balance calidad/frecuencia
        "horizonte":5,       # predice 5 velas adelante (75 min)
        "desc": "M15 · Trades 1-4h · SL $15-25 · Cierra antes 5PM MX"
    },
    # ── SWING H4 ──────────────────────────────────────────────────
    # Velas de 4 horas. Trades de 1-5 días.
    # SL amplio ($50-100), TP amplio ($100-200)
    # Pocas señales, alta calidad, paciencia de días
    "Swing": {
        "interval": "4h",    # datos en velas de 4h
        "period":   "180d",
        "label":    "H4",
        "atr_sl":   2.5,     # SL = 2.5×ATR (~$50-100)
        "atr_tp":   5.0,     # TP = 5×ATR (~$100-200) R:R 1:2
        "umbral":   0.006,   # detecta solo movimientos grandes +0.6%
        "min_score":5.5,     # umbral alto → solo señales premium
        "horizonte":10,      # predice 10 velas adelante (40h)
        "desc": "H4 · Trades 1-5 días · SL $50-100 · BOS D1 obligatorio"
    },
}

# ════════════════════════════════════════════════════════════════
#  MOTOR DE SEÑALES — SISTEMA DE 3 NIVELES
#
#  Nivel 1 FUERTE  — BOS + sesión + RSI div o OB + ML confirmado
#  Nivel 2 MEDIA   — BOS + sesión + ML confirmado (sin doble conf)
#  Nivel 3 DÉBIL   — Solo SMC o solo ML (informativa, no opera)
# ════════════════════════════════════════════════════════════════

def _score_mercado(df) -> tuple[float, str]:
    """
    Score 0-3 de actividad del mercado.
    0 = plano, 3 = muy activo
    """
    score = 0
    razones = []
    if 'ATR_rel' in df.columns:
        atr_rel = float(df['ATR_rel'].iloc[-1])
        if atr_rel >= 1.1:   score += 1.5; razones.append(f"ATR×{atr_rel:.2f} 🔥")
        elif atr_rel >= 0.8: score += 1.0; razones.append(f"ATR×{atr_rel:.2f} ✅")
        elif atr_rel >= 0.65: score += 0.5; razones.append(f"ATR×{atr_rel:.2f} ⚠️")
        else: razones.append(f"ATR×{atr_rel:.2f} ❌ plano")
    if 'BB_width' in df.columns:
        bb_w     = float(df['BB_width'].iloc[-1])
        bb_w_avg = float(df['BB_width'].rolling(20).mean().iloc[-1])
        if bb_w > bb_w_avg * 1.0: score += 0.5; razones.append("BB expandido ✅")
        elif bb_w < bb_w_avg * 0.6: razones.append("BB comprimido ❌")
    if 'Vol_ratio' in df.columns:
        vr = float(df['Vol_ratio'].iloc[-1])
        if vr >= 1.3: score += 1.0; razones.append(f"Vol×{vr:.1f} 🔥")
        elif vr >= 0.9: score += 0.5; razones.append(f"Vol×{vr:.1f} ✅")
    return score, " · ".join(razones) if razones else "Sin datos"

def _score_sesion(hora_mx: int) -> tuple[float, str]:
    """
    Score 0-2 según sesión activa.
    London+NY = 2, London = 1.5, NY tarde = 1, resto = 0
    """
    if 8  <= hora_mx < 12: return 2.0, "London+NY 🔥 (mejor ventana)"
    if 3  <= hora_mx < 5:  return 1.5, "London Open ✅"
    if 12 <= hora_mx < 15: return 1.0, "NY tarde ✅"
    if 7  <= hora_mx < 8:  return 0.8, "Pre-NY (aceptable)"
    return 0.0, "Fuera de ventana ❌"

def _score_smc(smc, pred) -> tuple[float, str]:
    """
    Score 0-3 de confirmación SMC.
    BOS = base, OB = confirmación, FVG/EQH = complemento
    """
    score = 0; razones = []
    has_bos = bool(smc['bos'])
    has_msb = bool(smc['msb'])
    if has_bos:
        bos_dir = 'ALCISTA' if any('ALCISTA' in b['tipo'] for b in smc['bos']) else 'BAJISTA'
        if (pred == 1 and bos_dir == 'ALCISTA') or (pred == -1 and bos_dir == 'BAJISTA'):
            score += 1.5; razones.append("BOS alineado 🔥")
        else:
            score += 0.3; razones.append("BOS opuesto ⚠️")
    if has_msb:
        msb_dir = 'ALCISTA' if any('ALCISTA' in m['tipo'] for m in smc['msb']) else 'BAJISTA'
        if (pred == 1 and msb_dir == 'ALCISTA') or (pred == -1 and msb_dir == 'BAJISTA'):
            score += 0.5; razones.append("MSB confirmado ✅")
    # Order Block cerca
    ob_alineado = any(
        ('ALCISTA' in ob['tipo'] and pred == 1) or ('BAJISTA' in ob['tipo'] and pred == -1)
        for ob in smc['order_blocks']
    )
    if ob_alineado: score += 1.0; razones.append("OB alineado ✅")
    # FVG cerca
    if smc.get('fvg'): score += 0.3; razones.append("FVG presente")
    return score, " · ".join(razones) if razones else "Sin estructura"

def _score_rsi(df, pred) -> tuple[float, str]:
    """
    Score 0-2 de confirmación RSI.
    Divergencia = señal de reversión real.
    """
    score = 0; razones = []
    if 'RSI' not in df.columns: return 0, "Sin RSI"
    rsi      = float(df['RSI'].iloc[-1])
    rsi_fast = float(df['RSI_fast'].iloc[-1]) if 'RSI_fast' in df.columns else rsi
    rsi_d    = float(df['RSI_delta'].iloc[-1]) if 'RSI_delta' in df.columns else 0

    # RSI en zona correcta
    if pred == 1:
        if 35 <= rsi <= 60:   score += 1.0; razones.append(f"RSI {rsi:.0f} zona alcista ✅")
        elif rsi < 35:        score += 1.5; razones.append(f"RSI {rsi:.0f} sobrevendido 🔥")
        elif rsi > 70:        score -= 0.5; razones.append(f"RSI {rsi:.0f} sobrecomprado ⚠️")
    elif pred == -1:
        if 40 <= rsi <= 65:   score += 1.0; razones.append(f"RSI {rsi:.0f} zona bajista ✅")
        elif rsi > 65:        score += 1.5; razones.append(f"RSI {rsi:.0f} sobrecomprado 🔥")
        elif rsi < 30:        score -= 0.5; razones.append(f"RSI {rsi:.0f} sobrevendido ⚠️")

    # Divergencia RSI — muy fiable
    if pred == 1  and rsi_d > 2:  score += 0.5; razones.append("RSI acelerando ✅")
    if pred == -1 and rsi_d < -2: score += 0.5; razones.append("RSI cayendo ✅")

    # RSI fast confirmando
    if pred == 1  and rsi_fast > rsi: score += 0.3; razones.append("RSI7 > RSI14 ✅")
    if pred == -1 and rsi_fast < rsi: score += 0.3; razones.append("RSI7 < RSI14 ✅")

    return max(0, score), " · ".join(razones) if razones else "RSI neutral"

def _nivel_señal(score_total: float) -> tuple[int, str, str]:
    """
    Convierte score total en nivel de señal.
    Retorna (nivel, label, color)
    """
    if score_total >= 6.0:
        return 1, "🔥 SEÑAL FUERTE",  "#4CAF82"
    elif score_total >= 3.5:
        return 2, "✅ SEÑAL MEDIA",   "#C8A96E"
    elif score_total >= 2.0:
        return 3, "📡 SEÑAL DÉBIL",   "#6B8FCE"
    else:
        return 0, "⏳ SIN SEÑAL",     "#555555"

def get_signal_oraculo(df, smc, features, m, sc, atr_sl, atr_tp):
    """
    Oráculo — sistema de 3 niveles.
    Opera en nivel 1 y 2. Nivel 3 = informativo. 0 = espera.
    """
    ul   = df[features].iloc[-1:]
    pred = int(m.predict(sc.transform(ul))[0])
    prob = m.predict_proba(sc.transform(ul))[0]
    conf = max(prob) * 100
    p    = float(df['Close'].iloc[-1])
    atr  = float(df['ATR'].iloc[-1])

    hora_mx = datetime.now(pytz.timezone('America/Mexico_City')).hour

    # ── Calcular scores ───────────────────────────────────────────
    s_mercado, r_mercado = _score_mercado(df)
    s_sesion,  r_sesion  = _score_sesion(hora_mx)
    s_smc,     r_smc     = _score_smc(smc, pred)
    s_rsi,     r_rsi     = _score_rsi(df, pred)
    s_ml = (conf / 100) * 2.0   # ML aporta máx 2 puntos

    score_total = s_mercado + s_sesion + s_smc + s_rsi + s_ml
    nivel, nivel_label, nivel_color = _nivel_señal(score_total)

    # Oráculo usa su propio min_score — más estricto que Gladiador
    min_score_modo = MC.get('min_score', 4.5) if 'MC' in dir() else 4.5
    if score_total < min_score_modo:
        pred = 0

    # Mercado completamente plano = no operar nunca
    if s_mercado < 0.5:
        pred = 0

    scores = {
        'total': round(score_total, 2),
        'mercado': round(s_mercado, 2),
        'sesion':  round(s_sesion,  2),
        'smc':     round(s_smc,     2),
        'rsi':     round(s_rsi,     2),
        'ml':      round(s_ml,      2),
        'nivel':   nivel,
        'label':   nivel_label,
        'color':   nivel_color,
        'razon_mercado': r_mercado,
        'razon_sesion':  r_sesion,
        'razon_smc':     r_smc,
        'razon_rsi':     r_rsi,
    }

    return int(pred), prob, p, atr, \
           round(p-atr*atr_sl,2), round(p+atr*atr_tp,2), \
           round(p+atr*atr_sl,2), round(p-atr*atr_tp,2), scores

def get_signal_gladiador(df, smc, features, m, sc, atr_sl, atr_tp):
    """
    Gladiador — más permisivo, opera desde nivel 2 con umbral menor.
    También usa micro entradas SMC cuando ML es lateral.
    """
    ul   = df[features].iloc[-1:]
    pred = int(m.predict(sc.transform(ul))[0])
    prob = m.predict_proba(sc.transform(ul))[0]
    conf = max(prob) * 100
    p    = float(df['Close'].iloc[-1])
    atr  = float(df['ATR'].iloc[-1])

    hora_mx = datetime.now(pytz.timezone('America/Mexico_City')).hour

    # Micro entrada SMC si ML dice lateral
    if pred == 0 and smc.get('gladiador_entry') and conf >= 38:
        ge = smc['gladiador_entry']
        if 'LONG' in ge:   pred = 1
        elif 'SHORT' in ge: pred = -1

    s_mercado, r_mercado = _score_mercado(df)
    s_sesion,  r_sesion  = _score_sesion(hora_mx)
    s_smc,     r_smc     = _score_smc(smc, pred)
    s_rsi,     r_rsi     = _score_rsi(df, pred)
    s_ml = (conf / 100) * 2.0

    score_total = s_mercado + s_sesion + s_smc + s_rsi + s_ml
    nivel, nivel_label, nivel_color = _nivel_señal(score_total)

    # Gladiador opera con threshold mucho más bajo que Oráculo
    min_score_modo = MC.get('min_score', 2.5) if 'MC' in dir() else 2.5
    if score_total < min_score_modo:
        pred = 0

    # Mercado muerto = no operar
    if s_mercado < 0.4:
        pred = 0

    scores = {
        'total': round(score_total, 2),
        'mercado': round(s_mercado, 2),
        'sesion':  round(s_sesion,  2),
        'smc':     round(s_smc,     2),
        'rsi':     round(s_rsi,     2),
        'ml':      round(s_ml,      2),
        'nivel':   nivel,
        'label':   nivel_label,
        'color':   nivel_color,
        'razon_mercado': r_mercado,
        'razon_sesion':  r_sesion,
        'razon_smc':     r_smc,
        'razon_rsi':     r_rsi,
    }

    return int(pred), prob, p, atr, \
           round(p-atr*atr_sl,2), round(p+atr*atr_tp,2), \
           round(p+atr*atr_sl,2), round(p-atr*atr_tp,2), scores

def calc_pos(capital, risk, entrada, sl):
    r = capital*(risk/100); d = abs(entrada-sl)
    if d==0: return 0,0
    return round(r/(d*100),2), round(r,2)

# ════════════════════════════════════════════════════════════════
#  GITHUB PERSISTENCE
# ════════════════════════════════════════════════════════════════
GH_FILE = "mimi_data.json"

def gh_load():
    if not GH_TOKEN or not GH_REPO: return {}
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_FILE}"
        r   = requests.get(url, headers={"Authorization":f"token {GH_TOKEN}"}, timeout=5)
        if r.status_code==200:
            return json.loads(base64.b64decode(r.json().get('content','')).decode())
    except: pass
    return {}

def gh_save(data):
    if not GH_TOKEN or not GH_REPO: return
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_FILE}"
        r   = requests.get(url, headers={"Authorization":f"token {GH_TOKEN}"}, timeout=5)
        sha = r.json().get('sha','') if r.status_code==200 else ''
        cnt = base64.b64encode(json.dumps(data, default=str).encode()).decode()
        payload = {"message":"MIMI-AI update","content":cnt}
        if sha: payload["sha"]=sha
        requests.put(url, headers={"Authorization":f"token {GH_TOKEN}"}, json=payload, timeout=5)
    except: pass

# ════════════════════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════════════════════
def send_tg(msg):
    if not TG_TOKEN or not TG_CHAT_ID: return False
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={'chat_id':TG_CHAT_ID,'text':msg,'parse_mode':'Markdown'}, timeout=5)
        return True
    except: return False

def get_tg_updates(offset=0):
    if not TG_TOKEN: return []
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?limit=10&offset={offset}", timeout=5)
        if r.status_code==200: return r.json().get('result',[])
    except: pass
    return []

def parse_tg_command(txt):
    t = txt.lower().strip()
    if any(w in t for w in ['entré','entre','sí entro','si entro','entro','long','short','sí','si']): return 'ENTRO'
    if any(w in t for w in ['no','no entro','cancelar']): return 'NO_ENTRO'
    if any(w in t for w in ['me quedo','quedo','mantener','hold']): return 'MANTENER'
    if any(w in t for w in ['salgo','salir','cerrar','exit','sal']): return 'SALIR'
    if any(w in t for w in ['estado','status','como voy','posicion']): return 'STATUS'
    if any(w in t for w in ['señal','signal','analiza']): return 'SEÑAL'
    if any(w in t for w in ['wyckoff','amd','acumulacion','manipulacion']): return 'WYCKOFF'
    return 'TEXTO_LIBRE'

def analizar_texto_libre(txt, precio, pred, prob, rsi, atr, smc, ET, conf, sl_long, tp_long, sl_short, tp_short, rr, wyckoff=None):
    t  = txt.lower()
    ot = [tr for tr in st.session_state.paper_trades if tr['estado']=='ABIERTO']
    if any(w in t for w in ['rsi','momentum']):
        return f"RSI: {rsi:.1f} — {'sobrecomprado ⚠️' if rsi>70 else 'sobrevendido ⚠️' if rsi<30 else 'neutral ✅'}"
    if any(w in t for w in ['smc','order block','ob','estructura']):
        ob_txt = f"\nOB: {smc['order_blocks'][-1]['tipo']} ${smc['order_blocks'][-1]['bottom']:,.2f}–${smc['order_blocks'][-1]['top']:,.2f}" if smc['order_blocks'] else ""
        return f"Bias SMC: {smc['bias']}{ob_txt}"
    if any(w in t for w in ['wyckoff','amd','fase']):
        if wyckoff:
            act = wyckoff['active']
            return f"Wyckoff AMD — Tendencia: {wyckoff['trend']}\n{'Señal activa: '+act['tipo']+' @ $'+str(act['entrada']) if act else 'Sin señal activa. Esperando ruptura.'}"
        return "Wyckoff no disponible ahorita."
    if any(w in t for w in ['sl','stop','riesgo']):
        return f"SL LONG: ${sl_long:,.2f} | SL SHORT: ${sl_short:,.2f}"
    if any(w in t for w in ['tp','objetivo','target']):
        return f"TP LONG: ${tp_long:,.2f} | TP SHORT: ${tp_short:,.2f} | R:R 1:{rr}"
    if ot:
        t2  = ot[0]
        pnl = (precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*100
        return f"Tienes {t2['dir']} abierto @ ${t2['entrada']:,.2f}\nActual: ${precio:,.2f} | P&L: {'+'if pnl>0 else ''}${pnl:.2f}\n{'🟢 Mantén' if pnl>0 else '🔴 Precaución'}"
    return f"Señal: {ET.get(pred)} {conf:.1f}% | ${precio:,.2f}\nComandos: entré · no · salgo · estado · señal · wyckoff · me quedo"

def process_tg_updates(precio, pred, prob, rsi, atr, sl_long, tp_long, sl_short, tp_short, rr, smc, conf, ET, risk_pct, wyckoff=None):
    updates = get_tg_updates(offset=st.session_state.last_tg_update)
    for u in updates:
        uid = u.get('update_id',0)
        if uid <= st.session_state.last_tg_update: continue
        st.session_state.last_tg_update = uid+1
        txt = u.get('message',{}).get('text','')
        if not txt: continue
        cmd = parse_tg_command(txt)
        ot  = [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
        mx  = pytz.timezone('America/Mexico_City')
        ah  = datetime.now(mx)

        if cmd=='ENTRO':
            if pred!=0 and not ot:
                sl_r = sl_long if pred==1 else sl_short
                tp_r = tp_long if pred==1 else tp_short
                lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, sl_r)
                st.session_state.paper_trades.append({
                    'id':len(st.session_state.paper_trades)+1,
                    'dir':'LONG 📈' if pred==1 else 'SHORT 📉',
                    'entrada':precio,'sl':sl_r,'tp':tp_r,'lotes':lot,'riesgo':risg,
                    'estado':'ABIERTO','fecha':ah.strftime('%d/%m %H:%M'),
                    'resultado':'PENDIENTE','pnl':0
                })
                send_tg(f"✅ *Trade registrado*\n{'LONG 📈' if pred==1 else 'SHORT 📉'} @ ${precio:,.2f}\nSL: ${sl_r:,.2f} | TP: ${tp_r:,.2f}\nLotes: {lot} | Riesgo: ${risg:.2f}\n\n_Responde 'estado' o 'salgo' cuando quieras._")
            elif ot: send_tg("⚠️ Ya tienes un trade abierto. Escríbeme 'estado'.")
            else: send_tg("⚠️ Señal LATERAL. Sin entrada clara.")

        elif cmd=='NO_ENTRO':
            send_tg(f"🏛️ Señal rechazada. {ET.get(pred)} @ ${precio:,.2f}\n_El estoico espera. La próxima señal llegará._")

        elif cmd=='MANTENER':
            if ot:
                t2  = ot[0]
                pnl = (precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*100
                send_tg(f"🏛️ *Posición activa*\n{t2['dir']} @ ${t2['entrada']:,.2f}\nActual: ${precio:,.2f}\nP&L: {'+'if pnl>0 else ''}${pnl:.2f}\nSL: ${t2['sl']:,.2f} | TP: ${t2['tp']:,.2f}\n\n{'🟢 Mantén mientras la estructura aguante.' if pnl>0 else '🔴 Evalúa si tu razón de entrada sigue válida.'}")
            else: send_tg("Sin trade abierto.")

        elif cmd=='SALIR':
            if ot:
                t2  = ot[0]
                pnl = (precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*100
                t2['estado']='CERRADO'; t2['pnl']=round(pnl,2)
                t2['resultado']='WIN ✅' if pnl>0 else 'LOSS ❌'
                send_tg(f"{'✅' if pnl>0 else '❌'} *Trade cerrado*\nSalida: ${precio:,.2f}\nP&L: {'+'if pnl>0 else ''}${pnl:.2f}\n\n_{'El sabio toma sus ganancias.' if pnl>0 else 'Una pérdida aceptada es una victoria de carácter.'}_")
            else: send_tg("Sin trade abierto.")

        elif cmd=='STATUS':
            if ot:
                t2  = ot[0]
                pnl = (precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*100
                est = "🚨 SL — SAL YA" if (('LONG' in t2['dir'] and precio<=t2['sl']) or ('SHORT' in t2['dir'] and precio>=t2['sl'])) else "🎯 TP — TOMA GANANCIA" if (('LONG' in t2['dir'] and precio>=t2['tp']) or ('SHORT' in t2['dir'] and precio<=t2['tp'])) else "🟢 MANTÉN" if pnl>0 else "🔴 PRECAUCIÓN"
                send_tg(f"👁️ *Estado*\n{t2['dir']} @ ${t2['entrada']:,.2f}\nActual: ${precio:,.2f}\nP&L: {'+'if pnl>0 else ''}${pnl:.2f}\n{est}\n\nResponde 'me quedo' o 'salgo'")
            else:
                send_tg(f"📊 Sin posición.\nSeñal: *{ET.get(pred)}* {conf:.1f}%\nPrecio: ${precio:,.2f}\nCapital: ${st.session_state.capital:,.2f}")

        elif cmd=='SEÑAL':
            sl_r = sl_long if pred>=0 else sl_short
            tp_r = tp_long if pred>=0 else tp_short
            wy_txt = ""
            if wyckoff and wyckoff['active']:
                wy_txt = f"\n🏺 Wyckoff: {wyckoff['active']['tipo']} @ ${wyckoff['active']['entrada']:,.2f}"
            send_tg(f"🏛️ *MIMI-AI — Señal*\n*{ET.get(pred)}* | {conf:.1f}%\nPrecio: ${precio:,.2f}\nSL: ${sl_r:,.2f} | TP: ${tp_r:,.2f} | R:R 1:{rr}\nSMC: {smc['bias']} | RSI: {rsi:.1f}{wy_txt}\n\n{'_Responde *entré* para registrar._' if pred!=0 else '_Mercado lateral — espera ruptura._'}")

        elif cmd=='WYCKOFF':
            if wyckoff:
                act = wyckoff['active']
                manip_str = f"\nManipulación: {wyckoff['manips'][-1]['tipo']} [{wyckoff['manips'][-1]['confianza']}]" if wyckoff['manips'] else ""
                send_tg(f"🏺 *Wyckoff AMD*\nTendencia: *{wyckoff['trend']}*\nZonas acumulación: {len(wyckoff['acc_zones'])}{manip_str}\n{'✅ Señal: '+act['tipo']+' @ $'+str(act['entrada'])+' | SL: $'+str(act['sl'])+' | TP: $'+str(act['tp']) if act else '⏳ Sin señal activa — esperando ruptura.'}")
            else: send_tg("Wyckoff no disponible.")

        elif cmd=='TEXTO_LIBRE':
            resp = analizar_texto_libre(txt, precio, pred, prob, rsi, atr, smc, ET, conf, sl_long, tp_long, sl_short, tp_short, rr, wyckoff)
            send_tg(f"🏛️ *MIMI-AI:*\n{resp}")

    sv2 = {'paper_trades':st.session_state.paper_trades,
            'signal_history':st.session_state.signal_history[-50:],
            'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
            'modo':st.session_state.modo,'last_tg_update':st.session_state.last_tg_update}
    gh_save(sv2)

# ════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════
if 'loaded' not in st.session_state:
    sv = gh_load()
    st.session_state.paper_trades   = sv.get('paper_trades',[])
    st.session_state.signal_history = sv.get('signal_history',[])
    st.session_state.capital        = sv.get('capital',1000.0)
    st.session_state.trade_style    = sv.get('trade_style','Day Trading')
    st.session_state.modo           = sv.get('modo','Oráculo 🏛️')
    st.session_state.tema           = sv.get('tema','Mármol Griego')
    st.session_state.risk_pct       = sv.get('risk_pct', 1.0)   # ← persiste
    st.session_state.chat_history   = []
    st.session_state.last_tg_update = sv.get('last_tg_update',0)
    st.session_state.loaded         = True

# ════════════════════════════════════════════════════════════════
#  TEMAS
# ════════════════════════════════════════════════════════════════
THEMES = {
    # Clásicos
    "Mármol Griego":   {"primary":"#C8A96E","secondary":"#8B6914","bg":"#0a0905","card":"#13100a"},
    "Bronce Estoico":  {"primary":"#CD7F32","secondary":"#8B4513","bg":"#080503","card":"#120a05"},
    "Lapislázuli":     {"primary":"#6B8FCE","secondary":"#3A5A9B","bg":"#03060f","card":"#070b18"},
    "Olimpo Oscuro":   {"primary":"#9B7FD4","secondary":"#6B4FA0","bg":"#060308","card":"#0d0614"},
    "Athena":          {"primary":"#7BAF9E","secondary":"#3D7A68","bg":"#030a08","card":"#06120f"},
    # Nuevos
    "Sangre de Toro":  {"primary":"#C0392B","secondary":"#922B21","bg":"#080202","card":"#120504"},
    "Plata Espartana": {"primary":"#BDC3C7","secondary":"#808B96","bg":"#050506","card":"#0d0d0f"},
    "Oro Micénico":    {"primary":"#F4D03F","secondary":"#D4AC0D","bg":"#08080a","card":"#131305"},
}
T = THEMES.get(st.session_state.get('tema','Mármol Griego'), THEMES["Mármol Griego"])

FRASES = [
    ("Warren Buffett","El mercado transfiere dinero del impaciente al paciente."),
    ("Marco Aurelio","Tienes poder sobre tu mente, no sobre los eventos externos."),
    ("Jesse Livermore","El dinero se hace sentado y esperando."),
    ("Epicteto","No busques que los eventos sucedan como deseas."),
    ("George Soros","Lo que importa es cuánto ganas cuando tienes razón."),
    ("Séneca","Sé avaro con tu tiempo. No permitas que nadie te lo quite."),
    ("Paul Tudor Jones","No hagas apuestas descomunales. Si pierdes no podrás jugar mañana."),
    ("Ed Seykota","Todo el mundo obtiene lo que quiere del mercado."),
    ("Ray Dalio","El mayor error es creer que el pasado reciente continuará."),
    ("Zenón de Citio","Tenemos dos orejas y una boca. Úsalas en esa proporción."),
    ("Marco Aurelio","Nunca desperdicies tiempo preguntándote qué tipo de persona ser. Sé esa persona."),
    ("Séneca","No es pobre el que tiene poco, sino el que desea mucho."),
]

# ════════════════════════════════════════════════════════════════
#  CSS — DISEÑO GRIEGO / ESTOICO
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Philosopher:ital,wght@0,400;0,700;1,400&display=swap');
* {{ font-family:'Philosopher',serif; }}
h1,h2,h3,h4 {{ font-family:'Cinzel',serif !important; color:{T['primary']} !important; letter-spacing:2px; }}
.stApp {{ background:{T['bg']} !important; }}
.stTabs [data-baseweb="tab"] {{ font-family:'Cinzel',serif; color:{T['primary']}99; font-size:.7em; letter-spacing:1px; }}
.stTabs [aria-selected="true"] {{ color:{T['primary']} !important; border-bottom:2px solid {T['primary']}; }}
.mimi-title {{ font-family:'Cinzel',serif; font-size:clamp(1.8rem,5vw,3rem); font-weight:900; letter-spacing:10px; text-align:center;
    background:linear-gradient(180deg,#E8D5A3 0%,{T['primary']} 50%,{T['secondary']} 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; filter:drop-shadow(0 0 24px {T['primary']}44); margin:8px 0; }}
.mimi-sub {{ text-align:center; font-family:'Philosopher',serif; font-style:italic; color:{T['primary']}77; font-size:.85em; letter-spacing:4px; }}
.greek-orn {{ text-align:center; color:{T['primary']}55; letter-spacing:8px; margin:6px 0; font-size:.9em; }}
.ticker-wrap {{ background:linear-gradient(90deg,{T['bg']},{T['card']},{T['bg']});
    border-top:1px solid {T['primary']}44; border-bottom:1px solid {T['primary']}44; overflow:hidden; padding:7px 0; margin:3px 0; }}
.ticker-label {{ font-family:'Cinzel',serif; color:{T['primary']}; font-size:10px; letter-spacing:2px;
    padding:0 14px; display:inline-block; border-right:1px solid {T['primary']}44; vertical-align:middle; }}
.t-s1 {{ display:inline-block; white-space:nowrap; animation:sc1 45s linear infinite; }}
.t-s2 {{ display:inline-block; white-space:nowrap; animation:sc1 62s linear infinite; }}
@keyframes sc1 {{ from{{transform:translateX(0)}} to{{transform:translateX(-50%)}} }}
.card {{ background:{T['card']}; border:1px solid {T['primary']}33; border-top:2px solid {T['primary']}66;
    border-radius:3px; padding:18px; margin:8px 0; box-shadow:0 4px 20px {T['primary']}0a; }}
.card-title {{ font-family:'Cinzel',serif; color:{T['primary']}; font-size:.78em; letter-spacing:3px;
    text-transform:uppercase; border-bottom:1px solid {T['primary']}33; padding-bottom:8px; margin-bottom:14px; }}
.sig-long  {{ color:#4CAF82; font-weight:700; font-size:1.4em; font-family:'Cinzel',serif; letter-spacing:4px; }}
.sig-short {{ color:#C0392B; font-weight:700; font-size:1.4em; font-family:'Cinzel',serif; letter-spacing:4px; }}
.sig-neu   {{ color:{T['primary']}; font-weight:700; font-size:1.4em; font-family:'Cinzel',serif; letter-spacing:4px; }}
.stoic-q {{ border-left:3px solid {T['primary']}; padding:14px 20px; margin:20px 0;
    background:{T['primary']}08; font-style:italic; color:{T['primary']}CC; font-size:1.05em; }}
.stoic-a {{ font-family:'Cinzel',serif; font-size:.72em; letter-spacing:2px; color:{T['primary']}77; margin-top:6px; }}
.smc-strong {{ color:#4CAF82; font-weight:600; }}
.smc-bear   {{ color:#C0392B; font-weight:600; }}
.smc-weak   {{ color:#C8A96E; }}
.modo-badge {{ font-family:'Cinzel',serif; font-size:.8em; letter-spacing:2px; padding:4px 12px;
    border-radius:2px; display:inline-block; margin:4px 0; }}
.wy-acc  {{ color:#7BAF9E; font-weight:600; }}
.wy-man  {{ color:#CD7F32; font-weight:600; }}
.wy-dist {{ color:#C0392B; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};letter-spacing:3px;text-align:center;">⚙ CONFIG</div>', unsafe_allow_html=True)
    st.markdown("---")
    nuevo_tema = st.selectbox("🏛️ Estilo", list(THEMES.keys()),
                               index=list(THEMES.keys()).index(st.session_state.get('tema','Mármol Griego')))
    if nuevo_tema != st.session_state.get('tema','Mármol Griego'):
        st.session_state.tema = nuevo_tema
        sv = gh_load(); sv['tema'] = nuevo_tema; gh_save(sv)
        st.rerun()

    nuevo_modo = st.selectbox("🎯 Modo", ["Oráculo 🏛️","Gladiador ⚔️"],
                               index=["Oráculo 🏛️","Gladiador ⚔️"].index(st.session_state.modo))
    MC = MODO_CONFIG[nuevo_modo]
    st.caption(MC['desc'])
    if nuevo_modo != st.session_state.modo:
        st.session_state.modo = nuevo_modo
        sv = gh_load(); sv['modo'] = nuevo_modo; gh_save(sv)
        send_tg(f"🏛️ *MIMI-AI* — Modo: *{nuevo_modo}*\n{MC['desc']}")

    nuevo_estilo = st.selectbox("📊 Estilo de Trading", ["Scalping","Day Trading","Swing"],
                                 index=["Scalping","Day Trading","Swing"].index(st.session_state.trade_style))
    _sc_preview = STYLE_CONFIG.get(nuevo_estilo, {})
    st.caption(_sc_preview.get('desc',''))
    if nuevo_estilo != st.session_state.trade_style:
        st.session_state.trade_style = nuevo_estilo
        sv = gh_load(); sv['trade_style'] = nuevo_estilo; gh_save(sv)
        send_tg(f"🏛️ Estilo: *{nuevo_estilo}* — {_sc_preview.get('desc','')}")

    risk_pct = st.slider("⚠️ Riesgo/trade (%)", 0.5, 5.0,
                          float(st.session_state.get('risk_pct', 1.0)), 0.5)
    if risk_pct != st.session_state.get('risk_pct', 1.0):
        st.session_state.risk_pct = risk_pct
        sv = gh_load(); sv['risk_pct'] = risk_pct; gh_save(sv)
    st.markdown("---")

    # ── MODO AUTOMÁTICO ───────────────────────────────────────────
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"] if "T" in dir() else "#C8A96E"}99;font-size:.8em;letter-spacing:2px;">🤖 MODO AUTO</div>', unsafe_allow_html=True)
    auto_mode = st.toggle("Activar señales automáticas", value=st.session_state.get('auto_mode', False))
    if auto_mode != st.session_state.get('auto_mode', False):
        st.session_state.auto_mode = auto_mode
        estado_txt = "activado" if auto_mode else "desactivado"
        send_tg(f"🤖 *Modo automático {estado_txt}*\n{'Las señales se enviarán a Telegram automáticamente.' if auto_mode else 'Solo análisis — sin envíos automáticos.'}")
    if auto_mode:
        st.caption("🟢 Señales ≥60% se envían solas a Telegram")
    else:
        st.caption("⚪ Solo análisis — sin envíos automáticos")
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📱 TELEGRAM</div>', unsafe_allow_html=True)
    st.caption("entré · no · salgo · estado · señal · wyckoff · me quedo")
    st.markdown("---")
    if FINNHUB_KEY:
        fuente_txt = "🟢 Finnhub — precio + datos + señales"
    elif TWELVEDATA_KEY:
        fuente_txt = "🟡 TwelveData (activo como fallback)"
    else:
        fuente_txt = "⚠️ Agrega FINNHUB_KEY en Secrets"
    st.caption(fuente_txt)
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📚 GUÍA</div>', unsafe_allow_html=True)
    for titulo, texto in [
        ("BOS / MSB","BOS=estructura confirmada. MSB=reversión. Base del análisis SMC."),
        ("Order Block","Última vela institucional antes de impulso. Zona de reacción."),
        ("EQH / EQL","Equal Highs/Lows = pools de liquidez donde barren stops."),
        ("FVG","Fair Value Gap: desequilibrio. El mercado suele regresar a llenarlo."),
        ("Wyckoff AMD","Acumulación=rango lateral. Manipulación=liquidity grab. Distribución=breakout real."),
        ("Oráculo","Requiere BOS + OB. Señales escasas pero precisas."),
        ("Gladiador","Rebotes en OB, FVG, EQH/EQL. Más trades, más riesgo."),
    ]:
        with st.expander(titulo): st.write(texto)

T  = THEMES.get(st.session_state.tema, THEMES["Mármol Griego"])
MC = MODO_CONFIG[st.session_state.modo]
SC = STYLE_CONFIG[st.session_state.trade_style]

# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
modo_color = '#C8A96E' if 'Oráculo' in st.session_state.modo else '#CD7F32'
st.markdown(f"""
<div class="greek-orn">─────── ✦ ───────</div>
<div class="mimi-title">MIMI · AI</div>
<div class="mimi-sub">XAU/USD · ML · SMC · ICT · WYCKOFF · BOS · OB · EQH/EQL</div>
<div style="text-align:center;margin:6px 0;">
  <span class="modo-badge" style="background:{modo_color}22;border:1px solid {modo_color}66;color:{modo_color};">{st.session_state.modo}</span>
  <span class="modo-badge" style="background:{T['primary']}11;border:1px solid {T['primary']}44;color:{T['primary']}99;margin-left:8px;">{st.session_state.trade_style.upper()} · {SC['label']}</span>
</div>
<div class="greek-orn" style="margin-top:6px;">─────── ✦ ───────</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  CARGA DE DATOS Y MODELO (caché 10min)
# ════════════════════════════════════════════════════════════════
INTERVALS = {"M5":"5m","M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}
PERIODS   = {"M5":"5d","M15":"10d","M30":"20d","H1":"60d","H4":"180d","D1":"2y"}

with st.spinner("🏛️ El Oráculo consulta los astros..."):
    raw = get_data(SC['interval'], SC['period'])
    if raw is None: raw = get_data("1d","2y")

if raw is None:
    st.warning("⚠️ Mercado cerrado — reabre domingo 6PM MX."); st.stop()

df  = add_ind(raw.to_json(orient='split'))
# Umbral viene del estilo de trading — scalping necesita umbral más bajo
_umbral = SC.get('umbral', MC.get('umbral', 0.003))
m_model, sc_model, features, df_trained = train_model(df.to_json(orient='split'), _umbral)
smc     = detect_smc_advanced(df, lookback=50)
wyckoff = run_wyckoff(df)

# Valores del modelo (lentos, cacheados)
rsi   = float(df['RSI'].iloc[-1])
atr   = float(df['ATR'].iloc[-1])
atr_rel = float(df['ATR_rel'].iloc[-1]) if 'ATR_rel' in df.columns else 1.0
# scores se calcula después de get_signal — inicializar aquí como placeholder
mercado_activo  = True   # se actualiza con scores después
mercado_razon   = "Calculando..." 
bb_up = float(df['BB_upper'].iloc[-1])
bb_low= float(df['BB_lower'].iloc[-1])
ema20 = float(df['EMA_20'].iloc[-1])
ema50 = float(df['EMA_50'].iloc[-1])

# Señal según modo
# SL/TP viene del estilo de trading — cada estilo tiene su propio ATR
_atr_sl = SC.get('atr_sl', MC['atr_sl'])
_atr_tp = SC.get('atr_tp', MC['atr_tp'])

if 'Gladiador' in st.session_state.modo:
    pred, prob, _, _, sl_long, tp_long, sl_short, tp_short, scores = get_signal_gladiador(
        df_trained, smc, features, m_model, sc_model, _atr_sl, _atr_tp)
else:
    pred, prob, _, _, sl_long, tp_long, sl_short, tp_short, scores = get_signal_oraculo(
        df_trained, smc, features, m_model, sc_model, _atr_sl, _atr_tp)

rr   = round(MC['atr_tp'] / MC['atr_sl'], 2)
conf = max(prob) * 100
ET   = {1:"LONG — ASCENSO", 0:"LATERAL — ESPERA", -1:"SHORT — DESCENSO"}
p_long  = round(float(prob[2] if len(prob)==3 else prob[1])*100,1)
p_short = round(float(prob[0])*100,1)
p_lat   = round(max(0,100-p_long-p_short-5),1)
p_shock = round(100-p_long-p_short-p_lat,1)

# ════════════════════════════════════════════════════════════════
#  PRECIO INTERNO — para lógica de paper trades / señales
#  (el precio visual lo actualiza el widget JS cada 2s)
# ════════════════════════════════════════════════════════════════
_p_fallback = get_precio_fallback()
precio       = _p_fallback if _p_fallback else float(df['Close'].iloc[-1])

mx_tz = pytz.timezone('America/Mexico_City')
ahora = datetime.now(mx_tz)
h     = ahora.hour

# ════════════════════════════════════════════════════════════════
#  AUTO PAPER TRADE — detecta SL/TP automáticamente
# ════════════════════════════════════════════════════════════════
for t in st.session_state.paper_trades:
    if t['estado']=='ABIERTO':
        if 'LONG' in t['dir']:
            if precio >= t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['tp']-t['entrada'])*t['lotes']*100,2); t['resultado']='WIN ✅'
                send_tg(f"🏛️ *TP ALCANZADO* 🟢\nLONG cerrado @ ${precio:,.2f}\nP&L: +${t['pnl']:.2f}\n\n_El sabio toma sus ganancias._")
            elif precio <= t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['sl']-t['entrada'])*t['lotes']*100,2); t['resultado']='LOSS ❌'
                send_tg(f"🏛️ *SL ALCANZADO* 🔴\nLONG cerrado @ ${precio:,.2f}\nP&L: ${t['pnl']:.2f}\n\n_Una pérdida aceptada es una victoria de carácter._")
        elif 'SHORT' in t['dir']:
            if precio <= t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['tp'])*t['lotes']*100,2); t['resultado']='WIN ✅'
                send_tg(f"🏛️ *TP ALCANZADO* 🟢\nSHORT cerrado @ ${precio:,.2f}\nP&L: +${t['pnl']:.2f}")
            elif precio >= t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['sl'])*t['lotes']*100,2); t['resultado']='LOSS ❌'
                send_tg(f"🏛️ *SL ALCANZADO* 🔴\nSHORT cerrado @ ${precio:,.2f}\nP&L: ${t['pnl']:.2f}")

cap = 1000.0 + sum(t.get('pnl',0) for t in st.session_state.paper_trades if t['estado']=='CERRADO')
st.session_state.capital = round(cap,2)

# Guardar señal
if not st.session_state.signal_history or st.session_state.signal_history[-1].get('precio')!=precio:
    sl_r=sl_long if pred>=0 else sl_short; tp_r=tp_long if pred>=0 else tp_short
    st.session_state.signal_history.append({
        'id':len(st.session_state.signal_history)+1,'fecha':ahora.strftime('%d/%m %H:%M'),
        'modo':st.session_state.modo,'estilo':st.session_state.trade_style,
        'direccion':ET.get(pred),'confianza':f"{conf:.1f}%",'precio':precio,
        'sl':sl_r,'tp':tp_r,'rsi':round(rsi,1),'smc':smc['bias'],'resultado':'PENDIENTE'})

sv2 = {
    'paper_trades':   st.session_state.paper_trades,
    'signal_history': st.session_state.signal_history[-50:],
    'capital':        st.session_state.capital,
    'trade_style':    st.session_state.trade_style,
    'modo':           st.session_state.modo,
    'tema':           st.session_state.get('tema','Mármol Griego'),
    'risk_pct':       st.session_state.get('risk_pct', 1.0),
    'last_tg_update': st.session_state.last_tg_update
}
gh_save(sv2)

# ── ALERTAS TELEGRAM AUTOMÁTICAS — TODAS las señales ─────────────
_auto_mode  = st.session_state.get('auto_mode', False)
_nivel_act  = scores.get('nivel', 0)
_nivel_lbl  = scores.get('label', '⏳ SIN SEÑAL')

# Guardar señal en GitHub siempre que haya dirección
if pred != 0:
    _signal = build_signal(pred, prob, precio, sl_long, tp_long,
                           sl_short, tp_short, conf,
                           smc['bias'], wyckoff['trend'])
    save_signal_github(_signal, GH_TOKEN, GH_REPO)

# Mandar a Telegram si modo auto ON — manda TODAS (Fuerte, Media y Débil)
if _auto_mode and pred != 0:
    _sl_r  = sl_long  if pred == 1 else sl_short
    _tp_r  = tp_long  if pred == 1 else tp_short
    _rr    = round(abs(_tp_r - precio) / abs(precio - _sl_r), 2) if abs(precio - _sl_r) > 0 else 0
    _icon  = "📈" if pred == 1 else "📉"
    _estilo = st.session_state.trade_style
    _nivel_emoji = "🔥" if _nivel_act == 1 else "✅" if _nivel_act == 2 else "📡"
    _operar = "Opera" if _nivel_act in [1,2] else "Solo informativa"
    send_tg(
        f"{_nivel_emoji} *MIMI\\-AI — {_nivel_lbl}*\n"
        f"{'─'*24}\n"
        f"{_icon} *{ET.get(pred)}* · {_estilo} · {SC['label']}\n"
        f"💰 Precio : `${precio:,.2f}`\n"
        f"🔴 SL     : `${_sl_r:,.2f}`\n"
        f"🟢 TP     : `${_tp_r:,.2f}`\n"
        f"📐 R:R    : `1:{_rr}`\n"
        f"📊 Conf   : `{conf:.1f}%` · Score `{scores.get('total',0):.1f}/10`\n"
        f"🏛️ SMC    : {smc['bias']} · BOS: {'✅' if smc['bos'] else '❌'} · OB: {len(smc['order_blocks'])}\n"
        f"🏺 Wyckoff: {wyckoff['trend']}\n"
        f"{'─'*24}\n"
        f"_{_operar}_\n"
        f"_Responde 'entré' · 'no' · 'estado'_"
    )

# Alerta Wyckoff AMD si hay señal activa (siempre que auto esté ON)
if _auto_mode and wyckoff.get('active') and wyckoff['active'].get('tipo'):
    _wa = wyckoff['active']
    _wa_icon = "📈" if _wa.get('tipo') == 'LONG' else "📉"
    send_tg(
        f"🏺 *WYCKOFF AMD — SEÑAL*\n"
        f"{_wa_icon} *{_wa['tipo']}* [{_wa.get('confianza','—')}]\n"
        f"Entrada : `${_wa.get('entrada',0):,.2f}`\n"
        f"SL      : `${_wa.get('sl',0):,.2f}`\n"
        f"TP      : `${_wa.get('tp',0):,.2f}`\n"
        f"Tendencia: {wyckoff['trend']}"
    )

# Procesar Telegram
process_tg_updates(precio, pred, prob, rsi, atr, sl_long, tp_long, sl_short, tp_short, rr, smc, conf, ET, risk_pct, wyckoff)

# ════════════════════════════════════════════════════════════════
#  BANNERS
# ════════════════════════════════════════════════════════════════
bc    = '#4CAF82' if pred==1 else '#C0392B' if pred==-1 else T['primary']
ge    = smc.get('gladiador_entry','')
ge_str= f"  ⚔️ MICRO: {ge.replace('_',' ')}  ·" if ge and 'Gladiador' in st.session_state.modo else ""
wy_str= f"  🏺 AMD: {wyckoff['active']['tipo']}  ·" if wyckoff.get('active') else ""
b1 = (f"  {st.session_state.modo}  ·  {ET.get(pred)}  ·  CONF: {conf:.1f}%  ·  ${precio:,.2f}  ·  SL: ${sl_long:,.2f}  ·  TP: ${tp_long:,.2f}  ·  R:R 1:{rr}  ·  ATR: {atr:.2f}{ge_str}{wy_str}  ")*2
b2 = (f"  RSI: {rsi:.1f}  ·  EMA20: ${ema20:,.2f}  ·  EMA50: ${ema50:,.2f}  ·  SMC: {smc['bias']}  ·  BOS: {len(smc['bos'])}  ·  OB: {len(smc['order_blocks'])}  ·  FVG: {len(smc['fvg'])}  ·  Wyckoff: {wyckoff['trend']}  ·  Precio JS live  ")*2
st.markdown(f"""
<div class="ticker-wrap">
  <span class="ticker-label">ORACLE</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s1" style="color:{bc};font-family:'Philosopher',serif;font-size:.85em;">{b1}</div>
  </div>
</div>
<div class="ticker-wrap">
  <span class="ticker-label">SMC·AMD</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s2" style="color:{T['primary']}99;font-family:'Philosopher',serif;font-size:.82em;">{b2}</div>
  </div>
</div>
<div class="greek-orn">── ✦ ── ✦ ── ✦ ──</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  WIDGET JS — PRECIO EN VIVO SIN RECARGAR STREAMLIT
#  Llama a TwelveData directo desde el navegador cada 2s
# ════════════════════════════════════════════════════════════════
_fh_key   = FINNHUB_KEY or ""
_td_key   = TWELVEDATA_KEY or ""
_primary  = T['primary']
_card     = T['card']
_bg       = T['bg']

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Philosopher&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:transparent; font-family:'Philosopher',serif; }}

  .wrap {{
    background: linear-gradient(90deg, {_card}, {_bg}, {_card});
    border: 1px solid {_primary}44;
    border-radius: 4px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }}

  .left {{ display:flex; flex-direction:column; gap:4px; }}

  .label {{
    font-family: 'Cinzel', serif;
    color: {_primary}88;
    font-size: 11px;
    letter-spacing: 3px;
  }}

  .price {{
    font-family: 'Cinzel', serif;
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 900;
    transition: color 0.3s, text-shadow 0.3s;
  }}

  .change {{
    font-family: 'Philosopher', serif;
    font-size: 1em;
    margin-left: 10px;
    transition: color 0.3s;
  }}

  .right {{
    text-align: right;
    font-family: 'Philosopher', serif;
    color: {_primary}88;
    font-size: 13px;
    line-height: 1.9;
  }}

  .dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4CAF82;
    margin-right: 5px;
    animation: pulse 1.5s infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:.4; transform:scale(1.3); }}
  }}

  .flash-up   {{ animation: fu .4s ease; }}
  .flash-down {{ animation: fd .4s ease; }}
  @keyframes fu {{ 0%{{opacity:.3}} 100%{{opacity:1}} }}
  @keyframes fd {{ 0%{{opacity:.3}} 100%{{opacity:1}} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <span class="label">
      <span class="dot" id="dot"></span>
      XAU/USD &nbsp;·&nbsp; <span id="src">—</span> &nbsp;·&nbsp; <span id="hora">--:--</span>
    </span>
    <div>
      <span class="price" id="precio" style="color:#C8A96E;">— — —</span>
      <span class="change" id="cambio" style="color:#C8A96E;"></span>
    </div>
  </div>
  <div class="right" id="hl">
    H: — &nbsp; L: —<br>
    <span style="font-size:11px;letter-spacing:1px;">Precio live JS · Modelo caché 10min</span>
  </div>
</div>

<script>
const FH_KEY = "{_fh_key}";
const TD_KEY = "{_td_key}";
let prevPrice  = null;
let prevClose  = null;   // precio de cierre anterior para calcular cambio
let ws         = null;
let wsAlive    = false;
let restTimer  = null;

// ══════════════════════════════════════════════════════════════
//  FUENTE 1: Finnhub WebSocket — tiempo real, gratis
//  El WebSocket manda cada trade de OANDA:XAU_USD al instante
// ══════════════════════════════════════════════════════════════
function conectarWS() {{
  if (!FH_KEY) {{ iniciarREST(); return; }}
  try {{
    ws = new WebSocket(`wss://ws.finnhub.io?token=${{FH_KEY}}`);

    ws.onopen = () => {{
      wsAlive = true;
      ws.send(JSON.stringify({{type:"subscribe", symbol:"OANDA:XAU_USD"}}));
      document.getElementById("src").textContent = "Finnhub WS ⚡";
    }};

    ws.onmessage = (e) => {{
      try {{
        const msg = JSON.parse(e.data);
        if (msg.type !== "trade" || !msg.data || !msg.data.length) return;
        // Tomar el trade más reciente
        const trades = msg.data;
        const last   = trades[trades.length - 1];
        const p      = parseFloat(last.p);
        if (!p || p < 100) return;

        // Guardar prevClose la primera vez
        if (!prevClose) prevClose = p;
        const ch    = p - prevClose;
        const chpct = (ch / prevClose * 100).toFixed(3);
        const ts    = new Date(last.t);
        const hh    = String(ts.getHours()).padStart(2,"0");
        const mm    = String(ts.getMinutes()).padStart(2,"0");

        actualizarUI({{
          precio:     p.toFixed(2),
          cambio:     ch.toFixed(2),
          cambio_pct: chpct,
          high:       p.toFixed(2),
          low:        p.toFixed(2),
          hora:       hh + ":" + mm,
          src:        "Finnhub WS ⚡"
        }});
      }} catch(err) {{}}
    }};

    ws.onerror = () => {{ wsAlive = false; iniciarREST(); }};

    ws.onclose = () => {{
      wsAlive = false;
      // Reconectar después de 5 segundos
      setTimeout(conectarWS, 5000);
    }};

    // Keepalive ping cada 20s (Finnhub desconecta si no hay actividad)
    setInterval(() => {{
      if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{type:"ping"}}));
      }}
    }}, 20000);

  }} catch(e) {{ iniciarREST(); }}
}}

// ══════════════════════════════════════════════════════════════
//  FUENTE 2: Finnhub REST — fallback si WS falla
//  Usa /forex/rates que da spot XAU/USD más preciso que /quote
// ══════════════════════════════════════════════════════════════
async function fetchFinnhubREST() {{
  if (!FH_KEY) return null;
  try {{
    // forex/rates?base=USD da la cantidad de USD por unidad de cada moneda
    // Para XAU necesitamos invertir: 1/XAU_rate = precio de 1 oz en USD
    const r = await fetch(
      `https://finnhub.io/api/v1/forex/rates?base=XAU&token=${{FH_KEY}}`,
      {{signal: AbortSignal.timeout(4000)}}
    );
    if (!r.ok) return null;
    const d   = await r.json();
    const usd = parseFloat(d?.quote?.USD);
    // d.quote.USD = cuántos USD vale 1 XAU = precio spot real
    if (!usd || usd < 100) return null;
    if (!prevClose) prevClose = usd;
    const ch    = usd - prevClose;
    const chpct = (ch / prevClose * 100).toFixed(3);
    const now   = new Date();
    return {{
      precio:     usd.toFixed(2),
      cambio:     ch.toFixed(2),
      cambio_pct: chpct,
      high:       usd.toFixed(2),
      low:        usd.toFixed(2),
      hora:       String(now.getHours()).padStart(2,"0")+":"+String(now.getMinutes()).padStart(2,"0"),
      src:        "Finnhub REST"
    }};
  }} catch(e) {{ return null; }}
}}

// ══════════════════════════════════════════════════════════════
//  FUENTE 3: TwelveData REST — segundo fallback
// ══════════════════════════════════════════════════════════════
async function fetchTD() {{
  if (!TD_KEY) return null;
  try {{
    const r = await fetch(
      `https://api.twelvedata.com/price?symbol=XAU/USD&apikey=${{TD_KEY}}`,
      {{signal: AbortSignal.timeout(4000)}}
    );
    if (!r.ok) return null;
    const d = await r.json();
    const p = parseFloat(d.price);
    if (!p || p < 100) return null;
    if (!prevClose) prevClose = p;
    const ch    = p - prevClose;
    const now   = new Date();
    return {{
      precio:     p.toFixed(2),
      cambio:     ch.toFixed(2),
      cambio_pct: (ch/prevClose*100).toFixed(3),
      high:       p.toFixed(2),
      low:        p.toFixed(2),
      hora:       String(now.getHours()).padStart(2,"0")+":"+String(now.getMinutes()).padStart(2,"0"),
      src:        "TwelveData"
    }};
  }} catch(e) {{ return null; }}
}}

// ══════════════════════════════════════════════════════════════
//  LOOP REST — se usa si WebSocket no está disponible
// ══════════════════════════════════════════════════════════════
async function tickREST() {{
  if (wsAlive) return;   // WS activo, no necesitamos REST
  let data = await fetchFinnhubREST();
  if (!data) data = await fetchTD();
  if (data) actualizarUI(data);
}}

function iniciarREST() {{
  if (restTimer) return;
  restTimer = setInterval(tickREST, 2000);
  tickREST();
}}

// ══════════════════════════════════════════════════════════════
//  ACTUALIZAR UI
// ══════════════════════════════════════════════════════════════
function actualizarUI(tick) {{
  if (!tick) return;
  const up    = parseFloat(tick.cambio) >= 0;
  const color = up ? "#4CAF82" : "#C0392B";
  const flecha= up ? "▲" : "▼";
  const sign  = up ? "+" : "";

  const el  = document.getElementById("precio");
  const was = prevPrice;
  prevPrice = tick.precio;

  // Flash animación en cada cambio
  if (was && was !== tick.precio) {{
    el.classList.remove("flash-up","flash-down");
    void el.offsetWidth;
    el.classList.add(up ? "flash-up" : "flash-down");
  }}

  el.style.color      = color;
  el.style.textShadow = `0 0 16px ${{color}}99`;
  el.textContent      = "$" + parseFloat(tick.precio).toLocaleString("en-US",
                        {{minimumFractionDigits:2, maximumFractionDigits:2}});

  const ch = document.getElementById("cambio");
  ch.style.color = color;
  ch.textContent = `${{flecha}} ${{Math.abs(parseFloat(tick.cambio)).toFixed(2)}} (${{sign}}${{tick.cambio_pct}}%)`;

  document.getElementById("hora").textContent = tick.hora;
  document.getElementById("src").textContent  = tick.src;
  document.getElementById("hl").innerHTML =
    `OANDA · XAU/USD Spot<br>` +
    `<span style="font-size:11px;letter-spacing:1px;">${{tick.src}} · Señales caché 5min</span>`;

  document.getElementById("dot").style.background = color;
}}

// ══════════════════════════════════════════════════════════════
//  ARRANCAR — WebSocket primero, REST como respaldo
// ══════════════════════════════════════════════════════════════
conectarWS();
// REST de respaldo si WS no conecta en 3s
setTimeout(() => {{ if (!wsAlive) iniciarREST(); }}, 3000);
</script>
</body>
</html>
""", height=90, scrolling=False)

# Valor de precio para lógica interna (paper trades, Telegram, etc.)
# Este no afecta el display — solo se usa internamente
precio_color = '#4CAF82'  # default, widget JS maneja el color visual

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("💰 Precio",f"${precio:,.2f}")
c2.metric("📊 RSI",f"{rsi:.1f}","SC" if rsi>70 else "SV" if rsi<30 else "OK")
c3.metric("⚡ ATR",f"{atr:.2f}",f"rel:{atr_rel:.2f}")
c4.metric("🎯 Señal","LONG" if pred==1 else "SHORT" if pred==-1 else "LAT",f"{conf:.1f}%")
c5.metric("📐 R:R",f"1:{rr}")
c6.metric("🏺 Wyckoff",wyckoff['trend'])
s_mercado_val = scores.get('mercado', 1.0) if 'scores' in dir() else 1.0
mercado_activo = s_mercado_val >= 0.5
c7.metric("🌊 Mercado","ACTIVO" if mercado_activo else "PLANO",
          f"Score:{s_mercado_val:.1f}")
st.markdown('<div class="greek-orn">── ✦ ──</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11 = st.tabs([
    "🎯 Señal","🏛️ SMC·ICT","🏺 Wyckoff","🌐 Multi·TF",
    "📋 Paper","📊 Gráfica","💬 Chat",
    "📈 Backtest","🔔 Alertas","📜 Historial","👁️ Monitor"])

# ── TAB 1: SEÑAL ──────────────────────────────────────────────────
with tab1:
    ca, cb = st.columns(2)
    with ca:
        # ── ESTADO DE ENTRADA ─────────────────────────────────────
        sl_r          = sl_long if pred >= 0 else sl_short
        tp_r          = tp_long if pred >= 0 else tp_short
        precio_señal  = float(df['Close'].iloc[-1])
        dist_entrada  = abs(precio - precio_señal)

        sl_tocado = pred != 0 and ((pred == 1 and precio <= sl_r) or (pred == -1 and precio >= sl_r))
        tp_tocado = pred != 0 and ((pred == 1 and precio >= tp_r) or (pred == -1 and precio <= tp_r))

        # Si hay señal y no tocó SL ni TP → siempre zona válida
        # El modelo recalculó hace menos de 5 min, el precio actual ES la entrada
        zona_valida = pred != 0 and not sl_tocado and not tp_tocado

        if pred == 0:
            estado  = "ESPERA";  ec = T['primary']
            elabel  = "⏳  ESPERA"
            esub    = "Sin dirección clara — no operes aún"
        elif sl_tocado:
            estado  = "SL";      ec = "#C0392B"
            elabel  = "🚨  SL TOCADO"
            esub    = "Stop Loss alcanzado — espera la siguiente señal"
        elif tp_tocado:
            estado  = "TP";      ec = "#4CAF82"
            elabel  = "🎯  TP ALCANZADO"
            esub    = "Objetivo alcanzado — señal completada"
        elif zona_valida:
            estado  = "ENTRA_YA"; ec = "#4CAF82"
            elabel  = "⚡  ENTRA YA"
            esub    = f"Señal activa · entra al precio actual ${precio:,.2f}"
        else:
            estado  = "ESPERA";  ec = T['primary']
            elabel  = "⏳  ESPERA"
            esub    = "Sin señal activa"

        st.markdown('<div class="card"><div class="card-title">SEÑAL DEL ORÁCULO</div>', unsafe_allow_html=True)

        # Scorecard del sistema de 3 niveles
        nivel       = scores.get('nivel', 0)
        nivel_label = scores.get('label', '⏳ SIN SEÑAL')
        nivel_color = scores.get('color', '#555555')
        score_total = scores.get('total', 0)

        # Badge de nivel
        st.markdown(f"""
<div style="margin:6px 0 10px 0;padding:12px 16px;border-radius:4px;
            border:2px solid {nivel_color}88;background:{nivel_color}14;">
  <div style="font-family:'Cinzel',serif;font-size:1.2em;font-weight:900;
              color:{nivel_color};letter-spacing:3px;">{nivel_label}</div>
  <div style="font-size:.75em;color:{nivel_color}99;margin-top:3px;letter-spacing:1px;">
    Score total: {score_total:.1f}/10 · {'Opera' if nivel <= 2 and nivel > 0 else 'Solo informativo' if nivel == 3 else 'Espera'}
  </div>
</div>""", unsafe_allow_html=True)

        # Desglose de scores
        if nivel > 0:
            cols_s = st.columns(5)
            score_items = [
                ("🌊", "Mercado", scores.get('mercado',0), 3.0),
                ("🕐", "Sesión",  scores.get('sesion', 0), 2.0),
                ("🏛️", "SMC",    scores.get('smc',    0), 3.0),
                ("📊", "RSI",     scores.get('rsi',    0), 2.0),
                ("🤖", "ML",      scores.get('ml',     0), 2.0),
            ]
            for col, (icon, name, val, max_val) in zip(cols_s, score_items):
                pct  = min(val/max_val, 1.0)
                clr  = '#4CAF82' if pct >= 0.7 else '#C8A96E' if pct >= 0.4 else '#C0392B'
                col.markdown(
                    f'<div style="text-align:center;padding:6px 4px;'
                    f'background:{T["card"]};border-radius:3px;'
                    f'border-top:3px solid {clr};">'
                    f'<div style="font-size:.7em;color:{T["primary"]}88;">{icon} {name}</div>'
                    f'<div style="font-size:1.1em;font-weight:700;color:{clr};">{val:.1f}</div>'
                    f'<div style="font-size:.65em;color:{T["primary"]}55;">/{max_val:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            # Razones detalladas
            with st.expander("📋 Ver detalle del análisis"):
                st.caption(f"🌊 Mercado: {scores.get('razon_mercado','')}")
                st.caption(f"🕐 Sesión:  {scores.get('razon_sesion','')}")
                st.caption(f"🏛️ SMC:     {scores.get('razon_smc','')}")
                st.caption(f"📊 RSI:     {scores.get('razon_rsi','')}")

        # Dirección
        st.markdown(
            f'<div style="margin-top:10px;" class="{"sig-long" if pred==1 else "sig-short" if pred==-1 else "sig-neu"}">'
            f'{ET.get(pred)}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="{"sig-long" if pred==1 else "sig-short" if pred==-1 else "sig-neu"}">'
            f'{ET.get(pred)}</div>',
            unsafe_allow_html=True)

        # Badge de estado — el más importante visualmente
        st.markdown(f"""
<div style="margin:10px 0 14px 0;padding:13px 16px;border-radius:4px;
            border:2px solid {ec}88;background:{ec}14;font-family:'Cinzel',serif;">
  <div style="font-size:1.25em;font-weight:900;color:{ec};letter-spacing:3px;">{elabel}</div>
  <div style="font-size:.76em;color:{ec}CC;margin-top:5px;letter-spacing:1px;">{esub}</div>
</div>""", unsafe_allow_html=True)

        # Barra SL → Precio → TP
        if pred != 0 and not sl_tocado and not tp_tocado:
            rango = abs(tp_r - sl_r)
            if rango > 0:
                pos_pct = max(0, min(100,
                    (precio - sl_r) / rango * 100 if pred == 1
                    else (sl_r - precio) / rango * 100))
                bc = '#4CAF82' if pos_pct > 55 else '#C8A96E' if pos_pct > 25 else '#C0392B'
                st.markdown(f"""
<div style="margin:6px 0 14px 0;">
  <div style="display:flex;justify-content:space-between;
              font-size:.72em;color:{T['primary']}88;margin-bottom:5px;">
    <span>🔴 SL ${sl_r:,.0f}</span>
    <span style="color:{bc};font-weight:700;">📍 ${precio:,.2f}</span>
    <span>🟢 TP ${tp_r:,.0f}</span>
  </div>
  <div style="background:{T['card']};border-radius:4px;height:9px;
              border:1px solid {T['primary']}33;overflow:hidden;">
    <div style="width:{pos_pct:.1f}%;height:100%;border-radius:4px;
                background:linear-gradient(90deg,#C0392B 0%,{bc} 100%);
                transition:width .5s ease;">
    </div>
  </div>
  <div style="text-align:center;font-size:.68em;
              color:{T['primary']}55;margin-top:3px;">
    Posición en rango SL→TP: {pos_pct:.1f}%
  </div>
</div>""", unsafe_allow_html=True)

        # Extras Gladiador / Wyckoff
        if 'Gladiador' in st.session_state.modo and smc.get('gladiador_entry'):
            ge_label = smc['gladiador_entry'].replace('_',' ')
            ge_color = '#4CAF82' if 'LONG' in smc['gladiador_entry'] else '#C0392B'
            st.markdown(
                f'<span style="font-family:Cinzel,serif;font-size:.8em;'
                f'color:{ge_color};letter-spacing:2px;">⚔️ MICRO: {ge_label}</span>',
                unsafe_allow_html=True)
        if wyckoff.get('active'):
            wa = wyckoff['active']
            wa_color = '#4CAF82' if wa['tipo'] == 'LONG' else '#C0392B'
            st.markdown(
                f'<span style="font-family:Cinzel,serif;font-size:.8em;'
                f'color:{wa_color};letter-spacing:2px;">'
                f'🏺 WYCKOFF: {wa["tipo"]} @ ${wa["entrada"]:,.2f}</span>',
                unsafe_allow_html=True)

        st.markdown(f"**Modo:** {st.session_state.modo}")
        if pred != 0:
            st.markdown(f"**Precio señal:** ${precio_señal:,.2f}  |  **Actual:** ${precio:,.2f}")
            st.markdown(f"**Stop Loss 🔴:** ${sl_r:,.2f}")
            st.markdown(f"**Take Profit 🟢:** ${tp_r:,.2f}")
        else:
            st.markdown(f"▲ LONG si rompe ${bb_up:,.2f}")
            st.markdown(f"▼ SHORT si rompe ${bb_low:,.2f}")

        lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, sl_r if sl_r else precio - atr)
        st.markdown(f"**Lotes:** {lot}  |  **Riesgo:** ${risg:.2f}  |  **R:R:** 1:{rr}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Botones — deshabilitados si SL/TP/ESPERA
        btn_off = estado in ['SL', 'TP', 'ESPERA']
        btn_txt = "⚡ ENTRA YA" if estado == 'ENTRA_YA' else "🎯 REGISTRAR ENTRADA"
        c_si, c_no = st.columns(2)
        if c_si.button(btn_txt, use_container_width=True, disabled=btn_off):
            ot = [t for t in st.session_state.paper_trades if t['estado'] == 'ABIERTO']
            if not ot:
                dir_str     = 'LONG 📈' if pred==1 else 'SHORT 📉' if pred==-1 else ('LONG 📈' if smc.get('gladiador_entry','').startswith('LONG') else 'SHORT 📉')
                actual_pred = pred if pred != 0 else (1 if 'LONG' in dir_str else -1)
                sl_r2 = sl_long if actual_pred == 1 else sl_short
                tp_r2 = tp_long if actual_pred == 1 else tp_short
                lot2, risg2 = calc_pos(st.session_state.capital, risk_pct, precio, sl_r2)
                st.session_state.paper_trades.append({
                    'id': len(st.session_state.paper_trades)+1, 'dir': dir_str,
                    'entrada': precio, 'sl': sl_r2, 'tp': tp_r2,
                    'lotes': lot2, 'riesgo': risg2, 'estado': 'ABIERTO',
                    'fecha': ahora.strftime('%d/%m %H:%M'), 'resultado': 'PENDIENTE', 'pnl': 0})
                send_tg(
                    f"🏛️ *Trade abierto — {st.session_state.modo}*\n"
                    f"{dir_str} @ ${precio:,.2f}\n"
                    f"SL: ${sl_r2:,.2f} | TP: ${tp_r2:,.2f}\nLotes: {lot2}")
                gh_save({**sv2, 'paper_trades': st.session_state.paper_trades})
                st.success("Trade registrado ✅"); st.rerun()
            else:
                st.warning("Ya tienes un trade abierto.")
        if c_no.button("❌ NO ENTRO", use_container_width=True):
            send_tg(f"🏛️ Rechazado: {ET.get(pred)} @ ${precio:,.2f}"); st.info("Rechazada")

    with cb:
        st.markdown('<div class="card"><div class="card-title">VENTANAS · HORA MX</div>', unsafe_allow_html=True)
        st.markdown(f"**Hora:** {ahora.strftime('%H:%M')}")
        for n,ini,fin,cal in [("London Open",3,5,"Alta"),("London+NY",8,11,"Máxima ✦"),("NY Tarde",12,14,"Media"),("NY Cierre",15,17,"Baja")]:
            st.markdown(f"{'🟢' if ini<=h<fin else '⚫'} **{ini:02d}–{fin:02d}** {n} [{cal}]")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">VARIANTES</div>', unsafe_allow_html=True)
        v1,v2,v3,v4 = st.columns(4)
        v1.metric("📈",f"{p_long}%"); v2.metric("📉",f"{p_short}%")
        v3.metric("➡️",f"{p_lat}%"); v4.metric("⚡",f"{p_shock}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: SMC ────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};font-size:.85em;letter-spacing:3px;margin-bottom:12px;">SMC — BIAS: {"📈 ALCISTA" if smc["bias"]=="ALCISTA" else "📉 BAJISTA" if smc["bias"]=="BAJISTA" else "➡️ NEUTRAL"} (score: {smc["bias_score"]})</div>', unsafe_allow_html=True)
    r1,r2 = st.columns(2)
    with r1:
        st.markdown('<div class="card"><div class="card-title">BOS · MSB</div>', unsafe_allow_html=True)
        for b in smc['bos'][-3:]:
            color = 'smc-strong' if 'ALCISTA' in b['tipo'] else 'smc-bear'
            st.markdown(f'<span class="{color}">● {b["tipo"]}</span> — ${b["nivel"]:,.2f} [{b["fuerza"]}]', unsafe_allow_html=True)
        if not smc['bos']: st.markdown("Sin BOS")
        for m2 in smc['msb'][-2:]:
            color = 'smc-strong' if 'ALCISTA' in m2['tipo'] else 'smc-bear'
            st.markdown(f'<span class="{color}">◆ {m2["tipo"]}</span> — ${m2["nivel"]:,.2f} [{m2["fuerza"]}]', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">EQH · EQL</div>', unsafe_allow_html=True)
        for e in smc['eqh'][-3:]:
            st.markdown(f'<span class="smc-bear">▲ EQH</span> — ${e["nivel"]:,.2f}', unsafe_allow_html=True)
        for e in smc['eql'][-3:]:
            st.markdown(f'<span class="smc-strong">▼ EQL</span> — ${e["nivel"]:,.2f}', unsafe_allow_html=True)
        if not smc['eqh'] and not smc['eql']: st.markdown("Sin EQH/EQL detectados")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="card"><div class="card-title">ORDER BLOCKS</div>', unsafe_allow_html=True)
        for ob in smc['order_blocks'][-4:]:
            color = 'smc-strong' if 'ALCISTA' in ob['tipo'] else 'smc-bear'
            near  = abs(precio-ob['mid'])/precio<0.003
            st.markdown(f'<span class="{color}">■ {ob["tipo"]}</span> [{ob["fuerza"]}]{"  ⚡ PRECIO CERCA" if near else ""}<br><span style="color:{T["primary"]}99;font-size:.85em;">  ${ob["bottom"]:,.2f}–${ob["top"]:,.2f}</span>', unsafe_allow_html=True)
        if not smc['order_blocks']: st.markdown("Sin OB")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">FVG · SWINGS</div>', unsafe_allow_html=True)
        for fv in smc['fvg'][-3:]:
            color = 'smc-strong' if 'ALCISTA' in fv['tipo'] else 'smc-bear'
            st.markdown(f'<span class="{color}">◇ {fv["tipo"]}</span> ${fv["bottom"]:,.2f}–${fv["top"]:,.2f} ({fv["size"]:.0f}pts)', unsafe_allow_html=True)
        for ls in smc['liquidity_swings'][-4:]:
            color = 'smc-bear' if 'HIGH' in ls['tipo'] else 'smc-strong'
            st.markdown(f'<span class="{color}">○ {ls["tipo"]}</span> — ${ls["nivel"]:,.2f}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">ICT KILLZONES · HORA MX</div>', unsafe_allow_html=True)
    for n,ini,fin,desc in [("Asian Range",19,23,"Acumulación"),("London Open",3,5,"Barrido liquidez asiática"),("NY Open",8,11,"✦ Mayor volatilidad"),("London Close",10,12,"Reversales frecuentes"),("NY PM",13,15,"Continuación o reversión")]:
        st.markdown(f"{'🟢' if ini<=h<fin else '⚪'} **{ini:02d}–{fin:02d} {n}** — {desc}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: WYCKOFF ────────────────────────────────────────────────
with tab3:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};font-size:.85em;letter-spacing:3px;margin-bottom:12px;">WYCKOFF AMD — TENDENCIA: {wyckoff["trend"]}</div>', unsafe_allow_html=True)

    wa1, wa2, wa3 = st.columns(3)
    wa1.metric("🏺 Tendencia", wyckoff['trend'])
    wa2.metric("📦 Zonas Acum.", len(wyckoff['acc_zones']))
    wa3.metric("⚡ Manipulaciones", len(wyckoff['manips']))

    wy1, wy2 = st.columns(2)

    with wy1:
        st.markdown('<div class="card"><div class="card-title">ESTRUCTURA — HH/HL/LH/LL</div>', unsafe_allow_html=True)
        if wyckoff['structure']:
            for s in wyckoff['structure'][-6:]:
                color = 'smc-strong' if s['tipo'] in ['HH','HL'] else 'smc-bear'
                st.markdown(f'<span class="{color}">◉ {s["tipo"]}</span> — ${s["nivel"]:,.2f}', unsafe_allow_html=True)
        else:
            st.markdown("Sin estructura detectada aún")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">ACUMULACIÓN — ZONAS WYCKOFF</div>', unsafe_allow_html=True)
        if wyckoff['acc_zones']:
            for z in wyckoff['acc_zones']:
                near = abs((z['support']+z['resistance'])/2 - precio)/precio < 0.005
                st.markdown(f'<span class="wy-acc">▭ RANGO {z["rango_pct"]:.2f}%</span>{"  ⚡ PRECIO AQUÍ" if near else ""}<br><span style="font-size:.85em;">  Soporte: ${z["support"]:,.2f} | Res: ${z["resistance"]:,.2f}</span>', unsafe_allow_html=True)
        else:
            st.markdown("Sin zonas de acumulación detectadas")
        st.markdown('</div>', unsafe_allow_html=True)

    with wy2:
        st.markdown('<div class="card"><div class="card-title">MANIPULACIÓN — LIQUIDITY GRABS</div>', unsafe_allow_html=True)
        if wyckoff['manips']:
            for mp in wyckoff['manips']:
                color = 'smc-strong' if mp['tipo']=='BULL_GRAB' else 'smc-bear'
                tipo_label = '🟢 BULL GRAB' if mp['tipo']=='BULL_GRAB' else '🔴 BEAR GRAB'
                st.markdown(f'<span class="{color}">⚡ {tipo_label}</span> [{mp["confianza"]}]<br><span style="font-size:.85em;">  Nivel: ${mp["nivel"]:,.2f}</span>', unsafe_allow_html=True)
        else:
            st.markdown("Sin manipulación detectada recientemente")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">DISTRIBUCIÓN — SEÑAL AMD</div>', unsafe_allow_html=True)
        if wyckoff.get('active'):
            wa = wyckoff['active']
            color = 'smc-strong' if wa['tipo']=='LONG' else 'smc-bear'
            st.markdown(f'<span class="{color}">✅ {wa["tipo"]} ACTIVO</span> [Conf: {wa["confianza"]}]', unsafe_allow_html=True)
            st.markdown(f"Entrada: **${wa['entrada']:,.2f}**")
            st.markdown(f"Stop Loss: **${wa['sl']:,.2f}**")
            st.markdown(f"Take Profit: **${wa['tp']:,.2f}**")
            st.markdown(f"R:R: **1:{wa['rr']}**")
        else:
            st.markdown("⏳ Sin señal AMD activa\n\nEl sistema espera ruptura después de manipulación confirmada.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Wyckoff en gráfica
    if wyckoff['acc_zones'] or wyckoff['manips']:
        st.markdown('<div class="card-title" style="font-family:Cinzel,serif;color:{};letter-spacing:3px;">ZONAS AMD EN GRÁFICA</div>'.format(T['primary']), unsafe_allow_html=True)
        dp = df.tail(100)
        fw = go.Figure()
        fw.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
            increasing_line_color='#4CAF82',decreasing_line_color='#C0392B',name="XAU"))
        # Zonas de acumulación
        for z in wyckoff['acc_zones']:
            fw.add_hrect(y0=z['support'],y1=z['resistance'],fillcolor='rgba(123,175,158,0.1)',line_width=0)
            fw.add_hline(y=z['support'],line_color='rgba(123,175,158,0.5)',line_dash='dot')
            fw.add_hline(y=z['resistance'],line_color='rgba(123,175,158,0.5)',line_dash='dot')
        # Niveles de manipulación
        for mp in wyckoff['manips']:
            col = 'rgba(76,175,130,0.6)' if mp['tipo']=='BULL_GRAB' else 'rgba(192,57,43,0.6)'
            fw.add_hline(y=mp['nivel'],line_color=col,line_dash='dash',
                         annotation_text=mp['tipo'],annotation_position="right")
        # Señal activa
        if wyckoff.get('active'):
            wa = wyckoff['active']
            fw.add_hline(y=wa['entrada'],line_color='#FFFFFF',line_dash='solid',annotation_text="ENTRADA")
            fw.add_hline(y=wa['sl'],line_color='#C0392B',line_dash='dash',annotation_text="SL")
            fw.add_hline(y=wa['tp'],line_color='#4CAF82',line_dash='dash',annotation_text="TP")
        fw.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',height=380,
            font=dict(color='#888',family='Philosopher,serif'),
            xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=20,b=0))
        fw.update_xaxes(gridcolor='rgba(200,169,110,0.07)')
        fw.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fw, use_container_width=True)
        st.caption("Verde = zona acumulación Wyckoff | Líneas = niveles manipulación | Blanco/Rojo/Verde = señal AMD activa")

# ── TAB 4: MULTI-TF ───────────────────────────────────────────────
with tab4:
    with st.spinner("Analizando timeframes..."):
        mtf_s, mtf_b, mtf_p = mtf_conf()
    m1,m2,m3 = st.columns(3)
    m1.metric("Bias",f"{'📈 LONG' if mtf_b=='LONG' else '📉 SHORT' if mtf_b=='SHORT' else '➡️ NEUTRAL'}")
    m2.metric("Confluencia",f"{mtf_p:.0f}%"); m3.metric("TFs",str(len(mtf_s)))
    for tfn,data in mtf_s.items():
        bc2 = "🟢" if data['bias']=='LONG' else "🔴" if data['bias']=='SHORT' else "🟡"
        st.markdown(f"{bc2} **{tfn}** — {data['bias']} | {'█'*data['score']}{'░'*(5-data['score'])} {data['score']}/5 | RSI:{data['rsi']:.1f} | ${data['precio']:,.2f}")
    if mtf_p>=60: st.success(f"✅ Confluencia fuerte: {mtf_b} ({mtf_p:.0f}%)")
    elif mtf_p<=40: st.error(f"🔴 Confluencia bajista ({mtf_p:.0f}%)")
    else: st.warning("⚠️ Sin confluencia clara")

# ── TAB 5: PAPER TRADING ──────────────────────────────────────────
with tab5:
    pm1,pm2,pm3 = st.columns(3)
    pm1.metric("💰 Capital",f"${st.session_state.capital:,.2f}")
    ct_c = [t for t in st.session_state.paper_trades if t['estado']=='CERRADO']
    w_p  = sum(1 for t in ct_c if 'WIN' in t.get('resultado',''))
    pm2.metric("Win Rate",f"{w_p/len(ct_c)*100:.0f}%" if ct_c else "—")
    pm3.metric("Trades",f"{len(ct_c)} cerrados")
    ot = [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot:
        t2  = ot[0]
        pnl = (precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*100
        est = "🚨 SAL YA" if (('LONG' in t2['dir'] and precio<=t2['sl']) or ('SHORT' in t2['dir'] and precio>=t2['sl'])) else "🎯 TP" if (('LONG' in t2['dir'] and precio>=t2['tp']) or ('SHORT' in t2['dir'] and precio<=t2['tp'])) else "🟢 MANTÉN" if pnl>0 else "🔴 PRECAUCIÓN"
        st.markdown(f'<div class="card"><div class="card-title">POSICIÓN ABIERTA</div>', unsafe_allow_html=True)
        st.markdown(f"**{t2['dir']}** @ ${t2['entrada']:,.2f} | Actual: ${precio:,.2f} | **{est}**")
        st.markdown(f"P&L: **${pnl:.2f}** | SL: ${t2['sl']:,.2f} | TP: ${t2['tp']:,.2f}")
        if st.button("Cerrar manualmente"):
            t2['estado']='CERRADO'; t2['pnl']=round(pnl,2); t2['resultado']='WIN ✅' if pnl>0 else 'LOSS ❌'
            send_tg(f"🏛️ Cerrado manualmente — P&L: {'+'if pnl>0 else ''}${pnl:.2f}")
            gh_save({**sv2,'paper_trades':st.session_state.paper_trades}); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Sin trade abierto. Usa '✅ ENTRO' o escríbele 'entré' al bot.")
    if ct_c:
        st.dataframe(pd.DataFrame(ct_c)[['fecha','dir','entrada','sl','tp','lotes','pnl','resultado']], use_container_width=True)
    if st.button("🗑️ Reiniciar paper trading"):
        st.session_state.paper_trades=[]; st.session_state.capital=1000.0
        gh_save({**sv2,'paper_trades':[],'capital':1000.0}); st.rerun()

# ── TAB 6: GRÁFICA ────────────────────────────────────────────────
with tab6:
    g1,g2,g3 = st.columns(3)
    tf_h = g1.selectbox("TF Histórico",list(INTERVALS.keys()),index=5,key="tfh")
    ct_g = g2.selectbox("Tipo",["Velas 🕯️","Línea 📈"],key="ct")
    tf_l = g3.selectbox("TF En Vivo",list(INTERVALS.keys()),index=1,key="tfl")
    dfc = get_data(INTERVALS[tf_h],PERIODS[tf_h])
    if dfc is not None:
        dfc = add_ind(dfc.to_json(orient='split')); dp = dfc.tail(120)
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25])
        if "Velas" in ct_g:
            fig.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
                increasing_line_color='#4CAF82',decreasing_line_color='#C0392B',name="XAU"),row=1,col=1)
        else:
            fig.add_trace(go.Scatter(x=dp.index,y=dp['Close'],line=dict(color=T['primary'],width=2),name="Precio"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_20'],line=dict(color='#C8A96E',width=1),name="EMA20"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_50'],line=dict(color='#7B9E87',width=1,dash='dot'),name="EMA50"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_upper'],line=dict(color='rgba(200,169,110,0.2)',width=1),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_lower'],line=dict(color='rgba(200,169,110,0.2)',width=1),fill='tonexty',fillcolor='rgba(200,169,110,0.04)',name="BB"),row=1,col=1)
        for ob in smc['order_blocks'][-2:]:
            color = 'rgba(76,175,130,0.15)' if 'ALCISTA' in ob['tipo'] else 'rgba(192,57,43,0.15)'
            fig.add_hrect(y0=ob['bottom'],y1=ob['top'],fillcolor=color,line_width=0,row=1,col=1)
        for e in smc['eqh'][-2:]:
            fig.add_hline(y=e['nivel'],line_color='rgba(192,57,43,0.5)',line_dash='dot',row=1,col=1)
        for e in smc['eql'][-2:]:
            fig.add_hline(y=e['nivel'],line_color='rgba(76,175,130,0.5)',line_dash='dot',row=1,col=1)
        # Zonas Wyckoff en gráfica histórica
        for z in wyckoff['acc_zones']:
            fig.add_hrect(y0=z['support'],y1=z['resistance'],fillcolor='rgba(123,175,158,0.07)',line_width=0,row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['RSI'],line=dict(color=T['primary'],width=1.5),name="RSI"),row=2,col=1)
        fig.add_hline(y=70,line_color='#C0392B',line_dash='dot',row=2,col=1)
        fig.add_hline(y=30,line_color='#4CAF82',line_dash='dot',row=2,col=1)
        fig.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            xaxis_rangeslider_visible=False,height=500,margin=dict(l=0,r=0,t=20,b=0),
            legend=dict(bgcolor='#000',bordercolor='#222',orientation='h'))
        fig.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fig.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Verde = OB Alcista · Rojo = OB Bajista · Azul turquesa = zona Wyckoff · Punteadas = EQH/EQL")

# ── TAB 7: CHAT — TrendSpider AI ─────────────────────────────────
with tab7:
    st.markdown(f"""
<div style="font-family:'Cinzel',serif;color:{T['primary']};font-size:.8em;
            letter-spacing:3px;margin-bottom:8px;">
    TRENDSPIDER XAUUSD AI · Motor de análisis profesional
</div>
<div style="font-size:.75em;color:{T['primary']}77;margin-bottom:12px;">
    Análisis multi-timeframe automático · Confluencia mínima 4 factores ·
    Solo XAUUSD spot · Mismo formato que TrendSpider profesional
</div>
""", unsafe_allow_html=True)

    # Contexto del mercado actual para pasar a la API
    _ctx_precio = precio
    _ctx_rsi    = rsi
    _ctx_atr    = atr
    _ctx_ema20  = ema20
    _ctx_ema50  = ema50
    _ctx_smc    = smc['bias']
    _ctx_bos    = "SÍ" if smc['bos'] else "NO"
    _ctx_ob     = len(smc['order_blocks'])
    _ctx_fvg    = len(smc['fvg'])
    _ctx_wy     = wyckoff['trend']
    _ctx_señal  = ET.get(pred)
    _ctx_conf   = conf
    _ctx_score  = scores.get('total', 0)
    _ctx_sl     = sl_long if pred >= 0 else sl_short
    _ctx_tp     = tp_long if pred >= 0 else tp_short
    _ctx_estilo = st.session_state.trade_style
    _ctx_modo   = st.session_state.modo
    _ctx_hora   = ahora.strftime('%H:%M')
    _ctx_sesion = scores.get('razon_sesion','—')
    _ctx_mercado= scores.get('razon_mercado','—')

    SYSTEM_PROMPT = f"""Eres TrendSpider XAUUSD AI — motor de análisis técnico profesional para XAUUSD (Oro Spot).
Actúas exactamente como TrendSpider: automatizado, preciso, visual, basado en confluencias altas.
Nunca das opiniones vagas. Respondes como un motor de alertas profesional, no como un humano.

DATOS EN TIEMPO REAL DE MIMI-AI (usa estos para tu análisis):
- Precio actual    : ${_ctx_precio:,.2f}
- Señal MIMI-AI    : {_ctx_señal} (confianza {_ctx_conf:.1f}% · score {_ctx_score:.1f}/10)
- RSI 14           : {_ctx_rsi:.1f}
- ATR 14           : {_ctx_atr:.2f}
- EMA 20           : ${_ctx_ema20:,.2f}
- EMA 50           : ${_ctx_ema50:,.2f}
- SMC Bias         : {_ctx_smc}
- BOS confirmado   : {_ctx_bos}
- Order Blocks     : {_ctx_ob}
- FVG detectados   : {_ctx_fvg}
- Wyckoff trend    : {_ctx_wy}
- SL sugerido      : ${_ctx_sl:,.2f}
- TP sugerido      : ${_ctx_tp:,.2f}
- Hora MX          : {_ctx_hora}
- Sesión           : {_ctx_sesion}
- Mercado          : {_ctx_mercado}
- Estilo trading   : {_ctx_estilo} · {_ctx_modo}

CAPACIDADES QUE SIEMPRE APLICAS:
- Análisis Multi-Timeframe (M5, M15, H1, H4) basado en los datos anteriores
- Auto Fibonacci de swings recientes
- Detección de patrones: Double Top/Bottom, H&S, Triángulos, Flags, Wedges, B&R, OB, FVG
- Análisis de velas: pinbar, engulfing, doji
- Indicadores: EMA 9/21, SuperTrend (10,3), RSI 14, MACD, ATR 14
- Correlación inversa con DXY
- Prioridad sesiones Londres + Nueva York

REGLAS ESTRICTAS:
1. Solo das señal con CONFLUENCIA ALTA (mínimo 4 factores alineados)
2. Si no hay setup claro → responde exactamente: "SIN SEÑAL VÁLIDA EN ESTE MOMENTO"
3. Nunca predices sin confluencia. Sé conservador.
4. Siempre incluye SL y TP basados en ATR o niveles estructurales.
5. Sin emoticonos. Sin frases de relleno. Directo y estructurado.

FORMATO OBLIGATORIO DE RESPUESTA:
**DIRECCIÓN:** LONG / SHORT

**Entrada:** [precio o rango]

**Stop Loss:** [precio] | Distancia: X ATR

**Take Profit:**
- TP1: [precio] (RR 1:2)
- TP2: [precio] (RR 1:3+)

**Confluencia:**
- MTFA: [estado M5, M15, H1, H4]
- Estructura: [trendline, fib, pattern]
- Indicadores: [estado]
- Volumen/Momentum: [estado]

**Probabilidad estimada:** Alta / Media

**Riesgo recomendado:** Máximo 0.5-1% por trade

**Notas:** (máximo 2 líneas)
"""

    def llamar_claude(historial: list, pregunta: str) -> str:
        """Llama a Claude API con contexto de mercado en tiempo real."""
        if not ANTHROPIC_KEY:
            return "⚠️ Agrega ANTHROPIC_KEY en Streamlit Secrets para activar el análisis IA."
        try:
            mensajes = []
            for m in historial[-6:]:  # últimos 6 mensajes de contexto
                rol = "user" if m['role'] == 'user' else "assistant"
                mensajes.append({"role": rol, "content": m['content']})
            mensajes.append({"role": "user", "content": pregunta})

            headers = {
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            }
            body = {
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system":     SYSTEM_PROMPT,
                "messages":   mensajes
            }
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=body, timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                return data['content'][0]['text']
            else:
                return f"Error API: {r.status_code} — {r.text[:200]}"
        except Exception as e:
            return f"Error de conexión: {e}"

    # Botones de análisis rápido
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}88;font-size:.72em;letter-spacing:2px;margin-bottom:8px;">ANÁLISIS RÁPIDO</div>', unsafe_allow_html=True)
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    quick_prompt = None
    if col_b1.button("📊 Analizar ahora", use_container_width=True):
        quick_prompt = f"Analiza el mercado XAU/USD ahora mismo con el precio actual de ${precio:,.2f} y dame señal completa en tu formato."
    if col_b2.button("🎯 ¿Entro o espero?", use_container_width=True):
        quick_prompt = f"El precio está en ${precio:,.2f} y la señal es {ET.get(pred)}. ¿Entro ahora o espero mejor setup? Dame análisis completo."
    if col_b3.button("🛡️ Gestión de riesgo", use_container_width=True):
        quick_prompt = f"Con precio en ${precio:,.2f}, ATR {atr:.2f}, dame los niveles exactos de SL y TP para {st.session_state.trade_style} y calcula el tamaño de posición para 1% de riesgo."
    if col_b4.button("📰 ¿Qué esperar hoy?", use_container_width=True):
        quick_prompt = f"Son las {ahora.strftime('%H:%M')} MX. Sesión: {scores.get('razon_sesion','—')}. Dame tu outlook para XAU/USD en las próximas 4 horas."

    # Historial de chat
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg['role'] == 'user' else "assistant"):
            st.markdown(msg['content'])

    # Input manual
    uin = st.chat_input("Consulta a TrendSpider AI... (ej: ¿hay señal ahora? ¿qué dice el H4?)")

    # Procesar input (botón o texto)
    _pregunta = quick_prompt or uin
    if _pregunta:
        st.session_state.chat_history.append({'role':'user','content':_pregunta})
        with st.chat_message("user"):
            st.markdown(_pregunta)
        with st.chat_message("assistant"):
            with st.spinner("TrendSpider AI analizando..."):
                resp = llamar_claude(st.session_state.chat_history[:-1], _pregunta)
            st.markdown(resp)
        st.session_state.chat_history.append({'role':'mimi','content':resp})
        # Mandar análisis a Telegram si modo auto ON
        if st.session_state.get('auto_mode', False) and _pregunta == quick_prompt:
            send_tg(f"🤖 *TrendSpider AI*\n{resp[:800]}")
        st.rerun()

# ── TAB 8: BACKTEST ───────────────────────────────────────────────
with tab8:
    @st.cache_data(ttl=3600)
    def backtest(df_json,asl,atp):
        df_b=pd.read_json(io.StringIO(df_json),orient='split')
        cap=1000.0; eq=[cap]; tds=[]
        ac=ta.volatility.average_true_range(df_b['High'],df_b['Low'],df_b['Close'])
        rc=ta.momentum.rsi(df_b['Close'],window=14); mc=ta.trend.macd_diff(df_b['Close'])
        e2=ta.trend.ema_indicator(df_b['Close'],window=20); e5=ta.trend.ema_indicator(df_b['Close'],window=50)
        i=50
        while i<len(df_b)-5:
            p=float(df_b['Close'].iloc[i]); atr_v=float(ac.iloc[i]) if not pd.isna(ac.iloc[i]) else 50
            rv=float(rc.iloc[i]) if not pd.isna(rc.iloc[i]) else 50; mh=float(mc.iloc[i]) if not pd.isna(mc.iloc[i]) else 0
            em2=float(e2.iloc[i]) if not pd.isna(e2.iloc[i]) else p; em5=float(e5.iloc[i]) if not pd.isna(e5.iloc[i]) else p
            sl_l=p-atr_v*asl; tp_l=p+atr_v*atp; sl_s=p+atr_v*asl; tp_s=p-atr_v*atp; d=0
            if p>em2 and p>em5 and rv<70 and mh>0: d=1
            elif p<em2 and p<em5 and rv>30 and mh<0: d=-1
            if d!=0:
                for j in range(1,6):
                    fp=float(df_b['Close'].iloc[i+j])
                    if d==1:
                        if fp>=tp_l: pnl=(tp_l-p)/p*cap*0.1; cap+=pnl; tds.append({'dir':'LONG','entrada':p,'salida':tp_l,'pnl':round(pnl,2),'res':'WIN'}); break
                        elif fp<=sl_l: pnl=(sl_l-p)/p*cap*0.1; cap+=pnl; tds.append({'dir':'LONG','entrada':p,'salida':sl_l,'pnl':round(pnl,2),'res':'LOSS'}); break
                    else:
                        if fp<=tp_s: pnl=(p-tp_s)/p*cap*0.1; cap+=pnl; tds.append({'dir':'SHORT','entrada':p,'salida':tp_s,'pnl':round(pnl,2),'res':'WIN'}); break
                        elif fp>=sl_s: pnl=(p-sl_s)/p*cap*0.1; cap+=pnl; tds.append({'dir':'SHORT','entrada':p,'salida':sl_s,'pnl':round(pnl,2),'res':'LOSS'}); break
                eq.append(cap); i+=5
            else: i+=1
        w=sum(1 for t in tds if t['res']=='WIN')
        return tds,eq,round(w/len(tds)*100,1) if tds else 0,round(cap-1000,2)

    with st.spinner("Simulando..."):
        bt_t,bt_e,bt_w,bt_p = backtest(df.to_json(orient='split'),MC['atr_sl'],MC['atr_tp'])
    bm1,bm2,bm3,bm4=st.columns(4)
    bm1.metric("Capital Inicial","$1,000"); bm2.metric("Capital Final",f"${1000+bt_p:,.2f}",f"{bt_p:+.2f}")
    bm3.metric("Win Rate",f"{bt_w:.1f}%"); bm4.metric("Trades",str(len(bt_t)))
    if bt_e:
        fe=go.Figure()
        fe.add_trace(go.Scatter(y=bt_e,fill='tozeroy',fillcolor=f'rgba(200,169,110,0.08)',line=dict(color=T['primary'],width=2)))
        fe.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            height=300,margin=dict(l=0,r=0,t=20,b=0),
            title=dict(text=f"CURVA DE CAPITAL — {st.session_state.modo}",font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        fe.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fe.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fe,use_container_width=True)
    if bt_t: st.dataframe(pd.DataFrame(bt_t[-20:]),use_container_width=True)

# ── TAB 9: ALERTAS ────────────────────────────────────────────────
with tab9:
    st.markdown(f'<div class="card-title" style="font-family:Cinzel,serif;color:{T["primary"]};letter-spacing:3px;">ALERTAS · TELEGRAM BIDIRECCIONAL</div>', unsafe_allow_html=True)
    st.markdown("""
| Comando | Acción |
|---------|--------|
| `entré` | Abre paper trade |
| `no` | Rechaza señal |
| `estado` | P&L actual |
| `me quedo` | Evalúa mantener |
| `salgo` | Cierra trade |
| `señal` | Señal actual |
| `wyckoff` | Análisis AMD |
| Pregunta libre | MIMI-AI responde |
""")
    col_a,col_b=st.columns(2)
    with col_a:
        al=st.checkbox("📈 Alertar LONG",value=True)
        as_=st.checkbox("📉 Alertar SHORT",value=True)
        ac2=st.slider("Confianza mínima (%)",30,90,50)
        aw=st.checkbox("⭐ Solo ventana activa",value=True)
    with col_b:
        if st.button("🧪 Prueba"):
            ok=send_tg(f"🏛️ *MIMI-AI Test* ✅\nModo: {st.session_state.modo}\nPrecio: ${precio:,.2f}\nWyckoff: {wyckoff['trend']}\nResponde 'señal' para ver señal.")
            st.success("Enviado ✅") if ok else st.error("Error — revisa Secrets")
        if st.button("📡 Enviar señal"):
            vens2=[(3,5),(8,11),(12,14),(15,17)]; ev=any(i<=h<f for i,f in vens2)
            if conf>=ac2 and ((pred==1 and al) or (pred==-1 and as_)) and ((not aw) or ev):
                sl_r=sl_long if pred>=0 else sl_short; tp_r=tp_long if pred>=0 else tp_short
                wy_msg=f"\n🏺 AMD: {wyckoff['active']['tipo']} @ ${wyckoff['active']['entrada']:,.2f}" if wyckoff.get('active') else ""
                ok2=send_tg(f"🏛️ *MIMI-AI — {st.session_state.modo}*\n🕐 {ahora.strftime('%H:%M')} MX · {st.session_state.trade_style}\n💰 ${precio:,.2f}\n🎯 *{ET.get(pred)}*\n📊 {conf:.1f}% | SMC: {smc['bias']}{wy_msg}\n🔴 SL: ${sl_r:,.2f}\n🟢 TP: ${tp_r:,.2f}\n📐 R:R: 1:{rr}\n\n_Responde 'entré' o 'no'_")
                st.success("Enviado ✅") if ok2 else st.error("Error")
            else: st.info("Condiciones no cumplidas")

# ── TAB 10: HISTORIAL ─────────────────────────────────────────────
with tab10:
    if st.session_state.signal_history:
        df_sh=pd.DataFrame(st.session_state.signal_history)
        st.markdown(f"**{len(df_sh)} señales registradas**")
        st.dataframe(df_sh,use_container_width=True)
        w_h=sum(1 for s in st.session_state.signal_history if 'WIN' in s.get('resultado',''))
        l_h=sum(1 for s in st.session_state.signal_history if 'LOSS' in s.get('resultado',''))
        if w_h+l_h>0: st.metric("Win Rate Real",f"{w_h/(w_h+l_h)*100:.1f}%",f"{w_h}W / {l_h}L")
        if st.button("🗑️ Limpiar historial"):
            st.session_state.signal_history=[]; gh_save({**sv2,'signal_history':[]}); st.rerun()
    else: st.info("Las señales se guardan automáticamente.")

# ── TAB 11: MONITOR ───────────────────────────────────────────────
with tab11:
    ot_m=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot_m:
        t_m=ot_m[0]
        pnl_m=(precio-t_m['entrada'])*(1 if 'LONG' in t_m['dir'] else -1)*t_m['lotes']*100
        pct_m=pnl_m/t_m['entrada']*100 if t_m['entrada'] else 0
        est_m="🚨 SAL YA" if (('LONG' in t_m['dir'] and precio<=t_m['sl']) or ('SHORT' in t_m['dir'] and precio>=t_m['sl'])) else "🎯 TP" if (('LONG' in t_m['dir'] and precio>=t_m['tp']) or ('SHORT' in t_m['dir'] and precio<=t_m['tp'])) else "🟢 MANTÉN" if pnl_m>0 else "🔴 PRECAUCIÓN"
        fm=go.Figure()
        fm.add_hline(y=t_m['tp'],line_color='#4CAF82',line_dash='dash',annotation_text=f"TP ${t_m['tp']:,.0f}")
        fm.add_hline(y=t_m['entrada'],line_color=T['primary'],line_width=2,annotation_text=f"ENTRADA ${t_m['entrada']:,.0f}")
        fm.add_hline(y=t_m['sl'],line_color='#C0392B',line_dash='dash',annotation_text=f"SL ${t_m['sl']:,.0f}")
        fm.add_hline(y=precio,line_color='#FFFFFF',line_dash='dot',annotation_text=f"ACTUAL ${precio:,.2f}")
        fm.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',height=260,
            font=dict(color='#888',family='Philosopher,serif'),margin=dict(l=0,r=0,t=30,b=0),
            title=dict(text=f"{t_m['dir']} | P&L: {'+'if pnl_m>0 else ''}${pnl_m:.2f} | {est_m}",
                       font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        st.plotly_chart(fm,use_container_width=True)
        mo1,mo2,mo3,mo4=st.columns(4)
        mo1.metric("Entrada",f"${t_m['entrada']:,.2f}"); mo2.metric("Actual",f"${precio:,.2f}")
        mo3.metric("P&L",f"${pnl_m:.2f}",f"{pct_m:+.3f}%"); mo4.metric("Estado",est_m)
        mc1,mc2=st.columns(2)
        if mc1.button("🔄 Actualizar"): st.cache_data.clear(); st.rerun()
        if mc2.button("📱 Pedir evaluación al bot"):
            send_tg(f"👁️ *Monitor*\n{t_m['dir']} @ ${t_m['entrada']:,.2f}\nActual: ${precio:,.2f}\nP&L: {'+'if pnl_m>0 else ''}${pnl_m:.2f}\n{est_m}\n\n'me quedo' o 'salgo'")
            st.success("Enviado")
    else:
        st.info("Sin posición abierta.")

# ════════════════════════════════════════════════════════════════
#  FRASE FINAL
# ════════════════════════════════════════════════════════════════
fr = random.choice(FRASES)
st.markdown(f"""
<div class="greek-orn" style="margin-top:24px;">─────── ✦ ───────</div>
<div class="stoic-q">{fr[1]}<div class="stoic-a">— {fr[0]}</div></div>
<div class="greek-orn">─────── ✦ ───────</div>
""", unsafe_allow_html=True)
