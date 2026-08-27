import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pytz
import time
from datetime import datetime, timedelta
import random
import warnings
import requests
import io
import json
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI", page_icon="🏛️", layout="wide")

# ── FRAGMENT — permite refrescar SOLO el precio, sin recargar la app entera
try:
    fragment_decorator = st.fragment
except AttributeError:
    fragment_decorator = st.experimental_fragment

# ── SECRETS ──────────────────────────────────────────────────────
try:
    TG_TOKEN   = st.secrets["TG_TOKEN"]
    TG_CHAT_ID = st.secrets["TG_CHAT_ID"]
    GH_TOKEN   = st.secrets["GITHUB_TOKEN"]
    GH_REPO    = st.secrets["GITHUB_REPO"]
except:
    TG_TOKEN = TG_CHAT_ID = GH_TOKEN = GH_REPO = ''

def _leer_secret(nombre, default=''):
    """Lectura robusta de Secrets: usa .get (no KeyError), y limpia espacios
    o saltos de línea invisibles que a veces trae el copy-paste del navegador."""
    try:
        val = st.secrets.get(nombre, default)
        return val.strip() if isinstance(val, str) else val
    except Exception:
        return default

GEMINI_API_KEY = _leer_secret("GEMINI_API_KEY", "")
GEMINI_MODEL   = _leer_secret("GEMINI_MODEL", "gemini-3.5-flash")

# ══════════════════════════════════════════════════════════════════
#  ESTRATEGIA POR PAR — Trend Following + Pullback (price action puro)
#  XAU/USD: EMA50/200 tendencia · EMA20/50 pullback · alterna IB Breakout
#  EUR/USD: EMA20/200 tendencia · EMA20 o Fib 50-61.8% pullback
# ══════════════════════════════════════════════════════════════════
PAIRS = {
    "XAU/USD 🥇": {
        "nombre": "Oro", "slug": "xauusd",
        "yf_symbol": "GC=F", "td_symbol": "XAU/USD",
        "dxy_symbol": "DX-Y.NYB", "contract_size": 100, "decimales": 2,
        "ema_trend": (50, 200), "ema_entry": 20,
        "sl_atr_mult": 1.5, "tp_rr": 2.5, "tp2_rr": 4.0,
        "usa_fib": False, "usa_ib": True,
    },
    "EUR/USD 💶": {
        "nombre": "Euro", "slug": "eurusd",
        "yf_symbol": "EURUSD=X", "td_symbol": "EUR/USD",
        "dxy_symbol": "DX-Y.NYB", "contract_size": 100000, "decimales": 5,
        "ema_trend": (20, 200), "ema_entry": 20,
        "sl_atr_mult": 1.2, "tp_rr": 2.0, "tp2_rr": 3.5,
        "usa_fib": True, "usa_ib": False,
    }
}

STYLE_TF = {
    "Scalping":    {"trend_interval":"1h",  "trend_period":"60d",  "entry_interval":"15m", "entry_period":"10d", "label":"M15/H1"},
    "Day Trading": {"trend_interval":"4h",  "trend_period":"180d", "entry_interval":"1h",  "entry_period":"60d", "label":"H1/H4"},
    "Swing":       {"trend_interval":"1d",  "trend_period":"2y",   "entry_interval":"4h",  "entry_period":"180d","label":"H4/D1"},
}

NOTICIAS_ALTO_IMPACTO = ["FOMC", "NFP (Nóminas no agrícolas)", "Decisión de tasas ECB", "CPI / Inflación", "Discurso de la Fed"]

PIP_SIZE = {"XAU/USD 🥇": 0.1, "EUR/USD 💶": 0.0001}
SCORE_MIN_ALERTA = 58
SCORE_MIN_ASIA = 80  # en sesión asiática solo se avisa si es MUY clara

def es_sesion_asiatica(hora_mx):
    return hora_mx in (19,20,21,22,23,0,1,2)

def a_pips(dist_precio, par):
    return dist_precio / PIP_SIZE.get(par, 0.0001)

def generar_escenarios(par, sen, df_entry, lookback=20):
    precio_ = sen['precio']; atr_ = sen['atr']
    resistencia = float(df_entry['High'].iloc[-lookback:].max())
    soporte     = float(df_entry['Low'].iloc[-lookback:].min())
    obj_arriba  = resistencia + atr_*2
    obj_abajo   = soporte - atr_*2
    return [
        f"Si rompe {pf(resistencia,par)} con fuerza → probable continuación hasta {pf(obj_arriba,par)}",
        f"Si rechaza en {pf(resistencia,par)} → posible retroceso hacia {pf(soporte,par)}",
        "Zona actual clave — esperar confirmación de ruptura o rechazo antes de decidir",
    ]

def pdec(par): return PAIRS[par]["decimales"]
def pf(x, par):
    try: return f"{x:,.{pdec(par)}f}"
    except Exception: return str(x)

# ── PRECIO EN VIVO ─────────────────────────────────────────────────
@st.cache_data(ttl=15)
def get_precio_vivo(td_symbol, yf_symbol):
    try:
        td_key = st.secrets.get("TWELVEDATA_KEY", "")
        if td_key:
            r = requests.get(f"https://api.twelvedata.com/quote?symbol={td_symbol}&apikey={td_key}", timeout=2)
            if r.status_code == 200:
                d = r.json()
                p = float(d.get("close", 0)); prev = float(d.get("previous_close", p))
                if p > 0:
                    return {'precio':p,'prev':prev,'cambio':round(p-prev,6),
                            'cambio_pct':round((p-prev)/prev*100,3) if prev else 0,
                            'high':float(d.get("high",p)),'low':float(d.get("low",p)),
                            'hora':d.get("datetime","")[:16],'fuente':'TwelveData'}
    except: pass
    try:
        df_tick = yf.download(yf_symbol, period="1d", interval="1m", progress=False)
        if df_tick is not None and len(df_tick) > 0:
            df_tick.columns = [c[0] if isinstance(c,tuple) else c for c in df_tick.columns]
            last = df_tick.iloc[-1]; prev_row = df_tick.iloc[-2] if len(df_tick)>1 else last
            p = float(last['Close']); prev = float(prev_row['Close'])
            return {'precio':p,'prev':prev,'cambio':round(p-prev,6),
                    'cambio_pct':round((p-prev)/prev*100,3) if prev else 0,
                    'high':float(last['High']),'low':float(last['Low']),
                    'hora':str(df_tick.index[-1])[11:16],'fuente':'yfinance 1m'}
    except: pass
    return None

@fragment_decorator(run_every=3)
def precios_en_vivo(par_activo):
    cols = st.columns(2)
    for i, par_key in enumerate(PAIRS.keys()):
        pc_i = PAIRS[par_key]
        tk = get_precio_vivo(pc_i['td_symbol'], pc_i['yf_symbol'])
        p_d   = tk['precio'] if tk else None
        c_d   = tk['cambio'] if tk else 0
        cp_d  = tk['cambio_pct'] if tk else 0
        hora_d = tk['hora'][11:16] if tk else "—"
        color = '#4CAF82' if c_d >= 0 else '#C0392B'
        flecha = '▲' if c_d >= 0 else '▼'
        activo_tag = " ⭐ ANALIZANDO" if par_key == par_activo else ""
        with cols[i]:
            if tk:
                st.markdown(f"""
                <div style="background:linear-gradient(90deg,{T['card']},{T['bg']},{T['card']});
                  border:1px solid {color}44;border-radius:4px;padding:12px 16px;margin:4px 0;">
                  <span style="font-family:'Cinzel',serif;color:{T['primary']}88;font-size:.72em;letter-spacing:2px;">{par_key} · EN VIVO · {hora_d}{activo_tag}</span><br>
                  <span style="font-family:'Cinzel',serif;font-size:clamp(1.2rem,3.5vw,1.9rem);font-weight:900;color:{color};
                    filter:drop-shadow(0 0 10px {color}66);">{pf(p_d,par_key)}</span>
                  <span style="font-family:'Philosopher',serif;color:{color};font-size:.9em;margin-left:10px;">
                    {flecha} {pf(abs(c_d),par_key)} ({cp_d:+.3f}%)
                  </span>
                  <div style="font-family:'Philosopher',serif;color:{T['primary']}77;font-size:.72em;margin-top:2px;">{tk['fuente']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"Sin datos en vivo — {par_key}")

# ── GITHUB PERSISTENCE (un archivo por par) ───────────────────────
def gh_file_for(par): return f"mimi_data_{PAIRS[par]['slug']}.json"

def gh_load(par):
    if not GH_TOKEN or not GH_REPO: return {}
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{gh_file_for(par)}"
        r = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}"}, timeout=5)
        if r.status_code == 200:
            return json.loads(base64.b64decode(r.json().get('content','')).decode())
    except: pass
    return {}

def gh_save(par, data):
    if not GH_TOKEN or not GH_REPO: return
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{gh_file_for(par)}"
        r   = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}"}, timeout=5)
        sha = r.json().get('sha','') if r.status_code == 200 else ''
        cnt = base64.b64encode(json.dumps(data, default=str).encode()).decode()
        payload = {"message":"MIMI-AI update","content":cnt}
        if sha: payload["sha"] = sha
        requests.put(url, headers={"Authorization": f"token {GH_TOKEN}"}, json=payload, timeout=5)
    except: pass

def gh_load_config():
    if not GH_TOKEN or not GH_REPO: return {}
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/mimi_config.json"
        r = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}"}, timeout=5)
        if r.status_code == 200:
            return json.loads(base64.b64decode(r.json().get('content','')).decode())
    except: pass
    return {}

def gh_save_config(cfg):
    if not GH_TOKEN or not GH_REPO: return
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/mimi_config.json"
        r   = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}"}, timeout=5)
        sha = r.json().get('sha','') if r.status_code == 200 else ''
        cnt = base64.b64encode(json.dumps(cfg, default=str).encode()).decode()
        payload = {"message":"MIMI-AI config","content":cnt}
        if sha: payload["sha"] = sha
        requests.put(url, headers={"Authorization": f"token {GH_TOKEN}"}, json=payload, timeout=5)
    except: pass

# ── SESSION STATE ─────────────────────────────────────────────────
if 'par_loaded' not in st.session_state:
    cfg = gh_load_config()
    st.session_state.par = cfg.get('par', "XAU/USD 🥇")
    st.session_state.tema = cfg.get('tema', "Mármol Griego")
    st.session_state.par_loaded = True

def cargar_estado_par(par):
    sv = gh_load(par)
    st.session_state.paper_trades    = sv.get('paper_trades', [])
    st.session_state.signal_history  = sv.get('signal_history', [])
    st.session_state.capital         = sv.get('capital', 1000.0)
    st.session_state.trade_style     = sv.get('trade_style', 'Day Trading')
    st.session_state.estrategia_xau  = sv.get('estrategia_xau', 'Trend + Pullback (EMA)')
    st.session_state.last_tg_update  = sv.get('last_tg_update', 0)
    st.session_state.last_entry_alert = sv.get('last_entry_alert', {})
    st.session_state.loaded_par      = par

if 'loaded_par' not in st.session_state or st.session_state.loaded_par != st.session_state.par:
    cargar_estado_par(st.session_state.par)
    st.session_state.chat_history = []

# ── THEMES ───────────────────────────────────────────────────────
# Mármol Griego = mármol blanco/plateado real (ya no dorado — ese look se
# lo quedó Bronce Estoico). Athena renombrado a Agamenón (negro + mármol/oro,
# inspirado en su armadura). Púrpura Imperial reemplazado por Grecia
# Contemporánea, basado en la paleta de 5 tonos que mandaste.
THEMES = {
    "Mármol Griego":       {"primary":"#E9E4D8","secondary":"#AFA28C","bg":"#0b0b0a","card":"#141311"},
    "Bronce Estoico":      {"primary":"#C8A96E","secondary":"#8B6914","bg":"#0a0905","card":"#13100a"},
    "Lapislázuli":         {"primary":"#6B8FCE","secondary":"#3A5A9B","bg":"#03060f","card":"#070b18"},
    "Olimpo Oscuro":       {"primary":"#9B7FD4","secondary":"#6B4FA0","bg":"#060308","card":"#0d0614"},
    "Agamenón":            {"primary":"#D4C9A8","secondary":"#5C5347","bg":"#050505","card":"#0d0d0c"},
    "Ónix Espartano":      {"primary":"#B33A3A","secondary":"#6E1F1F","bg":"#070505","card":"#100b0b"},
    "Laurel de Delfos":    {"primary":"#8FB08C","secondary":"#4C6B4A","bg":"#050a06","card":"#0a120b"},
    "Grecia Contemporánea":{"primary":"#6FA09C","secondary":"#B0A093","bg":"#070a09","card":"#0d1211",
                             "palette":["#D4E1E0","#9DBFBC","#6FA09C","#D3CCC0","#B0A093"]},
}
if 'tema' not in st.session_state: st.session_state.tema = "Mármol Griego"
T = THEMES.get(st.session_state.tema, THEMES["Mármol Griego"])

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
]

# ── ESTRATEGIA: TENDENCIA / PULLBACK / PATRONES / FIB / IB ────────
def detectar_tendencia(df, ema_fast_p, ema_slow_p):
    ef = ta.trend.ema_indicator(df['Close'], window=ema_fast_p)
    es = ta.trend.ema_indicator(df['Close'], window=ema_slow_p)
    precio = float(df['Close'].iloc[-1]); f = float(ef.iloc[-1]); s = float(es.iloc[-1])
    if precio > f and precio > s and f > s: return 'ALCISTA'
    if precio < f and precio < s and f < s: return 'BAJISTA'
    return 'RANGO'

def en_pullback(df, ema_period, tolerancia=0.0025):
    ema = ta.trend.ema_indicator(df['Close'], window=ema_period)
    precio = float(df['Close'].iloc[-1]); nivel = float(ema.iloc[-1])
    dist = abs(precio - nivel) / precio
    return dist <= tolerancia, nivel

def fib_levels(df, lookback=50):
    h = float(df['High'].iloc[-lookback:].max())
    l = float(df['Low'].iloc[-lookback:].min())
    diff = h - l
    return {'high':h,'low':l,'fib_50': h-diff*0.5,'fib_618': h-diff*0.618}

def en_zona_fib(precio, fib, direccion, tolerancia=0.0025):
    if direccion == 1:
        lo, hi = fib['fib_618'], fib['fib_50']
    else:
        lo, hi = fib['fib_50'], fib['fib_618']
    lo2, hi2 = min(lo,hi), max(lo,hi)
    margen = (hi2-lo2)*tolerancia*40 if (hi2-lo2) > 0 else precio*tolerancia
    return (lo2-margen) <= precio <= (hi2+margen)

def detectar_patron_vela(df):
    o,h,l,c = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    o_prev,c_prev = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])
    body = abs(c-o); rng = h-l
    upper_wick = h - max(o,c); lower_wick = min(o,c) - l
    if rng > 0:
        if lower_wick > body*2 and lower_wick > rng*0.45:
            return 'PIN BAR ALCISTA'
        if upper_wick > body*2 and upper_wick > rng*0.45:
            return 'PIN BAR BAJISTA'
    if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
        return 'ENGULFING ALCISTA'
    if c_prev > o_prev and c < o and c < o_prev and o > c_prev:
        return 'ENGULFING BAJISTA'
    return None

def detectar_estructura_rota(df, lookback=20):
    h = float(df['High'].iloc[-lookback:-1].max())
    l = float(df['Low'].iloc[-lookback:-1].min())
    c = float(df['Close'].iloc[-1])
    if c > h: return 'ALCISTA'
    if c < l: return 'BAJISTA'
    return None

@st.cache_data(ttl=1800)
def get_dxy_returns(period="60d"):
    try:
        d = yf.download("DX-Y.NYB", period=period, interval="1d", progress=False)
        d.columns = [c[0] if isinstance(c,tuple) else c for c in d.columns]
        d.dropna(inplace=True)
        return float(d['Close'].pct_change(1).iloc[-1])
    except Exception:
        return None

def initial_balance_breakout(df_m15):
    """Rango de la primera hora de la sesión de NY (08:00-09:00 hora MX) y su ruptura."""
    mx = pytz.timezone('America/Mexico_City')
    idx = pd.to_datetime(df_m15.index)
    try:
        idx_mx = idx.tz_convert(mx)
    except Exception:
        idx_mx = idx.tz_localize('UTC').tz_convert(mx)
    df_local = df_m15.copy(); df_local.index = idx_mx
    hoy = datetime.now(mx).date()
    ib = df_local[(df_local.index.date == hoy) & (df_local.index.hour == 8)]
    if len(ib) < 2: return None
    ib_high, ib_low = float(ib['High'].max()), float(ib['Low'].min())
    despues = df_local[(df_local.index.date == hoy) & (df_local.index.hour > 8)]
    if despues.empty: return {'direccion':None,'ib_high':ib_high,'ib_low':ib_low}
    precio_actual = float(despues['Close'].iloc[-1])
    if precio_actual > ib_high: direccion = 1
    elif precio_actual < ib_low: direccion = -1
    else: direccion = 0
    return {'direccion':direccion,'ib_high':ib_high,'ib_low':ib_low,'precio':precio_actual}

def generar_senal(par, df_trend, df_entry, usar_ib=False):
    cfg = PAIRS[par]
    tendencia = detectar_tendencia(df_trend, cfg['ema_trend'][0], cfg['ema_trend'][1])
    en_pb, ema_ref = en_pullback(df_entry, cfg['ema_entry'])
    patron   = detectar_patron_vela(df_entry)
    estructura = detectar_estructura_rota(df_entry)
    precio = float(df_entry['Close'].iloc[-1])
    atr    = float(ta.volatility.average_true_range(df_entry['High'],df_entry['Low'],df_entry['Close']).iloc[-1])

    fib = fib_levels(df_trend, 50) if cfg['usa_fib'] else None
    en_fib_long  = en_zona_fib(precio, fib, 1) if fib else False
    en_fib_short = en_zona_fib(precio, fib, -1) if fib else False

    ib = None
    if usar_ib and cfg['usa_ib']:
        ib = initial_balance_breakout(df_entry)

    direccion = 0; razon = []

    if ib and ib.get('direccion') in (1,-1):
        direccion = ib['direccion']
        razon.append(f"Initial Balance Breakout — ruptura del rango {pf(ib['ib_low'],par)}–{pf(ib['ib_high'],par)} de la primera hora NY")
    else:
        if tendencia == 'ALCISTA':
            zona_ok = en_pb or en_fib_long
            confirm_ok = patron == 'PIN BAR ALCISTA' or patron == 'ENGULFING ALCISTA' or estructura == 'ALCISTA'
            if zona_ok and confirm_ok:
                direccion = 1
                razon.append(f"Tendencia alcista — precio sobre EMA{cfg['ema_trend'][0]}/EMA{cfg['ema_trend'][1]}")
                razon.append(f"Pullback a EMA{cfg['ema_entry']}" if en_pb else "Retroceso en zona Fib 50–61.8%")
                if patron: razon.append(f"Confirmación: {patron}")
                if estructura: razon.append("Ruptura de microestructura a favor")
        elif tendencia == 'BAJISTA':
            zona_ok = en_pb or en_fib_short
            confirm_ok = patron == 'PIN BAR BAJISTA' or patron == 'ENGULFING BAJISTA' or estructura == 'BAJISTA'
            if zona_ok and confirm_ok:
                direccion = -1
                razon.append(f"Tendencia bajista — precio bajo EMA{cfg['ema_trend'][0]}/EMA{cfg['ema_trend'][1]}")
                razon.append(f"Pullback a EMA{cfg['ema_entry']}" if en_pb else "Rally en zona Fib 50–61.8%")
                if patron: razon.append(f"Confirmación: {patron}")
                if estructura: razon.append("Ruptura de microestructura a favor")

    sl_mult = cfg['sl_atr_mult']; tp_rr = cfg['tp_rr']; tp2_rr = cfg['tp2_rr']
    if direccion == 1:
        swing_ref = float(df_entry['Low'].iloc[-8:].min())
        sl = min(swing_ref, precio - atr*sl_mult)
        tp = precio + (precio - sl) * tp_rr
        tp2 = precio + (precio - sl) * tp2_rr
    elif direccion == -1:
        swing_ref = float(df_entry['High'].iloc[-8:].max())
        sl = max(swing_ref, precio + atr*sl_mult)
        tp = precio - (sl - precio) * tp_rr
        tp2 = precio - (sl - precio) * tp2_rr
    else:
        sl = precio - atr*sl_mult
        tp = precio + atr*sl_mult*tp_rr
        tp2 = precio + atr*sl_mult*tp2_rr

    return {
        'direccion':direccion,'precio':precio,'sl':round(sl,6),'tp':round(tp,6),'tp2':round(tp2,6),
        'tendencia':tendencia,'pullback':en_pb,'ema_ref':ema_ref,'patron':patron,
        'estructura':estructura,'atr':atr,'razon':razon,'fib':fib,
        'en_fib_long':en_fib_long,'en_fib_short':en_fib_short,'ib':ib
    }

def calcular_score(senal, dxy_corr_ok, noticia_bloqueando, rr):
    setup = 0
    if senal['tendencia'] in ('ALCISTA','BAJISTA'): setup += 4
    if senal['pullback'] or senal['en_fib_long'] or senal['en_fib_short']: setup += 3
    if senal['patron']: setup += 2
    if senal['estructura']: setup += 1
    setup = min(setup, 10)

    prob = 5
    if senal['direccion'] != 0: prob += 3
    if dxy_corr_ok: prob += 2
    prob = min(prob, 10)

    noticias_score = 3 if noticia_bloqueando else 9
    riesgo_score = 10 if rr >= 2.5 else 8 if rr >= 2 else 6 if rr >= 1.5 else 3

    total = setup*0.4 + prob*0.3 + noticias_score*0.15 + riesgo_score*0.15
    total_pct = round(total*10, 1)
    if total_pct >= 75: cat = "🟢 EXCELENTE"
    elif total_pct >= 60: cat = "🟢 BUENA — recomendada"
    elif total_pct >= 45: cat = "🟡 ACEPTABLE"
    elif total_pct >= 30: cat = "🟠 BAJA CALIDAD — solo agresivo"
    else: cat = "🔴 NO RECOMENDADA"
    return {'total':total_pct,'setup':setup,'prob':prob,'noticias':noticias_score,'riesgo':riesgo_score,'categoria':cat}

def calc_pos(capital, risk, entrada, sl, contract_size):
    r = capital * (risk / 100); d = abs(entrada - sl)
    if d == 0: return 0, 0
    return round(r/(d*contract_size), 4), round(r, 2)

# ── TELEGRAM ─────────────────────────────────────────────────────
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
        if r.status_code == 200: return r.json().get('result', [])
    except: pass
    return []

# ══════════════════════════════════════════════════════════════════
#  ANÁLISIS DE IMÁGENES (screenshots de gráficos) — Google Gemini
#  Usa el SDK nuevo "google-genai" (el viejo "google-generativeai" ya
#  fue descontinuado por Google). Requiere GEMINI_API_KEY en Secrets.
#  Modelo configurable vía GEMINI_MODEL en Secrets (por defecto
#  "gemini-3.5-flash" — OJO: gemini-1.5-flash y gemini-2.0-flash ya
#  fueron apagados por Google (junio 2026). Revisa
#  ai.google.dev/gemini-api/docs/deprecations
#  cada tanto porque Google cambia/retira modelos con frecuencia).
# ══════════════════════════════════════════════════════════════════
GEMINI_SYSTEM_PROMPT = """Eres MIMI-AI, analista técnico profesional especializado EXCLUSIVAMENTE en XAUUSD (oro) y EURUSD. Te muestran capturas de pantalla de gráficos de trading (velas japonesas, indicadores, líneas dibujadas a mano, etc.) y debes analizarlas con criterio profesional y disciplinado.

Para cada imagen que recibas, estructura tu respuesta así:

1. **Par y timeframe** — identifica XAUUSD o EURUSD y el timeframe aproximado si es visible. Si no puedes determinarlo con certeza, dilo explícitamente en vez de adivinar.
2. **Estructura de mercado** — tendencia visible (alcista / bajista / rango), máximos y mínimos relevantes (HH/HL/LH/LL si se distinguen claramente).
3. **Niveles clave** — soportes, resistencias, zonas de oferta/demanda o líneas de tendencia dibujadas en la imagen, con precios concretos si son legibles.
4. **Patrones e indicadores visibles** — velas relevantes (pin bar, engulfing, doble techo/piso, banderas, etc.) e indicadores visibles (RSI, MACD, EMAs, Bollinger) con su lectura actual.
5. **Escenarios (2-3, obligatorio)** — con el mismo estilo siempre: "Si rompe X con fuerza → probable continuación hasta Y", "Si rechaza en X → posible retroceso hacia Z", "Esperar confirmación en zona actual".
6. **Decisión — obligatorio, sin vaguedad** — termina SIEMPRE con una línea "Decisión:" seguida de UNA de estas tres formas concretas (nunca dejes la decisión abierta ni te quedes solo en "depende"):
   - Si el setup ya está confirmado ahora mismo: **"Decisión: Compra ahora"** o **"Decisión: Vende ahora"**, con el nivel de entrada.
   - Si falta un disparador claro: **"Decisión: Espera a que rompa/rechace en $X para comprar/vender"** — siempre con el nivel de precio EXACTO que activaría la entrada, nunca un "espera" sin número.
   - Solo usa **"Decisión: Sin operación"** si la imagen es demasiado ambigua o ilegible para dar ni siquiera un nivel de activación.
   Después de la Decisión agrega una calificación aproximada 0-100% de qué tan limpio está el setup (40% calidad técnica, 30% probabilidad direccional, 15% contexto/noticias — asume neutral si no tienes esa info, 15% gestión de riesgo visible) — esto es información de apoyo, la Decisión es lo principal.

Reglas estrictas:
- NUNCA inventes niveles de precio que no puedas justificar con lo que se ve en la imagen. Si está borrosa o incompleta, dilo.
- Si la imagen NO es un gráfico de trading, dilo directamente y no fuerces un análisis técnico falso.
- Sé directo, profesional y breve — usa viñetas y encabezados cortos, sin relleno.
- La Decisión debe ser accionable: evita frases como "podría subir o bajar" sin más — siempre da la condición o el nivel exacto que el usuario debe vigilar.
- Aun siendo directivo, no prometas resultados garantizados — el análisis técnico da probabilidades altas, no certezas absolutas.
- Responde siempre en español, con emojis moderados (🟢🔴📊🎯🔭⚠️) para que se lea fácil en Telegram.
- Máximo ~200 palabras salvo que te pidan explícitamente más detalle."""

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    """La caché se guarda por valor de api_key: si cambias el secret y
    reinicias, esta función se vuelve a evaluar en vez de quedarse
    pegada a un cliente viejo (o a None de antes de configurar la key)."""
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None

def descargar_foto_telegram(file_id):
    """Descarga la foto de mayor resolución que Telegram guardó para ese file_id."""
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getFile",
                          params={'file_id': file_id}, timeout=10)
        file_path = r.json()['result']['file_path']
        url = f"https://api.telegram.org/file/bot{TG_TOKEN}/{file_path}"
        img = requests.get(url, timeout=15)
        mime = 'image/png' if file_path.lower().endswith('.png') else 'image/jpeg'
        return img.content, mime
    except Exception:
        return None, None

GEMINI_MODELOS_RESPALDO = ["gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]

def analizar_imagen_grafica(image_bytes, mime_type, contexto_extra=""):
    """Manda la imagen + el system prompt fuerte a Gemini y regresa el análisis en texto.
    Si el modelo configurado ya no existe (Google los retira seguido), reintenta
    automáticamente con una lista de respaldo antes de rendirse."""
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY no está configurada en Secrets (o llegó vacía). Revisa Settings → Secrets en Streamlit Cloud."
    client = get_gemini_client(GEMINI_API_KEY)
    if client is None:
        return "⚠️ La key está presente pero no se pudo crear el cliente de Gemini — probablemente la key es inválida o falta instalar el paquete 'google-genai'."

    from google.genai import types
    partes = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        f"Analiza esta captura de gráfico de trading. {contexto_extra}".strip()
    ]
    intentos = [GEMINI_MODEL] + [m for m in GEMINI_MODELOS_RESPALDO if m != GEMINI_MODEL]
    ultimo_error = None
    for modelo in intentos:
        try:
            resp = client.models.generate_content(
                model=modelo,
                contents=partes,
                config=types.GenerateContentConfig(system_instruction=GEMINI_SYSTEM_PROMPT, temperature=0.3),
            )
            aviso = f"\n\n_(nota: {GEMINI_MODEL} ya no está disponible, usé {modelo} — actualiza GEMINI_MODEL en Secrets)_" if modelo != GEMINI_MODEL else ""
            return (resp.text or "No pude generar un análisis para esta imagen.") + aviso
        except Exception as e:
            ultimo_error = e
            if "404" in str(e) or "NOT_FOUND" in str(e) or "no longer available" in str(e):
                continue  # prueba el siguiente modelo de la lista
            break  # otro tipo de error (key inválida, red, etc.) — no tiene caso seguir probando
    return f"⚠️ Error analizando la imagen con Gemini (probé {len(intentos)} modelos): {ultimo_error}"

def parse_tg_command(txt):
    t = txt.lower().strip()
    if any(w in t for w in ['bitácora','bitacora','historial completo','resumen 2 dias','resumen dos dias','log']): return 'BITACORA'
    if any(w in t for w in ['entré','entre','sí entro','si entro','entro','long','short','sí','si']): return 'ENTRO'
    if any(w in t for w in ['no','no entro','no entré','cancelar','rechazar']): return 'NO_ENTRO'
    if any(w in t for w in ['me quedo','quedo','mantener','mantén','hold','seguir']): return 'MANTENER'
    if any(w in t for w in ['salgo','salir','cerrar','cierra','exit','sal']): return 'SALIR'
    if any(w in t for w in ['estado','status','como voy','cómo voy','posicion','posición']): return 'STATUS'
    if any(w in t for w in ['señal','signal','que dice','qué dice','analiza']): return 'SEÑAL'
    return 'TEXTO_LIBRE'

def process_tg_updates(par, senal, score, risk_pct, contract_size):
    updates = get_tg_updates(offset=st.session_state.last_tg_update)
    precio = senal['precio']; pred = senal['direccion']
    for u in updates:
        uid = u.get('update_id', 0)
        if uid <= st.session_state.last_tg_update: continue
        st.session_state.last_tg_update = uid + 1
        msg = u.get('message', {})

        # ── FOTO/SCREENSHOT DE GRÁFICO → análisis con Gemini ──────
        fotos = msg.get('photo')
        if fotos:
            file_id = fotos[-1]['file_id']  # última = mayor resolución
            caption = msg.get('caption', '')
            send_tg("📸 Recibí tu captura, analizándola...")
            img_bytes, mime = descargar_foto_telegram(file_id)
            if img_bytes:
                contexto = f"Contexto de la app: par activo {par}, precio actual {pf(precio,par)}. {caption}".strip()
                analisis = analizar_imagen_grafica(img_bytes, mime, contexto)
                send_tg(f"🖼️ *ANÁLISIS DE IMAGEN*\n\n{analisis}")
                try: st.toast("Análisis de imagen enviado", icon="🖼️")
                except Exception: pass
            else:
                send_tg("⚠️ No pude descargar la imagen, intenta reenviarla.")
            continue

        txt = msg.get('text', '')
        if not txt: continue
        cmd = parse_tg_command(txt)
        ot  = [t for t in st.session_state.paper_trades if t['estado'] == 'ABIERTO']
        mx  = pytz.timezone('America/Mexico_City'); ah = datetime.now(mx); tag = par.split()[0]

        if cmd == 'ENTRO':
            if pred != 0 and not ot:
                lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, senal['sl'], contract_size)
                st.session_state.paper_trades.append({
                    'id':len(st.session_state.paper_trades)+1,'par':par,
                    'dir':'LONG 📈' if pred==1 else 'SHORT 📉','entrada':precio,'sl':senal['sl'],
                    'tp':senal['tp'],'tp1':senal['tp'],'tp2':senal.get('tp2',senal['tp']),
                    'tp1_hit':False,'sl_warned':False,'sl_dist':abs(precio-senal['sl']),'last_followup_ts':_ahora_ts(),
                    'lotes':lot,'riesgo':risg,'estado':'ABIERTO','fecha':ah.strftime('%d/%m %H:%M'),
                    'resultado':'PENDIENTE','pnl':0,'score':score['total']})
                send_tg(f"✅ *Trade registrado — {tag}*\n{'LONG 📈' if pred==1 else 'SHORT 📉'} @ {pf(precio,par)}\nSL: {pf(senal['sl'],par)} | TP1: {pf(senal['tp'],par)} | TP2: {pf(senal.get('tp2',senal['tp']),par)}\nScore: {score['total']}%\nLotes: {lot} | Riesgo: ${risg:.2f}")
            elif ot: send_tg("⚠️ Ya tienes un trade abierto en este par.")
            else: send_tg("⚠️ No hay setup claro ahorita. Espera confirmación.")
        elif cmd == 'NO_ENTRO':
            send_tg(f"🏛️ Señal rechazada — {tag} @ {pf(precio,par)}\n_El estoico espera._")
        elif cmd == 'MANTENER':
            if ot:
                t = ot[0]; pnl = (precio - t['entrada']) * (1 if 'LONG' in t['dir'] else -1) * t['lotes'] * contract_size
                send_tg(f"🏛️ *Posición activa — {tag}*\n{t['dir']} @ {pf(t['entrada'],par)}\nActual: {pf(precio,par)}\nP&L: {'+' if pnl>0 else ''}${pnl:.2f}\n\n{'🟢 Mantén.' if pnl>0 else '🔴 Evalúa si el setup sigue válido.'}")
            else: send_tg("Sin trade abierto.")
        elif cmd == 'SALIR':
            if ot:
                t = ot[0]; pnl = (precio - t['entrada']) * (1 if 'LONG' in t['dir'] else -1) * t['lotes'] * contract_size
                t['estado']='CERRADO'; t['pnl']=round(pnl,2); t['resultado']='WIN ✅' if pnl>0 else 'LOSS ❌'
                send_tg(f"{'✅' if pnl>0 else '❌'} *Cerrado — {tag}*\nSalida: {pf(precio,par)}\nP&L: {'+' if pnl>0 else ''}${pnl:.2f}")
            else: send_tg("Sin trade abierto para cerrar.")
        elif cmd == 'STATUS':
            if ot:
                t = ot[0]; pnl = (precio - t['entrada']) * (1 if 'LONG' in t['dir'] else -1) * t['lotes'] * contract_size
                send_tg(f"👁️ *Estado — {tag}*\n{t['dir']} @ {pf(t['entrada'],par)}\nActual: {pf(precio,par)}\nP&L: {'+' if pnl>0 else ''}${pnl:.2f}\nSL: {pf(t['sl'],par)} | TP: {pf(t['tp'],par)}")
            else:
                send_tg(f"📊 Sin posición — {tag}. Score actual: {score['total']}% ({score['categoria']})\nCapital: ${st.session_state.capital:,.2f}")
        elif cmd == 'SEÑAL':
            send_tg(f"🏛️ *MIMI-AI — {tag}*\nTendencia: {senal['tendencia']}\nSetup: {' · '.join(senal['razon']) if senal['razon'] else 'Sin confluencia aún'}\nScore: {score['total']}% ({score['categoria']})\nPrecio: {pf(precio,par)} | SL: {pf(senal['sl'],par)} | TP: {pf(senal['tp'],par)}")
        elif cmd == 'BITACORA':
            send_tg(generar_bitacora_2dias())
        elif cmd == 'TEXTO_LIBRE':
            send_tg(f"🏛️ Score actual: {score['total']}% — {score['categoria']}\nTendencia: {senal['tendencia']}\nComandos: entré · no · salgo · estado · señal · me quedo · bitácora")

    sv2 = {'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history[-50:],
           'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
           'estrategia_xau':st.session_state.estrategia_xau,'last_tg_update':st.session_state.last_tg_update}
    gh_save(par, sv2)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Philosopher:ital,wght@0,400;0,700;1,400&display=swap');
* {{ font-family:'Philosopher',serif; }}
h1,h2,h3,h4 {{ font-family:'Cinzel',serif !important; color:{T['primary']} !important; letter-spacing:2px; }}
.stApp {{
    background:
        radial-gradient(circle at 8% 12%, {T['primary']}0d, transparent 32%),
        radial-gradient(circle at 92% 18%, {T['secondary']}0d, transparent 34%),
        radial-gradient(circle at 15% 88%, {T['secondary']}0a, transparent 30%),
        radial-gradient(circle at 90% 85%, {T['primary']}0a, transparent 32%),
        {T['bg']} !important;
    background-attachment:fixed;
}}
.stTabs [data-baseweb="tab"] {{ font-family:'Cinzel',serif; color:{T['primary']}99; font-size:.7em; letter-spacing:1px; }}
.stTabs [aria-selected="true"] {{ color:{T['primary']} !important; border-bottom:2px solid {T['primary']}; }}
.mimi-title {{ font-family:'Cinzel',serif; font-size:clamp(1.8rem,5vw,3rem); font-weight:900; letter-spacing:10px; text-align:center;
    background:linear-gradient(180deg,#E8D5A3 0%,{T['primary']} 50%,{T['secondary']} 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; filter:drop-shadow(0 0 24px {T['primary']}44); margin:8px 0; }}
.mimi-sub {{ text-align:center; font-family:'Philosopher',serif; font-style:italic; color:{T['primary']}77; font-size:.85em; letter-spacing:4px; }}
.greek-orn {{ text-align:center; color:{T['primary']}55; letter-spacing:8px; margin:6px 0; font-size:.9em; }}
.meander-divider {{
    height:10px; margin:4px 0 18px 0; opacity:.35;
    background-image:repeating-linear-gradient(90deg,
        {T['primary']} 0 3px, transparent 3px 6px, transparent 6px 9px, {T['primary']} 9px 12px,
        {T['primary']} 12px 15px, transparent 15px 24px);
    background-size:24px 10px; background-repeat:repeat-x; background-position:center;
    mask-image:linear-gradient(90deg,transparent,black 15%,black 85%,transparent);
    -webkit-mask-image:linear-gradient(90deg,transparent,black 15%,black 85%,transparent);
}}
.ticker-wrap {{ background:linear-gradient(90deg,{T['bg']},{T['card']},{T['bg']});
    border-top:1px solid {T['primary']}44; border-bottom:1px solid {T['primary']}44; overflow:hidden; padding:7px 0; margin:3px 0; }}
.ticker-label {{ font-family:'Cinzel',serif; color:{T['primary']}; font-size:10px; letter-spacing:2px;
    padding:0 14px; display:inline-block; border-right:1px solid {T['primary']}44; vertical-align:middle; }}
.t-s1 {{ display:inline-block; white-space:nowrap; animation:sc1 45s linear infinite; }}
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
.score-bar-bg {{ background:{T['primary']}22; border-radius:8px; height:22px; width:100%; overflow:hidden; }}
.score-bar-fill {{ height:22px; border-radius:8px; text-align:right; padding-right:8px; color:#000;
    font-family:'Cinzel',serif; font-size:.75em; font-weight:700; line-height:22px; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MOTOR DE NOTIFICACIONES AUTOMÁTICAS — App (toast) + Telegram
#  - Señal nueva (score >= 58%, o >=80% en sesión asiática): entrada,
#    SL, TP1, TP2, RR, calidad, razón + 2-3 escenarios
#  - Seguimiento cada 40 min (dentro de la ventana 30-60 pedida) con
#    escenarios y recomendación justificada (mantener/breakeven/parcial/salir)
#  - TP1 (parcial + SL a breakeven), TP2 (cierre total), SL alcanzado
#  - Aviso si el precio está a menos de 35-40 pips del SL
#  - Resumen diario a las 23:30 UTC
#  NOTA: esto corre mientras la pestaña del navegador esté abierta —
#  Streamlit no tiene un proceso en segundo plano 24/7 sin la página abierta.
# ══════════════════════════════════════════════════════════════════
def _ahora_ts():
    return datetime.now(pytz.timezone('America/Mexico_City')).timestamp()

def notificar(msg, icon="🏛️"):
    """Manda a Telegram y muestra un toast dentro de la App."""
    send_tg(msg)
    try:
        corto = msg.split("\n")[0].replace("*","")
        st.toast(corto, icon=icon)
    except Exception:
        pass

def log_alerta(sv, tipo, **kw):
    """Guarda un renglón de bitácora cada vez que se manda una alerta real por Telegram."""
    entry = {'fecha': datetime.now(pytz.timezone('America/Mexico_City')).strftime('%d/%m %H:%M'), 'tipo': tipo}
    entry.update(kw)
    sv.setdefault('alertas_enviadas', []).append(entry)
    sv['alertas_enviadas'] = sv['alertas_enviadas'][-150:]  # no crecer sin límite

def evaluar_y_notificar_par(par, es_par_activo, stf_usar):
    pc = PAIRS[par]
    df_t = get_data(pc['yf_symbol'], stf_usar['trend_interval'], stf_usar['trend_period'])
    df_e = get_data(pc['yf_symbol'], stf_usar['entry_interval'], stf_usar['entry_period'])
    if df_t is None or df_e is None:
        return

    usar_ib_local = pc['usa_ib'] and par == "XAU/USD 🥇" and st.session_state.get('estrategia_xau','') == "Initial Balance Breakout (NY Open)"
    sen = generar_senal(par, df_t, df_e, usar_ib=usar_ib_local)
    dxy_r = get_dxy_returns()
    dxy_ok = False
    if dxy_r is not None:
        dxy_ok = (sen['direccion']==1 and dxy_r<0) or (sen['direccion']==-1 and dxy_r>0)
    rr_ = abs(sen['tp']-sen['precio'])/abs(sen['precio']-sen['sl']) if sen['sl']!=sen['precio'] else 0
    sc_ = calcular_score(sen, dxy_ok, False, rr_)

    if es_par_activo:
        sv = {'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history,
              'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
              'estrategia_xau':st.session_state.estrategia_xau,'last_tg_update':st.session_state.last_tg_update,
              'last_entry_alert':st.session_state.get('last_entry_alert', {})}
    else:
        sv = gh_load(par)

    trades = sv.get('paper_trades', [])
    abiertos = [t for t in trades if t['estado']=='ABIERTO']
    ahora_ts = _ahora_ts()
    hora_mx = datetime.now(pytz.timezone('America/Mexico_City')).hour
    tag = par.split()[0]
    cambios = False

    # 1) ALERTA DE ENTRADA — máximo 1 operación abierta por par
    #    umbral normal 58%, pero en sesión asiática solo si es MUY clara (>=80%)
    umbral = SCORE_MIN_ASIA if es_sesion_asiatica(hora_mx) else SCORE_MIN_ALERTA
    if not abiertos and sen['direccion'] != 0 and sc_['total'] >= umbral:
        last_alert = sv.get('last_entry_alert', {})
        mismo_setup_reciente = last_alert.get('direccion') == sen['direccion'] and (ahora_ts - last_alert.get('ts', 0)) < 3600
        if not mismo_setup_reciente:
            dir_txt = "Compra" if sen['direccion']==1 else "Venta"
            emoji = "🟢" if sen['direccion']==1 else "🔴"
            razon_txt = " · ".join(sen['razon'][:2]) if sen['razon'] else "Setup técnico confirmado"
            escenarios = generar_escenarios(par, sen, df_e)
            esc_txt = "\n".join([f"• {e}" for e in escenarios])
            notificar(f"{emoji} *SEÑAL {tag}*\n"
                    f"Dirección: {dir_txt}\n"
                    f"Precio Entrada: {pf(sen['precio'],par)}\n"
                    f"Stop Loss: {pf(sen['sl'],par)}\n"
                    f"Take Profit 1: {pf(sen['tp'],par)}\n"
                    f"Take Profit 2: {pf(sen['tp2'],par)}\n"
                    f"RR: 1:{rr_:.1f}\n"
                    f"Calidad: {sc_['total']}%\n"
                    f"Razón: {razon_txt}\n\n"
                    f"🔭 *Escenarios:*\n{esc_txt}", icon="🎯")
            log_alerta(sv, 'SEÑAL', dir=dir_txt, precio=sen['precio'], score=sc_['total'])
            sv['last_entry_alert'] = {'direccion': sen['direccion'], 'ts': ahora_ts}
            if es_par_activo: st.session_state.last_entry_alert = sv['last_entry_alert']
            cambios = True

    # 2) SEGUIMIENTO / TP1 / TP2 / SL de posiciones abiertas de este par (máx. 1)
    precio_t = sen['precio']
    for t in abiertos:
        if t.get('par') != par: continue
        dirlong = 'LONG' in t['dir']
        pnl = (precio_t - t['entrada']) * (1 if dirlong else -1) * t['lotes'] * pc['contract_size']
        t.setdefault('tp1', t.get('tp'))
        t.setdefault('tp2', t.get('tp'))
        t.setdefault('tp1_hit', False)
        t.setdefault('sl_warned', False)
        t.setdefault('sl_dist', abs(t['entrada']-t['sl']))
        t.setdefault('last_followup_ts', ahora_ts)

        tocó_tp1 = (dirlong and precio_t >= t['tp1']) or (not dirlong and precio_t <= t['tp1'])
        if tocó_tp1 and not t['tp1_hit']:
            t['tp1_hit'] = True
            t['sl'] = t['entrada']
            notificar(f"🎯 *TP1 ALCANZADO — {tag}*\nParcial ganado ✅\nSL movido a breakeven ({pf(t['entrada'],par)}).\nSigue corriendo hacia TP2: {pf(t['tp2'],par)}", icon="🎯")
            log_alerta(sv, 'TP1', dir=t['dir'], precio=precio_t)
            cambios = True

        tocó_tp2 = (dirlong and precio_t >= t['tp2']) or (not dirlong and precio_t <= t['tp2'])
        tocó_sl  = (dirlong and precio_t <= t['sl']) or (not dirlong and precio_t >= t['sl'])
        if tocó_tp2:
            pnl_final = (t['tp2']-t['entrada'])*t['lotes']*pc['contract_size'] if dirlong else (t['entrada']-t['tp2'])*t['lotes']*pc['contract_size']
            t['estado']='CERRADO'; t['pnl']=round(pnl_final,2); t['resultado']='WIN ✅'
            notificar(f"🏁 *TP2 ALCANZADO — {tag}* 🟢\nOperación cerrada en {pf(t['tp2'],par)}\nP&L total: +${t['pnl']:.2f}", icon="🏁")
            log_alerta(sv, 'TP2', dir=t['dir'], precio=t['tp2'], pnl=t['pnl'])
            cambios = True
        elif tocó_sl:
            pnl_final = (t['sl']-t['entrada'])*t['lotes']*pc['contract_size'] if dirlong else (t['entrada']-t['sl'])*t['lotes']*pc['contract_size']
            t['estado']='CERRADO'; t['pnl']=round(pnl_final,2)
            t['resultado']='WIN ✅' if pnl_final>0 else 'LOSS ❌'
            notificar(f"🛑 *SL ALCANZADO — {tag}*{' (breakeven, parcial ya asegurado)' if t['tp1_hit'] else ''}\nOperación cerrada en {pf(t['sl'],par)}\nP&L: {'+' if pnl_final>0 else ''}${pnl_final:.2f}", icon="🛑")
            log_alerta(sv, 'SL', dir=t['dir'], precio=t['sl'], pnl=t['pnl'])
            cambios = True
        else:
            dist_sl = abs(precio_t - t['sl'])
            dist_sl_pips = a_pips(dist_sl, par)
            if dist_sl_pips < 40 and not t['sl_warned']:
                notificar(f"⚠️ *CERCA DEL SL — {tag}*\nPrecio actual: {pf(precio_t,par)}\nSL: {pf(t['sl'],par)}\nQuedan ~{dist_sl_pips:.0f} pips para el SL.", icon="⚠️")
                t['sl_warned'] = True
                cambios = True

            minutos_desde_followup = (ahora_ts - t['last_followup_ts']) / 60
            if minutos_desde_followup >= 40:
                dist_tp_pips = a_pips(abs(t['tp1']-precio_t), par)
                dist_sl_pips2 = a_pips(dist_sl, par)
                pct = pnl / (t['entrada']*t['lotes']*pc['contract_size']) * 100 if t['entrada'] else 0

                tendencia_actual = sen['tendencia']
                en_contra = (dirlong and tendencia_actual=='BAJISTA') or (not dirlong and tendencia_actual=='ALCISTA')
                patron_en_contra = sen['patron'] and ((dirlong and 'BAJISTA' in sen['patron']) or (not dirlong and 'ALCISTA' in sen['patron']))
                if en_contra:
                    recomendacion = "Salir ahora"; justificacion = "La tendencia se invirtió en contra de la posición."
                elif pnl > 0 and patron_en_contra:
                    recomendacion = "Cerrar parcial"; justificacion = "Vela de rechazo en contra mientras vas en ganancia."
                elif pnl > 0 and not t['tp1_hit']:
                    recomendacion = "Mover SL a breakeven"; justificacion = "Ya vas en ganancia, protege el capital antes de TP1."
                else:
                    recomendacion = "Mantener"; justificacion = "El setup original sigue válido."

                escenarios = generar_escenarios(par, sen, df_e)
                esc_txt = "\n".join([f"• {e}" for e in escenarios])
                notificar(f"📊 *ACTUALIZACIÓN {tag}*\n"
                        f"Precio actual: {pf(precio_t,par)}\n"
                        f"P&L: {'+' if pnl>0 else ''}${pnl:.2f} ({pct:+.2f}%)\n"
                        f"Distancia a TP1: ~{dist_tp_pips:.0f} pips\n"
                        f"Distancia a SL: ~{dist_sl_pips2:.0f} pips\n"
                        f"Recomendación: {recomendacion}\n"
                        f"Justificación: {justificacion}\n\n"
                        f"🔭 *Escenarios:*\n{esc_txt}", icon="📊")
                log_alerta(sv, 'SEGUIMIENTO', dir=t['dir'], precio=precio_t, pnl=round(pnl,2), recomendacion=recomendacion)
                t['last_followup_ts'] = ahora_ts
                cambios = True

    if cambios:
        sv['paper_trades'] = trades
        sv['capital'] = round(1000.0 + sum(x.get('pnl',0) for x in trades if x['estado']=='CERRADO'), 2)
        if es_par_activo:
            st.session_state.paper_trades = trades
            st.session_state.capital = sv['capital']
        gh_save(par, sv)

def generar_bitacora_2dias():
    """Junta todas las alertas reales mandadas por Telegram (señales, TP1/TP2/SL,
    seguimientos) de los últimos 2 días, para los dos pares."""
    mx = pytz.timezone('America/Mexico_City')
    hoy = datetime.now(mx)
    ayer = hoy - timedelta(days=1)
    fechas_validas = (hoy.strftime('%d/%m'), ayer.strftime('%d/%m'))
    emoji_map = {'SEÑAL':'🎯','TP1':'🎯','TP2':'🏁','SL':'🛑','SEGUIMIENTO':'📊'}
    lineas = ["📜 *BITÁCORA — últimos 2 días*"]
    total = 0
    for par_k in PAIRS.keys():
        alertas = gh_load(par_k).get('alertas_enviadas', [])
        recientes = [a for a in alertas if str(a.get('fecha','')).startswith(fechas_validas)]
        tag = par_k.split()[0]
        if not recientes:
            lineas.append(f"\n*{tag}* — sin eventos en este periodo.")
            continue
        lineas.append(f"\n*{tag}* — {len(recientes)} eventos")
        for a in recientes[-20:]:
            e = emoji_map.get(a.get('tipo'), '•')
            extra = f" {a['dir']}" if a.get('dir') else ""
            precio_txt = f" @ {pf(a['precio'],par_k)}" if 'precio' in a else ""
            score_txt = f" · {a['score']}%" if 'score' in a else ""
            pnl_txt = f" · P&L {'+' if a.get('pnl',0)>0 else ''}{a.get('pnl',0):.2f}" if 'pnl' in a else ""
            lineas.append(f"{a.get('fecha','')} {e} {a.get('tipo')}{extra}{precio_txt}{score_txt}{pnl_txt}")
        total += len(recientes)
    lineas.append(f"\nTotal: {total} eventos registrados en 2 días.")
    texto = "\n".join(lineas)
    if len(texto) > 3900:
        texto = texto[:3880] + "\n… (recortado — hay más actividad de la que cabe en un mensaje de Telegram)"
    return texto

def enviar_resumen_diario():
    mx = pytz.timezone('America/Mexico_City')
    hoy_mx_str = datetime.now(mx).strftime('%d/%m')
    lineas = ["📅 *RESUMEN DIARIO — MIMI-AI*"]
    for par_k in PAIRS.keys():
        pc = PAIRS[par_k]
        if par_k == st.session_state.par:
            trades = st.session_state.paper_trades
        else:
            trades = gh_load(par_k).get('paper_trades', [])
        abiertos = [t for t in trades if t['estado']=='ABIERTO']
        cerrados_hoy = [t for t in trades if t['estado']=='CERRADO' and str(t.get('fecha','')).startswith(hoy_mx_str)]
        cerrados_todos = [t for t in trades if t['estado']=='CERRADO']
        ultimos = cerrados_todos[-15:]
        wins = sum(1 for t in ultimos if 'WIN' in t.get('resultado',''))
        wr = round(wins/len(ultimos)*100,1) if ultimos else 0
        try:
            stf_d = STYLE_TF['Day Trading']
            df_td = get_data(pc['yf_symbol'], stf_d['trend_interval'], stf_d['trend_period'])
            tendencia = detectar_tendencia(df_td, pc['ema_trend'][0], pc['ema_trend'][1]) if df_td is not None else 'N/D'
        except Exception:
            tendencia = 'N/D'
        tag = par_k.split()[0]
        lineas.append(f"\n*{tag}*")
        lineas.append(f"Sesgo actual: {tendencia}")
        lineas.append(f"Abiertas: {len(abiertos)}")
        if cerrados_hoy:
            resumen_hoy = " · ".join([f"{x['resultado']} ({'+' if x.get('pnl',0)>0 else ''}{x.get('pnl',0):.2f})" for x in cerrados_hoy])
            lineas.append(f"Cerradas hoy: {resumen_hoy}")
        else:
            lineas.append("Cerradas hoy: ninguna")
        lineas.append(f"Winrate (últimas {len(ultimos)}): {wr}%")
    notificar("\n".join(lineas), icon="📅")

def revisar_resumen_diario():
    ahora_utc = datetime.now(pytz.utc)
    if ahora_utc.hour == 23 and ahora_utc.minute >= 30:
        hoy_str = ahora_utc.strftime('%Y-%m-%d')
        cfg = gh_load_config()
        if cfg.get('last_daily_summary') != hoy_str:
            try:
                enviar_resumen_diario()
            except Exception:
                pass
            cfg['par'] = st.session_state.par
            cfg['last_daily_summary'] = hoy_str
            gh_save_config(cfg)

@fragment_decorator(run_every=60)
def monitor_automatico(par_seleccionado, stf_activo):
    for par_k in PAIRS.keys():
        try:
            stf_usar = stf_activo if par_k == par_seleccionado else STYLE_TF['Day Trading']
            evaluar_y_notificar_par(par_k, par_k == par_seleccionado, stf_usar)
        except Exception:
            pass
    try:
        revisar_resumen_diario()
    except Exception:
        pass

# ── SIDEBAR ───────────────────────────────────────────────────────
# El menú de navegación ya NO vive aquí — se movió a la página
# principal (barra superior con categorías). La sidebar es solo CONFIG.
NAV_GROUPS = [
    ("Principal",    [("senal","Señal"), ("monitor","Monitor")]),
    ("Análisis",     [("estructura","Estructura"), ("multitf","Multi-TF"), ("grafica","Gráfica")]),
    ("Operar",       [("paper","Paper"), ("historial","Historial")]),
    ("Herramientas", [("chat","Chat"), ("alertas","Alertas"), ("backtest","Backtest")]),
]
if 'page' not in st.session_state:
    st.session_state.page = 'senal'
if 'nav_group_open' not in st.session_state:
    st.session_state.nav_group_open = None

with st.sidebar:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};letter-spacing:3px;text-align:center;">⚙ CONFIG</div><div class="meander-divider" style="margin:6px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("---")

    nuevo_par = st.selectbox("💱 Par", list(PAIRS.keys()), index=list(PAIRS.keys()).index(st.session_state.par))
    if nuevo_par != st.session_state.par:
        st.session_state.par = nuevo_par
        cfg_par = gh_load_config(); cfg_par['par'] = nuevo_par; cfg_par['tema'] = st.session_state.tema
        gh_save_config(cfg_par)
        cargar_estado_par(nuevo_par)
        st.rerun()

    nuevo_tema = st.selectbox("🏛️ Estilo visual", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.tema))
    if nuevo_tema != st.session_state.tema:
        st.session_state.tema = nuevo_tema
        cfg_tema = gh_load_config(); cfg_tema['tema'] = nuevo_tema; cfg_tema['par'] = st.session_state.par
        gh_save_config(cfg_tema)
        st.rerun()
    if THEMES[st.session_state.tema].get('palette'):
        swatches = "".join([f'<span style="display:inline-block;width:18px;height:18px;border-radius:3px;background:{c};margin-right:4px;border:1px solid #0004;"></span>' for c in THEMES[st.session_state.tema]['palette']])
        st.markdown(f'<div style="margin:-4px 0 8px 2px;">{swatches}</div>', unsafe_allow_html=True)

    if st.session_state.par == "XAU/USD 🥇":
        nueva_estrat = st.selectbox("📐 Estrategia", ["Trend + Pullback (EMA)", "Initial Balance Breakout (NY Open)"],
                                     index=["Trend + Pullback (EMA)","Initial Balance Breakout (NY Open)"].index(st.session_state.estrategia_xau))
        if nueva_estrat != st.session_state.estrategia_xau:
            st.session_state.estrategia_xau = nueva_estrat
            sv = gh_load(st.session_state.par); sv['estrategia_xau'] = nueva_estrat; gh_save(st.session_state.par, sv)
        st.caption("EMA: pullback a EMA20/50 en dirección de tendencia H4/H1. IB: ruptura del rango de la primera hora de NY.")
    else:
        st.caption("📐 Estrategia: Trend Pullback Continuation — pullback a EMA20 o zona Fib 50–61.8% en dirección de tendencia H4.")

    nuevo_estilo = st.selectbox("⏱️ Timeframes", list(STYLE_TF.keys()),
                                 index=list(STYLE_TF.keys()).index(st.session_state.trade_style))
    st.caption(f"Tendencia / Entrada: {STYLE_TF[nuevo_estilo]['label']}")
    if nuevo_estilo != st.session_state.trade_style:
        st.session_state.trade_style = nuevo_estilo
        sv = gh_load(st.session_state.par); sv['trade_style'] = nuevo_estilo; gh_save(st.session_state.par, sv)

    risk_pct = st.slider("⚠️ Riesgo/trade (%)", 0.25, 1.0, 1.0, 0.25)
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📰 NOTICIAS ALTO IMPACTO</div>', unsafe_allow_html=True)
    noticia_hoy = st.checkbox("Hay noticia de alto impacto ahora (FOMC/NFP/ECB/CPI)", value=False)
    forzar_pese_noticia = False
    if noticia_hoy:
        forzar_pese_noticia = st.checkbox("Forzar señal de todas formas", value=False)
    st.caption("Por defecto se evita operar en noticias de alto impacto salvo que tú lo pidas.")
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">🤖 MONITOREO AUTOMÁTICO</div>', unsafe_allow_html=True)
    st.caption("Ambos pares se revisan cada 60s mientras esta pestaña esté abierta. Umbral de señal: 58% (80% en sesión asiática). Máx. 1 operación abierta por par. Resumen diario a las 23:30 UTC.")
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📱 TELEGRAM COMANDOS</div>', unsafe_allow_html=True)
    st.caption("entré · no · salgo · estado · señal · me quedo · bitácora")
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📚 GUÍA</div>', unsafe_allow_html=True)
    for titulo, texto in [
        ("Tendencia (EMA)","Precio sobre ambas EMAs de tendencia = alcista. Bajo ambas = bajista. Si no, rango — no se opera."),
        ("Pullback","Retroceso del precio hacia la EMA de entrada antes de continuar la tendencia — punto de entrada de mejor R:R."),
        ("Fibonacci 50–61.8%","Para EUR/USD: zona de retroceso típica antes de que el precio retome la tendencia."),
        ("Pin Bar / Engulfing","Velas de rechazo que confirman que el pullback terminó y el precio retoma dirección."),
        ("Initial Balance Breakout","Rango de la primera hora de NY (08–09 hora MX). Ruptura de ese rango = señal de continuación."),
        ("Score 0-100%","40% calidad del setup, 30% probabilidad direccional, 15% noticias, 15% gestión de riesgo."),
        ("DXY","Índice del dólar. Correlación inversa con EUR/USD — se usa como filtro de probabilidad."),
    ]:
        with st.expander(titulo): st.write(texto)

T = THEMES.get(st.session_state.tema, THEMES["Mármol Griego"])
PAR = st.session_state.par
PC  = PAIRS[PAR]
CONTRACT = PC['contract_size']
STF = STYLE_TF[st.session_state.trade_style]

# ── TOP NAV — categorías arriba de todo, ancladas mientras deslizas ──
NAV_GROUP_OF = {}
for _g, _items in NAV_GROUPS:
    for _k, _label in _items:
        NAV_GROUP_OF[_k] = _g

with st.container(key="mimi_topnav"):
    st.markdown(f"""
    <style>
    .st-key-mimi_topnav {{
        position:sticky; top:0; z-index:9999;
        background:linear-gradient(180deg,{T['bg']}f2,{T['bg']}d8);
        backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
        border-bottom:1px solid {T['primary']}33;
        padding:12px 6px 10px 6px; margin:-1rem -1rem 26px -1rem;
        position:relative;
    }}
    .st-key-mimi_topnav::after {{
        content:''; position:absolute; left:0; right:0; bottom:-8px; height:8px;
        background-image:repeating-linear-gradient(90deg,
            {T['primary']} 0 3px, transparent 3px 6px, transparent 6px 9px, {T['primary']} 9px 12px,
            {T['primary']} 12px 15px, transparent 15px 24px);
        background-size:24px 8px; background-repeat:repeat-x; background-position:center;
        opacity:.4;
    }}
    .st-key-mimi_topnav button {{
        background:transparent !important; border:none !important;
        border-bottom:2px solid transparent !important; border-radius:0 !important;
        color:{T['primary']}80 !important;
        font-family:'Cinzel',serif !important; letter-spacing:2.5px !important;
        font-size:.82em !important; text-transform:uppercase !important;
        padding:7px 4px !important; box-shadow:none !important;
        transition:all .25s ease !important; width:100% !important;
    }}
    .st-key-mimi_topnav button:hover {{
        color:{T['primary']} !important; border-bottom:2px solid {T['primary']}77 !important;
    }}
    .st-key-mimi_topnav button[kind="primary"] {{
        background:transparent !important; color:{T['primary']} !important;
        border-bottom:2px solid {T['primary']} !important; font-weight:700 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    grupo_abierto = st.session_state.nav_group_open
    if grupo_abierto is None:
        cols_nav = st.columns(len(NAV_GROUPS))
        for i, (g, items) in enumerate(NAV_GROUPS):
            with cols_nav[i]:
                es_grupo_activo = NAV_GROUP_OF.get(st.session_state.page) == g
                if st.button(g, key=f"navcat_{g}", use_container_width=True,
                             type=("primary" if es_grupo_activo else "secondary")):
                    st.session_state.nav_group_open = g
                    st.rerun()
    else:
        items_abiertos = dict(NAV_GROUPS)[grupo_abierto]
        cols_nav = st.columns([0.6] + [1]*len(items_abiertos))
        with cols_nav[0]:
            if st.button("‹ Menú", key="navcat_back", use_container_width=True):
                st.session_state.nav_group_open = None
                st.rerun()
        for i, (k, label) in enumerate(items_abiertos):
            with cols_nav[i+1]:
                activo = st.session_state.page == k
                if st.button(label, key=f"navitem_{k}", use_container_width=True,
                             type=("primary" if activo else "secondary")):
                    st.session_state.page = k
                    st.session_state.nav_group_open = None
                    st.rerun()

# ── NOTICIAS — ticker con lo más reciente de oro y EUR/USD (yfinance, gratis) ──
@st.cache_data(ttl=600)
def obtener_noticias():
    titulares = []
    for simbolo in ["GC=F", "EURUSD=X"]:
        try:
            items = yf.Ticker(simbolo).news or []
            for it in items[:4]:
                # yfinance ha cambiado el formato de 'news' varias veces —
                # se intenta leer de las dos estructuras conocidas.
                contenido = it.get('content', it)
                titulo = contenido.get('title') or it.get('title')
                if titulo:
                    titulares.append(titulo.strip())
        except Exception:
            continue
    vistos, unicos = set(), []
    for t in titulares:
        if t not in vistos:
            vistos.add(t); unicos.append(t)
    return unicos[:8]

def render_ticker_noticias():
    noticias = obtener_noticias()
    if noticias:
        texto = "   📰   ".join(noticias)
    else:
        texto = "No hay noticias relevantes de oro o EUR/USD en este momento."
    contenido = (f"  📰 NOTICIAS · ORO & EUR/USD  ·   {texto}   ")*2
    st.markdown(f"""
    <div class="ticker-wrap">
      <span class="ticker-label">NEWS</span>
      <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
        <div class="t-s1" style="color:{T['primary']}cc;font-family:'Philosopher',serif;font-size:.82em;">{contenido}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_data(yf_symbol, interval, period):
    df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
    df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df) >= 60: return df
    df2 = yf.download(yf_symbol, period="2y", interval="1d", progress=False)
    df2.columns = [c[0] if isinstance(c,tuple) else c for c in df2.columns]
    df2.dropna(inplace=True)
    return df2 if len(df2) >= 60 else None

with st.spinner("🏛️ El Oráculo consulta los astros..."):
    df_trend = get_data(PC['yf_symbol'], STF['trend_interval'], STF['trend_period'])
    df_entry = get_data(PC['yf_symbol'], STF['entry_interval'], STF['entry_period'])

if df_trend is None or df_entry is None:
    st.warning("⚠️ Mercado cerrado o sin datos — intenta más tarde."); st.stop()

usar_ib = PC['usa_ib'] and st.session_state.get('estrategia_xau','') == "Initial Balance Breakout (NY Open)"
senal = generar_senal(PAR, df_trend, df_entry, usar_ib=usar_ib)

dxy_ret = get_dxy_returns()
dxy_corr_ok = False
if dxy_ret is not None:
    if PAR == "EUR/USD 💶":
        dxy_corr_ok = (senal['direccion']==1 and dxy_ret<0) or (senal['direccion']==-1 and dxy_ret>0)
    else:
        dxy_corr_ok = (senal['direccion']==1 and dxy_ret<0) or (senal['direccion']==-1 and dxy_ret>0)

noticia_bloqueando = noticia_hoy and not forzar_pese_noticia
if noticia_bloqueando:
    senal['direccion'] = 0
    senal['razon'] = ["⛔ Señal bloqueada — noticia de alto impacto en curso"]

rr_actual = 0
if senal['sl'] != senal['precio']:
    rr_actual = abs(senal['tp']-senal['precio']) / abs(senal['precio']-senal['sl'])

score = calcular_score(senal, dxy_corr_ok, noticia_bloqueando, rr_actual)

precio = senal['precio']; pred = senal['direccion']
ET = {1:"LONG — ASCENSO", 0:"SIN SETUP CLARO — ESPERA", -1:"SHORT — DESCENSO"}
mx_tz = pytz.timezone('America/Mexico_City'); ahora = datetime.now(mx_tz); h = ahora.hour

process_tg_updates(PAR, senal, score, risk_pct, CONTRACT)

# ── MONITOR AUTOMÁTICO — entrada, seguimiento, TP1/TP2/SL, cerca-de-SL ──
# corre también aquí en el rerun completo (por si el fragmento aún no dispara)
try:
    evaluar_y_notificar_par(PAR, True, STF)
except Exception:
    pass

if not st.session_state.signal_history or st.session_state.signal_history[-1].get('precio') != precio:
    st.session_state.signal_history.append({
        'id':len(st.session_state.signal_history)+1,'fecha':ahora.strftime('%d/%m %H:%M'),'par':PAR,
        'estilo':st.session_state.trade_style,'direccion':ET.get(pred),'score':score['total'],
        'precio':precio,'sl':senal['sl'],'tp':senal['tp'],'tp2':senal['tp2'],'tendencia':senal['tendencia'],'resultado':'PENDIENTE'})

sv2 = {'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history[-50:],
       'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
       'estrategia_xau':st.session_state.estrategia_xau,'last_tg_update':st.session_state.last_tg_update,
       'last_entry_alert':st.session_state.get('last_entry_alert', {})}
gh_save(PAR, sv2)

# ── BANNER ────────────────────────────────────────────────────────
bc = '#4CAF82' if pred==1 else '#C0392B' if pred==-1 else T['primary']
b1 = (f"  {PAR}  ·  {ET.get(pred)}  ·  SCORE: {score['total']}%  ·  {pf(precio,PAR)}  ·  SL: {pf(senal['sl'],PAR)}  ·  TP: {pf(senal['tp'],PAR)}  ·  Tendencia: {senal['tendencia']}  ")*2
st.markdown(f"""
<div class="ticker-wrap">
  <span class="ticker-label">ORACLE</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s1" style="color:{bc};font-family:'Philosopher',serif;font-size:.85em;">{b1}</div>
  </div>
</div>
<div class="greek-orn">── ✦ ── ✦ ── ✦ ──</div>
""", unsafe_allow_html=True)

# ── PRECIO EN VIVO — se actualiza solo cada 3s, SIN recargar la app ──
precios_en_vivo(PAR)
monitor_automatico(PAR, STF)
render_ticker_noticias()

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("📈 Tendencia", senal['tendencia'])
c2.metric("🎯 Señal","LONG" if pred==1 else "SHORT" if pred==-1 else "ESPERA")
c3.metric("⭐ Score", f"{score['total']}%", score['categoria'].split(' ')[0])
c4.metric("📐 R:R", f"1:{rr_actual:.2f}" if rr_actual else "—")
c5.metric("💰 Capital", f"${st.session_state.capital:,.2f}")
c6.metric("⏱️ Timeframes", STF['label'])
st.markdown('<div class="greek-orn">── ✦ ──</div><div class="meander-divider"></div>', unsafe_allow_html=True)

# ── CONTENIDO — controlado por la barra de navegación superior ─────

# ── TAB 1: SEÑAL ──────────────────────────────────────────────────
if st.session_state.page == 'senal':
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="card"><div class="card-title">SEÑAL DEL ORÁCULO</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{"sig-long" if pred==1 else "sig-short" if pred==-1 else "sig-neu"}">{ET.get(pred)}</div>', unsafe_allow_html=True)
        st.markdown(f"**Par:** {PAR}  |  **Precio:** {pf(precio,PAR)}  |  **Timeframes:** {STF['label']}")
        if senal['razon']:
            st.markdown("**Justificación:**")
            for rz in senal['razon']: st.markdown(f"- {rz}")
        else:
            st.markdown("Sin confluencia de tendencia + pullback + confirmación todavía. El estoico espera.")
        if pred != 0:
            st.markdown(f"**Stop Loss 🔴:** {pf(senal['sl'],PAR)}")
            st.markdown(f"**Take Profit 1 🟢:** {pf(senal['tp'],PAR)}  (R:R 1:{rr_actual:.2f})")
            st.markdown(f"**Take Profit 2 🟢🟢:** {pf(senal['tp2'],PAR)}")
        lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, senal['sl'] if senal['sl']!=precio else precio-senal['atr'], CONTRACT)
        st.markdown(f"**Lotes sugeridos:** {lot}  |  **Riesgo:** ${risg:.2f} ({risk_pct}%)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Barra de score
        sc_color = '#4CAF82' if score['total']>=60 else '#C8A96E' if score['total']>=45 else '#C0392B'
        st.markdown('<div class="card"><div class="card-title">CALIFICACIÓN DE LA OPERACIÓN</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score['total']}%;background:{sc_color};">{score['total']}%</div></div>
        <div style="margin-top:8px;color:{T['primary']};font-family:Cinzel,serif;font-size:.85em;">{score['categoria']}</div>
        <div style="margin-top:6px;color:{T['primary']}99;font-size:.85em;">
        Setup técnico: {score['setup']}/10 · Probabilidad: {score['prob']}/10 · Noticias: {score['noticias']}/10 · Gestión de riesgo: {score['riesgo']}/10
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        c_si, c_no = st.columns(2)
        if c_si.button("✅ ENTRO", use_container_width=True):
            ot = [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
            if not ot and pred != 0:
                lot2, risg2 = calc_pos(st.session_state.capital, risk_pct, precio, senal['sl'], CONTRACT)
                st.session_state.paper_trades.append({
                    'id':len(st.session_state.paper_trades)+1,'par':PAR,
                    'dir':'LONG 📈' if pred==1 else 'SHORT 📉','entrada':precio,'sl':senal['sl'],
                    'tp':senal['tp'],'tp1':senal['tp'],'tp2':senal['tp2'],
                    'tp1_hit':False,'sl_warned':False,'sl_dist':abs(precio-senal['sl']),'last_followup_ts':_ahora_ts(),
                    'lotes':lot2,'riesgo':risg2,'estado':'ABIERTO','fecha':ahora.strftime('%d/%m %H:%M'),
                    'resultado':'PENDIENTE','pnl':0,'score':score['total']})
                send_tg(f"🟢 *SEÑAL {PAR.split()[0]}*\nDirección: {'Compra' if pred==1 else 'Venta'}\nPrecio de entrada: {pf(precio,PAR)}\nStop Loss: {pf(senal['sl'],PAR)}\nTake Profit 1: {pf(senal['tp'],PAR)}\nTake Profit 2: {pf(senal['tp2'],PAR)}\nRatio RR: 1:{rr_actual:.1f}\nCalidad: {score['total']}%\nRazón: {' · '.join(senal['razon'][:2]) if senal['razon'] else 'Entrada manual'}")
                gh_save(PAR, {**sv2,'paper_trades':st.session_state.paper_trades}); st.success("Trade registrado ✅"); st.rerun()
            elif ot: st.warning("Ya tienes un trade abierto en este par.")
            else: st.warning("No hay setup claro ahorita.")
        if c_no.button("❌ NO ENTRO", use_container_width=True):
            send_tg(f"🏛️ Señal rechazada — {PAR} @ {pf(precio,PAR)}")
            st.info("Rechazada")

    with cb:
        st.markdown('<div class="card"><div class="card-title">VENTANAS · HORA MX</div>', unsafe_allow_html=True)
        st.markdown(f"**Hora:** {ahora.strftime('%H:%M')}")
        for n, ini, fin, cal in [("London Open",3,5,"Alta"),("NY Open (IB)",8,9,"Máxima ✦"),("London+NY",8,11,"Máxima"),("NY Cierre",15,17,"Baja")]:
            st.markdown(f"{'🟢' if ini<=h<fin else '⚫'} **{ini:02d}–{fin:02d}** {n} [{cal}]")
        st.markdown('</div>', unsafe_allow_html=True)

        if PC['usa_fib'] and senal['fib']:
            st.markdown('<div class="card"><div class="card-title">ZONA FIBONACCI (H4)</div>', unsafe_allow_html=True)
            st.markdown(f"High: {pf(senal['fib']['high'],PAR)}  |  Low: {pf(senal['fib']['low'],PAR)}")
            st.markdown(f"Fib 50%: {pf(senal['fib']['fib_50'],PAR)}  |  Fib 61.8%: {pf(senal['fib']['fib_618'],PAR)}")
            st.markdown('</div>', unsafe_allow_html=True)

        if usar_ib and senal.get('ib'):
            st.markdown('<div class="card"><div class="card-title">INITIAL BALANCE (08–09 MX)</div>', unsafe_allow_html=True)
            ibd = senal['ib']
            st.markdown(f"Rango: {pf(ibd['ib_low'],PAR)} — {pf(ibd['ib_high'],PAR)}")
            st.markdown(f"Ruptura: {'🟢 LONG' if ibd.get('direccion')==1 else '🔴 SHORT' if ibd.get('direccion')==-1 else '➡️ Dentro del rango'}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">DXY · CORRELACIÓN</div>', unsafe_allow_html=True)
        if dxy_ret is not None:
            st.markdown(f"Retorno DXY (1d): {dxy_ret*100:+.2f}%")
            st.markdown(f"Confluencia con la señal: {'✅ A favor' if dxy_corr_ok else '⚠️ Sin confirmar'}")
        else:
            st.markdown("Sin datos de DXY disponibles ahorita.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: ESTRUCTURA ──────────────────────────────────────────────
if st.session_state.page == 'estructura':
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};font-size:.85em;letter-spacing:3px;margin-bottom:12px;">ESTRUCTURA — {PAR}</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="card"><div class="card-title">TENDENCIA</div>', unsafe_allow_html=True)
        color = '#4CAF82' if senal['tendencia']=='ALCISTA' else '#C0392B' if senal['tendencia']=='BAJISTA' else T['primary']
        st.markdown(f'<span style="color:{color};font-family:Cinzel,serif;font-size:1.2em;letter-spacing:2px;">{senal["tendencia"]}</span>', unsafe_allow_html=True)
        st.markdown(f"EMA{PC['ema_trend'][0]} / EMA{PC['ema_trend'][1]} en timeframe de tendencia ({STF['trend_interval']})")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">PULLBACK</div>', unsafe_allow_html=True)
        st.markdown(f"¿En zona de pullback (EMA{PC['ema_entry']})? {'✅ Sí' if senal['pullback'] else '❌ No'}")
        st.markdown(f"Nivel EMA{PC['ema_entry']}: {pf(senal['ema_ref'],PAR)}")
        if PC['usa_fib']:
            st.markdown(f"¿En zona Fib? Long: {'✅' if senal['en_fib_long'] else '❌'} · Short: {'✅' if senal['en_fib_short'] else '❌'}")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="card"><div class="card-title">CONFIRMACIÓN</div>', unsafe_allow_html=True)
        st.markdown(f"Patrón de vela: **{senal['patron'] or 'Ninguno detectado'}**")
        st.markdown(f"Ruptura de microestructura: **{senal['estructura'] or 'Ninguna'}**")
        st.markdown(f"ATR actual: {pf(senal['atr'],PAR)}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">GESTIÓN DE RIESGO</div>', unsafe_allow_html=True)
        st.markdown(f"SL: {pf(senal['sl'],PAR)}  |  TP: {pf(senal['tp'],PAR)}")
        st.markdown(f"R:R actual: 1:{rr_actual:.2f} (mínimo aceptable 1:2)")
        st.markdown(f"Riesgo configurado: {risk_pct}% del capital por operación")
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: MULTI-TF (confluencia genérica, independiente de la estrategia principal) ──
if st.session_state.page == 'multitf':
    @st.cache_data(ttl=600)
    def mtf_conf(yf_symbol):
        sigs = {}
        for name, iv, per in [("D1","1d","2y"),("H4","4h","180d"),("H1","1h","60d"),("M15","15m","10d")]:
            try:
                d = yf.download(yf_symbol, period=per, interval=iv, progress=False)
                d.columns = [c[0] if isinstance(c,tuple) else c for c in d.columns]
                d.dropna(inplace=True)
                if len(d) < 30: continue
                e20 = ta.trend.ema_indicator(d['Close'], window=20); e50 = ta.trend.ema_indicator(d['Close'], window=50)
                rsi = ta.momentum.rsi(d['Close'], window=14)
                p, em20, em50, r = float(d['Close'].iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1]), float(rsi.iloc[-1])
                score_ = sum([p>em20, p>em50, em20>em50, r>50])
                sigs[name] = {'score':score_,'bias':'LONG' if score_>=3 else 'SHORT' if score_<=1 else 'NEUTRAL','rsi':r,'precio':p}
            except: pass
        if not sigs: return sigs, 'NEUTRAL', 50
        total = sum(s['score'] for s in sigs.values())
        pct = total / (len(sigs)*4) * 100
        return sigs, 'LONG' if pct>=60 else 'SHORT' if pct<=40 else 'NEUTRAL', pct

    with st.spinner("Analizando timeframes..."):
        mtf_s, mtf_b, mtf_p = mtf_conf(PC['yf_symbol'])
    m1,m2,m3 = st.columns(3)
    m1.metric("Bias",f"{'📈 LONG' if mtf_b=='LONG' else '📉 SHORT' if mtf_b=='SHORT' else '➡️ NEUTRAL'}")
    m2.metric("Confluencia",f"{mtf_p:.0f}%"); m3.metric("TFs analizados",str(len(mtf_s)))
    for tfn,data in mtf_s.items():
        bc2 = "🟢" if data['bias']=='LONG' else "🔴" if data['bias']=='SHORT' else "🟡"
        st.markdown(f"{bc2} **{tfn}** — {data['bias']} | RSI:{data['rsi']:.1f} | {pf(data['precio'],PAR)}")

# ── TAB 4: PAPER TRADING ──────────────────────────────────────────
if st.session_state.page == 'paper':
    pm1,pm2,pm3 = st.columns(3)
    pm1.metric("💰 Capital", f"${st.session_state.capital:,.2f}")
    ct_c = [t for t in st.session_state.paper_trades if t['estado']=='CERRADO']
    w_p  = sum(1 for t in ct_c if 'WIN' in t.get('resultado',''))
    pm2.metric("Win Rate", f"{w_p/len(ct_c)*100:.0f}%" if ct_c else "—")
    pm3.metric("Trades", f"{len(ct_c)} cerrados")

    ot = [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot:
        t = ot[0]; pnl = (precio-t['entrada'])*(1 if 'LONG' in t['dir'] else -1)*t['lotes']*CONTRACT
        est = "🚨 SL — SAL YA" if (('LONG' in t['dir'] and precio<=t['sl']) or ('SHORT' in t['dir'] and precio>=t['sl'])) else "🎯 TP ALCANZADO" if (('LONG' in t['dir'] and precio>=t['tp']) or ('SHORT' in t['dir'] and precio<=t['tp'])) else "🟢 MANTÉN" if pnl>0 else "🔴 PRECAUCIÓN"
        st.markdown(f'<div class="card"><div class="card-title">POSICIÓN ABIERTA — {PAR}</div>', unsafe_allow_html=True)
        st.markdown(f"**{t['dir']}** @ {pf(t['entrada'],PAR)} | Actual: {pf(precio,PAR)} | **{est}**")
        st.markdown(f"P&L: **${pnl:.2f}** | SL: {pf(t['sl'],PAR)} | TP: {pf(t['tp'],PAR)} | Lotes: {t['lotes']}")
        if st.button("Cerrar manualmente"):
            t['estado']='CERRADO'; t['pnl']=round(pnl,2); t['resultado']='WIN ✅' if pnl>0 else 'LOSS ❌'
            send_tg(f"✋ *ORDEN CERRADA MANUALMENTE — {PAR.split()[0]}*\n{t['dir']} @ {pf(t['entrada'],PAR)}\nCierre: {pf(precio,PAR)}\nP&L: {'+' if pnl>0 else ''}${pnl:.2f}")
            gh_save(PAR, {**sv2,'paper_trades':st.session_state.paper_trades,'capital':st.session_state.capital}); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Sin trade abierto en este par.")
    if ct_c:
        st.dataframe(pd.DataFrame(ct_c)[['fecha','dir','entrada','sl','tp','lotes','pnl','resultado']], use_container_width=True)
    if st.button("🗑️ Reiniciar paper trading"):
        st.session_state.paper_trades=[]; st.session_state.capital=1000.0
        gh_save(PAR, {**sv2,'paper_trades':[],'capital':1000.0}); st.rerun()

# ── TAB 5: GRÁFICA ────────────────────────────────────────────────
if st.session_state.page == 'grafica':
    dp = df_entry.tail(120).copy()
    dp['EMA_e'] = ta.trend.ema_indicator(dp['Close'], window=PC['ema_entry'])
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25])
    fig.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
        increasing_line_color='#4CAF82',decreasing_line_color='#C0392B',name=PAR),row=1,col=1)
    fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_e'],line=dict(color='#C8A96E',width=1),name=f"EMA{PC['ema_entry']}"),row=1,col=1)
    if PC['usa_fib'] and senal['fib']:
        fig.add_hline(y=senal['fib']['fib_50'],line_color='rgba(200,169,110,0.5)',line_dash='dot',row=1,col=1)
        fig.add_hline(y=senal['fib']['fib_618'],line_color='rgba(200,169,110,0.5)',line_dash='dot',row=1,col=1)
    if usar_ib and senal.get('ib'):
        fig.add_hline(y=senal['ib']['ib_high'],line_color='rgba(76,175,130,0.5)',line_dash='dash',row=1,col=1)
        fig.add_hline(y=senal['ib']['ib_low'],line_color='rgba(192,57,43,0.5)',line_dash='dash',row=1,col=1)
    rsi_g = ta.momentum.rsi(dp['Close'], window=14)
    fig.add_trace(go.Scatter(x=dp.index,y=rsi_g,line=dict(color=T['primary'],width=1.5),name="RSI"),row=2,col=1)
    fig.add_hline(y=70,line_color='#C0392B',line_dash='dot',row=2,col=1)
    fig.add_hline(y=30,line_color='#4CAF82',line_dash='dot',row=2,col=1)
    fig.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
        xaxis_rangeslider_visible=False,height=500,margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(bgcolor='#000',bordercolor='#222',orientation='h'))
    fig.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fig.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Timeframe de entrada: {STF['entry_interval']} · EMA{PC['ema_entry']} en dorado" + (" · líneas Fib punteadas" if PC['usa_fib'] else "") + (" · rango IB en verde/rojo" if usar_ib else ""))

# ── TAB 6: CHAT ───────────────────────────────────────────────────
if st.session_state.page == 'chat':
    st.caption("💡 Escribe una pregunta o sube una captura de gráfico (XAUUSD/EURUSD) para un análisis híbrido.")

    with st.expander("🔧 Debug Gemini (temporal — bórralo cuando ya funcione)"):
        if GEMINI_API_KEY:
            st.success(f"GEMINI_API_KEY detectada ✅ — longitud: {len(GEMINI_API_KEY)} caracteres, empieza con: `{GEMINI_API_KEY[:6]}...`")
        else:
            st.error("GEMINI_API_KEY NO detectada ❌ — revisa Settings → Secrets en Streamlit Cloud (nombre exacto, sin espacios, con comillas).")
        st.caption(f"Modelo configurado (GEMINI_MODEL): `{GEMINI_MODEL}`")
        cliente_debug = get_gemini_client(GEMINI_API_KEY)
        st.write("Cliente Gemini inicializado:", "✅ Sí" if cliente_debug else "❌ No")
        if GEMINI_API_KEY and not cliente_debug:
            st.warning("La key llegó pero el cliente no se pudo crear — revisa que el paquete `google-genai` esté en requirements.txt y que la key sea válida.")

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg['role']=='user' else "assistant"):
            if msg.get('image'):
                st.image(msg['image'], width=350)
            if msg.get('content'):
                st.markdown(msg['content'])

    # ── SUBIR CAPTURA DE GRÁFICO — 2 formas ─────────────────────────
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        img_file = st.file_uploader("📸 Subir archivo (o arrástralo aquí)", type=['png','jpg','jpeg'], key="chat_img_uploader")
        st.caption("También puedes arrastrar la imagen directo aquí — no hace falta ni siquiera abrir el explorador de archivos.")
    with col_up2:
        st.markdown("**📋 O pega la imagen (Ctrl+V)**")
        with st.container(key="paste_bridge_wrap"):
            st.text_area("Pegar imagen", key="paste_bridge", height=68,
                         placeholder="Copia la captura (Win+Shift+S, Cmd+Shift+4, etc.) y da clic aquí + Ctrl+V",
                         label_visibility="collapsed")
        st.caption("Experimental: si tu navegador no lo detecta, usa el uploader de la izquierda.")
        st.html("""
        <style>
        .st-key-paste_bridge_wrap textarea { cursor: text; }
        </style>
        <script>
        (function(){
            function attach(){
                var wrap = document.querySelector('.st-key-paste_bridge_wrap');
                if(!wrap) return;
                var ta = wrap.querySelector('textarea');
                if(!ta || ta.dataset.pasteBound) return;
                ta.dataset.pasteBound = "1";
                ta.addEventListener('paste', function(e){
                    var items = (e.clipboardData || {}).items || [];
                    for (var i=0; i<items.length; i++){
                        if (items[i].type.indexOf('image') === 0){
                            var file = items[i].getAsFile();
                            var reader = new FileReader();
                            reader.onload = function(evt){
                                var dataUrl = evt.target.result;
                                var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                                setter.call(ta, dataUrl);
                                ta.dispatchEvent(new Event('input', {bubbles:true}));
                                setTimeout(function(){ ta.blur(); }, 50);
                            };
                            reader.readAsDataURL(file);
                            e.preventDefault();
                            break;
                        }
                    }
                });
            }
            var tries = 0;
            var iv = setInterval(function(){ attach(); tries++; if (tries > 25) clearInterval(iv); }, 300);
        })();
        </script>
        """, unsafe_allow_javascript=True)

    pegado = st.session_state.get('paste_bridge', '')
    if pegado and pegado.startswith('data:image'):
        paste_sig = str(len(pegado))
        if st.session_state.get('last_paste_sig') != paste_sig:
            st.session_state.last_paste_sig = paste_sig
            try:
                header, b64data = pegado.split(',', 1)
                mime_pegado = header.split(':')[1].split(';')[0]
                img_bytes_pegado = base64.b64decode(b64data)
                st.session_state['_pegado_pendiente'] = (img_bytes_pegado, mime_pegado)
            except Exception:
                pass

    img_file_bytes = None
    img_file_mime = None
    origen_nueva_img = False

    if img_file is not None:
        file_sig = f"{img_file.name}_{img_file.size}"
        if st.session_state.get('last_chat_image_sig') != file_sig:
            st.session_state.last_chat_image_sig = file_sig
            img_file_bytes = img_file.getvalue()
            img_file_mime = img_file.type or ("image/png" if img_file.name.lower().endswith("png") else "image/jpeg")
            origen_nueva_img = True
    elif st.session_state.get('_pegado_pendiente'):
        img_file_bytes, img_file_mime = st.session_state.pop('_pegado_pendiente')
        origen_nueva_img = True

    if origen_nueva_img and img_file_bytes:
        img_bytes = img_file_bytes
        mime = img_file_mime
        nombre_img = img_file.name if img_file is not None else "captura pegada"
        if True:
            st.session_state.chat_history.append({'role':'user','content':f"📸 {nombre_img}", 'image':img_bytes})

            with st.spinner("🏛️ Analizando la captura con Gemini y cruzándola con la estrategia..."):
                contexto = (f"Contexto de la app en este momento — Par activo: {PAR}. "
                            f"Precio actual: {pf(precio,PAR)}. Tendencia detectada por la estrategia (EMA{PC['ema_trend'][0]}/EMA{PC['ema_trend'][1]}): {senal['tendencia']}. "
                            f"Señal actual del sistema: {ET.get(pred)}. Score del setup actual: {score['total']}% ({score['categoria']}). "
                            f"Toma en cuenta esta información al analizar la imagen y compárala contra lo que ves en la captura.")
                analisis_gemini = analizar_imagen_grafica(img_bytes, mime, contexto)

            # ── Combinar el veredicto de Gemini con la señal del sistema ──
            gl = analisis_gemini.lower()
            dice_comprar = 'decisión: compra' in gl or ('compra' in gl and 'vende' not in gl)
            dice_vender  = 'decisión: vende'   in gl or ('vende'  in gl and 'compra' not in gl)
            concuerda = None
            if pred == 1 and dice_comprar and not dice_vender: concuerda = True
            elif pred == -1 and dice_vender and not dice_comprar: concuerda = True
            elif (pred == 1 and dice_vender) or (pred == -1 and dice_comprar): concuerda = False

            combinacion = [
                f"**Par activo en la app:** {PAR}",
                f"**Tendencia (EMA{PC['ema_trend'][0]}/EMA{PC['ema_trend'][1]}):** {senal['tendencia']}",
                f"**Señal actual del sistema:** {ET.get(pred)} — Score {score['total']}% ({score['categoria']})",
            ]
            if senal['pullback'] or senal.get('en_fib_long') or senal.get('en_fib_short'):
                combinacion.append("**Pullback:** el precio está en zona de entrada válida según la estrategia")
            if senal['patron']:
                combinacion.append(f"**Patrón detectado por el sistema:** {senal['patron']}")
            if concuerda is True:
                combinacion.append("✅ El análisis de Gemini **coincide** con la señal del sistema.")
            elif concuerda is False:
                combinacion.append("⚠️ El análisis de Gemini **no coincide** con la señal del sistema — revisa con cuidado antes de operar.")
            else:
                combinacion.append("El sistema aún no confirma dirección — usa el nivel de activación de abajo, no te quedes solo esperando sin condición.")

            # Decisión final: siempre accionable, con nivel exacto si no hay setup confirmado todavía
            if concuerda is False:
                recomendacion_final = "Sin operación por ahora — la imagen y el sistema no coinciden, espera más confirmación"
            elif pred == 1:
                recomendacion_final = f"Compra ahora en {pf(precio,PAR)} — SL {pf(senal['sl'],PAR)} / TP1 {pf(senal['tp'],PAR)}"
            elif pred == -1:
                recomendacion_final = f"Vende ahora en {pf(precio,PAR)} — SL {pf(senal['sl'],PAR)} / TP1 {pf(senal['tp'],PAR)}"
            else:
                escenarios_niv = generar_escenarios(PAR, senal, df_entry)
                resistencia_niv = float(df_entry['High'].iloc[-20:].max())
                soporte_niv = float(df_entry['Low'].iloc[-20:].min())
                recomendacion_final = (f"Espera a que rompa {pf(resistencia_niv,PAR)} para comprar, "
                                        f"o a que rompa {pf(soporte_niv,PAR)} para vender — ahí es donde se define.")

            resp_final = (
                f"### 🖼️ Análisis de la captura (Gemini)\n{analisis_gemini}\n\n"
                f"### 📊 Combinación con la estrategia actual\n" + "\n".join(f"- {c}" for c in combinacion) + "\n\n"
                f"### 🎯 Decisión: **{recomendacion_final}**\n"
                f"Calidad aproximada: **{score['total']}%** ({score['categoria']})"
            )
            st.session_state.chat_history.append({'role':'mimi','content':resp_final})
            st.rerun()

    # ── CHAT DE TEXTO (comandos normales, sin cambios) ─────────────
    uin = st.chat_input("Consulta al Oráculo, o sube una captura arriba...")
    if uin:
        t = uin.lower()
        if 'score' in t or 'calificaci' in t:
            resp = f"Score actual: {score['total']}% — {score['categoria']}\nSetup {score['setup']}/10 · Prob {score['prob']}/10 · Noticias {score['noticias']}/10 · Riesgo {score['riesgo']}/10"
        elif 'tendencia' in t:
            resp = f"Tendencia: {senal['tendencia']} (EMA{PC['ema_trend'][0]}/EMA{PC['ema_trend'][1]})"
        elif 'sl' in t or 'stop' in t:
            resp = f"SL: {pf(senal['sl'],PAR)}"
        elif 'tp' in t or 'objetivo' in t:
            resp = f"TP: {pf(senal['tp'],PAR)} | R:R 1:{rr_actual:.2f}"
        else:
            resp = f"Señal: {ET.get(pred)} | Score {score['total']}% | {pf(precio,PAR)}\nComandos: entré · no · salgo · estado · señal · me quedo\nTambién puedes subir una captura de gráfico arriba 📸"
        st.session_state.chat_history.append({'role':'user','content':uin})
        st.session_state.chat_history.append({'role':'mimi','content':resp})
        st.rerun()

# ── TAB 7: BACKTEST (misma lógica de tendencia+pullback, replay histórico) ──
if st.session_state.page == 'backtest':
    @st.cache_data(ttl=3600)
    def backtest(df_json, ema_fast, ema_slow, ema_entry, sl_mult, tp_rr):
        df_b = pd.read_json(io.StringIO(df_json), orient='split')
        ef = ta.trend.ema_indicator(df_b['Close'], window=ema_fast)
        es = ta.trend.ema_indicator(df_b['Close'], window=ema_slow)
        ee = ta.trend.ema_indicator(df_b['Close'], window=ema_entry)
        atr_s = ta.volatility.average_true_range(df_b['High'],df_b['Low'],df_b['Close'])
        cap=1000.0; eq=[cap]; tds=[]
        i = max(ema_slow, 30)
        while i < len(df_b)-5:
            p = float(df_b['Close'].iloc[i])
            f,s,e = float(ef.iloc[i]), float(es.iloc[i]), float(ee.iloc[i])
            atr_v = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else p*0.001
            en_pb = abs(p-e)/p <= 0.0025
            d = 0
            if p>f and p>s and f>s and en_pb: d = 1
            elif p<f and p<s and f<s and en_pb: d = -1
            if d != 0:
                sl = p - atr_v*sl_mult if d==1 else p + atr_v*sl_mult
                tp = p + (p-sl)*tp_rr if d==1 else p - (sl-p)*tp_rr
                for j in range(1,6):
                    fp = float(df_b['Close'].iloc[i+j])
                    if d==1:
                        if fp>=tp: pnl=(tp-p)/p*cap*0.1; cap+=pnl; tds.append({'dir':'LONG','entrada':p,'salida':tp,'pnl':round(pnl,2),'res':'WIN'}); break
                        elif fp<=sl: pnl=(sl-p)/p*cap*0.1; cap+=pnl; tds.append({'dir':'LONG','entrada':p,'salida':sl,'pnl':round(pnl,2),'res':'LOSS'}); break
                    else:
                        if fp<=tp: pnl=(p-tp)/p*cap*0.1; cap+=pnl; tds.append({'dir':'SHORT','entrada':p,'salida':tp,'pnl':round(pnl,2),'res':'WIN'}); break
                        elif fp>=sl: pnl=(p-sl)/p*cap*0.1; cap+=pnl; tds.append({'dir':'SHORT','entrada':p,'salida':sl,'pnl':round(pnl,2),'res':'LOSS'}); break
                eq.append(cap); i += 5
            else: i += 1
        w = sum(1 for t in tds if t['res']=='WIN')
        return tds, eq, round(w/len(tds)*100,1) if tds else 0, round(cap-1000,2)

    with st.spinner("Simulando estrategia de tendencia + pullback..."):
        bt_t,bt_e,bt_w,bt_p = backtest(df_trend.to_json(orient='split'), PC['ema_trend'][0], PC['ema_trend'][1], PC['ema_entry'], PC['sl_atr_mult'], PC['tp_rr'])
    bm1,bm2,bm3,bm4 = st.columns(4)
    bm1.metric("Capital Inicial","$1,000"); bm2.metric("Capital Final",f"${1000+bt_p:,.2f}",f"{bt_p:+.2f}")
    bm3.metric("Win Rate",f"{bt_w:.1f}%"); bm4.metric("Trades",str(len(bt_t)))
    if bt_e:
        fe=go.Figure()
        fe.add_trace(go.Scatter(y=bt_e,fill='tozeroy',fillcolor='rgba(200,169,110,0.08)',line=dict(color=T['primary'],width=2)))
        fe.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            height=300,margin=dict(l=0,r=0,t=20,b=0),title=dict(text=f"CURVA DE CAPITAL — {PAR} · Trend+Pullback",font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        st.plotly_chart(fe,use_container_width=True)
    if bt_t: st.dataframe(pd.DataFrame(bt_t[-20:]),use_container_width=True)

# ── TAB 8: ALERTAS ────────────────────────────────────────────────
if st.session_state.page == 'alertas':
    st.markdown('<div class="card-title" style="font-family:Cinzel,serif;color:#C8A96E;letter-spacing:3px;">ALERTAS · TELEGRAM BIDIRECCIONAL</div>', unsafe_allow_html=True)
    st.markdown(f"**Par actual: {PAR}.** Comandos: `entré` `no` `estado` `me quedo` `salgo` `señal`")
    col_a, col_b = st.columns(2)
    with col_a:
        ac2 = st.slider("Score mínimo para alertar (%)", 30, 90, 58)
    with col_b:
        if st.button("🧪 Prueba"):
            ok=send_tg(f"🏛️ MIMI-AI Test ✅\nPar: {PAR}\nPrecio: {pf(precio,PAR)}")
            st.success("Enviado ✅") if ok else st.error("Error — revisa Secrets")
        if st.button("📡 Enviar señal ahora"):
            if score['total']>=ac2 and pred!=0:
                ok2=send_tg(f"🏛️ *MIMI-AI — {PAR}*\n🕐 {ahora.strftime('%H:%M')} MX\n💰 {pf(precio,PAR)}\n🎯 {ET.get(pred)}\n⭐ Score: {score['total']}% ({score['categoria']})\n🔴 SL: {pf(senal['sl'],PAR)}\n🟢 TP: {pf(senal['tp'],PAR)}\n\n_Responde 'entré' o 'no'_")
                st.success("Enviado ✅") if ok2 else st.error("Error")
            else: st.info("Condiciones no cumplidas (sin setup o score bajo)")

# ── TAB 9: HISTORIAL ──────────────────────────────────────────────
if st.session_state.page == 'historial':
    if st.session_state.signal_history:
        df_sh = pd.DataFrame(st.session_state.signal_history)
        st.markdown(f"**{len(df_sh)} señales registradas — {PAR}**")
        st.dataframe(df_sh, use_container_width=True)
        if st.button("🗑️ Limpiar"):
            st.session_state.signal_history=[]; gh_save(PAR, {**sv2,'signal_history':[]}); st.rerun()
    else: st.info("Las señales se guardan automáticamente al cargar la app.")

# ── TAB 10: MONITOR ───────────────────────────────────────────────
if st.session_state.page == 'monitor':
    ot_m=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot_m:
        t_m=ot_m[0]; pnl_m=(precio-t_m['entrada'])*(1 if 'LONG' in t_m['dir'] else -1)*t_m['lotes']*CONTRACT
        est_m="🚨 SL — SAL YA" if (('LONG' in t_m['dir'] and precio<=t_m['sl']) or ('SHORT' in t_m['dir'] and precio>=t_m['sl'])) else "🎯 TP — TOMA GANANCIA" if (('LONG' in t_m['dir'] and precio>=t_m['tp']) or ('SHORT' in t_m['dir'] and precio<=t_m['tp'])) else "🟢 MANTÉN" if pnl_m>0 else "🔴 PRECAUCIÓN"
        fm=go.Figure()
        fm.add_hline(y=t_m['tp'],line_color='#4CAF82',line_dash='dash',annotation_text=f"TP {pf(t_m['tp'],PAR)}")
        fm.add_hline(y=t_m['entrada'],line_color=T['primary'],line_width=2,annotation_text=f"ENTRADA {pf(t_m['entrada'],PAR)}")
        fm.add_hline(y=t_m['sl'],line_color='#C0392B',line_dash='dash',annotation_text=f"SL {pf(t_m['sl'],PAR)}")
        fm.add_hline(y=precio,line_color='#FFFFFF',line_dash='dot',annotation_text=f"ACTUAL {pf(precio,PAR)}")
        fm.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',height=260,
            font=dict(color='#888',family='Philosopher,serif'),margin=dict(l=0,r=0,t=30,b=0),
            title=dict(text=f"{PAR} · {t_m['dir']} | P&L: {'+'if pnl_m>0 else ''}${pnl_m:.2f} | {est_m}",font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        st.plotly_chart(fm,use_container_width=True)
        mo1,mo2,mo3,mo4=st.columns(4)
        mo1.metric("Entrada",pf(t_m['entrada'],PAR)); mo2.metric("Actual",pf(precio,PAR))
        mo3.metric("P&L",f"${pnl_m:.2f}"); mo4.metric("Estado",est_m)
        if st.button("🔄 Actualizar precio"): st.cache_data.clear(); st.rerun()
    else:
        st.info(f"Sin posición abierta en {PAR}.")

# ── FRASE FINAL ───────────────────────────────────────────────────
fr = random.choice(FRASES)
st.markdown(f"""
<div class="greek-orn" style="margin-top:24px;">─────── ✦ ───────</div>
<div class="stoic-q">{fr[1]}<div class="stoic-a">— {fr[0]}</div></div>
<div class="greek-orn">─────── ✦ ───────</div>
""", unsafe_allow_html=True)
