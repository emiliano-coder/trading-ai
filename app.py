import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
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
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except:
    HAS_AUTOREFRESH = False
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI", page_icon="🏛️", layout="wide")

# ── AUTO REFRESH — precio cada 30s ────────────────────────────────
if HAS_AUTOREFRESH:
    st_autorefresh(interval=30000, limit=None, key="price_tick")

# ── SECRETS ──────────────────────────────────────────────────────
try:
    TG_TOKEN   = st.secrets["TG_TOKEN"]
    TG_CHAT_ID = st.secrets["TG_CHAT_ID"]
    GH_TOKEN   = st.secrets["GITHUB_TOKEN"]
    GH_REPO    = st.secrets["GITHUB_REPO"]
except:
    TG_TOKEN = TG_CHAT_ID = GH_TOKEN = GH_REPO = ''

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
        "sl_atr_mult": 1.5, "tp_rr": 2.5,
        "usa_fib": False, "usa_ib": True,
    },
    "EUR/USD 💶": {
        "nombre": "Euro", "slug": "eurusd",
        "yf_symbol": "EURUSD=X", "td_symbol": "EUR/USD",
        "dxy_symbol": "DX-Y.NYB", "contract_size": 100000, "decimales": 5,
        "ema_trend": (20, 200), "ema_entry": 20,
        "sl_atr_mult": 1.2, "tp_rr": 2.0,
        "usa_fib": True, "usa_ib": False,
    }
}

STYLE_TF = {
    "Scalping":    {"trend_interval":"1h",  "trend_period":"60d",  "entry_interval":"15m", "entry_period":"10d", "label":"M15/H1"},
    "Day Trading": {"trend_interval":"4h",  "trend_period":"180d", "entry_interval":"1h",  "entry_period":"60d", "label":"H1/H4"},
    "Swing":       {"trend_interval":"1d",  "trend_period":"2y",   "entry_interval":"4h",  "entry_period":"180d","label":"H4/D1"},
}

NOTICIAS_ALTO_IMPACTO = ["FOMC", "NFP (Nóminas no agrícolas)", "Decisión de tasas ECB", "CPI / Inflación", "Discurso de la Fed"]

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
    st.session_state.par_loaded = True

def cargar_estado_par(par):
    sv = gh_load(par)
    st.session_state.paper_trades    = sv.get('paper_trades', [])
    st.session_state.signal_history  = sv.get('signal_history', [])
    st.session_state.capital         = sv.get('capital', 1000.0)
    st.session_state.trade_style     = sv.get('trade_style', 'Day Trading')
    st.session_state.estrategia_xau  = sv.get('estrategia_xau', 'Trend + Pullback (EMA)')
    st.session_state.last_tg_update  = sv.get('last_tg_update', 0)
    st.session_state.loaded_par      = par

if 'loaded_par' not in st.session_state or st.session_state.loaded_par != st.session_state.par:
    cargar_estado_par(st.session_state.par)
    st.session_state.chat_history = []

# ── THEMES ───────────────────────────────────────────────────────
THEMES = {
    "Mármol Griego":  {"primary":"#C8A96E","secondary":"#8B6914","bg":"#0a0905","card":"#13100a"},
    "Bronce Estoico": {"primary":"#CD7F32","secondary":"#8B4513","bg":"#080503","card":"#120a05"},
    "Lapislázuli":    {"primary":"#6B8FCE","secondary":"#3A5A9B","bg":"#03060f","card":"#070b18"},
    "Olimpo Oscuro":  {"primary":"#9B7FD4","secondary":"#6B4FA0","bg":"#060308","card":"#0d0614"},
    "Athena":         {"primary":"#7BAF9E","secondary":"#3D7A68","bg":"#030a08","card":"#06120f"},
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

    sl_mult = cfg['sl_atr_mult']; tp_rr = cfg['tp_rr']
    if direccion == 1:
        swing_ref = float(df_entry['Low'].iloc[-8:].min())
        sl = min(swing_ref, precio - atr*sl_mult)
        tp = precio + (precio - sl) * tp_rr
    elif direccion == -1:
        swing_ref = float(df_entry['High'].iloc[-8:].max())
        sl = max(swing_ref, precio + atr*sl_mult)
        tp = precio - (sl - precio) * tp_rr
    else:
        sl = precio - atr*sl_mult
        tp = precio + atr*sl_mult*tp_rr

    return {
        'direccion':direccion,'precio':precio,'sl':round(sl,6),'tp':round(tp,6),
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

def parse_tg_command(txt):
    t = txt.lower().strip()
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
        txt = u.get('message', {}).get('text', '')
        if not txt: continue
        cmd = parse_tg_command(txt)
        ot  = [t for t in st.session_state.paper_trades if t['estado'] == 'ABIERTO']
        mx  = pytz.timezone('America/Mexico_City'); ah = datetime.now(mx); tag = par.split()[0]

        if cmd == 'ENTRO':
            if pred != 0 and not ot:
                lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, senal['sl'], contract_size)
                st.session_state.paper_trades.append({
                    'id':len(st.session_state.paper_trades)+1,'par':par,
                    'dir':'LONG 📈' if pred==1 else 'SHORT 📉','entrada':precio,'sl':senal['sl'],'tp':senal['tp'],
                    'lotes':lot,'riesgo':risg,'estado':'ABIERTO','fecha':ah.strftime('%d/%m %H:%M'),
                    'resultado':'PENDIENTE','pnl':0,'score':score['total']})
                send_tg(f"✅ *Trade registrado — {tag}*\n{'LONG 📈' if pred==1 else 'SHORT 📉'} @ {pf(precio,par)}\nSL: {pf(senal['sl'],par)} | TP: {pf(senal['tp'],par)}\nScore: {score['total']}%\nLotes: {lot} | Riesgo: ${risg:.2f}")
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
        elif cmd == 'TEXTO_LIBRE':
            send_tg(f"🏛️ Score actual: {score['total']}% — {score['categoria']}\nTendencia: {senal['tendencia']}\nComandos: entré · no · salgo · estado · señal · me quedo")

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

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};letter-spacing:3px;text-align:center;">⚙ CONFIG</div>', unsafe_allow_html=True)
    st.markdown("---")

    nuevo_par = st.selectbox("💱 Par", list(PAIRS.keys()), index=list(PAIRS.keys()).index(st.session_state.par))
    if nuevo_par != st.session_state.par:
        st.session_state.par = nuevo_par
        gh_save_config({'par': nuevo_par})
        cargar_estado_par(nuevo_par)
        st.rerun()

    nuevo_tema = st.selectbox("🏛️ Estilo visual", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.tema))
    if nuevo_tema != st.session_state.tema:
        st.session_state.tema = nuevo_tema; st.rerun()

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
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📱 TELEGRAM COMANDOS</div>', unsafe_allow_html=True)
    st.caption("entré · no · salgo · estado · señal · me quedo")
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

# ── HEADER ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="greek-orn">─────── ✦ ───────</div>
<div class="mimi-title">MIMI · AI</div>
<div class="mimi-sub">{PAR} · Trend Following · Pullback · Price Action</div>
<div class="greek-orn" style="margin-top:6px;">─────── ✦ ───────</div>
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

# ── AUTO PAPER TRADE RESULT ────────────────────────────────────────
for t in st.session_state.paper_trades:
    if t['estado'] == 'ABIERTO':
        if 'LONG' in t['dir']:
            if precio >= t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['tp']-t['entrada'])*t['lotes']*CONTRACT,2); t['resultado']='WIN ✅'
                send_tg(f"🏛️ TP alcanzado 🟢 — {PAR}\nLONG cerrado @ {pf(precio,PAR)}\nP&L: +${t['pnl']:.2f}")
            elif precio <= t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['sl']-t['entrada'])*t['lotes']*CONTRACT,2); t['resultado']='LOSS ❌'
                send_tg(f"🏛️ SL alcanzado 🔴 — {PAR}\nLONG cerrado @ {pf(precio,PAR)}\nP&L: ${t['pnl']:.2f}")
        elif 'SHORT' in t['dir']:
            if precio <= t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['tp'])*t['lotes']*CONTRACT,2); t['resultado']='WIN ✅'
                send_tg(f"🏛️ TP alcanzado 🟢 — {PAR}\nSHORT cerrado @ {pf(precio,PAR)}\nP&L: +${t['pnl']:.2f}")
            elif precio >= t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['sl'])*t['lotes']*CONTRACT,2); t['resultado']='LOSS ❌'
                send_tg(f"🏛️ SL alcanzado 🔴 — {PAR}\nSHORT cerrado @ {pf(precio,PAR)}\nP&L: ${t['pnl']:.2f}")

cap = 1000.0 + sum(t.get('pnl',0) for t in st.session_state.paper_trades if t['estado']=='CERRADO')
st.session_state.capital = round(cap, 2)

if not st.session_state.signal_history or st.session_state.signal_history[-1].get('precio') != precio:
    st.session_state.signal_history.append({
        'id':len(st.session_state.signal_history)+1,'fecha':ahora.strftime('%d/%m %H:%M'),'par':PAR,
        'estilo':st.session_state.trade_style,'direccion':ET.get(pred),'score':score['total'],
        'precio':precio,'sl':senal['sl'],'tp':senal['tp'],'tendencia':senal['tendencia'],'resultado':'PENDIENTE'})

sv2 = {'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history[-50:],
       'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
       'estrategia_xau':st.session_state.estrategia_xau,'last_tg_update':st.session_state.last_tg_update}
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

# ── PRECIO EN VIVO ───────────────────────────────────────────────
tick = get_precio_vivo(PC['td_symbol'], PC['yf_symbol'])
precio_display = tick['precio'] if tick else precio
cambio_display = tick['cambio'] if tick else 0
cambio_pct_display = tick['cambio_pct'] if tick else 0
hora_display = tick['hora'][11:16] if tick else "—"
precio_color = '#4CAF82' if cambio_display >= 0 else '#C0392B'
flecha = '▲' if cambio_display >= 0 else '▼'

st.markdown(f"""
<div style="background:linear-gradient(90deg,{T['card']},{T['bg']},{T['card']});
  border:1px solid {precio_color}44;border-radius:4px;padding:12px 20px;
  margin:8px 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
  <div>
    <span style="font-family:'Cinzel',serif;color:{T['primary']}88;font-size:.75em;letter-spacing:3px;">{PAR} · EN VIVO · {hora_display}</span><br>
    <span style="font-family:'Cinzel',serif;font-size:clamp(1.4rem,4vw,2.2rem);font-weight:900;color:{precio_color};
      filter:drop-shadow(0 0 12px {precio_color}66);">{pf(precio_display,PAR)}</span>
    <span style="font-family:'Philosopher',serif;color:{precio_color};font-size:1em;margin-left:12px;">
      {flecha} {pf(abs(cambio_display),PAR)} ({cambio_pct_display:+.3f}%)
    </span>
  </div>
  <div style="text-align:right;font-family:'Philosopher',serif;color:{T['primary']}88;font-size:.82em;line-height:1.8;">
    {STF['label']} · Riesgo {risk_pct}% · Auto-refresh 30s · {tick['fuente'] if tick else '—'}
  </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("💰 Precio Vivo", pf(precio_display,PAR), f"{pf(cambio_display,PAR)} ({cambio_pct_display:+.3f}%)")
c2.metric("📈 Tendencia", senal['tendencia'])
c3.metric("🎯 Señal","LONG" if pred==1 else "SHORT" if pred==-1 else "ESPERA")
c4.metric("⭐ Score", f"{score['total']}%", score['categoria'].split(' ')[0])
c5.metric("📐 R:R", f"1:{rr_actual:.2f}" if rr_actual else "—")
c6.metric("💰 Capital", f"${st.session_state.capital:,.2f}")
st.markdown('<div class="greek-orn">── ✦ ──</div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
    "🎯 Señal","📐 Estructura","🌐 Multi·TF","📋 Paper","📊 Gráfica",
    "💬 Chat","📈 Backtest","🔔 Alertas","📜 Historial","👁️ Monitor"])

# ── TAB 1: SEÑAL ──────────────────────────────────────────────────
with tab1:
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
            st.markdown(f"**Take Profit 🟢:** {pf(senal['tp'],PAR)}  (R:R 1:{rr_actual:.2f})")
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
                    'dir':'LONG 📈' if pred==1 else 'SHORT 📉','entrada':precio,'sl':senal['sl'],'tp':senal['tp'],
                    'lotes':lot2,'riesgo':risg2,'estado':'ABIERTO','fecha':ahora.strftime('%d/%m %H:%M'),
                    'resultado':'PENDIENTE','pnl':0,'score':score['total']})
                send_tg(f"🏛️ *Trade abierto — {PAR}*\n{'LONG 📈' if pred==1 else 'SHORT 📉'} @ {pf(precio,PAR)}\nSL: {pf(senal['sl'],PAR)} | TP: {pf(senal['tp'],PAR)}\nScore: {score['total']}%")
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
with tab2:
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
with tab3:
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
with tab4:
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
with tab5:
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
with tab6:
    st.caption("💡 Este chat también funciona por Telegram.")
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg['role']=='user' else "assistant"):
            st.markdown(msg['content'])
    uin = st.chat_input("Consulta al Oráculo...")
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
            resp = f"Señal: {ET.get(pred)} | Score {score['total']}% | {pf(precio,PAR)}\nComandos: entré · no · salgo · estado · señal · me quedo"
        st.session_state.chat_history.append({'role':'user','content':uin})
        st.session_state.chat_history.append({'role':'mimi','content':resp})
        st.rerun()

# ── TAB 7: BACKTEST (misma lógica de tendencia+pullback, replay histórico) ──
with tab7:
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
with tab8:
    st.markdown('<div class="card-title" style="font-family:Cinzel,serif;color:#C8A96E;letter-spacing:3px;">ALERTAS · TELEGRAM BIDIRECCIONAL</div>', unsafe_allow_html=True)
    st.markdown(f"**Par actual: {PAR}.** Comandos: `entré` `no` `estado` `me quedo` `salgo` `señal`")
    col_a, col_b = st.columns(2)
    with col_a:
        ac2 = st.slider("Score mínimo para alertar (%)", 30, 90, 60)
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
with tab9:
    if st.session_state.signal_history:
        df_sh = pd.DataFrame(st.session_state.signal_history)
        st.markdown(f"**{len(df_sh)} señales registradas — {PAR}**")
        st.dataframe(df_sh, use_container_width=True)
        if st.button("🗑️ Limpiar"):
            st.session_state.signal_history=[]; gh_save(PAR, {**sv2,'signal_history':[]}); st.rerun()
    else: st.info("Las señales se guardan automáticamente al cargar la app.")

# ── TAB 10: MONITOR ───────────────────────────────────────────────
with tab10:
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
