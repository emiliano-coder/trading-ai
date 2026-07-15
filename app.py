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
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except:
    HAS_AUTOREFRESH = False
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI", page_icon="🏛️", layout="wide")

# ── AUTO REFRESH — precio cada 30s, todo lo demás cacheado ───────
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
#  CONFIGURACIÓN DE PARES — cada par trae su propio símbolo, modo
#  de señal, tamaño de contrato y features de ML específicos
# ══════════════════════════════════════════════════════════════════
PAIRS = {
    "XAU/USD 🥇": {
        "nombre": "Oro", "slug": "xauusd",
        "yf_symbol": "GC=F", "td_symbol": "XAU/USD",
        "dxy_symbol": None,          # el oro no usa correlación DXY como feature
        "contract_size": 100,        # 1 lote = 100 oz
        "decimales": 2,
        "modos": {
            "Oráculo 🏛️": {"umbral":0.003,"atr_sl":1.5,"atr_tp":2.5,"min_conf":55,"require_smc":True,"require_adx":False,
                "desc":"Señales precisas. Requiere BOS + OB + EMA alineados."},
            "Gladiador ⚔️": {"umbral":0.001,"atr_sl":0.6,"atr_tp":1.2,"min_conf":40,"require_smc":False,"require_adx":False,
                "desc":"Micro entradas. Rebotes en OB, FVG, EQH/EQL. Más trades, más riesgo."}
        }
    },
    "EUR/USD 💶": {
        "nombre": "Euro", "slug": "eurusd",
        "yf_symbol": "EURUSD=X", "td_symbol": "EUR/USD",
        "dxy_symbol": "DX-Y.NYB",    # índice dólar — correlación inversa fuerte con EUR/USD
        "contract_size": 100000,     # 1 lote estándar = 100,000 unidades
        "decimales": 5,
        "modos": {
            "Oráculo 🏛️": {"umbral":0.0015,"atr_sl":1.3,"atr_tp":2.2,"min_conf":55,"require_smc":True,"require_adx":True,
                "desc":"Requiere BOS + OB + ADX>20 (tendencia real, evita rango sin fuerza)."},
            "Gladiador ⚔️": {"umbral":0.0006,"atr_sl":0.6,"atr_tp":1.2,"min_conf":40,"require_smc":False,"require_adx":False,
                "desc":"Rebotes en Londres/NY overlap. Más trades, SL corto."}
        }
    }
}

def pdec(par):
    return PAIRS[par]["decimales"]

def pf(x, par):
    """Formatea precio con los decimales correctos según el par."""
    try:
        return f"{x:,.{pdec(par)}f}"
    except Exception:
        return str(x)

# ── PRECIO EN VIVO — dual source, sin bloquear el modelo ────────
@st.cache_data(ttl=15)
def get_precio_vivo(td_symbol, yf_symbol):
    try:
        td_key = st.secrets.get("TWELVEDATA_KEY", "")
        if td_key:
            r = requests.get(
                f"https://api.twelvedata.com/quote?symbol={td_symbol}&apikey={td_key}",
                timeout=2)
            if r.status_code == 200:
                d = r.json()
                p     = float(d.get("close", 0))
                prev  = float(d.get("previous_close", p))
                if p > 0:
                    return {
                        'precio': p, 'prev': prev,
                        'cambio': round(p-prev, 6),
                        'cambio_pct': round((p-prev)/prev*100, 3) if prev else 0,
                        'high': float(d.get("high", p)),
                        'low':  float(d.get("low", p)),
                        'open': float(d.get("open", p)),
                        'hora': d.get("datetime", "")[:16],
                        'fuente': 'TwelveData', 'vivo': True
                    }
    except: pass

    try:
        df_tick = yf.download(yf_symbol, period="1d", interval="1m", progress=False)
        if df_tick is not None and len(df_tick) > 0:
            df_tick.columns = [c[0] if isinstance(c,tuple) else c for c in df_tick.columns]
            last = df_tick.iloc[-1]; prev_row = df_tick.iloc[-2] if len(df_tick)>1 else last
            p = float(last['Close']); prev = float(prev_row['Close'])
            return {
                'precio': p, 'prev': prev,
                'cambio': round(p-prev,6),
                'cambio_pct': round((p-prev)/prev*100,3) if prev else 0,
                'high':  float(last['High']),
                'low':   float(last['Low']),
                'open':  float(last['Open']),
                'hora':  str(df_tick.index[-1])[11:16],
                'fuente':'yfinance 1m', 'vivo': True
            }
    except: pass
    return None

# ── GITHUB PERSISTENCE (un archivo distinto por par) ─────────────
def gh_file_for(par):
    return f"mimi_data_{PAIRS[par]['slug']}.json"

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
    """Guarda solo la preferencia de qué par estás viendo, para que se quede fija."""
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
    """Carga capital/trades/historial/modo/estilo específicos del par seleccionado."""
    sv = gh_load(par)
    st.session_state.paper_trades    = sv.get('paper_trades', [])
    st.session_state.signal_history  = sv.get('signal_history', [])
    st.session_state.capital         = sv.get('capital', 1000.0)
    st.session_state.trade_style     = sv.get('trade_style', 'Day Trading')
    st.session_state.modo            = sv.get('modo', 'Oráculo 🏛️')
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
    ("Marco Aurelio","Nunca desperdicies tiempo preguntándote qué tipo de persona ser. Sé esa persona."),
    ("Séneca","No es pobre el que tiene poco, sino el que desea mucho."),
]

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

def process_tg_updates(par, precio, pred, prob, rsi, atr, sl_long, tp_long, sl_short, tp_short, rr, smc, conf, ET, risk_pct, contract_size):
    updates = get_tg_updates(offset=st.session_state.last_tg_update)
    for u in updates:
        uid = u.get('update_id', 0)
        if uid <= st.session_state.last_tg_update: continue
        st.session_state.last_tg_update = uid + 1
        txt = u.get('message', {}).get('text', '')
        if not txt: continue
        cmd = parse_tg_command(txt)
        ot  = [t for t in st.session_state.paper_trades if t['estado'] == 'ABIERTO']
        mx  = pytz.timezone('America/Mexico_City')
        ah  = datetime.now(mx)
        tag = par.split()[0]

        if cmd == 'ENTRO':
            if pred != 0 and not ot:
                sl_r = sl_long if pred==1 else sl_short
                tp_r = tp_long if pred==1 else tp_short
                lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, sl_r, contract_size)
                st.session_state.paper_trades.append({
                    'id':len(st.session_state.paper_trades)+1,'par':par,
                    'dir':'LONG 📈' if pred==1 else 'SHORT 📉',
                    'entrada':precio,'sl':sl_r,'tp':tp_r,'lotes':lot,'riesgo':risg,
                    'estado':'ABIERTO','fecha':ah.strftime('%d/%m %H:%M'),
                    'resultado':'PENDIENTE','pnl':0
                })
                send_tg(f"✅ *Trade registrado — {tag}*\n{'LONG 📈' if pred==1 else 'SHORT 📉'} @ {pf(precio,par)}\nSL: {pf(sl_r,par)} | TP: {pf(tp_r,par)}\nLotes: {lot} | Riesgo: ${risg:.2f}\n\nEscríbeme 'salgo' cuando quieras cerrar o 'estado' para ver cómo vas.")
            elif ot: send_tg("⚠️ Ya tienes un trade abierto en este par. Escríbeme 'estado' para verlo.")
            else: send_tg("⚠️ La señal actual es LATERAL. No hay entrada clara.")

        elif cmd == 'NO_ENTRO':
            send_tg(f"🏛️ Señal rechazada — {tag}. {ET.get(pred)} @ {pf(precio,par)}\n_El estoico espera. La próxima señal llegará._")

        elif cmd == 'MANTENER':
            if ot:
                t  = ot[0]
                pnl = (precio - t['entrada']) * (1 if 'LONG' in t['dir'] else -1) * t['lotes'] * contract_size
                send_tg(f"🏛️ *Posición activa — {tag}*\n{t['dir']} @ {pf(t['entrada'],par)}\nActual: {pf(precio,par)}\nP&L: {'+'if pnl>0 else ''}${pnl:.2f}\nSL: {pf(t['sl'],par)} | TP: {pf(t['tp'],par)}\n\n{'🟢 Posición ganadora — mantén mientras la estructura aguante.' if pnl>0 else '🔴 Posición en negativo — evalúa si tu razón de entrada sigue válida.'}")
            else: send_tg("Sin trade abierto. Escríbeme 'señal' para ver la señal actual.")

        elif cmd == 'SALIR':
            if ot:
                t   = ot[0]
                pnl = (precio - t['entrada']) * (1 if 'LONG' in t['dir'] else -1) * t['lotes'] * contract_size
                t['estado'] = 'CERRADO'
                t['pnl']    = round(pnl, 2)
                t['resultado'] = 'WIN ✅' if pnl > 0 else 'LOSS ❌'
                send_tg(f"{'✅' if pnl>0 else '❌'} *Trade cerrado por orden del usuario — {tag}*\nPrecio de salida: {pf(precio,par)}\nP&L: {'+'if pnl>0 else ''}${pnl:.2f}\n\n_{'El sabio toma sus ganancias.' if pnl>0 else 'Una pérdida aceptada es una victoria de carácter.'}_")
            else: send_tg("Sin trade abierto para cerrar.")

        elif cmd == 'STATUS':
            if ot:
                t   = ot[0]
                pnl = (precio - t['entrada']) * (1 if 'LONG' in t['dir'] else -1) * t['lotes'] * contract_size
                est = "🟢 MANTÉN" if pnl > 0 else "🔴 PRECAUCIÓN"
                if ('LONG' in t['dir'] and precio <= t['sl']) or ('SHORT' in t['dir'] and precio >= t['sl']): est = "🚨 SL — SAL YA"
                elif ('LONG' in t['dir'] and precio >= t['tp']) or ('SHORT' in t['dir'] and precio <= t['tp']): est = "🎯 TP — TOMA GANANCIA"
                send_tg(f"👁️ *Estado de posición — {tag}*\n{t['dir']} @ {pf(t['entrada'],par)}\nActual: {pf(precio,par)}\nP&L: {'+'if pnl>0 else ''}${pnl:.2f}\nSL: {pf(t['sl'],par)} | TP: {pf(t['tp'],par)}\n\n{est}\n\nResponde 'me quedo' o 'salgo'")
            else:
                send_tg(f"📊 *Sin posición abierta — {tag}*\nSeñal actual: *{ET.get(pred)}*\nConfianza: {conf:.1f}%\nPrecio: {pf(precio,par)}\n\nCapital: ${st.session_state.capital:,.2f}")

        elif cmd == 'SEÑAL':
            sl_r = sl_long if pred >= 0 else sl_short
            tp_r = tp_long if pred >= 0 else tp_short
            send_tg(f"🏛️ *MIMI-AI — Señal {tag}*\n*{ET.get(pred)}* | {conf:.1f}%\nPrecio: {pf(precio,par)}\nSL: {pf(sl_r,par)} | TP: {pf(tp_r,par)}\nR:R: 1:{rr}\nSMC: {smc['bias']}\nRSI: {rsi:.1f}\n\n{'Responde *entré* para registrar el trade.' if pred != 0 else 'Mercado lateral — espera ruptura.'}")

        elif cmd == 'TEXTO_LIBRE':
            resp = analizar_texto_libre(txt, par, precio, pred, prob, rsi, atr, smc, ET, conf, sl_long, tp_long, sl_short, tp_short, rr)
            send_tg(f"🏛️ *MIMI-AI responde ({tag}):*\n{resp}")

    sv2 = {'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history[-50:],
            'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
            'modo':st.session_state.modo,'last_tg_update':st.session_state.last_tg_update}
    gh_save(par, sv2)

def analizar_texto_libre(txt, par, precio, pred, prob, rsi, atr, smc, ET, conf, sl_long, tp_long, sl_short, tp_short, rr):
    t = txt.lower()
    ot = [tr for tr in st.session_state.paper_trades if tr['estado']=='ABIERTO']
    if any(w in t for w in ['rsi','momento','momentum']): return f"RSI: {rsi:.1f} — {'sobrecomprado ⚠️' if rsi>70 else 'sobrevendido ⚠️' if rsi<30 else 'neutral ✅'}"
    if any(w in t for w in ['smc','order block','ob','estructura']): return f"Bias SMC: {smc['bias']}\nOB: {smc['order_blocks'][-1]['tipo']} {pf(smc['order_blocks'][-1]['bottom'],par)}–{pf(smc['order_blocks'][-1]['top'],par)}" if smc['order_blocks'] else f"Bias SMC: {smc['bias']}. Sin OB claros."
    if any(w in t for w in ['sl','stop','riesgo']): return f"SL LONG: {pf(sl_long,par)} | SL SHORT: {pf(sl_short,par)}"
    if any(w in t for w in ['tp','objetivo','target']): return f"TP LONG: {pf(tp_long,par)} | TP SHORT: {pf(tp_short,par)} | R:R 1:{rr}"
    if any(w in t for w in ['adx','tendencia','fuerza']): return "ADX mide la fuerza de la tendencia (>20 = tendencia real, <20 = rango sin fuerza)."
    if any(w in t for w in ['dxy','dolar','dólar']): return "El EUR/USD tiene correlación inversa fuerte con el índice del dólar (DXY) — si el dólar sube, el euro normalmente baja."
    if ot:
        t2  = ot[0]; pnl = (precio - t2['entrada']) * (1 if 'LONG' in t2['dir'] else -1) * t2['lotes'] * PAIRS[par]['contract_size']
        return f"Tienes {t2['dir']} abierto @ {pf(t2['entrada'],par)}\nActual: {pf(precio,par)} | P&L: {'+'if pnl>0 else ''}${pnl:.2f}\n{'🟢 Mantén' if pnl>0 else '🔴 Precaución'}"
    return f"Señal: {ET.get(pred)} {conf:.1f}% | {pf(precio,par)}\nComandos: entré · no · salgo · estado · señal · me quedo"

# ── ADVANCED SMC ──────────────────────────────────────────────────
def detect_smc_advanced(df, lookback=50):
    res = {
        'bos': [], 'msb': [], 'order_blocks': [], 'fvg': [],
        'eqh': [], 'eql': [], 'liquidity_swings': [],
        'bias': 'NEUTRAL', 'bias_score': 0,
        'gladiador_entry': None
    }
    if len(df) < lookback + 5: return res

    H = df['High'].values
    L = df['Low'].values
    C = df['Close'].values
    O = df['Open'].values
    V = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))

    swing_highs = []
    swing_lows  = []
    for i in range(3, len(df)-3):
        if H[i] > H[i-1] and H[i] > H[i-2] and H[i] > H[i-3] and H[i] > H[i+1] and H[i] > H[i+2] and H[i] > H[i+3]:
            swing_highs.append((i, H[i]))
        if L[i] < L[i-1] and L[i] < L[i-2] and L[i] < L[i-3] and L[i] < L[i+1] and L[i] < L[i+2] and L[i] < L[i+3]:
            swing_lows.append((i, L[i]))

    swing_highs = swing_highs[-8:]
    swing_lows  = swing_lows[-8:]

    for sh in swing_highs[-4:]:
        res['liquidity_swings'].append({'tipo':'SWING HIGH','nivel':round(sh[1],5),'idx':sh[0]})
    for sl in swing_lows[-4:]:
        res['liquidity_swings'].append({'tipo':'SWING LOW','nivel':round(sl[1],5),'idx':sl[0]})

    precio_actual = C[-1]
    if len(swing_highs) >= 2:
        prev_sh = swing_highs[-2][1]
        last_sh = swing_highs[-1][1]
        if last_sh > prev_sh:
            res['bos'].append({'tipo':'BOS ALCISTA','nivel':round(last_sh,5),'fuerza':'FUERTE' if last_sh > prev_sh * 1.002 else 'DÉBIL'})
            res['bias_score'] += 2
        elif last_sh < prev_sh:
            res['msb'].append({'tipo':'MSB BAJISTA','nivel':round(last_sh,5),'fuerza':'FUERTE' if last_sh < prev_sh * 0.998 else 'DÉBIL'})
            res['bias_score'] -= 2

    if len(swing_lows) >= 2:
        prev_sl = swing_lows[-2][1]
        last_sl = swing_lows[-1][1]
        if last_sl < prev_sl:
            res['bos'].append({'tipo':'BOS BAJISTA','nivel':round(last_sl,5),'fuerza':'FUERTE' if last_sl < prev_sl * 0.998 else 'DÉBIL'})
            res['bias_score'] -= 2
        elif last_sl > prev_sl:
            res['msb'].append({'tipo':'MSB ALCISTA','nivel':round(last_sl,5),'fuerza':'FUERTE' if last_sl > prev_sl * 1.002 else 'DÉBIL'})
            res['bias_score'] += 2

    avg_body = np.mean([abs(C[j]-O[j]) for j in range(-lookback,-1)])
    avg_vol  = np.mean(V[-lookback:]) if V is not None else 1

    for i in range(-lookback+2, -2):
        body   = abs(C[i] - O[i])
        vol_i  = V[i] if V is not None else 1
        next_move = C[i+2] - C[i]
        if body > avg_body * 1.2:
            strength = 'FUERTE' if body > avg_body * 2 or vol_i > avg_vol * 1.5 else 'NORMAL'
            if C[i] < O[i] and next_move > avg_body * 1.5:
                res['order_blocks'].append({
                    'tipo':'OB ALCISTA','top':round(O[i],5),'bottom':round(C[i],5),
                    'mid':round((O[i]+C[i])/2,5),'fuerza':strength
                })
                res['bias_score'] += 1
            elif C[i] > O[i] and next_move < -avg_body * 1.5:
                res['order_blocks'].append({
                    'tipo':'OB BAJISTA','top':round(C[i],5),'bottom':round(O[i],5),
                    'mid':round((O[i]+C[i])/2,5),'fuerza':strength
                })
                res['bias_score'] -= 1

    res['order_blocks'] = res['order_blocks'][-5:]

    tolerance = precio_actual * 0.0015

    for i in range(len(swing_highs)):
        for j in range(i+1, len(swing_highs)):
            if abs(swing_highs[i][1] - swing_highs[j][1]) < tolerance:
                lvl = round((swing_highs[i][1] + swing_highs[j][1]) / 2, 5)
                res['eqh'].append({'nivel':lvl,'tipo':'EQH — Liquidez Alcista','descripcion':'Zona de stop hunts probables arriba'})

    for i in range(len(swing_lows)):
        for j in range(i+1, len(swing_lows)):
            if abs(swing_lows[i][1] - swing_lows[j][1]) < tolerance:
                lvl = round((swing_lows[i][1] + swing_lows[j][1]) / 2, 5)
                res['eql'].append({'nivel':lvl,'tipo':'EQL — Liquidez Bajista','descripcion':'Zona de stop hunts probables abajo'})

    for i in range(-lookback, -2):
        gap_size = abs(L[i+2] - H[i])
        if gap_size > precio_actual * 0.001:
            if L[i+2] > H[i]:
                res['fvg'].append({'tipo':'FVG ALCISTA','top':round(L[i+2],5),'bottom':round(H[i],5),'size':round(gap_size,5)})
            elif H[i+2] < L[i]:
                gap_size2 = abs(L[i] - H[i+2])
                res['fvg'].append({'tipo':'FVG BAJISTA','top':round(L[i],5),'bottom':round(H[i+2],5),'size':round(gap_size2,5)})
    res['fvg'] = res['fvg'][-4:]

    if res['bias_score'] >= 2:   res['bias'] = 'ALCISTA'
    elif res['bias_score'] <= -2: res['bias'] = 'BAJISTA'
    else:                         res['bias'] = 'NEUTRAL'

    near_ob_bull = any(ob['bottom'] <= precio_actual <= ob['top'] * 1.001 for ob in res['order_blocks'] if 'ALCISTA' in ob['tipo'])
    near_ob_bear = any(ob['bottom'] * 0.999 <= precio_actual <= ob['top'] for ob in res['order_blocks'] if 'BAJISTA' in ob['tipo'])
    near_eqh = any(abs(e['nivel'] - precio_actual) / precio_actual < 0.002 for e in res['eqh'])
    near_eql = any(abs(e['nivel'] - precio_actual) / precio_actual < 0.002 for e in res['eql'])
    near_fvg_bull = any(f['bottom'] <= precio_actual <= f['top'] for f in res['fvg'] if 'ALCISTA' in f['tipo'])
    near_fvg_bear = any(f['bottom'] <= precio_actual <= f['top'] for f in res['fvg'] if 'BAJISTA' in f['tipo'])

    if near_ob_bull or near_fvg_bull: res['gladiador_entry'] = 'LONG_REBOTE'
    elif near_ob_bear or near_fvg_bear: res['gladiador_entry'] = 'SHORT_REBOTE'
    elif near_eqh: res['gladiador_entry'] = 'SHORT_LIQUIDEZ'
    elif near_eql: res['gladiador_entry'] = 'LONG_LIQUIDEZ'

    return res

STYLE_CONFIG = {
    "Scalping":    {"interval":"5m",  "period":"5d",   "label":"M5"},
    "Day Trading": {"interval":"15m", "period":"10d",  "label":"M15"},
    "Swing":       {"interval":"4h",  "period":"180d", "label":"H4"},
}

def get_signal_oraculo(df, smc, features, m, sc, atr_sl, atr_tp, require_adx=False):
    ul   = df[features].iloc[-1:]
    pred = m.predict(sc.transform(ul))[0]
    prob = m.predict_proba(sc.transform(ul))[0]
    p    = float(df['Close'].iloc[-1])
    atr  = float(df['ATR'].iloc[-1])

    smc_confirms_long  = smc['bias'] == 'ALCISTA' and any('ALCISTA' in ob['tipo'] for ob in smc['order_blocks'])
    smc_confirms_short = smc['bias'] == 'BAJISTA' and any('BAJISTA' in ob['tipo'] for ob in smc['order_blocks'])
    has_bos            = bool(smc['bos'])

    if pred == 1 and not (smc_confirms_long or has_bos):  pred = 0
    if pred == -1 and not (smc_confirms_short or has_bos): pred = 0

    if require_adx and 'ADX' in df.columns:
        adx_v = float(df['ADX'].iloc[-1])
        if adx_v < 20:
            pred = 0  # sin tendencia real, no hay señal Oráculo

    return int(pred), prob, p, atr, round(p-atr*atr_sl,5), round(p+atr*atr_tp,5), round(p+atr*atr_sl,5), round(p-atr*atr_tp,5)

def get_signal_gladiador(df, smc, features, m, sc, atr_sl, atr_tp, require_adx=False):
    ul   = df[features].iloc[-1:]
    pred = m.predict(sc.transform(ul))[0]
    prob = m.predict_proba(sc.transform(ul))[0]
    p    = float(df['Close'].iloc[-1])
    atr  = float(df['ATR'].iloc[-1])

    if pred == 0 and smc['gladiador_entry']:
        ge = smc['gladiador_entry']
        if 'LONG' in ge:   pred = 1
        elif 'SHORT' in ge: pred = -1

    return int(pred), prob, p, atr, round(p-atr*atr_sl,5), round(p+atr*atr_tp,5), round(p+atr*atr_sl,5), round(p-atr*atr_tp,5)

def calc_pos(capital, risk, entrada, sl, contract_size):
    r = capital * (risk / 100)
    d = abs(entrada - sl)
    if d == 0: return 0, 0
    return round(r/(d*contract_size), 4), round(r, 2)

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
.smc-weak   {{ color:#C8A96E; }}
.smc-bear   {{ color:#C0392B; font-weight:600; }}
.modo-badge {{ font-family:'Cinzel',serif; font-size:.8em; letter-spacing:2px; padding:4px 12px;
    border-radius:2px; display:inline-block; margin:4px 0; }}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};letter-spacing:3px;text-align:center;">⚙ CONFIG</div>', unsafe_allow_html=True)
    st.markdown("---")

    nuevo_par = st.selectbox("💱 Par", list(PAIRS.keys()),
                              index=list(PAIRS.keys()).index(st.session_state.par))
    if nuevo_par != st.session_state.par:
        st.session_state.par = nuevo_par
        gh_save_config({'par': nuevo_par})
        cargar_estado_par(nuevo_par)
        st.rerun()

    nuevo_tema = st.selectbox("🏛️ Estilo", list(THEMES.keys()),
                               index=list(THEMES.keys()).index(st.session_state.tema))
    if nuevo_tema != st.session_state.tema:
        st.session_state.tema = nuevo_tema; st.rerun()

    PC = PAIRS[st.session_state.par]
    nuevo_modo = st.selectbox("🎯 Modo", list(PC['modos'].keys()),
                               index=list(PC['modos'].keys()).index(st.session_state.modo))
    MC = PC['modos'][nuevo_modo]
    st.caption(MC['desc'])
    if nuevo_modo != st.session_state.modo:
        st.session_state.modo = nuevo_modo
        sv = gh_load(st.session_state.par); sv['modo'] = nuevo_modo; gh_save(st.session_state.par, sv)

    nuevo_estilo = st.selectbox("📊 Estilo", ["Scalping","Day Trading","Swing"],
                                 index=["Scalping","Day Trading","Swing"].index(st.session_state.trade_style))
    if nuevo_estilo != st.session_state.trade_style:
        st.session_state.trade_style = nuevo_estilo
        sv = gh_load(st.session_state.par); sv['trade_style'] = nuevo_estilo; gh_save(st.session_state.par, sv)

    risk_pct = st.slider("⚠️ Riesgo/trade (%)", 0.5, 5.0, 1.0, 0.5)
    st.markdown("---")

    if st.session_state.par == "EUR/USD 💶":
        st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">💶 INDICADORES EUR/USD</div>', unsafe_allow_html=True)
        st.caption("Este par usa ADX (fuerza de tendencia) y correlación con el índice del dólar (DXY) como features extra — no aplica al oro.")
        st.markdown("---")

    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📱 TELEGRAM COMANDOS</div>', unsafe_allow_html=True)
    st.caption("entré · no · salgo · estado · señal · me quedo")
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📚 GUÍA</div>', unsafe_allow_html=True)
    guia_items = [
        ("BOS / MSB","BOS=estructura confirmada en dirección. MSB=reversión de estructura. Base principal del análisis."),
        ("Order Block (OB)","Última vela institucional antes de impulso. Zona de reacción de alta probabilidad."),
        ("EQH / EQL","Equal Highs/Lows = pools de liquidez. El precio va a buscar esos stops antes de moverse."),
        ("Liquidity Swings","Máximos y mínimos relevantes donde hay órdenes acumuladas."),
        ("FVG","Fair Value Gap: desequilibrio de precio. El mercado suele regresar a llenarlo."),
        ("ADX","Fuerza de tendencia. >20 = tendencia real. <20 = rango sin dirección clara."),
        ("Modo Oráculo","Requiere BOS + OB confirmados (y ADX en EUR/USD). Señales escasas pero de alta calidad."),
        ("Modo Gladiador","Rebotes en OB, FVG, EQH/EQL. Más trades, SL ajustado, mayor riesgo."),
        ("Scalping","M5. Killzone NY Open. Máxima concentración requerida."),
        ("Day Trading","M15-H1. Cierra antes de 5PM MX."),
        ("Swing","H4-D1. Paciencia de días."),
    ]
    for titulo, texto in guia_items:
        with st.expander(titulo): st.write(texto)

T = THEMES.get(st.session_state.tema, THEMES["Mármol Griego"])
PAR = st.session_state.par
PC  = PAIRS[PAR]
MC  = PC['modos'][st.session_state.modo]
SC  = STYLE_CONFIG[st.session_state.trade_style]
CONTRACT = PC['contract_size']

# ── HEADER ────────────────────────────────────────────────────────
modo_color = '#C8A96E' if 'Oráculo' in st.session_state.modo else '#CD7F32'
st.markdown(f"""
<div class="greek-orn">─────── ✦ ───────</div>
<div class="mimi-title">MIMI · AI</div>
<div class="mimi-sub">{PAR} · ML · SMC · ICT · BOS · OB · EQH/EQL</div>
<div style="text-align:center;margin:6px 0;">
  <span class="modo-badge" style="background:{modo_color}22;border:1px solid {modo_color}66;color:{modo_color};">{st.session_state.modo}</span>
  <span class="modo-badge" style="background:{T['primary']}11;border:1px solid {T['primary']}44;color:{T['primary']}99;margin-left:8px;">{st.session_state.trade_style.upper()} · {SC['label']}</span>
</div>
<div class="greek-orn" style="margin-top:6px;">─────── ✦ ───────</div>
""", unsafe_allow_html=True)

# ── DATA & MODEL ──────────────────────────────────────────────────
INTERVALS = {"M5":"5m","M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}
PERIODS   = {"M5":"5d","M15":"10d","M30":"20d","H1":"60d","H4":"180d","D1":"2y"}

@st.cache_data(ttl=300)
def get_data(yf_symbol, interval="1d", period="2y"):
    df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
    df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df) >= 50: return df
    df2 = yf.download(yf_symbol, period="2y", interval="1d", progress=False)
    df2.columns = [c[0] if isinstance(c,tuple) else c for c in df2.columns]
    df2.dropna(inplace=True)
    return df2 if len(df2) >= 50 else None

@st.cache_data(ttl=1800)
def get_dxy_returns(period="2y"):
    """Retorno diario del índice dólar — solo se usa como feature para EUR/USD."""
    try:
        d = yf.download("DX-Y.NYB", period=period, interval="1d", progress=False)
        d.columns = [c[0] if isinstance(c,tuple) else c for c in d.columns]
        d.dropna(inplace=True)
        ret = d['Close'].pct_change(1)
        ret.index = pd.to_datetime(ret.index).tz_localize(None)
        return ret
    except Exception:
        return None

@st.cache_data(ttl=300)
def add_ind(df_json, usar_dxy=False):
    df = pd.read_json(io.StringIO(df_json), orient='split')
    df['EMA_20']    = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50']    = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200']   = ta.trend.ema_indicator(df['Close'], window=200)
    df['RSI']       = ta.momentum.rsi(df['Close'], window=14)
    df['MACD']      = ta.trend.macd(df['Close'])
    df['MACD_hist'] = ta.trend.macd_diff(df['Close'])
    df['BB_upper']  = ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower']  = ta.volatility.bollinger_lband(df['Close'])
    df['BB_width']  = (df['BB_upper'] - df['BB_lower']) / df['Close']
    df['ATR']       = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Stoch_K']   = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume']) if 'Volume' in df.columns else 0
    df['ADX']       = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['Dist_EMA20']  = (df['Close'] - df['EMA_20'])  / df['Close'] * 100
    df['Dist_EMA50']  = (df['Close'] - df['EMA_50'])  / df['Close'] * 100
    df['Dist_EMA200'] = (df['Close'] - df['EMA_200']) / df['Close'] * 100
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)

    if usar_dxy:
        dxy = get_dxy_returns()
        if dxy is not None:
            idx = pd.to_datetime(df.index).tz_localize(None)
            dxy_aligned = dxy.reindex(idx, method='ffill')
            df['DXY_Return_1d'] = dxy_aligned.values

    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=600)
def train_model(df_json, umbral, usar_dxy=False):
    df = pd.read_json(io.StringIO(df_json), orient='split')
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    df['Target'] = 0
    df.loc[df['Future_Return'] >  umbral, 'Target'] =  1
    df.loc[df['Future_Return'] < -umbral, 'Target'] = -1
    df.dropna(inplace=True)
    feats = ['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K','ADX',
             'Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    if usar_dxy and 'DXY_Return_1d' in df.columns:
        feats.append('DXY_Return_1d')
    feats = [f for f in feats if f in df.columns]
    X, y = df[feats], df['Target']
    sc   = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.2, random_state=42, shuffle=False)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1); rf.fit(Xtr, ytr)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42); gb.fit(Xtr, ytr)
    m  = rf if accuracy_score(yte, rf.predict(Xte)) >= accuracy_score(yte, gb.predict(Xte)) else gb
    return m, sc, feats, df

@st.cache_data(ttl=600)
def mtf_conf(yf_symbol):
    sigs = {}
    for name, iv, per in [("D1","1d","2y"),("H4","4h","180d"),("H1","1h","60d"),("M15","15m","10d")]:
        try:
            df = yf.download(yf_symbol, period=per, interval=iv, progress=False)
            df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if len(df) < 30: continue
            e20  = ta.trend.ema_indicator(df['Close'], window=20)
            e50  = ta.trend.ema_indicator(df['Close'], window=50)
            rsi  = ta.momentum.rsi(df['Close'], window=14)
            macd = ta.trend.macd(df['Close']); md = ta.trend.macd_diff(df['Close'])
            p, em20, em50 = float(df['Close'].iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1])
            r = float(rsi.iloc[-1]); m = float(macd.iloc[-1]) if macd is not None else 0
            mh = float(md.iloc[-1]) if md is not None else 0
            score = sum([p>em20, p>em50, em20>em50, r>50, m>0 and mh>0])
            sigs[name] = {'score':score, 'bias':'LONG' if score>=3 else 'SHORT' if score<=1 else 'NEUTRAL', 'rsi':r, 'precio':p}
        except: pass
    if not sigs: return sigs, 'NEUTRAL', 50
    total = sum(s['score'] for s in sigs.values())
    pct   = total / (len(sigs)*5) * 100
    return sigs, 'LONG' if pct>=60 else 'SHORT' if pct<=40 else 'NEUTRAL', pct

# ── LOAD DATA ─────────────────────────────────────────────────────
USAR_DXY = PC['dxy_symbol'] is not None

with st.spinner("🏛️ El Oráculo consulta los astros..."):
    raw = get_data(PC['yf_symbol'], SC['interval'], SC['period'])
    if raw is None: raw = get_data(PC['yf_symbol'], "1d","2y")

if raw is None:
    st.warning("⚠️ Mercado cerrado o sin datos — intenta más tarde."); st.stop()

df  = add_ind(raw.to_json(orient='split'), USAR_DXY)
m_model, sc_model, features, df_trained = train_model(df.to_json(orient='split'), MC['umbral'], USAR_DXY)
smc = detect_smc_advanced(df, lookback=50)

precio = float(df['Close'].iloc[-1])
rsi    = float(df['RSI'].iloc[-1])
atr    = float(df['ATR'].iloc[-1])
adx    = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0
bb_up  = float(df['BB_upper'].iloc[-1])
bb_low = float(df['BB_lower'].iloc[-1])
ema20  = float(df['EMA_20'].iloc[-1])
ema50  = float(df['EMA_50'].iloc[-1])

if 'Gladiador' in st.session_state.modo:
    pred, prob, _, _, sl_long, tp_long, sl_short, tp_short = get_signal_gladiador(df_trained, smc, features, m_model, sc_model, MC['atr_sl'], MC['atr_tp'], MC.get('require_adx',False))
else:
    pred, prob, _, _, sl_long, tp_long, sl_short, tp_short = get_signal_oraculo(df_trained, smc, features, m_model, sc_model, MC['atr_sl'], MC['atr_tp'], MC.get('require_adx',False))

rr   = round(MC['atr_tp'] / MC['atr_sl'], 2)
conf = max(prob) * 100
ET   = {1:"LONG — ASCENSO", 0:"LATERAL — ESPERA", -1:"SHORT — DESCENSO"}
p_long  = round(float(prob[2] if len(prob)==3 else prob[1])*100, 1)
p_short = round(float(prob[0])*100, 1)
p_lat   = round(max(0, 100-p_long-p_short-5), 1)
p_shock = round(100-p_long-p_short-p_lat, 1)
mx_tz   = pytz.timezone('America/Mexico_City')
ahora   = datetime.now(mx_tz); h = ahora.hour

# ── PROCESS TELEGRAM ─────────────────────────────────────────────
process_tg_updates(PAR, precio, pred, prob, rsi, atr, sl_long, tp_long, sl_short, tp_short, rr, smc, conf, ET, risk_pct, CONTRACT)

# ── AUTO PAPER TRADE RESULT ───────────────────────────────────────
for t in st.session_state.paper_trades:
    if t['estado'] == 'ABIERTO':
        if 'LONG' in t['dir']:
            if precio >= t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['tp']-t['entrada'])*t['lotes']*CONTRACT,2); t['resultado']='WIN ✅'
                send_tg(f"🏛️ *TP ALCANZADO* 🟢 — {PAR}\nLONG cerrado @ {pf(precio,PAR)}\nP&L: +${t['pnl']:.2f}\n\n_El sabio toma sus ganancias._")
            elif precio <= t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['sl']-t['entrada'])*t['lotes']*CONTRACT,2); t['resultado']='LOSS ❌'
                send_tg(f"🏛️ *SL ALCANZADO* 🔴 — {PAR}\nLONG cerrado @ {pf(precio,PAR)}\nP&L: ${t['pnl']:.2f}\n\n_Una pérdida aceptada es una victoria de carácter._")
        elif 'SHORT' in t['dir']:
            if precio <= t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['tp'])*t['lotes']*CONTRACT,2); t['resultado']='WIN ✅'
                send_tg(f"🏛️ *TP ALCANZADO* 🟢 — {PAR}\nSHORT cerrado @ {pf(precio,PAR)}\nP&L: +${t['pnl']:.2f}")
            elif precio >= t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['sl'])*t['lotes']*CONTRACT,2); t['resultado']='LOSS ❌'
                send_tg(f"🏛️ *SL ALCANZADO* 🔴 — {PAR}\nSHORT cerrado @ {pf(precio,PAR)}\nP&L: ${t['pnl']:.2f}")

cap = 1000.0 + sum(t.get('pnl',0) for t in st.session_state.paper_trades if t['estado']=='CERRADO')
st.session_state.capital = round(cap, 2)

if not st.session_state.signal_history or st.session_state.signal_history[-1].get('precio') != precio:
    sl_r = sl_long if pred>=0 else sl_short; tp_r = tp_long if pred>=0 else tp_short
    st.session_state.signal_history.append({
        'id':len(st.session_state.signal_history)+1,'fecha':ahora.strftime('%d/%m %H:%M'),'par':PAR,
        'modo':st.session_state.modo,'estilo':st.session_state.trade_style,
        'direccion':ET.get(pred),'confianza':f"{conf:.1f}%",'precio':precio,
        'sl':sl_r,'tp':tp_r,'rsi':round(rsi,1),'adx':round(adx,1),'smc':smc['bias'],'resultado':'PENDIENTE'})

sv2 = {'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history[-50:],
       'capital':st.session_state.capital,'trade_style':st.session_state.trade_style,
       'modo':st.session_state.modo,'last_tg_update':st.session_state.last_tg_update}
gh_save(PAR, sv2)

# ── BANNERS ───────────────────────────────────────────────────────
bc  = '#4CAF82' if pred==1 else '#C0392B' if pred==-1 else T['primary']
ge  = smc.get('gladiador_entry','')
ge_str = f"  ⚔️ MICRO: {ge.replace('_',' ')}  ·" if ge and 'Gladiador' in st.session_state.modo else ""
adx_str = f"  ·  ADX: {adx:.1f}" if USAR_DXY else ""
b1  = (f"  {PAR}  ·  {st.session_state.modo}  ·  SEÑAL: {ET.get(pred)}  ·  CONF: {conf:.1f}%  ·  {pf(precio,PAR)}  ·  SL: {pf(sl_long,PAR)}  ·  TP: {pf(tp_long,PAR)}  ·  R:R 1:{rr}  ·  ATR: {atr:.5f}{adx_str}{ge_str}  ")*2
b2  = (f"  RSI: {rsi:.1f}  ·  EMA20: {pf(ema20,PAR)}  ·  EMA50: {pf(ema50,PAR)}  ·  SMC: {smc['bias']}  ·  BOS: {len(smc['bos'])}  ·  OB: {len(smc['order_blocks'])}  ·  FVG: {len(smc['fvg'])}  ·  EQH: {len(smc['eqh'])}  ·  EQL: {len(smc['eql'])}  ")*2
st.markdown(f"""
<div class="ticker-wrap">
  <span class="ticker-label">ORACLE</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s1" style="color:{bc};font-family:'Philosopher',serif;font-size:.85em;">{b1}</div>
  </div>
</div>
<div class="ticker-wrap">
  <span class="ticker-label">SMC</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s2" style="color:{T['primary']}99;font-family:'Philosopher',serif;font-size:.82em;">{b2}</div>
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
    Modo: {st.session_state.modo} · {st.session_state.trade_style} · Auto-refresh 3s · {tick['fuente'] if tick else '—'}
  </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("💰 Precio Vivo", pf(precio_display,PAR), f"{pf(cambio_display,PAR)} ({cambio_pct_display:+.3f}%)")
c2.metric("📊 RSI",f"{rsi:.1f}","Sobrecomprado" if rsi>70 else "Sobrevendido" if rsi<30 else "Normal")
c3.metric("⚡ ADX" if USAR_DXY else "⚡ ATR", f"{adx:.1f}" if USAR_DXY else f"{atr:.5f}", "Tendencia" if adx>20 else "Rango" if USAR_DXY else None)
c4.metric("🎯 Señal","LONG" if pred==1 else "SHORT" if pred==-1 else "LATERAL",f"{conf:.1f}%")
c5.metric("📐 R:R",f"1:{rr}")
c6.metric("💰 Capital",f"${st.session_state.capital:,.2f}")
st.markdown('<div class="greek-orn">── ✦ ──</div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
    "🎯 Señal","🏛️ SMC·ICT","🌐 Multi·TF","📋 Paper","📊 Gráfica",
    "💬 Chat","📈 Backtest","🔔 Alertas","📜 Historial","👁️ Monitor"])

# ── TAB 1: SEÑAL ──────────────────────────────────────────────────
with tab1:
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="card"><div class="card-title">SEÑAL DEL ORÁCULO</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{"sig-long" if pred==1 else "sig-short" if pred==-1 else "sig-neu"}">{ET.get(pred)}</div>', unsafe_allow_html=True)

        if 'Gladiador' in st.session_state.modo and smc.get('gladiador_entry'):
            ge_label = smc['gladiador_entry'].replace('_',' ')
            ge_color = '#4CAF82' if 'LONG' in smc['gladiador_entry'] else '#C0392B'
            st.markdown(f'<span style="font-family:Cinzel,serif;font-size:.8em;color:{ge_color};letter-spacing:2px;">⚔️ MICRO ENTRADA: {ge_label}</span>', unsafe_allow_html=True)

        sl_r = sl_long if pred>=0 else sl_short
        tp_r = tp_long if pred>=0 else tp_short
        st.markdown(f"**Par:** {PAR}  |  **Modo:** {st.session_state.modo}  |  **Precio:** {pf(precio,PAR)}")
        if USAR_DXY:
            st.markdown(f"**ADX:** {adx:.1f} {'✅ tendencia' if adx>20 else '⚠️ rango sin fuerza'}")
        if pred != 0:
            st.markdown(f"**Stop Loss 🔴:** {pf(sl_r,PAR)}")
            st.markdown(f"**Take Profit 🟢:** {pf(tp_r,PAR)}")
        else:
            if smc.get('gladiador_entry') and 'Gladiador' in st.session_state.modo:
                st.markdown(f"**Micro zona activa:** {smc['gladiador_entry'].replace('_',' ')}")
            else:
                st.markdown(f"▲ LONG si rompe {pf(bb_up,PAR)}")
                st.markdown(f"▼ SHORT si rompe {pf(bb_low,PAR)}")
        lot, risg = calc_pos(st.session_state.capital, risk_pct, precio, sl_r if sl_r else precio-atr, CONTRACT)
        st.markdown(f"**Lotes:** {lot}  |  **Riesgo:** ${risg:.2f}  |  **R:R:** 1:{rr}")
        st.markdown('</div>', unsafe_allow_html=True)

        c_si, c_no = st.columns(2)
        if c_si.button("✅ ENTRO", use_container_width=True):
            ot = [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
            if not ot:
                dir_str = 'LONG 📈' if pred==1 else 'SHORT 📉' if pred==-1 else ('LONG 📈' if smc.get('gladiador_entry','').startswith('LONG') else 'SHORT 📉')
                actual_pred = pred if pred != 0 else (1 if 'LONG' in dir_str else -1)
                sl_r2 = sl_long if actual_pred==1 else sl_short
                tp_r2 = tp_long if actual_pred==1 else tp_short
                lot2, risg2 = calc_pos(st.session_state.capital, risk_pct, precio, sl_r2, CONTRACT)
                st.session_state.paper_trades.append({
                    'id':len(st.session_state.paper_trades)+1,'par':PAR,'dir':dir_str,
                    'entrada':precio,'sl':sl_r2,'tp':tp_r2,'lotes':lot2,'riesgo':risg2,
                    'estado':'ABIERTO','fecha':ahora.strftime('%d/%m %H:%M'),'resultado':'PENDIENTE','pnl':0})
                send_tg(f"🏛️ *Trade abierto — {PAR} · {st.session_state.modo}*\n{dir_str} @ {pf(precio,PAR)}\nSL: {pf(sl_r2,PAR)} | TP: {pf(tp_r2,PAR)}\nLotes: {lot2}\n\nEscríbeme 'estado' para monitorear o 'salgo' para cerrar.")
                gh_save(PAR, {**sv2,'paper_trades':st.session_state.paper_trades}); st.success("Trade registrado ✅"); st.rerun()
            else: st.warning("Ya tienes un trade abierto en este par. Escríbele 'estado' al bot.")
        if c_no.button("❌ NO ENTRO", use_container_width=True):
            send_tg(f"🏛️ Señal rechazada — {PAR}: {ET.get(pred)} @ {pf(precio,PAR)}")
            st.info("Rechazada")

    with cb:
        st.markdown('<div class="card"><div class="card-title">VENTANAS · HORA MX</div>', unsafe_allow_html=True)
        st.markdown(f"**Hora:** {ahora.strftime('%H:%M')}")
        for n, ini, fin, cal in [("London Open",3,5,"Alta"),("London+NY",8,11,"Máxima ✦"),("NY Tarde",12,14,"Media"),("NY Cierre",15,17,"Baja")]:
            st.markdown(f"{'🟢' if ini<=h<fin else '⚫'} **{ini:02d}–{fin:02d}** {n} [{cal}]")
        if PAR == "EUR/USD 💶":
            st.caption("Para EUR/USD, el traslape Londres+NY (08–11 MX) suele tener el mejor volumen y respeto de niveles.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">VARIANTES</div>', unsafe_allow_html=True)
        v1,v2,v3,v4 = st.columns(4)
        v1.metric("📈",f"{p_long}%"); v2.metric("📉",f"{p_short}%")
        v3.metric("➡️",f"{p_lat}%"); v4.metric("⚡",f"{p_shock}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: SMC ADVANCED ───────────────────────────────────────────
with tab2:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};font-size:.85em;letter-spacing:3px;margin-bottom:12px;">ANÁLISIS SMC — {PAR} — BIAS: {"📈 ALCISTA" if smc["bias"]=="ALCISTA" else "📉 BAJISTA" if smc["bias"]=="BAJISTA" else "➡️ NEUTRAL"} (score: {smc["bias_score"]})</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:
        st.markdown('<div class="card"><div class="card-title">BOS · MSB · ESTRUCTURA</div>', unsafe_allow_html=True)
        if smc['bos']:
            for b in smc['bos'][-3:]:
                color = 'smc-strong' if 'ALCISTA' in b['tipo'] else 'smc-bear'
                st.markdown(f'<span class="{color}">● {b["tipo"]}</span> — {pf(b["nivel"],PAR)} <span style="color:{T["primary"]}77;font-size:.85em;">[{b["fuerza"]}]</span>', unsafe_allow_html=True)
        else: st.markdown("Sin BOS confirmado")
        if smc['msb']:
            for m2 in smc['msb'][-2:]:
                color = 'smc-strong' if 'ALCISTA' in m2['tipo'] else 'smc-bear'
                st.markdown(f'<span class="{color}">◆ {m2["tipo"]}</span> — {pf(m2["nivel"],PAR)} <span style="color:{T["primary"]}77;font-size:.85em;">[{m2["fuerza"]}]</span>', unsafe_allow_html=True)
        else: st.markdown("Sin MSB detectado")
        if USAR_DXY:
            st.markdown(f'<div style="margin-top:10px;color:{T["primary"]}99;font-size:.85em;">ADX actual: <b>{adx:.1f}</b> {"— tendencia con fuerza ✅" if adx>20 else "— rango, sin fuerza ⚠️"}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">EQH · EQL · LIQUIDEZ</div>', unsafe_allow_html=True)
        if smc['eqh']:
            for e in smc['eqh'][-3:]:
                st.markdown(f'<span class="smc-bear">▲ EQH</span> — {pf(e["nivel"],PAR)} <span style="color:{T["primary"]}77;font-size:.8em;">stops arriba</span>', unsafe_allow_html=True)
        else: st.markdown("Sin EQH detectados")
        if smc['eql']:
            for e in smc['eql'][-3:]:
                st.markdown(f'<span class="smc-strong">▼ EQL</span> — {pf(e["nivel"],PAR)} <span style="color:{T["primary"]}77;font-size:.8em;">stops abajo</span>', unsafe_allow_html=True)
        else: st.markdown("Sin EQL detectados")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="card"><div class="card-title">ORDER BLOCKS</div>', unsafe_allow_html=True)
        if smc['order_blocks']:
            for ob in smc['order_blocks'][-4:]:
                color = 'smc-strong' if 'ALCISTA' in ob['tipo'] else 'smc-bear'
                near  = abs(precio - ob['mid']) / precio < 0.003
                near_str = ' ⚡ PRECIO CERCA' if near else ''
                st.markdown(f'<span class="{color}">■ {ob["tipo"]}</span> [{ob["fuerza"]}]{near_str}<br><span style="color:{T["primary"]}99;font-size:.85em;">  {pf(ob["bottom"],PAR)} — {pf(ob["top"],PAR)} | Mid: {pf(ob["mid"],PAR)}</span>', unsafe_allow_html=True)
        else: st.markdown("Sin OB detectados")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">FVG · SWING HIGHS/LOWS</div>', unsafe_allow_html=True)
        if smc['fvg']:
            for fv in smc['fvg'][-3:]:
                color = 'smc-strong' if 'ALCISTA' in fv['tipo'] else 'smc-bear'
                st.markdown(f'<span class="{color}">◇ {fv["tipo"]}</span> {pf(fv["bottom"],PAR)}–{pf(fv["top"],PAR)}', unsafe_allow_html=True)
        else: st.markdown("Sin FVG")
        if smc['liquidity_swings']:
            for ls in smc['liquidity_swings'][-4:]:
                color = 'smc-bear' if 'HIGH' in ls['tipo'] else 'smc-strong'
                st.markdown(f'<span class="{color}">○ {ls["tipo"]}</span> — {pf(ls["nivel"],PAR)}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if 'Gladiador' in st.session_state.modo:
        if smc.get('gladiador_entry'):
            ge = smc['gladiador_entry'].replace('_',' ')
            gc = '#4CAF82' if 'LONG' in smc['gladiador_entry'] else '#C0392B'
            st.markdown(f'<div style="border:1px solid {gc}44;background:{gc}11;padding:12px;border-radius:3px;font-family:Cinzel,serif;color:{gc};font-size:.85em;letter-spacing:2px;">⚔️ GLADIADOR — MICRO ENTRADA ACTIVA: {ge}</div>', unsafe_allow_html=True)
        else:
            st.info("⚔️ Gladiador: Sin zona micro activa ahorita.")

    st.markdown('<div class="card"><div class="card-title">ICT KILLZONES · HORA MX</div>', unsafe_allow_html=True)
    for n,ini,fin,desc in [("Asian Range",19,23,"Acumulación de liquidez"),("London Open",3,5,"Barrido liquidez asiática"),("NY Open",8,11,"✦ Mayor volatilidad — mejor R:R"),("London Close",10,12,"Reversales frecuentes"),("NY PM",13,15,"Continuación o reversión AM")]:
        st.markdown(f"{'🟢' if ini<=h<fin else '⚪'} **{ini:02d}–{fin:02d} {n}** — {desc}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: MULTI-TF ───────────────────────────────────────────────
with tab3:
    with st.spinner("Analizando timeframes..."):
        mtf_s, mtf_b, mtf_p = mtf_conf(PC['yf_symbol'])
    m1,m2,m3 = st.columns(3)
    m1.metric("Bias",f"{'📈 LONG' if mtf_b=='LONG' else '📉 SHORT' if mtf_b=='SHORT' else '➡️ NEUTRAL'}")
    m2.metric("Confluencia",f"{mtf_p:.0f}%"); m3.metric("TFs analizados",str(len(mtf_s)))
    for tfn,data in mtf_s.items():
        bc2 = "🟢" if data['bias']=='LONG' else "🔴" if data['bias']=='SHORT' else "🟡"
        st.markdown(f"{bc2} **{tfn}** — {data['bias']} | {'█'*data['score']}{'░'*(5-data['score'])} {data['score']}/5 | RSI:{data['rsi']:.1f} | {pf(data['precio'],PAR)}")
    if mtf_p>=60: st.success(f"✅ Confluencia fuerte: {mtf_b} ({mtf_p:.0f}%)")
    elif mtf_p<=40: st.error(f"🔴 Confluencia bajista ({mtf_p:.0f}%)")
    else: st.warning("⚠️ Sin confluencia clara — mercado indeciso")

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
            send_tg(f"🏛️ Cerrado manualmente ({PAR}) — P&L: {'+'if pnl>0 else ''}${pnl:.2f}")
            gh_save(PAR, {**sv2,'paper_trades':st.session_state.paper_trades,'capital':st.session_state.capital}); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Sin trade abierto en este par. Usa '✅ ENTRO' en Señal o escríbele 'entré' al bot de Telegram.")
    if ct_c:
        st.dataframe(pd.DataFrame(ct_c)[['fecha','dir','entrada','sl','tp','lotes','pnl','resultado']], use_container_width=True)
    if st.button("🗑️ Reiniciar paper trading"):
        st.session_state.paper_trades=[]; st.session_state.capital=1000.0
        gh_save(PAR, {**sv2,'paper_trades':[],'capital':1000.0}); st.rerun()

# ── TAB 5: GRÁFICA ────────────────────────────────────────────────
with tab5:
    g1,g2,g3 = st.columns(3)
    tf_h = g1.selectbox("TF Histórico",list(INTERVALS.keys()),index=5,key="tfh")
    ct_g = g2.selectbox("Tipo",["Velas 🕯️","Línea 📈"],key="ct")
    tf_l = g3.selectbox("TF En Vivo",list(INTERVALS.keys()),index=1,key="tfl")

    dfc = get_data(PC['yf_symbol'], INTERVALS[tf_h], PERIODS[tf_h])
    if dfc is not None:
        dfc = add_ind(dfc.to_json(orient='split'), USAR_DXY); dp = dfc.tail(120)
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25])
        if "Velas" in ct_g:
            fig.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
                increasing_line_color='#4CAF82',decreasing_line_color='#C0392B',name=PAR),row=1,col=1)
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
        fig.add_trace(go.Scatter(x=dp.index,y=dp['RSI'],line=dict(color=T['primary'],width=1.5),name="RSI"),row=2,col=1)
        fig.add_hline(y=70,line_color='#C0392B',line_dash='dot',row=2,col=1)
        fig.add_hline(y=30,line_color='#4CAF82',line_dash='dot',row=2,col=1)
        fig.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            xaxis_rangeslider_visible=False,height=500,margin=dict(l=0,r=0,t=20,b=0),
            legend=dict(bgcolor='#000',bordercolor='#222',orientation='h'))
        fig.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fig.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Zonas verdes = OB Alcista | Zonas rojas = OB Bajista | Líneas punteadas = EQH/EQL")

    dfl = get_data(PC['yf_symbol'], INTERVALS[tf_l], PERIODS[tf_l])
    if dfl is not None:
        dlp = dfl.tail(80); fig2 = go.Figure()
        if "Velas" in ct_g:
            fig2.add_trace(go.Candlestick(x=dlp.index,open=dlp['Open'],high=dlp['High'],low=dlp['Low'],close=dlp['Close'],
                increasing_line_color='#4CAF82',decreasing_line_color='#C0392B'))
        else:
            fig2.add_trace(go.Scatter(x=dlp.index,y=dlp['Close'],line=dict(color=T['primary'],width=2)))
        fig2.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888'),
            xaxis_rangeslider_visible=False,height=280,margin=dict(l=0,r=0,t=10,b=0))
        fig2.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fig2.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fig2,use_container_width=True)
        u = dfl.iloc[-1]; lv1,lv2,lv3 = st.columns(3)
        lv1.metric("Último",pf(float(u['Close']),PAR)); lv2.metric("Máximo",pf(float(u['High']),PAR)); lv3.metric("Mínimo",pf(float(u['Low']),PAR))

# ── TAB 6: CHAT ───────────────────────────────────────────────────
with tab6:
    st.caption("💡 Este chat también funciona por Telegram. Escríbele directamente al bot: entré · no · salgo · estado · señal · me quedo · o cualquier pregunta")
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg['role']=='user' else "assistant"):
            st.markdown(msg['content'])
    uin = st.chat_input("Consulta al Oráculo... o escríbele al bot de Telegram")
    if uin:
        resp = analizar_texto_libre(uin, PAR, precio, pred, prob, rsi, atr, smc, ET, conf, sl_long, tp_long, sl_short, tp_short, rr)
        st.session_state.chat_history.append({'role':'user','content':uin})
        st.session_state.chat_history.append({'role':'mimi','content':resp})
        st.rerun()

# ── TAB 7: BACKTEST ───────────────────────────────────────────────
with tab7:
    @st.cache_data(ttl=3600)
    def backtest(df_json, asl, atp):
        df_b = pd.read_json(io.StringIO(df_json), orient='split')
        cap=1000.0; eq=[cap]; tds=[]
        ac=ta.volatility.average_true_range(df_b['High'],df_b['Low'],df_b['Close'])
        rc=ta.momentum.rsi(df_b['Close'],window=14); mc=ta.trend.macd_diff(df_b['Close'])
        e2=ta.trend.ema_indicator(df_b['Close'],window=20); e5=ta.trend.ema_indicator(df_b['Close'],window=50)
        i=50
        while i<len(df_b)-5:
            p=float(df_b['Close'].iloc[i]); atr_v=float(ac.iloc[i]) if not pd.isna(ac.iloc[i]) else p*0.001
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
        bt_t,bt_e,bt_w,bt_p = backtest(df.to_json(orient='split'), MC['atr_sl'], MC['atr_tp'])
    bm1,bm2,bm3,bm4 = st.columns(4)
    bm1.metric("Capital Inicial","$1,000"); bm2.metric("Capital Final",f"${1000+bt_p:,.2f}",f"{bt_p:+.2f}")
    bm3.metric("Win Rate",f"{bt_w:.1f}%"); bm4.metric("Trades",str(len(bt_t)))
    if bt_e:
        fe=go.Figure()
        fe.add_trace(go.Scatter(y=bt_e,fill='tozeroy',fillcolor='rgba(200,169,110,0.08)',line=dict(color=T['primary'],width=2)))
        fe.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            height=300,margin=dict(l=0,r=0,t=20,b=0),title=dict(text=f"CURVA DE CAPITAL — {PAR} · {st.session_state.modo}",font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        fe.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fe.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fe,use_container_width=True)
    if bt_t: st.dataframe(pd.DataFrame(bt_t[-20:]),use_container_width=True)

# ── TAB 8: ALERTAS ────────────────────────────────────────────────
with tab8:
    st.markdown('<div class="card-title" style="font-family:Cinzel,serif;color:#C8A96E;letter-spacing:3px;">ALERTAS · TELEGRAM BIDIRECCIONAL</div>', unsafe_allow_html=True)
    st.markdown(f"""
**El bot ya está activo y escuchando (par actual: {PAR}).** Escríbele a tu bot en Telegram:

| Comando | Acción |
|---------|--------|
| `entré` o `sí` | Registra el trade en paper trading |
| `no` | Rechaza la señal actual |
| `estado` | Ve tu posición actual con P&L |
| `me quedo` | MIMI-AI evalúa si mantener |
| `salgo` | Cierra el trade activo |
| `señal` | Pide la señal actual |
| Cualquier pregunta | MIMI-AI responde |
""")
    col_a, col_b = st.columns(2)
    with col_a:
        al = st.checkbox("📈 Alertar LONG", value=True)
        as_ = st.checkbox("📉 Alertar SHORT", value=True)
        ac2 = st.slider("Confianza mínima (%)", 30, 90, 50)
        aw  = st.checkbox("⭐ Solo ventana activa", value=True)
    with col_b:
        if st.button("🧪 Prueba"):
            ok=send_tg(f"🏛️ *MIMI-AI Test* ✅\nPar: {PAR}\nModo: {st.session_state.modo}\nPrecio: {pf(precio,PAR)}\nResponde 'señal' para ver la señal actual.")
            st.success("Enviado ✅") if ok else st.error("Error — revisa Secrets")
        if st.button("📡 Enviar señal ahora"):
            vens2=[(3,5),(8,11),(12,14),(15,17)]; ev=any(i<=h<f for i,f in vens2)
            if conf>=ac2 and ((pred==1 and al) or (pred==-1 and as_)) and ((not aw) or ev):
                sl_r=sl_long if pred>=0 else sl_short; tp_r=tp_long if pred>=0 else tp_short
                ge_msg = f"\n⚔️ Micro zona: {smc['gladiador_entry'].replace('_',' ')}" if smc.get('gladiador_entry') and 'Gladiador' in st.session_state.modo else ""
                ok2=send_tg(f"🏛️ *MIMI-AI — {PAR} · {st.session_state.modo}*\n🕐 {ahora.strftime('%H:%M')} MX · {st.session_state.trade_style}\n💰 {pf(precio,PAR)}\n🎯 *{ET.get(pred)}*\n📊 {conf:.1f}% | SMC: {smc['bias']}{ge_msg}\n🔴 SL: {pf(sl_r,PAR)}\n🟢 TP: {pf(tp_r,PAR)}\n📐 R:R: 1:{rr}\n\n_Responde 'entré' o 'no'_")
                st.success("Enviado ✅") if ok2 else st.error("Error")
            else: st.info("Condiciones no cumplidas")
        if st.button("👁️ Pedir estado al bot"):
            ot2=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
            if ot2:
                t2=ot2[0]; pnl2=(precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*CONTRACT
                send_tg(f"👁️ *Estado de posición — {PAR}*\n{t2['dir']} @ {pf(t2['entrada'],PAR)}\nActual: {pf(precio,PAR)}\nP&L: {'+'if pnl2>0 else ''}${pnl2:.2f}\nSL: {pf(t2['sl'],PAR)} | TP: {pf(t2['tp'],PAR)}\n\n{'🟢 Mantén si la estructura aguanta.' if pnl2>0 else '🔴 Evalúa si el motivo de entrada sigue válido.'}\n\nResponde 'me quedo' o 'salgo'")
                st.success("Estado enviado al bot")
            else: send_tg(f"📊 Sin posición abierta ({PAR}).\nSeñal: {ET.get(pred)} {conf:.1f}%\nCapital: ${st.session_state.capital:,.2f}"); st.info("Sin trade abierto")

# ── TAB 9: HISTORIAL ──────────────────────────────────────────────
with tab9:
    if st.session_state.signal_history:
        df_sh = pd.DataFrame(st.session_state.signal_history)
        st.markdown(f"**{len(df_sh)} señales registradas — {PAR}**")
        st.dataframe(df_sh, use_container_width=True)
        w_h=sum(1 for s in st.session_state.signal_history if 'WIN' in s.get('resultado',''))
        l_h=sum(1 for s in st.session_state.signal_history if 'LOSS' in s.get('resultado',''))
        if w_h+l_h>0: st.metric("Win Rate Real",f"{w_h/(w_h+l_h)*100:.1f}%",f"{w_h}W / {l_h}L")
        if st.button("🗑️ Limpiar"):
            st.session_state.signal_history=[]; gh_save(PAR, {**sv2,'signal_history':[]}); st.rerun()
    else: st.info("Las señales se guardan automáticamente al cargar la app.")

# ── TAB 10: MONITOR ───────────────────────────────────────────────
with tab10:
    ot_m=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot_m:
        t_m=ot_m[0]; pnl_m=(precio-t_m['entrada'])*(1 if 'LONG' in t_m['dir'] else -1)*t_m['lotes']*CONTRACT
        pct_m=pnl_m/t_m['entrada']*100 if t_m['entrada'] else 0
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
        mo3.metric("P&L",f"${pnl_m:.2f}",f"{pct_m:+.3f}%"); mo4.metric("Estado",est_m)
        mc1,mc2=st.columns(2)
        if mc1.button("🔄 Actualizar precio"): st.cache_data.clear(); st.rerun()
        if mc2.button("📱 Pedir evaluación al bot"):
            send_tg(f"👁️ *Monitor activo — {PAR}*\n{t_m['dir']} @ {pf(t_m['entrada'],PAR)}\nActual: {pf(precio,PAR)}\nP&L: {'+'if pnl_m>0 else ''}${pnl_m:.2f}\nSMC Bias: {smc['bias']}\n{est_m}\n\nResponde 'me quedo' o 'salgo'")
            st.success("Enviado al bot")
    else:
        st.info(f"Sin posición abierta en {PAR}.")

# ── FRASE FINAL ───────────────────────────────────────────────────
fr = random.choice(FRASES)
st.markdown(f"""
<div class="greek-orn" style="margin-top:24px;">─────── ✦ ───────</div>
<div class="stoic-q">{fr[1]}<div class="stoic-a">— {fr[0]}</div></div>
<div class="greek-orn">─────── ✦ ───────</div>
""", unsafe_allow_html=True)
