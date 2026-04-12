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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI", page_icon="⚔️", layout="wide")

THEMES = {
    "Dorado":     {"primary":"#FFD700","secondary":"#FFA500","bg":"#0d0900","card":"#1a1200"},
    "Neon Verde": {"primary":"#00FF88","secondary":"#00CC66","bg":"#001a0e","card":"#002a16"},
    "Cyan":       {"primary":"#00FFFF","secondary":"#00CCCC","bg":"#001a1a","card":"#002626"},
    "Magenta":    {"primary":"#FF00FF","secondary":"#CC00CC","bg":"#1a001a","card":"#2a002a"},
    "Lavanda":    {"primary":"#B57BFF","secondary":"#8A4FFF","bg":"#0d0019","card":"#1a0030"},
    "Rosa":       {"primary":"#FF69B4","secondary":"#FF1493","bg":"#1a0010","card":"#2a001a"},
    "Menta":      {"primary":"#98FF98","secondary":"#66CC66","bg":"#001a00","card":"#002a00"},
}

# ── CARGAR SECRETS ───────────────────────────────────────────────
try:
    _tg_token   = st.secrets["TG_TOKEN"]
    _tg_chat_id = st.secrets["TG_CHAT_ID"]
except: 
    _tg_token   = ''
    _tg_chat_id = ''

if 'paper_trades'   not in st.session_state: st.session_state.paper_trades   = []
if 'chat_history'   not in st.session_state: st.session_state.chat_history   = []
if 'account_size'   not in st.session_state: st.session_state.account_size   = 1000.0
if 'signal_history' not in st.session_state: st.session_state.signal_history  = []
if 'monitor_active' not in st.session_state: st.session_state.monitor_active  = False
if 'monitor_data'   not in st.session_state: st.session_state.monitor_data    = {}
if 'tg_token'       not in st.session_state: st.session_state.tg_token        = _tg_token
if 'tg_chat_id'     not in st.session_state: st.session_state.tg_chat_id      = _tg_chat_id

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Config")
    tema         = st.selectbox("🎨 Tema", list(THEMES.keys()))
    account_size = st.number_input("💰 Capital ($)", min_value=100.0, value=st.session_state.account_size, step=100.0)
    risk_pct     = st.slider("⚠️ Riesgo/trade (%)", 0.5, 5.0, 1.0, 0.5)
    st.session_state.account_size = account_size
    st.markdown("---")
    st.markdown("## 📚 Guía")
    with st.expander("¿Qué es MIMI-AI?"): st.write("Sistema de trading ML + SMC + ICT + Multi-TF para XAU/USD.")
    with st.expander("SL y TP"):          st.write("SL = dónde cierras si va en tu contra. TP = precio objetivo de ganancia.")
    with st.expander("RSI"):              st.write(">70 sobrecomprado. <30 sobrevendido. 30–70 zona neutral.")
    with st.expander("ATR"):              st.write("Volatilidad promedio. A mayor ATR, movimientos más grandes.")
    with st.expander("SMC / ICT"):        st.write("Smart Money Concepts: detecta dónde operan los institucionales.")
    with st.expander("Multi-TF"):         st.write("Confluencia de M15, H1, H4 y D1 para confirmar dirección.")
    with st.expander("Paper Trading"):    st.write("Simula operaciones sin dinero real para validar la IA.")
    with st.expander("Order Block (OB)"): st.write("Última vela de impulso fuerte antes de un movimiento. Zona de reacción.")
    with st.expander("BOS / CHoCH"):      st.write("Break of Structure = confirmación de tendencia. CHoCH = posible reversión.")
    with st.expander("FVG"):              st.write("Fair Value Gap: hueco entre velas. El precio suele regresar a llenarlo.")
    with st.expander("Killzones ICT"):    st.write("Ventanas de máxima manipulación institucional. NY Open 08–11 MX es la mejor.")

T = THEMES[tema]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600&display=swap');
*{{font-family:'Rajdhani',sans-serif;}}
h1,h2,h3{{font-family:'Orbitron',monospace!important;color:{T['primary']}!important;}}
.stApp{{background:{T['bg']}!important;}}
.mimi-title{{
  font-family:'Orbitron',monospace;font-size:clamp(1.5rem,5vw,3rem);font-weight:900;text-align:center;
  background:linear-gradient(90deg,{T['secondary']},{T['primary']},{T['secondary']});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-size:200%;
  animation:shimmer 3s infinite;filter:drop-shadow(0 0 20px {T['primary']}88);
}}
@keyframes shimmer{{0%,100%{{background-position:0%}}50%{{background-position:100%}}}}
.ticker-wrap{{background:#000;border:1px solid {T['primary']}44;overflow:hidden;padding:8px 0;margin:4px 0;border-radius:4px;}}
.ticker-label{{color:{T['primary']};font-family:'Orbitron',monospace;font-size:11px;padding:0 12px;
  display:inline-block;line-height:2;border-right:1px solid {T['primary']}44;margin-right:12px;vertical-align:top;}}
.t-scroll{{display:inline-block;white-space:nowrap;animation:scroll1 40s linear infinite;}}
.t-scroll2{{display:inline-block;white-space:nowrap;animation:scroll1 58s linear infinite;}}
@keyframes scroll1{{from{{transform:translateX(0)}}to{{transform:translateX(-50%)}}}}
.card{{background:{T['card']};border:1px solid {T['primary']}33;border-radius:8px;padding:16px;margin:8px 0;box-shadow:0 0 12px {T['primary']}11;}}
.sig-long{{color:#00FF88;font-weight:700;font-size:1.3em;}}
.sig-short{{color:#FF4444;font-weight:700;font-size:1.3em;}}
.sig-neu{{color:#FFB800;font-weight:700;font-size:1.3em;}}
.stoic{{border-left:3px solid {T['primary']};padding:12px 16px;margin:16px 0;background:{T['primary']}11;
  font-style:italic;color:{T['primary']};font-size:1.05em;}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mimi-title">⚔ MIMI-AI ⚔</div>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center;color:{T["primary"]}88;font-size:.85em;margin-top:-8px;">XAU/USD · ML + SMC + ICT + Multi-TF · Estoico · Preciso</p>', unsafe_allow_html=True)

INTERVALS = {"M5":"5m","M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}
PERIODS   = {"M5":"5d","M15":"10d","M30":"20d","H1":"60d","H4":"180d","D1":"2y"}

@st.cache_data(ttl=300)
def get_data(interval="1d", period="2y"):
    df = yf.download("GC=F", period=period, interval=interval, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df) >= 50: return df
    df2 = yf.download("GC=F", period="2y", interval="1d", progress=False)
    df2.columns = [c[0] if isinstance(c, tuple) else c for c in df2.columns]
    df2.dropna(inplace=True)
    return df2 if len(df2) >= 50 else None

@st.cache_data(ttl=300)
def add_indicators(df_json):
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
    df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Dist_EMA20']  = (df['Close'] - df['EMA_20'])  / df['Close'] * 100
    df['Dist_EMA50']  = (df['Close'] - df['EMA_50'])  / df['Close'] * 100
    df['Dist_EMA200'] = (df['Close'] - df['EMA_200']) / df['Close'] * 100
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=3600)
def get_signal():
    raw = get_data("1d","2y")
    if raw is None: return None
    df = add_indicators(raw.to_json(orient='split'))
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    df['Target'] = 0
    df.loc[df['Future_Return'] >  0.003,'Target'] =  1
    df.loc[df['Future_Return'] < -0.003,'Target'] = -1
    df.dropna(inplace=True)
    features = ['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K',
                'Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    features = [f for f in features if f in df.columns]
    X,y = df[features], df['Target']
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=.2,random_state=42,shuffle=False)
    rf = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1)
    rf.fit(Xtr,ytr)
    gb = GradientBoostingClassifier(n_estimators=200,max_depth=5,random_state=42)
    gb.fit(Xtr,ytr)
    m = rf if accuracy_score(yte,rf.predict(Xte))>=accuracy_score(yte,gb.predict(Xte)) else gb
    ul = df[features].iloc[-1:]
    pred = m.predict(sc.transform(ul))[0]
    prob = m.predict_proba(sc.transform(ul))[0]
    return dict(df=df,pred=int(pred),prob=prob.tolist(),
                precio=float(df['Close'].iloc[-1]),rsi=float(df['RSI'].iloc[-1]),
                atr=float(df['ATR'].iloc[-1]),bb_up=float(df['BB_upper'].iloc[-1]),
                bb_low=float(df['BB_lower'].iloc[-1]),macd=float(df['MACD'].iloc[-1]),
                macd_h=float(df['MACD_hist'].iloc[-1]),ema20=float(df['EMA_20'].iloc[-1]),
                ema50=float(df['EMA_50'].iloc[-1]),features=features)

# ── SMC / ICT ────────────────────────────────────────────────────
def detect_smc(df, lookback=30):
    res = {'order_blocks':[],'bos':[],'choch':[],'fvg':[],'liquidity':[],'bias':'NEUTRAL'}
    if len(df) < lookback+5: return res
    H,L,C,O = df['High'].values,df['Low'].values,df['Close'].values,df['Open'].values
    sh,sl = [],[]
    for i in range(2,len(df)-2):
        if H[i]>H[i-1] and H[i]>H[i-2] and H[i]>H[i+1] and H[i]>H[i+2]: sh.append((i,H[i]))
        if L[i]<L[i-1] and L[i]<L[i-2] and L[i]<L[i+1] and L[i]<L[i+2]: sl.append((i,L[i]))
    if len(sh)>=2:
        if sh[-1][1]>sh[-2][1]: res['bos'].append({'tipo':'BOS ALCISTA','nivel':sh[-1][1]}); res['bias']='ALCISTA'
        else: res['choch'].append({'tipo':'CHoCH BAJISTA','nivel':sh[-1][1]}); res['bias']='BAJISTA'
    if len(sl)>=2:
        if sl[-1][1]<sl[-2][1]: res['bos'].append({'tipo':'BOS BAJISTA','nivel':sl[-1][1]})
        else: res['choch'].append({'tipo':'CHoCH ALCISTA','nivel':sl[-1][1]})
    avg_b = np.mean([abs(C[j]-O[j]) for j in range(-lookback,-1)])
    for i in range(-lookback,-3):
        b = abs(C[i]-O[i])
        if b > avg_b*1.5:
            t = 'OB ALCISTA' if C[i]>O[i] else 'OB BAJISTA'
            res['order_blocks'].append({'tipo':t,'top':H[i],'bottom':L[i]})
    for i in range(-lookback,-2):
        if L[i+2]>H[i]: res['fvg'].append({'tipo':'FVG ALCISTA','top':L[i+2],'bottom':H[i]})
        elif H[i+2]<L[i]: res['fvg'].append({'tipo':'FVG BAJISTA','top':L[i],'bottom':H[i+2]})
    if len(sh)>=2:
        for i in range(len(sh)-1):
            if abs(sh[i][1]-sh[-1][1])/sh[-1][1]<0.002: res['liquidity'].append({'tipo':'EQH Liquidez','nivel':sh[i][1]})
    if len(sl)>=2:
        for i in range(len(sl)-1):
            if abs(sl[i][1]-sl[-1][1])/sl[-1][1]<0.002: res['liquidity'].append({'tipo':'EQL Liquidez','nivel':sl[i][1]})
    return res

@st.cache_data(ttl=600)
def mtf_confluence():
    tfs = [("D1","1d","2y"),("H4","4h","180d"),("H1","1h","60d"),("M15","15m","10d")]
    signals = {}
    for name,iv,per in tfs:
        try:
            df = yf.download("GC=F",period=per,interval=iv,progress=False)
            df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if len(df)<30: continue
            e20  = ta.trend.ema_indicator(df['Close'],window=20)
            e50  = ta.trend.ema_indicator(df['Close'],window=50)
            rsi  = ta.momentum.rsi(df['Close'],window=14)
            macd = ta.trend.macd(df['Close'])
            md   = ta.trend.macd_diff(df['Close'])
            p,em20,em50 = float(df['Close'].iloc[-1]),float(e20.iloc[-1]),float(e50.iloc[-1])
            r,m,mh = float(rsi.iloc[-1]),float(macd.iloc[-1]) if macd is not None else 0,float(md.iloc[-1]) if md is not None else 0
            score = sum([p>em20,p>em50,em20>em50,r>50,m>0 and mh>0])
            signals[name] = {'score':score,'bias':'LONG' if score>=3 else 'SHORT' if score<=1 else 'NEUTRAL','rsi':r,'precio':p}
        except: pass
    if not signals: return signals,'NEUTRAL',50
    total = sum(s['score'] for s in signals.values())
    pct   = total/(len(signals)*5)*100
    bias  = 'LONG' if pct>=60 else 'SHORT' if pct<=40 else 'NEUTRAL'
    return signals,bias,pct

@st.cache_data(ttl=3600)
def get_calendar():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",timeout=5)
        ev = r.json()
        return [e for e in ev if e.get('impact')=='High' and 'USD' in e.get('country','')][:4]
    except: return []

def calc_posicion(capital,riesgo_pct,entrada,sl):
    riesgo = capital*(riesgo_pct/100)
    dist   = abs(entrada-sl)
    if dist==0: return 0,0
    lotes  = riesgo/(dist*100)
    return round(lotes,2),round(riesgo,2)

# ── CARGAR DATOS ─────────────────────────────────────────────────
with st.spinner("⚔️ MIMI-AI iniciando..."):
    sd = get_signal()

if sd is None:
    st.warning("⚠️ Mercado cerrado — mostrando últimos datos disponibles. Reabre el domingo 6 PM MX.")
    # Usar datos históricos D1 como fallback
    import yfinance as _yf
    _df = _yf.download("GC=F", period="2y", interval="1d", progress=False)
    _df.columns = [c[0] if isinstance(c, tuple) else c for c in _df.columns]
    _df.dropna(inplace=True)
    _df['EMA_20']    = ta.trend.ema_indicator(_df['Close'], window=20)
    _df['EMA_50']    = ta.trend.ema_indicator(_df['Close'], window=50)
    _df['EMA_200']   = ta.trend.ema_indicator(_df['Close'], window=200)
    _df['RSI']       = ta.momentum.rsi(_df['Close'], window=14)
    _df['MACD']      = ta.trend.macd(_df['Close'])
    _df['MACD_hist'] = ta.trend.macd_diff(_df['Close'])
    _df['BB_upper']  = ta.volatility.bollinger_hband(_df['Close'])
    _df['BB_lower']  = ta.volatility.bollinger_lband(_df['Close'])
    _df['BB_width']  = (_df['BB_upper'] - _df['BB_lower']) / _df['Close']
    _df['ATR']       = ta.volatility.average_true_range(_df['High'], _df['Low'], _df['Close'])
    _df['Stoch_K']   = ta.momentum.stoch(_df['High'], _df['Low'], _df['Close'])
    _df['OBV']       = ta.volume.on_balance_volume(_df['Close'], _df['Volume'])
    _df['Dist_EMA20']  = (_df['Close'] - _df['EMA_20'])  / _df['Close'] * 100
    _df['Dist_EMA50']  = (_df['Close'] - _df['EMA_50'])  / _df['Close'] * 100
    _df['Dist_EMA200'] = (_df['Close'] - _df['EMA_200']) / _df['Close'] * 100
    _df['Return_1d'] = _df['Close'].pct_change(1)
    _df['Return_3d'] = _df['Close'].pct_change(3)
    _df['Return_5d'] = _df['Close'].pct_change(5)
    _df['Future_Return'] = _df['Close'].pct_change(5).shift(-5)
    _df['Target'] = 0
    _df.loc[_df['Future_Return'] >  0.003,'Target'] =  1
    _df.loc[_df['Future_Return'] < -0.003,'Target'] = -1
    _df.dropna(inplace=True)
    _features = ['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K',
                 'Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    _features = [f for f in _features if f in _df.columns]
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    _sc = StandardScaler()
    _Xs = _sc.fit_transform(_df[_features])
    _Xtr,_Xte,_ytr,_yte = train_test_split(_Xs,_df['Target'],test_size=.2,random_state=42,shuffle=False)
    _m = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)
    _m.fit(_Xtr,_ytr)
    _ul   = _df[_features].iloc[-1:]
    _pred = _m.predict(_sc.transform(_ul))[0]
    _prob = _m.predict_proba(_sc.transform(_ul))[0]
    sd = dict(df=_df, pred=int(_pred), prob=_prob.tolist(),
              precio=float(_df['Close'].iloc[-1]), rsi=float(_df['RSI'].iloc[-1]),
              atr=float(_df['ATR'].iloc[-1]), bb_up=float(_df['BB_upper'].iloc[-1]),
              bb_low=float(_df['BB_lower'].iloc[-1]), macd=float(_df['MACD'].iloc[-1]),
              macd_h=float(_df['MACD_hist'].iloc[-1]), ema20=float(_df['EMA_20'].iloc[-1]),
              ema50=float(_df['EMA_50'].iloc[-1]), features=_features)

pred,prob,precio,rsi,atr = sd['pred'],sd['prob'],sd['precio'],sd['rsi'],sd['atr']
bb_up,bb_low,ema20,ema50,df = sd['bb_up'],sd['bb_low'],sd['ema20'],sd['ema50'],sd['df']
sl_long,sl_short  = precio-atr*1.5, precio+atr*1.5
tp_long,tp_short  = precio+atr*2.5, precio-atr*2.5
rr = round(atr*2.5/(atr*1.5),2)
ET = {1:"📈 LONG",0:"➡️ LATERAL",-1:"📉 SHORT"}
p_long  = round(float(prob[2] if len(prob)==3 else prob[1])*100,1)
p_short = round(float(prob[0])*100,1)
p_lat   = round(max(0,100-p_long-p_short-5),1)
p_shock = round(100-p_long-p_short-p_lat,1)
cond_rsi = "Sobrecomprado ⚠️" if rsi>70 else "Sobrevendido ⚠️" if rsi<30 else "RSI Normal ✅"
smc = detect_smc(df)

# ── BANNERS ───────────────────────────────────────────────────────
b1 = f"  ⚔️ SEÑAL: {ET.get(pred)}  •  CONFIANZA: {max(prob)*100:.1f}%  •  PRECIO: ${precio:,.2f}  •  SL: ${sl_long:,.2f}  •  TP: ${tp_long:,.2f}  •  R:R 1:{rr}  •  ATR: {atr:.2f}  " * 2
b2 = f"  📊 RSI: {rsi:.1f} — {cond_rsi}  •  EMA20: ${ema20:,.2f}  •  EMA50: ${ema50:,.2f}  •  BB↑: ${bb_up:,.2f}  •  BB↓: ${bb_low:,.2f}  •  SMC Bias: {smc['bias']}  " * 2
bc = '#00FF88' if pred==1 else '#FF4444' if pred==-1 else '#FFB800'
st.markdown(f"""
<div class="ticker-wrap">
  <span class="ticker-label">SEÑAL</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 80px)">
    <div class="t-scroll" style="color:{bc}">{b1}</div>
  </div>
</div>
<div class="ticker-wrap">
  <span class="ticker-label">TÉCNICO</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 80px)">
    <div class="t-scroll2" style="color:#aaa">{b2}</div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("💰 XAU/USD",  f"${precio:,.2f}")
c2.metric("📊 RSI",      f"{rsi:.1f}", "Sobrecomprado" if rsi>70 else "Sobrevendido" if rsi<30 else "Normal")
c3.metric("⚡ ATR",       f"{atr:.2f}")
c4.metric("🎯 Señal ML", ET.get(pred), f"{max(prob)*100:.1f}%")
c5.metric("📐 R:R",      f"1:{rr}")

st.markdown("---")

# ── GUARDAR SEÑAL EN HISTORIAL ────────────────────────────────────
def save_signal(pred, prob, precio, rsi, atr):
    mx = pytz.timezone('America/Mexico_City')
    entry = {
        'fecha': datetime.now(mx).strftime('%d/%m/%Y %H:%M'),
        'direccion': ET.get(pred),
        'confianza': f"{max(prob)*100:.1f}%",
        'precio': precio,
        'rsi': round(rsi,1),
        'atr': round(atr,2),
        'sl': round(precio-atr*1.5,2) if pred>=0 else round(precio+atr*1.5,2),
        'tp': round(precio+atr*2.5,2) if pred>=0 else round(precio-atr*2.5,2),
        'resultado': 'PENDIENTE'
    }
    if not st.session_state.signal_history or st.session_state.signal_history[-1]['precio'] != precio:
        st.session_state.signal_history.append(entry)

save_signal(pred, prob, precio, rsi, atr)

# ── TELEGRAM ─────────────────────────────────────────────────────
def send_telegram(token, chat_id, mensaje):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={'chat_id': chat_id, 'text': mensaje, 'parse_mode':'Markdown'}, timeout=5)
        return True
    except: return False

# ── BACKTEST ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def run_backtest(df_json):
    df = pd.read_json(io.StringIO(df_json), orient='split')
    capital = 1000.0
    equity  = [capital]
    trades  = []
    atr_col = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    rsi_col = ta.momentum.rsi(df['Close'], window=14)
    macd_col= ta.trend.macd_diff(df['Close'])
    e20     = ta.trend.ema_indicator(df['Close'], window=20)
    e50     = ta.trend.ema_indicator(df['Close'], window=50)
    i = 50
    while i < len(df)-5:
        p   = float(df['Close'].iloc[i])
        atr = float(atr_col.iloc[i]) if not pd.isna(atr_col.iloc[i]) else 50
        rsi = float(rsi_col.iloc[i]) if not pd.isna(rsi_col.iloc[i]) else 50
        mh  = float(macd_col.iloc[i]) if not pd.isna(macd_col.iloc[i]) else 0
        em20= float(e20.iloc[i]) if not pd.isna(e20.iloc[i]) else p
        em50= float(e50.iloc[i]) if not pd.isna(e50.iloc[i]) else p
        sl_l= p - atr*1.5
        tp_l= p + atr*2.5
        sl_s= p + atr*1.5
        tp_s= p - atr*2.5
        direccion = 0
        if p>em20 and p>em50 and rsi<70 and mh>0: direccion=1
        elif p<em20 and p<em50 and rsi>30 and mh<0: direccion=-1
        if direccion != 0:
            for j in range(1,6):
                fp = float(df['Close'].iloc[i+j])
                if direccion==1:
                    if fp>=tp_l:
                        pnl=(tp_l-p)/p*capital*0.1; capital+=pnl
                        trades.append({'fecha':str(df.index[i])[:10],'dir':'LONG','entrada':p,'salida':tp_l,'pnl':round(pnl,2),'resultado':'WIN'})
                        break
                    elif fp<=sl_l:
                        pnl=(sl_l-p)/p*capital*0.1; capital+=pnl
                        trades.append({'fecha':str(df.index[i])[:10],'dir':'LONG','entrada':p,'salida':sl_l,'pnl':round(pnl,2),'resultado':'LOSS'})
                        break
                else:
                    if fp<=tp_s:
                        pnl=(p-tp_s)/p*capital*0.1; capital+=pnl
                        trades.append({'fecha':str(df.index[i])[:10],'dir':'SHORT','entrada':p,'salida':tp_s,'pnl':round(pnl,2),'resultado':'WIN'})
                        break
                    elif fp>=sl_s:
                        pnl=(p-sl_s)/p*capital*0.1; capital+=pnl
                        trades.append({'fecha':str(df.index[i])[:10],'dir':'SHORT','entrada':p,'salida':sl_s,'pnl':round(pnl,2),'resultado':'LOSS'})
                        break
            equity.append(capital)
            i += 5
        else:
            i += 1
    wins   = sum(1 for t in trades if t['resultado']=='WIN')
    losses = len(trades)-wins
    wr     = wins/len(trades)*100 if trades else 0
    return trades, equity, round(wr,1), round(capital-1000,2)

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["🎯 Señal","🏦 SMC/ICT","🌐 Multi-TF","📋 Paper Trading","📊 Gráfica","💬 Chat","📈 Backtest","🔔 Alertas","📜 Historial","👁️ Monitor"])

# ─ TAB 1: SEÑAL ──────────────────────────────────────────────────
with tab1:
    ca,cb = st.columns(2)
    with ca:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Señal Principal")
        cls = "sig-long" if pred==1 else "sig-short" if pred==-1 else "sig-neu"
        st.markdown(f'<div class="{cls}">{ET.get(pred)}</div>', unsafe_allow_html=True)
        st.markdown(f"**Precio:** ${precio:,.2f}")
        if pred==1:
            st.markdown(f"**Entrada:** ${precio:,.2f}")
            st.markdown(f"**Stop Loss 🔴:** ${sl_long:,.2f}  (−{atr*1.5:.2f})")
            st.markdown(f"**Take Profit 🟢:** ${tp_long:,.2f}  (+{atr*2.5:.2f})")
        elif pred==-1:
            st.markdown(f"**Entrada:** ${precio:,.2f}")
            st.markdown(f"**Stop Loss 🔴:** ${sl_short:,.2f}  (+{atr*1.5:.2f})")
            st.markdown(f"**Take Profit 🟢:** ${tp_short:,.2f}  (−{atr*2.5:.2f})")
        else:
            st.markdown("**Sin entrada clara. Espera ruptura.**")
            st.markdown(f"▲ LONG si rompe ${bb_up:,.2f}")
            st.markdown(f"▼ SHORT si rompe ${bb_low:,.2f}")
        sl_ref = sl_long if pred>=0 else sl_short
        lotes,riesgo_usd = calc_posicion(account_size,risk_pct,precio,sl_ref)
        st.markdown(f"**Lotes sugeridos:** {lotes}  |  **Riesgo:** ${riesgo_usd:.2f}")
        st.markdown(f"**R:R:** 1:{rr}")
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🕐 Ventanas")
        mx   = pytz.timezone('America/Mexico_City')
        hora = datetime.now(mx)
        h    = hora.hour
        vens = [("Londres abre",3,5,"Alta"),("Londres + NY",8,11,"Máxima ⭐"),("NY tarde",12,14,"Media"),("Cierre NY",15,17,"Baja")]
        st.markdown(f"**Hora MX:** {hora.strftime('%H:%M')}")
        for n,ini,fin,cal in vens:
            a = ini<=h<fin
            st.markdown(f"{'🟢' if a else '⚫'} **{ini:02d}–{fin:02d}** {n} [{cal}]")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📅 Calendario Económico")
        cal = get_calendar()
        if cal:
            for e in cal: st.markdown(f"⚡ **{e.get('title','')}** — {e.get('date','')[:10]}")
        else: st.markdown("Sin eventos de alto impacto detectados")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔭 Variantes del Mercado")
    v1,v2,v3,v4 = st.columns(4)
    v1.metric("📈 Alcista",  f"{p_long}%",  f"Rompe ${bb_up:,.0f}")
    v2.metric("📉 Bajista",  f"{p_short}%", f"Rompe ${bb_low:,.0f}")
    v3.metric("➡️ Lateral",  f"{p_lat}%",   "Sin ruptura")
    v4.metric("⚡ Shock",    f"{p_shock}%", "Evento macro")
    st.markdown('</div>', unsafe_allow_html=True)

# ─ TAB 2: SMC / ICT ──────────────────────────────────────────────
with tab2:
    st.subheader("🏦 Smart Money Concepts / ICT")
    s1,s2 = st.columns(2)
    with s1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**Bias:** {'📈 ALCISTA' if smc['bias']=='ALCISTA' else '📉 BAJISTA' if smc['bias']=='BAJISTA' else '➡️ NEUTRAL'}")
        st.markdown("**Break of Structure (BOS):**")
        if smc['bos']:
            for b in smc['bos'][-2:]:
                st.markdown(f"{'🟢' if 'ALCISTA' in b['tipo'] else '🔴'} {b['tipo']} — ${b['nivel']:,.2f}")
        else: st.markdown("Sin BOS detectado")
        st.markdown("**Change of Character (CHoCH):**")
        if smc['choch']:
            for c in smc['choch'][-2:]:
                st.markdown(f"{'🟢' if 'ALCISTA' in c['tipo'] else '🔴'} {c['tipo']} — ${c['nivel']:,.2f}")
        else: st.markdown("Sin CHoCH detectado")
        st.markdown('</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Order Blocks (OB):**")
        if smc['order_blocks']:
            for ob in smc['order_blocks'][-3:]:
                st.markdown(f"{'🟢' if 'ALCISTA' in ob['tipo'] else '🔴'} {ob['tipo']} — ${ob['bottom']:,.2f}–${ob['top']:,.2f}")
        else: st.markdown("Sin OB detectados")
        st.markdown("**Fair Value Gaps (FVG):**")
        if smc['fvg']:
            for fv in smc['fvg'][-3:]:
                st.markdown(f"{'🟢' if 'ALCISTA' in fv['tipo'] else '🔴'} {fv['tipo']} — ${fv['bottom']:,.2f}–${fv['top']:,.2f}")
        else: st.markdown("Sin FVG detectados")
        st.markdown("**Zonas de Liquidez:**")
        if smc['liquidity']:
            for lq in smc['liquidity'][-2:]: st.markdown(f"⚡ {lq['tipo']} — ${lq['nivel']:,.2f}")
        else: st.markdown("Sin liquidez detectada")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⏰ ICT Killzones (Hora México)")
    kzs = [("Asian Range",19,23,"Acumulación de liquidez asiática"),
           ("London Open",3,5,"Barrido de liquidez asiática — primeras tendencias"),
           ("NY Open / Macro",8,11,"⭐ Mayor volatilidad — mejor R:R — killzone ICT principal"),
           ("London Close",10,12,"Reversales frecuentes al cierre de Londres"),
           ("NY PM",13,15,"Continuación o reversión de la tendencia AM")]
    for n,ini,fin,desc in kzs:
        a = ini<=h<fin
        st.markdown(f"{'🟢' if a else '⚪'} **{ini:02d}:00–{fin:02d}:00 {n}** — {desc}")
    st.markdown('</div>', unsafe_allow_html=True)

# ─ TAB 3: MULTI-TF ───────────────────────────────────────────────
with tab3:
    st.subheader("🌐 Confluencia Multi-Timeframe")
    with st.spinner("Analizando timeframes..."):
        mtf_sigs,mtf_bias,mtf_conf = mtf_confluence()
    m1,m2,m3 = st.columns(3)
    m1.metric("Bias General",  f"{'📈 LONG' if mtf_bias=='LONG' else '📉 SHORT' if mtf_bias=='SHORT' else '➡️ NEUTRAL'}")
    m2.metric("Confluencia",   f"{mtf_conf:.0f}%")
    m3.metric("TFs analizados",str(len(mtf_sigs)))
    st.markdown("---")
    for tfn,data in mtf_sigs.items():
        bc2 = "🟢" if data['bias']=='LONG' else "🔴" if data['bias']=='SHORT' else "🟡"
        bar = "█"*data['score'] + "░"*(5-data['score'])
        st.markdown(f"{bc2} **{tfn}** — {data['bias']} | [{bar}] {data['score']}/5 | RSI: {data['rsi']:.1f} | ${data['precio']:,.2f}")
    if mtf_conf >= 60:
        st.success(f"✅ Confluencia STRONG hacia {mtf_bias} ({mtf_conf:.0f}%)")
    elif mtf_conf <= 40:
        st.error(f"🔴 Confluencia SHORT ({mtf_conf:.0f}%)")
    else:
        st.warning("⚠️ Sin confluencia clara — mercado indeciso")

# ─ TAB 4: PAPER TRADING ──────────────────────────────────────────
with tab4:
    st.subheader("📋 Paper Trading")
    p1,p2 = st.columns(2)
    with p1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Nueva operación simulada:**")
        pt_dir  = st.selectbox("Dirección",["LONG 📈","SHORT 📉"])
        pt_ent  = st.number_input("Precio entrada", value=float(precio), step=1.0)
        pt_sl_d = sl_long if "LONG" in pt_dir else sl_short
        pt_tp_d = tp_long if "LONG" in pt_dir else tp_short
        pt_sl   = st.number_input("Stop Loss",   value=round(pt_sl_d,2), step=1.0)
        pt_tp   = st.number_input("Take Profit", value=round(pt_tp_d,2), step=1.0)
        pt_lot,pt_riesgo = calc_posicion(account_size,risk_pct,pt_ent,pt_sl)
        st.markdown(f"Lotes: **{pt_lot}** | Riesgo: **${pt_riesgo:.2f}**")
        if st.button("📌 Abrir Trade"):
            st.session_state.paper_trades.append({
                'id':len(st.session_state.paper_trades)+1,'dir':pt_dir,
                'entrada':pt_ent,'sl':pt_sl,'tp':pt_tp,'lotes':pt_lot,
                'riesgo':pt_riesgo,'estado':'ABIERTO',
                'fecha':datetime.now().strftime('%d/%m %H:%M')
            })
            st.success("Trade abierto ✅")
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Trades abiertos:**")
        open_t = [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
        if open_t:
            for t in open_t:
                pnl = (precio-t['entrada'])*(1 if 'LONG' in t['dir'] else -1)*t['lotes']*100
                st.markdown(f"{'🟢' if pnl>0 else '🔴'} **#{t['id']}** {t['dir']} | ${t['entrada']:,.2f} | P&L: **${pnl:.2f}**")
                if st.button(f"Cerrar #{t['id']}", key=f"c_{t['id']}"):
                    t['estado']='CERRADO'; t['pnl']=pnl; st.rerun()
        else: st.markdown("Sin trades abiertos")
        st.markdown('</div>', unsafe_allow_html=True)
    closed = [t for t in st.session_state.paper_trades if t['estado']=='CERRADO']
    if closed:
        st.markdown("**Historial:**")
        df_h = pd.DataFrame(closed)[['id','fecha','dir','entrada','sl','tp','lotes','pnl','estado']]
        st.dataframe(df_h, use_container_width=True)
        tot_pnl = sum(t.get('pnl',0) for t in closed)
        wins    = sum(1 for t in closed if t.get('pnl',0)>0)
        st.metric("P&L Total", f"${tot_pnl:.2f}", f"Win rate: {wins/len(closed)*100:.0f}%")

# ─ TAB 5: GRÁFICA ────────────────────────────────────────────────
with tab5:
    g1,g2,g3 = st.columns(3)
    tf_h   = g1.selectbox("TF Histórico", list(INTERVALS.keys()), index=5, key="tfh")
    ct     = g2.selectbox("Tipo", ["Velas 🕯️","Línea 📈"], key="ct")
    tf_l   = g3.selectbox("TF En Vivo",  list(INTERVALS.keys()), index=1, key="tfl")

    dfc = get_data(INTERVALS[tf_h], PERIODS[tf_h])
    if dfc is not None:
        dfc = add_indicators(dfc.to_json(orient='split'))
        dp  = dfc.tail(120)
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25])
        if "Velas" in ct:
            fig.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
                increasing_line_color='#00FF88',decreasing_line_color='#FF4444',name="XAU/USD"),row=1,col=1)
        else:
            fig.add_trace(go.Scatter(x=dp.index,y=dp['Close'],line=dict(color=T['primary'],width=2),name="Precio"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_20'],line=dict(color='#FFB800',width=1),name="EMA20"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_50'],line=dict(color='#00FFFF',width=1,dash='dot'),name="EMA50"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_upper'],line=dict(color='rgba(255,255,255,0.13)',width=1),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_lower'],line=dict(color='rgba(255,255,255,0.13)',width=1),fill='tonexty',fillcolor='rgba(255,255,255,0.03)',name="BB"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['RSI'],line=dict(color=T['primary'],width=1.5),name="RSI"),row=2,col=1)
        fig.add_hline(y=70,line_color='#FF4444',line_dash='dot',row=2,col=1)
        fig.add_hline(y=30,line_color='#00FF88',line_dash='dot',row=2,col=1)
        fig.update_layout(paper_bgcolor='#000',plot_bgcolor='#0a0a0a',font=dict(color='#aaa'),
            xaxis_rangeslider_visible=False,height=500,margin=dict(l=0,r=0,t=20,b=0),
            legend=dict(bgcolor='#000',bordercolor='#333',orientation='h'))
        fig.update_xaxes(gridcolor='rgba(30,30,30,0.8)'); fig.update_yaxes(gridcolor='rgba(30,30,30,0.8)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**📡 En Vivo:**")
    dfl = get_data(INTERVALS[tf_l], PERIODS[tf_l])
    if dfl is not None:
        dlp = dfl.tail(80)
        fig2 = go.Figure()
        if "Velas" in ct:
            fig2.add_trace(go.Candlestick(x=dlp.index,open=dlp['Open'],high=dlp['High'],low=dlp['Low'],close=dlp['Close'],
                increasing_line_color='#00FF88',decreasing_line_color='#FF4444'))
        else:
            fig2.add_trace(go.Scatter(x=dlp.index,y=dlp['Close'],line=dict(color=T['primary'],width=2)))
        fig2.update_layout(paper_bgcolor='#000',plot_bgcolor='#0a0a0a',font=dict(color='#aaa'),
            xaxis_rangeslider_visible=False,height=300,margin=dict(l=0,r=0,t=10,b=0))
        fig2.update_xaxes(gridcolor='rgba(30,30,30,0.8)'); fig2.update_yaxes(gridcolor='rgba(30,30,30,0.8)')
        st.plotly_chart(fig2, use_container_width=True)
        u = dfl.iloc[-1]
        lv1,lv2,lv3 = st.columns(3)
        lv1.metric("Último",f"${float(u['Close']):,.2f}")
        lv2.metric("Máximo",f"${float(u['High']):,.2f}")
        lv3.metric("Mínimo",f"${float(u['Low']):,.2f}")

# ─ TAB 6: CHAT ───────────────────────────────────────────────────
with tab6:
    st.subheader("💬 Chat MIMI-AI")
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg['role']=='user' else "assistant"):
            st.markdown(msg['content'])

    def mimi(q):
        q = q.lower().strip()
        smc2 = detect_smc(df)
        if any(w in q for w in ['señal','signal','dirección','hacia donde','tendencia','predicción']):
            return f"Señal actual: **{ET.get(pred)}** con {max(prob)*100:.1f}% confianza. Precio: ${precio:,.2f}. SMC Bias: {smc2['bias']}. {'Busca LONG si rompe $'+str(round(bb_up,2)) if pred>=0 else 'Busca SHORT si rompe $'+str(round(bb_low,2))}."
        elif any(w in q for w in ['sl','stop loss','cuanto arriesgo','stop']):
            return f"SL LONG: ${sl_long:,.2f} (−{atr*1.5:.2f}). SL SHORT: ${sl_short:,.2f} (+{atr*1.5:.2f}). Calculado con 1.5× ATR ({atr:.2f})."
        elif any(w in q for w in ['tp','take profit','objetivo','target']):
            return f"TP LONG: ${tp_long:,.2f} (+{atr*2.5:.2f}). TP SHORT: ${tp_short:,.2f} (−{atr*2.5:.2f}). R:R actual: 1:{rr}."
        elif any(w in q for w in ['rsi']):
            c = "sobrecomprado — posible corrección" if rsi>70 else "sobrevendido — posible rebote" if rsi<30 else "neutral, sin extremo"
            return f"RSI: {rsi:.1f} — {c}."
        elif any(w in q for w in ['smc','smart money','order block','ob','bloque']):
            r2 = f"Bias SMC: **{smc2['bias']}**. "
            if smc2['order_blocks']: ob=smc2['order_blocks'][-1]; r2+=f"Último OB: {ob['tipo']} en ${ob['bottom']:,.2f}–${ob['top']:,.2f}. "
            if smc2['bos']: r2+=f"BOS: {smc2['bos'][-1]['tipo']}."
            return r2
        elif any(w in q for w in ['fvg','fair value','gap','hueco']):
            if smc2['fvg']: fv=smc2['fvg'][-1]; return f"FVG: {fv['tipo']} entre ${fv['bottom']:,.2f}–${fv['top']:,.2f}. El precio suele regresar a llenarlo."
            return "Sin FVG activos ahora."
        elif any(w in q for w in ['bos','choch','estructura']):
            r2=""
            if smc2['bos']: r2+=f"BOS: {smc2['bos'][-1]['tipo']} en ${smc2['bos'][-1]['nivel']:,.2f}. "
            if smc2['choch']: r2+=f"CHoCH: {smc2['choch'][-1]['tipo']} en ${smc2['choch'][-1]['nivel']:,.2f}."
            return r2 if r2 else "Sin estructuras claras detectadas."
        elif any(w in q for w in ['ict','killzone','sesión','sesion']):
            return "ICT opera en killzones: London Open 03–05 MX y NY Open 08–11 MX son las principales. La NY Open es la de mayor R:R."
        elif any(w in q for w in ['confluencia','multi','timeframe']):
            return "Revisa la pestaña 🌐 Multi-TF para ver el bias de M15, H1, H4 y D1 y el score de confluencia."
        elif any(w in q for w in ['scalping']):
            return "Scalping: usa M5–M15 con confluencia H1. ICT NY Open 08–11 MX. SL ajustado 0.5× ATR. Mínimo R:R 1:1.5."
        elif any(w in q for w in ['day trading','intraday','intradía']):
            return "Day trading: M15–H1. Entra en killzones. SL basado en estructura SMC. Evita cargar posiciones overnight."
        elif any(w in q for w in ['swing']):
            return "Swing: H4–D1. Sigue el bias SMC diario. ATR en D1 como referencia de SL. Trades de 3–10 días."
        elif any(w in q for w in ['rr','r:r','risk reward','riesgo beneficio']):
            return f"R:R actual: 1:{rr}. Con 50% de win rate necesitas R:R mínimo 1:1 para no perder. Con 40% de win rate necesitas 1:1.5 para ser rentable."
        elif any(w in q for w in ['oro','gold','xau','por que sube','por que baja']):
            return "El oro sube: dólar baja, inflación sube, incertidumbre geopolítica, Fed reduce tasas. Baja: dólar se fortalece, tasas reales suben. Monitorea DXY como correlación inversa."
        elif any(w in q for w in ['atr','volatilidad']):
            return f"ATR: {atr:.2f} — el oro se mueve ~{atr:.0f} puntos por vela D1 en promedio. Ajusta SL y tamaño de posición según el ATR actual."
        elif any(w in q for w in ['probabilidad','prob','porcentaje']):
            return f"ML: LONG {p_long}% | SHORT {p_short}% | LATERAL {p_lat}% | Shock {p_shock}%. Entrenado con 2 años de datos históricos."
        elif any(w in q for w in ['como funciona','que eres','que es mimi','explica']):
            return "Soy MIMI-AI: combino ML (Random Forest + Gradient Boosting), SMC/ICT automatizado, confluencia multi-timeframe y análisis técnico para darte la señal más precisa en XAU/USD."
        elif any(w in q for w in ['paper','simulado','practica']):
            return "Paper Trading está en la pestaña 📋. Simula trades sin dinero real para medir la precisión del modelo antes de operar en vivo."
        elif any(w in q for w in ['lotes','tamaño','posicion','posición','cuanto entro']):
            l,r = calc_posicion(account_size,risk_pct,precio,sl_long)
            return f"Con capital ${account_size:.0f} y riesgo {risk_pct}%: {l} lotes = ${r:.2f} en riesgo. Calculado según ATR y SL actual."
        else:
            return f"${precio:,.2f} | {ET.get(pred)} {max(prob)*100:.1f}% | RSI {rsi:.1f} | SMC: {smc2['bias']}. Pregúntame: señal, SL, TP, SMC, FVG, BOS, ICT, scalping, swing, probabilidades, oro."

    user_in = st.chat_input("Pregúntale a MIMI-AI...")
    if user_in:
        st.session_state.chat_history.append({'role':'user','content':user_in})
        resp = mimi(user_in)
        st.session_state.chat_history.append({'role':'mimi','content':resp})
        st.rerun()

# ── FRASE ─────────────────────────────────────────────────────────
frases = ["El mercado revela lo que eres. No lo que quieres.",
          "No controlas el precio. Controlas tu reacción.",
          "La paciencia no es debilidad. Es claridad.",
          "Una pérdida aceptada a tiempo es una victoria de carácter.",
          "El ruido es abundante. La señal, escasa.",
          "El trader que no acepta la incertidumbre, ya perdió.",
          "Opera el plan. No las emociones."]
st.markdown(f'<div class="stoic">🪨 {random.choice(frases)}</div>', unsafe_allow_html=True)

# ─ TAB 7: BACKTEST ───────────────────────────────────────────────
with tab7:
    st.subheader("📈 Backtest Visual")
    with st.spinner("Corriendo backtest..."):
        bt_trades, bt_equity, bt_wr, bt_pnl = run_backtest(df.to_json(orient='split'))
    bm1,bm2,bm3,bm4 = st.columns(4)
    bm1.metric("Capital Inicial", "$1,000")
    bm2.metric("Capital Final",   f"${1000+bt_pnl:,.2f}", f"{bt_pnl:+.2f}")
    bm3.metric("Win Rate",        f"{bt_wr:.1f}%")
    bm4.metric("Total Trades",    str(len(bt_trades)))
    if bt_equity:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(y=bt_equity, fill='tozeroy',
            fillcolor='rgba(0,255,136,0.1)', line=dict(color='#00FF88',width=2), name="Capital"))
        fig_eq.update_layout(paper_bgcolor='#000', plot_bgcolor='#0a0a0a',
            font=dict(color='#aaa'), height=300, margin=dict(l=0,r=0,t=20,b=0),
            title=dict(text="Curva de Capital", font=dict(color='#FFD700')))
        fig_eq.update_xaxes(gridcolor='rgba(30,30,30,0.8)')
        fig_eq.update_yaxes(gridcolor='rgba(30,30,30,0.8)')
        st.plotly_chart(fig_eq, use_container_width=True)
    if bt_trades:
        df_bt = pd.DataFrame(bt_trades[-30:])
        st.markdown("**Últimos 30 trades:**")
        st.dataframe(df_bt, use_container_width=True)

# ─ TAB 8: ALERTAS TELEGRAM ───────────────────────────────────────
with tab8:
    st.subheader("🔔 Alertas por Telegram")
    st.markdown("""
**Cómo configurar Telegram:**
1. Abre Telegram y busca **@BotFather**
2. Escribe `/newbot` y sigue los pasos
3. Copia el **token** que te da
4. Busca **@userinfobot** en Telegram y escríbele — te da tu **Chat ID**
5. Pega ambos aquí abajo
""")
    tg_token   = st.text_input("🤖 Token del Bot", value=st.session_state.tg_token, type="password")
    tg_chat_id = st.text_input("💬 Chat ID",        value=st.session_state.tg_chat_id)
    if tg_token: st.session_state.tg_token   = tg_token
    if tg_chat_id: st.session_state.tg_chat_id = tg_chat_id

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Condiciones para alertar:**")
        alert_long   = st.checkbox("📈 Señal LONG",   value=True)
        alert_short  = st.checkbox("📉 Señal SHORT",  value=True)
        alert_conf   = st.slider("Confianza mínima (%)", 30, 90, 50)
        alert_window = st.checkbox("⭐ Solo en ventana activa", value=True)
    with col_b:
        if st.button("🧪 Enviar mensaje de prueba"):
            if tg_token and tg_chat_id:
                msg = f"⚔️ *MIMI-AI Test*\nConexión exitosa ✅\nPrecio XAU/USD: ${precio:,.2f}"
                ok = send_telegram(tg_token, tg_chat_id, msg)
                st.success("✅ Mensaje enviado") if ok else st.error("❌ Error — revisa token y chat ID")
            else:
                st.warning("Falta token o chat ID")

        if st.button("📡 Enviar señal actual"):
            if tg_token and tg_chat_id:
                mx2  = pytz.timezone('America/Mexico_City')
                hora2= datetime.now(mx2).strftime('%H:%M')
                msg2 = f"""⚔️ *MIMI-AI — Señal*
🕐 {hora2} MX
💰 Precio: *${precio:,.2f}*
🎯 Señal: *{ET.get(pred)}*
📊 Confianza: *{max(prob)*100:.1f}%*
📈 RSI: {rsi:.1f}
🔴 SL: ${sl_long:,.2f}
🟢 TP: ${tp_long:,.2f}
📐 R:R: 1:{rr}
🧠 SMC Bias: {smc['bias']}"""
                ok2 = send_telegram(tg_token, tg_chat_id, msg2)
                st.success("✅ Señal enviada a Telegram") if ok2 else st.error("❌ Error al enviar")
            else:
                st.warning("Configura token y chat ID primero")

    # Auto-alert logic
    if tg_token and tg_chat_id:
        conf_ok   = max(prob)*100 >= alert_conf
        dir_ok    = (pred==1 and alert_long) or (pred==-1 and alert_short)
        mx3       = pytz.timezone('America/Mexico_City')
        h3        = datetime.now(mx3).hour
        vens2     = [(3,5),(8,11),(12,14),(15,17)]
        en_ventana= any(i<=h3<f for i,f in vens2)
        win_ok    = (not alert_window) or en_ventana
        if conf_ok and dir_ok and win_ok:
            st.success(f"✅ Condiciones cumplidas — señal lista para alertar ({max(prob)*100:.1f}% confianza)")
        else:
            reasons = []
            if not conf_ok:   reasons.append(f"confianza {max(prob)*100:.1f}% < {alert_conf}%")
            if not dir_ok:    reasons.append("dirección no seleccionada")
            if not win_ok:    reasons.append("fuera de ventana activa")
            st.info(f"Sin alerta: {', '.join(reasons)}")

# ─ TAB 9: HISTORIAL DE SEÑALES ───────────────────────────────────
with tab9:
    st.subheader("📜 Historial de Señales")
    if st.session_state.signal_history:
        df_sh = pd.DataFrame(st.session_state.signal_history)
        st.markdown(f"**{len(df_sh)} señales registradas esta sesión**")
        # Colorear resultados
        st.dataframe(df_sh, use_container_width=True)
        # Permitir marcar resultado
        st.markdown("**Actualizar resultado de señal:**")
        sig_idx = st.number_input("# Señal (desde 1)", min_value=1,
                                   max_value=len(st.session_state.signal_history), value=1, step=1)
        sig_res = st.selectbox("Resultado", ["PENDIENTE","WIN ✅","LOSS ❌","BREAKEVEN ➡️"])
        if st.button("Actualizar"):
            st.session_state.signal_history[sig_idx-1]['resultado'] = sig_res
            st.success("Actualizado ✅"); st.rerun()
        wins_h  = sum(1 for s in st.session_state.signal_history if 'WIN' in s['resultado'])
        losses_h= sum(1 for s in st.session_state.signal_history if 'LOSS' in s['resultado'])
        if wins_h+losses_h > 0:
            wr_h = wins_h/(wins_h+losses_h)*100
            st.metric("Win Rate Real (sesión)", f"{wr_h:.1f}%", f"{wins_h}W / {losses_h}L")
        if st.button("🗑️ Limpiar historial"):
            st.session_state.signal_history = []; st.rerun()
    else:
        st.info("Sin señales registradas aún. Se guardan automáticamente al cargar la app.")

# ─ TAB 10: MONITOR DE POSICIÓN ───────────────────────────────────
with tab10:
    st.subheader("👁️ Monitor de Posición")
    mx4   = pytz.timezone('America/Mexico_City')
    ahora = datetime.now(mx4)
    h4    = ahora.hour
    mercado_abierto = not (h4 >= 17 and ahora.weekday() == 4) and not (ahora.weekday() in [5,6] and h4 < 18)

    if not mercado_abierto:
        st.warning("⚠️ Mercado cerrado. El monitor funciona cuando el mercado esté abierto (Dom 6PM – Vie 4PM MX).")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Configurar posición:**")
    mon1, mon2 = st.columns(2)
    with mon1:
        mon_dir     = st.selectbox("Dirección", ["LONG 📈","SHORT 📉"], key="mon_dir")
        mon_entrada = st.number_input("Precio de entrada", value=float(precio), step=0.5, key="mon_ent")
        mon_sl      = st.number_input("Stop Loss", value=round(sl_long if "LONG" in mon_dir else sl_short,2), step=0.5, key="mon_sl")
        mon_tp      = st.number_input("Take Profit", value=round(tp_long if "LONG" in mon_dir else tp_short,2), step=0.5, key="mon_tp")
    with mon2:
        mon_estilo  = st.selectbox("Estilo", ["Scalping (2 min)","Day Trading (5 min)","Swing (10 min)","Personalizado"])
        intervalos  = {"Scalping (2 min)":2,"Day Trading (5 min)":5,"Swing (10 min)":10}
        mon_interval= intervalos.get(mon_estilo, st.number_input("Minutos", 1, 60, 5) if mon_estilo=="Personalizado" else 5)
        mon_checks  = st.slider("Número de revisiones", 3, 20, 6)
        mon_lotes, mon_riesgo = calc_posicion(account_size, risk_pct, mon_entrada, mon_sl)
        st.markdown(f"**Lotes:** {mon_lotes} | **Riesgo:** ${mon_riesgo:.2f}")
        dist_sl = abs(mon_entrada - mon_sl)
        dist_tp = abs(mon_tp - mon_entrada)
        mon_rr  = round(dist_tp/dist_sl,2) if dist_sl>0 else 0
        st.markdown(f"**R:R:** 1:{mon_rr}")

    pnl_actual = (precio - mon_entrada) * (1 if "LONG" in mon_dir else -1)
    pnl_pct    = pnl_actual/mon_entrada*100
    estado_mon = "🟢 MANTÉN" if pnl_actual>0 else "🔴 PRECAUCIÓN"
    if ("LONG" in mon_dir and precio <= mon_sl) or ("SHORT" in mon_dir and precio >= mon_sl):
        estado_mon = "🚨 SL ALCANZADO — SAL YA"
    elif ("LONG" in mon_dir and precio >= mon_tp) or ("SHORT" in mon_dir and precio <= mon_tp):
        estado_mon = "🎯 TP ALCANZADO — TOMA GANANCIA"
    st.markdown(f"**Estado actual:** {estado_mon}")
    st.markdown(f"**Precio actual:** ${precio:,.2f} | **P&L estimado:** {pnl_pct:+.3f}%")

    fig_mon = go.Figure()
    fig_mon.add_hline(y=mon_tp,      line_color='#00FF88', line_dash='dash', annotation_text=f"TP ${mon_tp:,.0f}")
    fig_mon.add_hline(y=mon_entrada, line_color='#FFD700', line_width=2,     annotation_text=f"ENTRADA ${mon_entrada:,.0f}")
    fig_mon.add_hline(y=mon_sl,      line_color='#FF4444', line_dash='dash', annotation_text=f"SL ${mon_sl:,.0f}")
    fig_mon.add_hline(y=precio,      line_color='#FFFFFF', line_dash='dot',  annotation_text=f"ACTUAL ${precio:,.2f}")
    fig_mon.update_layout(paper_bgcolor='#000', plot_bgcolor='#0a0a0a',
        font=dict(color='#aaa'), height=250, margin=dict(l=0,r=0,t=30,b=0),
        title=dict(text="Visualización de Posición", font=dict(color='#FFD700')))
    st.plotly_chart(fig_mon, use_container_width=True)

    if mercado_abierto:
        if st.button("🔄 Actualizar precio ahora"):
            st.cache_data.clear(); st.rerun()
        st.info(f"💡 La página se actualiza manualmente. Para monitoreo automático cada {mon_interval} min activa Auto-refresh en el menú de Streamlit (⋮ arriba a la derecha → Settings → Run on save).")
    st.markdown('</div>', unsafe_allow_html=True)
