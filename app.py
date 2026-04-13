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
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI", page_icon="🏛️", layout="wide")

# ── SECRETS ──────────────────────────────────────────────────────
try:
    TG_TOKEN   = st.secrets["TG_TOKEN"]
    TG_CHAT_ID = st.secrets["TG_CHAT_ID"]
    GH_TOKEN   = st.secrets["GITHUB_TOKEN"]
    GH_REPO    = st.secrets["GITHUB_REPO"]
except:
    TG_TOKEN = TG_CHAT_ID = GH_TOKEN = GH_REPO = ''

# ── GITHUB PERSISTENCE ───────────────────────────────────────────
GH_FILE = "mimi_data.json"

def gh_load():
    if not GH_TOKEN or not GH_REPO: return {}
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_FILE}"
        r = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}"}, timeout=5)
        if r.status_code == 200:
            content = r.json().get('content','')
            return json.loads(base64.b64decode(content).decode())
    except: pass
    return {}

def gh_save(data):
    if not GH_TOKEN or not GH_REPO: return
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_FILE}"
        r = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}"}, timeout=5)
        sha = r.json().get('sha','') if r.status_code == 200 else ''
        content = base64.b64encode(json.dumps(data, default=str).encode()).decode()
        payload = {"message": "MIMI-AI update", "content": content}
        if sha: payload["sha"] = sha
        requests.put(url, headers={"Authorization": f"token {GH_TOKEN}"},
                     json=payload, timeout=5)
    except: pass

# ── SESSION STATE ─────────────────────────────────────────────────
if 'loaded' not in st.session_state:
    saved = gh_load()
    st.session_state.paper_trades   = saved.get('paper_trades', [])
    st.session_state.signal_history = saved.get('signal_history', [])
    st.session_state.capital        = saved.get('capital', 1000.0)
    st.session_state.trade_style    = saved.get('trade_style', 'Day Trading')
    st.session_state.chat_history   = []
    st.session_state.loaded         = True

# ── THEMES ───────────────────────────────────────────────────────
THEMES = {
    "Mármol Griego": {"primary":"#C8A96E","secondary":"#8B6914","bg":"#0a0905","card":"#13100a","accent":"#7B9E87"},
    "Bronce Estoico": {"primary":"#CD7F32","secondary":"#8B4513","bg":"#080503","card":"#120a05","accent":"#6B8E8B"},
    "Lapislázuli":   {"primary":"#6B8FCE","secondary":"#3A5A9B","bg":"#03060f","card":"#070b18","accent":"#C8A96E"},
    "Olimpo Oscuro": {"primary":"#9B7FD4","secondary":"#6B4FA0","bg":"#060308","card":"#0d0614","accent":"#C8A96E"},
    "Athena":        {"primary":"#7BAF9E","secondary":"#3D7A68","bg":"#030a08","card":"#06120f","accent":"#C8A96E"},
}
if 'tema' not in st.session_state: st.session_state.tema = "Mármol Griego"
T = THEMES.get(st.session_state.tema, THEMES["Mármol Griego"])

FRASES_TRADING = [
    ("Warren Buffett", "El mercado es un dispositivo para transferir dinero del impaciente al paciente."),
    ("Marco Aurelio", "Tienes poder sobre tu mente, no sobre los eventos externos. Date cuenta de esto y encontrarás la fuerza."),
    ("Jesse Livermore", "El dinero no se hace pensando. Se hace sentado y esperando."),
    ("Epicteto", "No busques que los eventos sucedan como deseas. Desea que sucedan como son."),
    ("George Soros", "No importa si tienes razón o no. Lo que importa es cuánto ganas cuando tienes razón."),
    ("Séneca", "Sé avaro con tu tiempo. Úsalo bien. No permitas que nadie te lo quite."),
    ("Paul Tudor Jones", "No hagas apuestas descomunales. Si te equivocas no podrás jugar mañana."),
    ("Ed Seykota", "Todo el mundo obtiene lo que quiere del mercado."),
    ("Epicteto", "La riqueza no está en tener grandes posesiones, sino en tener pocas necesidades."),
    ("Ray Dalio", "El mayor error es creer que lo que sucedió en el pasado reciente continuará."),
    ("Zenón de Citio", "Tenemos dos orejas y una boca. Úsalas en esa proporción."),
    ("Stan Weinstein", "El mercado siempre tiene razón. Las opiniones no valen nada."),
    ("Séneca", "No es pobre el que tiene poco, sino el que desea mucho."),
    ("Bruce Kovner", "Los traders novatos arriesgan el 25% de su cuenta en cada trade."),
    ("Marco Aurelio", "Nunca desperdicies tu tiempo preguntándote qué tipo de persona deberías ser. Sé esa persona."),
]

def send_telegram(mensaje):
    if not TG_TOKEN or not TG_CHAT_ID: return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': mensaje, 'parse_mode':'Markdown'}, timeout=5)
        return True
    except: return False

def get_telegram_updates():
    if not TG_TOKEN: return []
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?limit=5&offset=-5"
        r = requests.get(url, timeout=5)
        if r.status_code == 200: return r.json().get('result', [])
    except: pass
    return []

def check_telegram_response():
    for u in reversed(get_telegram_updates()):
        txt = u.get('message', {}).get('text', '').lower().strip()
        if txt in ['entré','entre','sí','si','entrar','long','short']: return 'ENTRO'
        elif txt in ['no','no entré','no entre','salir','cancelar']: return 'NO_ENTRO'
    return None

# ── CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Philosopher:ital,wght@0,400;0,700;1,400&display=swap');
* {{ font-family: 'Philosopher', serif; }}
h1,h2,h3,h4 {{ font-family: 'Cinzel', serif !important; color: {T['primary']} !important; letter-spacing: 2px; }}
.stApp {{ background: {T['bg']} !important; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Cinzel', serif; color: {T['primary']}99; font-size: 0.72em; letter-spacing:1px; }}
.stTabs [aria-selected="true"] {{ color: {T['primary']} !important; border-bottom: 2px solid {T['primary']}; }}
.mimi-title {{
    font-family: 'Cinzel', serif; font-size: clamp(1.8rem,5vw,3rem); font-weight: 900;
    letter-spacing: 10px; text-align: center;
    background: linear-gradient(180deg, #E8D5A3 0%, {T['primary']} 50%, {T['secondary']} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 24px {T['primary']}44); margin: 8px 0;
}}
.mimi-sub {{ text-align:center; font-family:'Philosopher',serif; font-style:italic; color:{T['primary']}77; font-size:.85em; letter-spacing:4px; }}
.greek-orn {{ text-align:center; color:{T['primary']}55; letter-spacing:8px; margin:6px 0; font-size:.9em; }}
.ticker-wrap {{
    background: linear-gradient(90deg,{T['bg']},{T['card']},{T['bg']});
    border-top:1px solid {T['primary']}44; border-bottom:1px solid {T['primary']}44;
    overflow:hidden; padding:7px 0; margin:3px 0;
}}
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
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]};letter-spacing:3px;text-align:center;font-size:1em;">⚙ CONFIG</div>', unsafe_allow_html=True)
    st.markdown("---")
    nuevo_tema = st.selectbox("🏛️ Estilo", list(THEMES.keys()),
                               index=list(THEMES.keys()).index(st.session_state.tema))
    if nuevo_tema != st.session_state.tema:
        st.session_state.tema = nuevo_tema; st.rerun()
    nuevo_estilo = st.selectbox("📊 Trading Style",["Scalping","Day Trading","Swing"],
        index=["Scalping","Day Trading","Swing"].index(st.session_state.trade_style))
    if nuevo_estilo != st.session_state.trade_style:
        st.session_state.trade_style = nuevo_estilo
        d = gh_load(); d['trade_style'] = nuevo_estilo; gh_save(d)
        send_telegram(f"🏛️ *MIMI-AI* — Estilo: *{nuevo_estilo}*")
    risk_pct = st.slider("⚠️ Riesgo/trade (%)", 0.5, 5.0, 1.0, 0.5)
    st.markdown("---")
    st.markdown(f'<div style="font-family:Cinzel,serif;color:{T["primary"]}99;font-size:.8em;letter-spacing:2px;">📚 GUÍA</div>', unsafe_allow_html=True)
    for t,d in [("¿Qué es MIMI-AI?","Oráculo de trading ML+SMC+ICT para XAU/USD. Filosofía estoica."),
                ("SL y TP","SL=límite de pérdida. TP=objetivo de ganancia. Respétalos siempre."),
                ("RSI",">70 sobrecomprado. <30 sobrevendido. El exceso se corrige."),
                ("SMC/ICT","Smart Money: sigue al dinero institucional."),
                ("OB","Order Block: zona de reacción institucional probable."),
                ("BOS/CHoCH","BOS=tendencia confirmada. CHoCH=posible reversión."),
                ("FVG","Fair Value Gap: el precio suele regresar a llenarlo."),
                ("Multi-TF","Mayor confluencia entre timeframes = mayor probabilidad."),
                ("Scalping","M5. Killzone NY Open. Máxima concentración."),
                ("Day Trading","M15-H1. Cierra antes de las 5PM MX."),
                ("Swing","H4-D1. Paciencia de días.")]:
        with st.expander(t): st.write(d)

T = THEMES.get(st.session_state.tema, THEMES["Mármol Griego"])

# ── HEADER ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="greek-orn">─────── ✦ ───────</div>
<div class="mimi-title">MIMI · AI</div>
<div class="mimi-sub">XAU/USD · ML · SMC · ICT · Multi-TF</div>
<div class="mimi-sub" style="font-size:.7em;margin-top:2px;">{st.session_state.trade_style.upper()} · {st.session_state.tema.upper()}</div>
<div class="greek-orn" style="margin-top:6px;">─────── ✦ ───────</div>
""", unsafe_allow_html=True)

# ── STYLE CONFIG ──────────────────────────────────────────────────
SC = {"Scalping":{"interval":"5m","period":"5d","atr_sl":1.0,"atr_tp":1.5},
      "Day Trading":{"interval":"15m","period":"10d","atr_sl":1.5,"atr_tp":2.5},
      "Swing":{"interval":"4h","period":"180d","atr_sl":2.0,"atr_tp":3.5}}[st.session_state.trade_style]
INTERVALS={"M5":"5m","M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}
PERIODS={"M5":"5d","M15":"10d","M30":"20d","H1":"60d","H4":"180d","D1":"2y"}

@st.cache_data(ttl=300)
def get_data(interval="1d",period="2y"):
    df=yf.download("GC=F",period=period,interval=interval,progress=False)
    df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df)>=50: return df
    df2=yf.download("GC=F",period="2y",interval="1d",progress=False)
    df2.columns=[c[0] if isinstance(c,tuple) else c for c in df2.columns]
    df2.dropna(inplace=True)
    return df2 if len(df2)>=50 else None

@st.cache_data(ttl=300)
def add_ind(df_json):
    df=pd.read_json(io.StringIO(df_json),orient='split')
    df['EMA_20']=ta.trend.ema_indicator(df['Close'],window=20)
    df['EMA_50']=ta.trend.ema_indicator(df['Close'],window=50)
    df['EMA_200']=ta.trend.ema_indicator(df['Close'],window=200)
    df['RSI']=ta.momentum.rsi(df['Close'],window=14)
    df['MACD']=ta.trend.macd(df['Close'])
    df['MACD_hist']=ta.trend.macd_diff(df['Close'])
    df['BB_upper']=ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower']=ta.volatility.bollinger_lband(df['Close'])
    df['BB_width']=(df['BB_upper']-df['BB_lower'])/df['Close']
    df['ATR']=ta.volatility.average_true_range(df['High'],df['Low'],df['Close'])
    df['Stoch_K']=ta.momentum.stoch(df['High'],df['Low'],df['Close'])
    df['OBV']=ta.volume.on_balance_volume(df['Close'],df['Volume'])
    df['Dist_EMA20']=(df['Close']-df['EMA_20'])/df['Close']*100
    df['Dist_EMA50']=(df['Close']-df['EMA_50'])/df['Close']*100
    df['Dist_EMA200']=(df['Close']-df['EMA_200'])/df['Close']*100
    df['Return_1d']=df['Close'].pct_change(1)
    df['Return_3d']=df['Close'].pct_change(3)
    df['Return_5d']=df['Close'].pct_change(5)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=600)
def get_signal(iv,per,asl,atp):
    raw=get_data(iv,per)
    if raw is None: raw=get_data("1d","2y")
    if raw is None: return None
    df=add_ind(raw.to_json(orient='split'))
    df['Future_Return']=df['Close'].pct_change(5).shift(-5)
    df['Target']=0
    df.loc[df['Future_Return']>0.003,'Target']=1
    df.loc[df['Future_Return']<-0.003,'Target']=-1
    df.dropna(inplace=True)
    feats=['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K','Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    feats=[f for f in feats if f in df.columns]
    X,y=df[feats],df['Target']
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=.2,random_state=42,shuffle=False)
    rf=RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1); rf.fit(Xtr,ytr)
    gb=GradientBoostingClassifier(n_estimators=200,max_depth=5,random_state=42); gb.fit(Xtr,ytr)
    m=rf if accuracy_score(yte,rf.predict(Xte))>=accuracy_score(yte,gb.predict(Xte)) else gb
    ul=df[feats].iloc[-1:]
    pred=m.predict(sc.transform(ul))[0]; prob=m.predict_proba(sc.transform(ul))[0]
    p=float(df['Close'].iloc[-1]); atr=float(df['ATR'].iloc[-1])
    return dict(df=df,pred=int(pred),prob=prob.tolist(),precio=p,rsi=float(df['RSI'].iloc[-1]),
                atr=atr,bb_up=float(df['BB_upper'].iloc[-1]),bb_low=float(df['BB_lower'].iloc[-1]),
                macd=float(df['MACD'].iloc[-1]),macd_h=float(df['MACD_hist'].iloc[-1]),
                ema20=float(df['EMA_20'].iloc[-1]),ema50=float(df['EMA_50'].iloc[-1]),
                sl_long=round(p-atr*asl,2),tp_long=round(p+atr*atp,2),
                sl_short=round(p+atr*asl,2),tp_short=round(p-atr*atp,2),
                rr=round(atp/asl,2),features=feats)

def detect_smc(df,lookback=30):
    res={'order_blocks':[],'bos':[],'choch':[],'fvg':[],'bias':'NEUTRAL'}
    if len(df)<lookback+5: return res
    H,L,C,O=df['High'].values,df['Low'].values,df['Close'].values,df['Open'].values
    sh,sl=[],[]
    for i in range(2,len(df)-2):
        if H[i]>H[i-1] and H[i]>H[i-2] and H[i]>H[i+1] and H[i]>H[i+2]: sh.append((i,H[i]))
        if L[i]<L[i-1] and L[i]<L[i-2] and L[i]<L[i+1] and L[i]<L[i+2]: sl.append((i,L[i]))
    if len(sh)>=2:
        if sh[-1][1]>sh[-2][1]: res['bos'].append({'tipo':'BOS ALCISTA','nivel':sh[-1][1]}); res['bias']='ALCISTA'
        else: res['choch'].append({'tipo':'CHoCH BAJISTA','nivel':sh[-1][1]}); res['bias']='BAJISTA'
    if len(sl)>=2:
        if sl[-1][1]<sl[-2][1]: res['bos'].append({'tipo':'BOS BAJISTA','nivel':sl[-1][1]})
        else: res['choch'].append({'tipo':'CHoCH ALCISTA','nivel':sl[-1][1]})
    avg_b=np.mean([abs(C[j]-O[j]) for j in range(-lookback,-1)])
    for i in range(-lookback,-3):
        if abs(C[i]-O[i])>avg_b*1.5:
            res['order_blocks'].append({'tipo':'OB ALCISTA' if C[i]>O[i] else 'OB BAJISTA','top':H[i],'bottom':L[i]})
    for i in range(-lookback,-2):
        if L[i+2]>H[i]: res['fvg'].append({'tipo':'FVG ALCISTA','top':L[i+2],'bottom':H[i]})
        elif H[i+2]<L[i]: res['fvg'].append({'tipo':'FVG BAJISTA','top':L[i],'bottom':H[i+2]})
    return res

@st.cache_data(ttl=600)
def mtf_conf():
    tfs=[("D1","1d","2y"),("H4","4h","180d"),("H1","1h","60d"),("M15","15m","10d")]
    sigs={}
    for name,iv,per in tfs:
        try:
            df=yf.download("GC=F",period=per,interval=iv,progress=False)
            df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if len(df)<30: continue
            e20=ta.trend.ema_indicator(df['Close'],window=20); e50=ta.trend.ema_indicator(df['Close'],window=50)
            rsi=ta.momentum.rsi(df['Close'],window=14); macd=ta.trend.macd(df['Close']); md=ta.trend.macd_diff(df['Close'])
            p,em20,em50=float(df['Close'].iloc[-1]),float(e20.iloc[-1]),float(e50.iloc[-1])
            r=float(rsi.iloc[-1]); m=float(macd.iloc[-1]) if macd is not None else 0; mh=float(md.iloc[-1]) if md is not None else 0
            score=sum([p>em20,p>em50,em20>em50,r>50,m>0 and mh>0])
            sigs[name]={'score':score,'bias':'LONG' if score>=3 else 'SHORT' if score<=1 else 'NEUTRAL','rsi':r,'precio':p}
        except: pass
    if not sigs: return sigs,'NEUTRAL',50
    total=sum(s['score'] for s in sigs.values()); pct=total/(len(sigs)*5)*100
    return sigs,'LONG' if pct>=60 else 'SHORT' if pct<=40 else 'NEUTRAL',pct

def calc_pos(capital,risk,entrada,sl):
    r=capital*(risk/100); d=abs(entrada-sl)
    if d==0: return 0,0
    return round(r/(d*100),2),round(r,2)

# ── LOAD ──────────────────────────────────────────────────────────
with st.spinner("🏛️ El Oráculo consulta los astros..."):
    sd=get_signal(SC['interval'],SC['period'],SC['atr_sl'],SC['atr_tp'])

if sd is None:
    st.warning("⚠️ Mercado cerrado — reabre domingo 6PM MX. Mostrando última señal guardada.")
    st.stop()

pred,prob,precio,rsi,atr=sd['pred'],sd['prob'],sd['precio'],sd['rsi'],sd['atr']
bb_up,bb_low,ema20,ema50,df=sd['bb_up'],sd['bb_low'],sd['ema20'],sd['ema50'],sd['df']
sl_long,tp_long,sl_short,tp_short,rr=sd['sl_long'],sd['tp_long'],sd['sl_short'],sd['tp_short'],sd['rr']
ET={1:"LONG — ASCENSO",0:"LATERAL — ESPERA",-1:"SHORT — DESCENSO"}
p_long=round(float(prob[2] if len(prob)==3 else prob[1])*100,1)
p_short=round(float(prob[0])*100,1); p_lat=round(max(0,100-p_long-p_short-5),1); p_shock=round(100-p_long-p_short-p_lat,1)
conf=max(prob)*100; smc=detect_smc(df)
mx_tz=pytz.timezone('America/Mexico_City'); ahora=datetime.now(mx_tz); h=ahora.hour

# ── AUTO SIGNAL SAVE ──────────────────────────────────────────────
if not st.session_state.signal_history or st.session_state.signal_history[-1].get('precio')!=precio:
    sl_r=sl_long if pred>=0 else sl_short; tp_r=tp_long if pred>=0 else tp_short
    st.session_state.signal_history.append({
        'id':len(st.session_state.signal_history)+1,'fecha':ahora.strftime('%d/%m/%Y %H:%M'),
        'estilo':st.session_state.trade_style,'direccion':ET.get(pred),'confianza':f"{conf:.1f}%",
        'precio':precio,'sl':sl_r,'tp':tp_r,'rsi':round(rsi,1),'resultado':'PENDIENTE'})

# ── AUTO PAPER TRADE RESULT ───────────────────────────────────────
for t in st.session_state.paper_trades:
    if t['estado']=='ABIERTO':
        if 'LONG' in t['dir']:
            if precio>=t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['tp']-t['entrada'])*t['lotes']*100,2); t['resultado']='WIN ✅'
                send_telegram(f"🏛️ *MIMI-AI — TP ALCANZADO*\n🟢 +${t['pnl']:.2f}")
            elif precio<=t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['sl']-t['entrada'])*t['lotes']*100,2); t['resultado']='LOSS ❌'
                send_telegram(f"🏛️ *MIMI-AI — SL ALCANZADO*\n🔴 ${t['pnl']:.2f}")
        elif 'SHORT' in t['dir']:
            if precio<=t['tp']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['tp'])*t['lotes']*100,2); t['resultado']='WIN ✅'
                send_telegram(f"🏛️ *MIMI-AI — TP ALCANZADO*\n🟢 +${t['pnl']:.2f}")
            elif precio>=t['sl']:
                t['estado']='CERRADO'; t['pnl']=round((t['entrada']-t['sl'])*t['lotes']*100,2); t['resultado']='LOSS ❌'
                send_telegram(f"🏛️ *MIMI-AI — SL ALCANZADO*\n🔴 ${t['pnl']:.2f}")

cap=1000.0+sum(t.get('pnl',0) for t in st.session_state.paper_trades if t['estado']=='CERRADO')
st.session_state.capital=round(cap,2)

# ── TELEGRAM ENTRY CHECK ──────────────────────────────────────────
tg_r=check_telegram_response()
if tg_r=='ENTRO' and pred!=0 and not [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']:
    sl_r=sl_long if pred==1 else sl_short; tp_r=tp_long if pred==1 else tp_short
    lot,risg=calc_pos(st.session_state.capital,risk_pct,precio,sl_r)
    st.session_state.paper_trades.append({
        'id':len(st.session_state.paper_trades)+1,'dir':'LONG 📈' if pred==1 else 'SHORT 📉',
        'entrada':precio,'sl':sl_r,'tp':tp_r,'lotes':lot,'riesgo':risg,
        'estado':'ABIERTO','fecha':ahora.strftime('%d/%m %H:%M'),'resultado':'PENDIENTE','pnl':0})
    send_telegram(f"✅ *Trade registrado via Telegram*\n{'LONG' if pred==1 else 'SHORT'} ${precio:,.2f} → TP ${tp_r:,.2f}")

sd2={'paper_trades':st.session_state.paper_trades,'signal_history':st.session_state.signal_history[-50:],
     'capital':st.session_state.capital,'trade_style':st.session_state.trade_style}
gh_save(sd2)

# ── BANNERS ───────────────────────────────────────────────────────
bc='#4CAF82' if pred==1 else '#C0392B' if pred==-1 else T['primary']
b1=(f"  SEÑAL: {ET.get(pred)}  ·  CONFIANZA: {conf:.1f}%  ·  PRECIO: ${precio:,.2f}  ·  SL: ${sl_long:,.2f}  ·  TP: ${tp_long:,.2f}  ·  R:R 1:{rr}  ·  ATR: {atr:.2f}  ·  SMC: {smc['bias']}  ")*2
b2=(f"  RSI: {rsi:.1f}  ·  EMA20: ${ema20:,.2f}  ·  EMA50: ${ema50:,.2f}  ·  BB↑: ${bb_up:,.2f}  ·  BB↓: ${bb_low:,.2f}  ·  ESTILO: {st.session_state.trade_style.upper()}  ·  CAPITAL: ${st.session_state.capital:,.2f}  ")*2
st.markdown(f"""
<div class="ticker-wrap">
  <span class="ticker-label">ORACLE</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s1" style="color:{bc};font-family:'Philosopher',serif;font-size:.85em;">{b1}</div>
  </div>
</div>
<div class="ticker-wrap">
  <span class="ticker-label">TÉCNICO</span>
  <div style="overflow:hidden;display:inline-block;width:calc(100% - 90px)">
    <div class="t-s2" style="color:{T['primary']}99;font-family:'Philosopher',serif;font-size:.82em;">{b2}</div>
  </div>
</div>
<div class="greek-orn">── ✦ ── ✦ ── ✦ ──</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("🏛️ XAU/USD",f"${precio:,.2f}")
c2.metric("📊 RSI",f"{rsi:.1f}","Sobrecomprado" if rsi>70 else "Sobrevendido" if rsi<30 else "Normal")
c3.metric("⚡ ATR",f"{atr:.2f}")
c4.metric("🎯 Señal","LONG" if pred==1 else "SHORT" if pred==-1 else "LATERAL",f"{conf:.1f}%")
c5.metric("📐 R:R",f"1:{rr}")
c6.metric("💰 Capital",f"${st.session_state.capital:,.2f}")
st.markdown('<div class="greek-orn">── ✦ ──</div>',unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10=st.tabs([
    "🎯 Señal","🏛️ SMC·ICT","🌐 Multi·TF","📋 Paper","📊 Gráfica",
    "💬 Chat","📈 Backtest","🔔 Alertas","📜 Historial","👁️ Monitor"])

with tab1:
    ca,cb=st.columns(2)
    with ca:
        st.markdown('<div class="card"><div class="card-title">SEÑAL DEL ORÁCULO</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="{"sig-long" if pred==1 else "sig-short" if pred==-1 else "sig-neu"}">{ET.get(pred)}</div>',unsafe_allow_html=True)
        st.markdown(f"**Estilo:** {st.session_state.trade_style}  |  **Precio:** ${precio:,.2f}")
        sl_r=sl_long if pred>=0 else sl_short; tp_r=tp_long if pred>=0 else tp_short
        if pred!=0:
            st.markdown(f"**Stop Loss 🔴:** ${sl_r:,.2f}")
            st.markdown(f"**Take Profit 🟢:** ${tp_r:,.2f}")
        else:
            st.markdown(f"▲ LONG si rompe ${bb_up:,.2f}"); st.markdown(f"▼ SHORT si rompe ${bb_low:,.2f}")
        lot,risg=calc_pos(st.session_state.capital,risk_pct,precio,sl_r)
        st.markdown(f"**Lotes:** {lot}  |  **Riesgo:** ${risg:.2f}  |  **R:R:** 1:{rr}")
        st.markdown('</div>',unsafe_allow_html=True)
        if pred!=0:
            c_si,c_no=st.columns(2)
            if c_si.button("✅ ENTRO",use_container_width=True):
                if not [t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']:
                    st.session_state.paper_trades.append({
                        'id':len(st.session_state.paper_trades)+1,'dir':'LONG 📈' if pred==1 else 'SHORT 📉',
                        'entrada':precio,'sl':sl_r,'tp':tp_r,'lotes':lot,'riesgo':risg,
                        'estado':'ABIERTO','fecha':ahora.strftime('%d/%m %H:%M'),'resultado':'PENDIENTE','pnl':0})
                    send_telegram(f"🏛️ *MIMI-AI — TRADE ABIERTO*\n{'LONG 📈' if pred==1 else 'SHORT 📉'}\nEntrada: ${precio:,.2f}\nSL: ${sl_r:,.2f}\nTP: ${tp_r:,.2f}\nLotes: {lot}")
                    gh_save({**sd2,'paper_trades':st.session_state.paper_trades})
                    st.success("Trade registrado ✅"); st.rerun()
                else: st.warning("Ya tienes un trade abierto")
            if c_no.button("❌ NO ENTRO",use_container_width=True):
                send_telegram(f"🏛️ *MIMI-AI* — Señal rechazada: {ET.get(pred)} ${precio:,.2f}")
                st.info("Rechazada")
    with cb:
        st.markdown('<div class="card"><div class="card-title">VENTANAS · HORA MX</div>',unsafe_allow_html=True)
        st.markdown(f"**Hora:** {ahora.strftime('%H:%M')}")
        for n,ini,fin,cal in [("London Open",3,5,"Alta"),("London+NY",8,11,"Máxima ✦"),("NY Tarde",12,14,"Media"),("NY Cierre",15,17,"Baja")]:
            st.markdown(f"{'🟢' if ini<=h<fin else '⚫'} **{ini:02d}–{fin:02d}** {n} [{cal}]")
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">VARIANTES</div>',unsafe_allow_html=True)
        v1,v2,v3,v4=st.columns(4)
        v1.metric("📈",f"{p_long}%",f">${bb_up:,.0f}"); v2.metric("📉",f"{p_short}%",f"<${bb_low:,.0f}")
        v3.metric("➡️",f"{p_lat}%","Rango"); v4.metric("⚡",f"{p_shock}%","Evento")
        st.markdown('</div>',unsafe_allow_html=True)

with tab2:
    s1,s2=st.columns(2)
    with s1:
        st.markdown('<div class="card"><div class="card-title">ESTRUCTURA DE MERCADO</div>',unsafe_allow_html=True)
        st.markdown(f"**Bias SMC:** {'📈 ALCISTA' if smc['bias']=='ALCISTA' else '📉 BAJISTA' if smc['bias']=='BAJISTA' else '➡️ NEUTRAL'}")
        for b in smc['bos'][-2:]: st.markdown(f"{'🟢' if 'ALCISTA' in b['tipo'] else '🔴'} {b['tipo']} — ${b['nivel']:,.2f}")
        if not smc['bos']: st.markdown("Sin BOS")
        for c in smc['choch'][-2:]: st.markdown(f"{'🟡' if 'ALCISTA' in c['tipo'] else '🟠'} {c['tipo']} — ${c['nivel']:,.2f}")
        if not smc['choch']: st.markdown("Sin CHoCH")
        st.markdown('</div>',unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="card"><div class="card-title">ORDER BLOCKS · FVG</div>',unsafe_allow_html=True)
        for ob in smc['order_blocks'][-3:]: st.markdown(f"{'🟢' if 'ALCISTA' in ob['tipo'] else '🔴'} {ob['tipo']} ${ob['bottom']:,.2f}–${ob['top']:,.2f}")
        if not smc['order_blocks']: st.markdown("Sin OB")
        for fv in smc['fvg'][-3:]: st.markdown(f"{'🟢' if 'ALCISTA' in fv['tipo'] else '🔴'} {fv['tipo']} ${fv['bottom']:,.2f}–${fv['top']:,.2f}")
        if not smc['fvg']: st.markdown("Sin FVG")
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">ICT KILLZONES · HORA MX</div>',unsafe_allow_html=True)
    for n,ini,fin,desc in [("Asian Range",19,23,"Acumulación de liquidez"),("London Open",3,5,"Barrido liquidez asiática"),("NY Open",8,11,"✦ Mayor volatilidad — mejor R:R"),("London Close",10,12,"Reversales frecuentes"),("NY PM",13,15,"Continuación/reversión AM")]:
        st.markdown(f"{'🟢' if ini<=h<fin else '⚪'} **{ini:02d}–{fin:02d} {n}** — {desc}")
    st.markdown('</div>',unsafe_allow_html=True)

with tab3:
    with st.spinner("Analizando timeframes..."):
        mtf_s,mtf_b,mtf_p=mtf_conf()
    m1,m2,m3=st.columns(3)
    m1.metric("Bias",f"{'📈 LONG' if mtf_b=='LONG' else '📉 SHORT' if mtf_b=='SHORT' else '➡️ NEUTRAL'}")
    m2.metric("Confluencia",f"{mtf_p:.0f}%"); m3.metric("TFs",str(len(mtf_s)))
    for tfn,data in mtf_s.items():
        bc2="🟢" if data['bias']=='LONG' else "🔴" if data['bias']=='SHORT' else "🟡"
        st.markdown(f"{bc2} **{tfn}** — {data['bias']} | {'█'*data['score']}{'░'*(5-data['score'])} {data['score']}/5 | RSI:{data['rsi']:.1f} | ${data['precio']:,.2f}")
    if mtf_p>=60: st.success(f"✅ Confluencia fuerte: {mtf_b} ({mtf_p:.0f}%)")
    elif mtf_p<=40: st.error(f"🔴 Confluencia bajista ({mtf_p:.0f}%)")
    else: st.warning("⚠️ Sin confluencia clara")

with tab4:
    pm1,pm2,pm3=st.columns(3)
    pm1.metric("💰 Capital",f"${st.session_state.capital:,.2f}")
    ct_c=[t for t in st.session_state.paper_trades if t['estado']=='CERRADO']
    w_p=sum(1 for t in ct_c if 'WIN' in t.get('resultado',''))
    pm2.metric("Win Rate",f"{w_p/len(ct_c)*100:.0f}%" if ct_c else "—")
    pm3.metric("Trades",f"{len(ct_c)} cerrados")
    ot=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot:
        t=ot[0]; pnl=(precio-t['entrada'])*(1 if 'LONG' in t['dir'] else -1)*t['lotes']*100
        pct=pnl/t['entrada']*100 if t['entrada'] else 0
        est="🚨 SL — SAL YA" if ('LONG' in t['dir'] and precio<=t['sl']) or ('SHORT' in t['dir'] and precio>=t['sl']) else "🎯 TP ALCANZADO" if ('LONG' in t['dir'] and precio>=t['tp']) or ('SHORT' in t['dir'] and precio<=t['tp']) else "🟢 MANTÉN" if pnl>0 else "🔴 PRECAUCIÓN"
        st.markdown(f'<div class="card"><div class="card-title">POSICIÓN ABIERTA</div>',unsafe_allow_html=True)
        st.markdown(f"**{t['dir']}** | Entrada: ${t['entrada']:,.2f} | Actual: ${precio:,.2f} | **{est}**")
        st.markdown(f"P&L: **${pnl:.2f}** ({pct:+.3f}%) | SL: ${t['sl']:,.2f} | TP: ${t['tp']:,.2f}")
        if st.button("Cerrar manualmente"):
            t['estado']='CERRADO'; t['pnl']=round(pnl,2); t['resultado']='WIN ✅' if pnl>0 else 'LOSS ❌'
            send_telegram(f"🏛️ Cerrado manualmente — P&L: {'+'if pnl>0 else ''}${pnl:.2f}")
            gh_save({**sd2,'paper_trades':st.session_state.paper_trades,'capital':st.session_state.capital}); st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)
    else:
        st.info("Sin trade abierto. Usa '✅ ENTRO' en Señal o escríbele 'entré' al bot de Telegram.")
    if ct_c:
        st.dataframe(pd.DataFrame(ct_c)[['fecha','dir','entrada','sl','tp','lotes','pnl','resultado']],use_container_width=True)
    if st.button("🗑️ Reiniciar paper trading"):
        st.session_state.paper_trades=[]; st.session_state.capital=1000.0
        gh_save({**sd2,'paper_trades':[],'capital':1000.0}); st.rerun()

with tab5:
    g1,g2,g3=st.columns(3)
    tf_h=g1.selectbox("TF Histórico",list(INTERVALS.keys()),index=5,key="tfh")
    ct_g=g2.selectbox("Tipo",["Velas 🕯️","Línea 📈"],key="ct")
    tf_l=g3.selectbox("TF En Vivo",list(INTERVALS.keys()),index=1,key="tfl")
    dfc=get_data(INTERVALS[tf_h],PERIODS[tf_h])
    if dfc is not None:
        dfc=add_ind(dfc.to_json(orient='split')); dp=dfc.tail(120)
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25])
        if "Velas" in ct_g:
            fig.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
                increasing_line_color='#4CAF82',decreasing_line_color='#C0392B',name="XAU"),row=1,col=1)
        else:
            fig.add_trace(go.Scatter(x=dp.index,y=dp['Close'],line=dict(color=T['primary'],width=2),name="Precio"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_20'],line=dict(color='#C8A96E',width=1),name="EMA20"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['EMA_50'],line=dict(color='#7B9E87',width=1,dash='dot'),name="EMA50"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_upper'],line=dict(color='rgba(200,169,110,0.2)',width=1),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_lower'],line=dict(color='rgba(200,169,110,0.2)',width=1),fill='tonexty',fillcolor='rgba(200,169,110,0.04)',name="BB"),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['RSI'],line=dict(color=T['primary'],width=1.5),name="RSI"),row=2,col=1)
        fig.add_hline(y=70,line_color='#C0392B',line_dash='dot',row=2,col=1)
        fig.add_hline(y=30,line_color='#4CAF82',line_dash='dot',row=2,col=1)
        fig.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            xaxis_rangeslider_visible=False,height=500,margin=dict(l=0,r=0,t=20,b=0),
            legend=dict(bgcolor='#000',bordercolor='#222',orientation='h'))
        fig.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fig.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fig,use_container_width=True)
    dfl=get_data(INTERVALS[tf_l],PERIODS[tf_l])
    if dfl is not None:
        dlp=dfl.tail(80)
        fig2=go.Figure()
        if "Velas" in ct_g:
            fig2.add_trace(go.Candlestick(x=dlp.index,open=dlp['Open'],high=dlp['High'],low=dlp['Low'],close=dlp['Close'],
                increasing_line_color='#4CAF82',decreasing_line_color='#C0392B'))
        else:
            fig2.add_trace(go.Scatter(x=dlp.index,y=dlp['Close'],line=dict(color=T['primary'],width=2)))
        fig2.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888'),
            xaxis_rangeslider_visible=False,height=280,margin=dict(l=0,r=0,t=10,b=0))
        fig2.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fig2.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fig2,use_container_width=True)
        u=dfl.iloc[-1]; lv1,lv2,lv3=st.columns(3)
        lv1.metric("Último",f"${float(u['Close']):,.2f}"); lv2.metric("Máximo",f"${float(u['High']):,.2f}"); lv3.metric("Mínimo",f"${float(u['Low']):,.2f}")

with tab6:
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg['role']=='user' else "assistant"):
            st.markdown(msg['content'])
    def mimi_r(q):
        q=q.lower()
        s2=detect_smc(df)
        ct_c2=[t for t in st.session_state.paper_trades if t['estado']=='CERRADO']
        w2=sum(1 for t in ct_c2 if 'WIN' in t.get('resultado',''))
        if any(w in q for w in ['señal','hacia donde','tendencia','dirección']):
            return f"**{ET.get(pred)}** — {conf:.1f}% confianza. ${precio:,.2f}. SMC: {s2['bias']}."
        elif any(w in q for w in ['sl','stop','riesgo']): return f"SL LONG: ${sl_long:,.2f} | SL SHORT: ${sl_short:,.2f}. {SC['atr_sl']}× ATR para {st.session_state.trade_style}."
        elif any(w in q for w in ['tp','take','objetivo']): return f"TP LONG: ${tp_long:,.2f} | TP SHORT: ${tp_short:,.2f}. R:R: 1:{rr}."
        elif 'rsi' in q: return f"RSI: {rsi:.1f} — {'sobrecomprado' if rsi>70 else 'sobrevendido' if rsi<30 else 'neutral'}."
        elif any(w in q for w in ['smc','order block','ob']): return f"Bias SMC: {s2['bias']}. OB: {s2['order_blocks'][-1]['tipo']} ${s2['order_blocks'][-1]['bottom']:,.2f}–${s2['order_blocks'][-1]['top']:,.2f}." if s2['order_blocks'] else f"Bias SMC: {s2['bias']}. Sin OB activos."
        elif any(w in q for w in ['fvg','gap','fair']): return f"FVG: {s2['fvg'][-1]['tipo']} ${s2['fvg'][-1]['bottom']:,.2f}–${s2['fvg'][-1]['top']:,.2f}." if s2['fvg'] else "Sin FVG activos."
        elif any(w in q for w in ['salgo','salir','quedo','mantener','cerrar']):
            ot2=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
            if ot2:
                t2=ot2[0]; pnl2=(precio-t2['entrada'])*(1 if 'LONG' in t2['dir'] else -1)*t2['lotes']*100
                if ('LONG' in t2['dir'] and precio>=t2['tp']) or ('SHORT' in t2['dir'] and precio<=t2['tp']): return f"TP alcanzado — SAL y toma la ganancia. +${pnl2:.2f}"
                elif ('LONG' in t2['dir'] and precio<=t2['sl']) or ('SHORT' in t2['dir'] and precio>=t2['sl']): return f"SL alcanzado — SAL YA sin dudar. ${pnl2:.2f}"
                return f"Posición {'ganadora' if pnl2>0 else 'perdedora'}: ${pnl2:.2f}. {'Mantén si la estructura sigue válida.' if pnl2>0 else 'Evalúa si tu razón de entrada sigue siendo válida.'}"
            return "Sin trade abierto."
        elif any(w in q for w in ['capital','cuanto tengo','ganancia']): return f"Capital: ${st.session_state.capital:,.2f}. Win rate: {w2/len(ct_c2)*100:.0f}% ({w2}W/{len(ct_c2)-w2}L)." if ct_c2 else f"Capital: ${st.session_state.capital:,.2f}. Sin trades cerrados aún."
        elif any(w in q for w in ['lotes','tamaño','cuanto entro']): l2,r2=calc_pos(st.session_state.capital,risk_pct,precio,sl_long); return f"Con ${st.session_state.capital:.0f} y {risk_pct}% riesgo: {l2} lotes = ${r2:.2f} en riesgo."
        elif 'scalping' in q: return f"Scalping: M5, killzone NY Open 08–11 MX. SL {SC['atr_sl']}× ATR. TP {SC['atr_tp']}× ATR. Requiere máxima concentración."
        elif 'swing' in q: return "Swing: H4–D1. Paciencia de días. SL amplio según estructura. Menos estrés, más tiempo."
        elif any(w in q for w in ['oro','gold','xau','sube','baja']): return "El oro sube: dólar débil, inflación alta, geopolítica, Fed dovish. Baja: dólar fuerte, tasas reales altas. Correlación inversa con DXY."
        else: return f"Señal: **{ET.get(pred)}** {conf:.1f}% | RSI: {rsi:.1f} | SMC: {s2['bias']} | Capital: ${st.session_state.capital:,.2f}. Pregúntame: señal, SL, TP, SMC, FVG, salir, capital, lotes, oro, scalping, swing."
    uin=st.chat_input("Consulta al Oráculo...")
    if uin:
        st.session_state.chat_history.append({'role':'user','content':uin})
        st.session_state.chat_history.append({'role':'mimi','content':mimi_r(uin)})
        st.rerun()

with tab7:
    @st.cache_data(ttl=3600)
    def backtest(df_json,asl,atp):
        df_b=pd.read_json(io.StringIO(df_json),orient='split')
        cap=1000.0; eq=[cap]; tds=[]
        ac=ta.volatility.average_true_range(df_b['High'],df_b['Low'],df_b['Close'])
        rc=ta.momentum.rsi(df_b['Close'],window=14); mc=ta.trend.macd_diff(df_b['Close'])
        e2=ta.trend.ema_indicator(df_b['Close'],window=20); e5=ta.trend.ema_indicator(df_b['Close'],window=50)
        i=50
        while i<len(df_b)-5:
            p=float(df_b['Close'].iloc[i]); atr=float(ac.iloc[i]) if not pd.isna(ac.iloc[i]) else 50
            rv=float(rc.iloc[i]) if not pd.isna(rc.iloc[i]) else 50; mh=float(mc.iloc[i]) if not pd.isna(mc.iloc[i]) else 0
            em2=float(e2.iloc[i]) if not pd.isna(e2.iloc[i]) else p; em5=float(e5.iloc[i]) if not pd.isna(e5.iloc[i]) else p
            sl_l=p-atr*asl; tp_l=p+atr*atp; sl_s=p+atr*asl; tp_s=p-atr*atp; d=0
            if p>em2 and p>em5 and rv<70 and mh>0: d=1
            elif p<em2 and p<em5 and rv>30 and mh<0: d=-1
            if d!=0:
                for j in range(1,6):
                    fp=float(df_b['Close'].iloc[i+j])
                    if d==1:
                        if fp>=tp_l: pnl=(tp_l-p)/p*cap*0.1; cap+=pnl; tds.append({'fecha':str(df_b.index[i])[:10],'dir':'LONG','entrada':p,'salida':tp_l,'pnl':round(pnl,2),'res':'WIN'}); break
                        elif fp<=sl_l: pnl=(sl_l-p)/p*cap*0.1; cap+=pnl; tds.append({'fecha':str(df_b.index[i])[:10],'dir':'LONG','entrada':p,'salida':sl_l,'pnl':round(pnl,2),'res':'LOSS'}); break
                    else:
                        if fp<=tp_s: pnl=(p-tp_s)/p*cap*0.1; cap+=pnl; tds.append({'fecha':str(df_b.index[i])[:10],'dir':'SHORT','entrada':p,'salida':tp_s,'pnl':round(pnl,2),'res':'WIN'}); break
                        elif fp>=sl_s: pnl=(p-sl_s)/p*cap*0.1; cap+=pnl; tds.append({'fecha':str(df_b.index[i])[:10],'dir':'SHORT','entrada':p,'salida':sl_s,'pnl':round(pnl,2),'res':'LOSS'}); break
                eq.append(cap); i+=5
            else: i+=1
        w=sum(1 for t in tds if t['res']=='WIN')
        return tds,eq,round(w/len(tds)*100,1) if tds else 0,round(cap-1000,2)
    with st.spinner("Simulando 2 años..."):
        bt_t,bt_e,bt_w,bt_p=backtest(df.to_json(orient='split'),SC['atr_sl'],SC['atr_tp'])
    bm1,bm2,bm3,bm4=st.columns(4)
    bm1.metric("Capital Inicial","$1,000"); bm2.metric("Capital Final",f"${1000+bt_p:,.2f}",f"{bt_p:+.2f}")
    bm3.metric("Win Rate",f"{bt_w:.1f}%"); bm4.metric("Trades",str(len(bt_t)))
    if bt_e:
        fe=go.Figure()
        fe.add_trace(go.Scatter(y=bt_e,fill='tozeroy',fillcolor=f'rgba(200,169,110,0.08)',line=dict(color=T['primary'],width=2),name="Capital"))
        fe.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',font=dict(color='#888',family='Philosopher,serif'),
            height=300,margin=dict(l=0,r=0,t=20,b=0),title=dict(text="CURVA DE CAPITAL",font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        fe.update_xaxes(gridcolor='rgba(200,169,110,0.07)'); fe.update_yaxes(gridcolor='rgba(200,169,110,0.07)')
        st.plotly_chart(fe,use_container_width=True)
    if bt_t: st.dataframe(pd.DataFrame(bt_t[-20:]),use_container_width=True)

with tab8:
    st.markdown('<div class="card-title" style="font-family:Cinzel,serif;color:#C8A96E;letter-spacing:3px;">ALERTAS · TELEGRAM</div>',unsafe_allow_html=True)
    st.markdown("""
**Comandos que entiende el bot:**
- `entré` o `sí` → registra el trade automáticamente en paper trading
- `no` o `no entré` → rechaza la señal
- Recibe alertas automáticas de SL y TP tocados
""")
    col_a,col_b=st.columns(2)
    with col_a:
        al=st.checkbox("📈 LONG",value=True); as_=st.checkbox("📉 SHORT",value=True)
        ac2=st.slider("Confianza mínima (%)",30,90,50); aw=st.checkbox("⭐ Solo ventana activa",value=True)
    with col_b:
        if st.button("🧪 Prueba"):
            ok=send_telegram(f"🏛️ *MIMI-AI Test* ✅\nPrecio: ${precio:,.2f}")
            st.success("Enviado ✅") if ok else st.error("Error — revisa Secrets")
        if st.button("📡 Enviar señal"):
            vens2=[(3,5),(8,11),(12,14),(15,17)]; ev=any(i<=h<f for i,f in vens2)
            if conf>=ac2 and ((pred==1 and al) or (pred==-1 and as_)) and ((not aw) or ev):
                sl_r2=sl_long if pred>=0 else sl_short; tp_r2=tp_long if pred>=0 else tp_short
                ok2=send_telegram(f"🏛️ *MIMI-AI — Oráculo*\n🕐 {ahora.strftime('%H:%M')} MX · {st.session_state.trade_style}\n💰 ${precio:,.2f}\n🎯 *{ET.get(pred)}*\n📊 {conf:.1f}%\n🔴 SL: ${sl_r2:,.2f}\n🟢 TP: ${tp_r2:,.2f}\n📐 R:R: 1:{rr}\n🏛️ SMC: {smc['bias']}\n\n_Responde 'entré' para registrar o 'no' para rechazar_")
                st.success("Enviado ✅") if ok2 else st.error("Error")
            else: st.info("Condiciones no cumplidas")

with tab9:
    if st.session_state.signal_history:
        df_sh=pd.DataFrame(st.session_state.signal_history)
        st.markdown(f"**{len(df_sh)} señales registradas**")
        st.dataframe(df_sh,use_container_width=True)
        w_h=sum(1 for s in st.session_state.signal_history if 'WIN' in s.get('resultado',''))
        l_h=sum(1 for s in st.session_state.signal_history if 'LOSS' in s.get('resultado',''))
        if w_h+l_h>0: st.metric("Win Rate Real",f"{w_h/(w_h+l_h)*100:.1f}%",f"{w_h}W / {l_h}L")
        if st.button("🗑️ Limpiar"):
            st.session_state.signal_history=[]; gh_save({**sd2,'signal_history':[]}); st.rerun()
    else: st.info("Las señales se guardan automáticamente al cargar la app.")

with tab10:
    ot_m=[t for t in st.session_state.paper_trades if t['estado']=='ABIERTO']
    if ot_m:
        t_m=ot_m[0]; pnl_m=(precio-t_m['entrada'])*(1 if 'LONG' in t_m['dir'] else -1)*t_m['lotes']*100
        pct_m=pnl_m/t_m['entrada']*100 if t_m['entrada'] else 0
        est_m="🟢 MANTÉN" if pnl_m>0 else "🔴 PRECAUCIÓN"
        if ('LONG' in t_m['dir'] and precio<=t_m['sl']) or ('SHORT' in t_m['dir'] and precio>=t_m['sl']): est_m="🚨 SL — SAL YA"
        elif ('LONG' in t_m['dir'] and precio>=t_m['tp']) or ('SHORT' in t_m['dir'] and precio<=t_m['tp']): est_m="🎯 TP — TOMA GANANCIA"
        fm=go.Figure()
        fm.add_hline(y=t_m['tp'],line_color='#4CAF82',line_dash='dash',annotation_text=f"TP ${t_m['tp']:,.0f}")
        fm.add_hline(y=t_m['entrada'],line_color=T['primary'],line_width=2,annotation_text=f"ENTRADA ${t_m['entrada']:,.0f}")
        fm.add_hline(y=t_m['sl'],line_color='#C0392B',line_dash='dash',annotation_text=f"SL ${t_m['sl']:,.0f}")
        fm.add_hline(y=precio,line_color='#FFFFFF',line_dash='dot',annotation_text=f"ACTUAL ${precio:,.2f}")
        fm.update_layout(paper_bgcolor='#000',plot_bgcolor='#050300',height=260,font=dict(color='#888',family='Philosopher,serif'),
            margin=dict(l=0,r=0,t=30,b=0),
            title=dict(text=f"{t_m['dir']} | P&L: {'+'if pnl_m>0 else ''}${pnl_m:.2f} | {est_m}",font=dict(color=T['primary'],family='Cinzel,serif',size=11)))
        st.plotly_chart(fm,use_container_width=True)
        mo1,mo2,mo3,mo4=st.columns(4)
        mo1.metric("Entrada",f"${t_m['entrada']:,.2f}"); mo2.metric("Actual",f"${precio:,.2f}")
        mo3.metric("P&L",f"${pnl_m:.2f}",f"{pct_m:+.3f}%"); mo4.metric("Estado",est_m)
        if st.button("🔄 Actualizar"): st.cache_data.clear(); st.rerun()
    else: st.info("Sin posición abierta. Confirma un trade en Señal.")

# ── FRASE FINAL ───────────────────────────────────────────────────
fr=random.choice(FRASES_TRADING)
st.markdown(f"""
<div class="greek-orn" style="margin-top:24px;">─────── ✦ ───────</div>
<div class="stoic-q">{fr[1]}<div class="stoic-a">— {fr[0]}</div></div>
<div class="greek-orn">─────── ✦ ───────</div>
""",unsafe_allow_html=True)
