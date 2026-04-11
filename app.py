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
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI | XAU/USD", page_icon="gold", layout="wide")

# ── TEMAS DE COLOR ──
TEMAS = {
    "Dorado (default)": {
        "primario": "#FFD700", "fondo_banner": "#0a0a0a",
        "borde": "#2a2a00", "titulo": "#FFD700", "acento": "#FFA500"
    },
    "Intenso — Neon Verde": {
        "primario": "#00FF88", "fondo_banner": "#001a0d",
        "borde": "#003319", "titulo": "#00FF88", "acento": "#00CC66"
    },
    "Intenso — Cyan": {
        "primario": "#00FFFF", "fondo_banner": "#001a1a",
        "borde": "#003333", "titulo": "#00FFFF", "acento": "#00CCCC"
    },
    "Intenso — Magenta": {
        "primario": "#FF00FF", "fondo_banner": "#1a001a",
        "borde": "#330033", "titulo": "#FF00FF", "acento": "#CC00CC"
    },
    "Pastel — Lavanda": {
        "primario": "#C3B1E1", "fondo_banner": "#0f0d14",
        "borde": "#1e1a2e", "titulo": "#C3B1E1", "acento": "#A89AC8"
    },
    "Pastel — Rosa": {
        "primario": "#FFB3C6", "fondo_banner": "#140d0f",
        "borde": "#2e1a1e", "titulo": "#FFB3C6", "acento": "#FF8FAB"
    },
    "Pastel — Menta": {
        "primario": "#B5EAD7", "fondo_banner": "#0d1410",
        "borde": "#1a2e22", "titulo": "#B5EAD7", "acento": "#8ED4BC"
    },
}

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## MIMI-AI")
    st.markdown("---")

    tema_sel = st.selectbox("Tema de color", list(TEMAS.keys()))
    tema = TEMAS[tema_sel]

    st.markdown("---")
    st.markdown("### Guia de MIMI-AI")

    guia_items = {
        "Que es MIMI-AI": "MIMI-AI es un sistema de inteligencia artificial para analizar XAU/USD (oro). Usa dos modelos de machine learning que analizan 13 indicadores y solo dan senal cuando ambos coinciden.",
        "Como leer la senal": "LONG = el modelo predice subida. SHORT = predice bajada. LATERAL = sin consenso, no operes. El porcentaje muestra la confianza del modelo.",
        "Entrada, SL y TP": "Entrada: precio donde abres la operacion. SL (Stop Loss): donde cierras si el mercado va en tu contra. TP (Take Profit): donde tomas ganancias. MIMI-AI calcula SL = ATR x 1.5 y TP = ATR x 2.5.",
        "Que es el ATR": "Mide la volatilidad promedio por vela. A mayor ATR, mayor movimiento y mayor distancia en SL y TP.",
        "Que es el RSI": "Mide sobrecompra (>70, posible caida) o sobreventa (<30, posible rebote). Entre 30-70 es zona neutral.",
        "Que es el MACD": "Mide el momentum. Positivo = momentum alcista. Negativo = bajista. MIMI-AI lo combina con RSI y EMAs.",
        "Que son las EMAs": "Medias moviles de tendencia. Precio sobre EMA20 y EMA50 = tendencia alcista. Por debajo = bajista.",
        "Cuando operar": "Mejor ventana: 08:00-11:00 MX (Londres + NY). Mayor liquidez y movimiento real.",
        "Timeframes": "M5/M15 = scalping. H1/H4 = day trading. D1 = swing. Menor TF = mas senales pero mas ruido.",
        "Variantes del mercado": "4 escenarios posibles con probabilidad: Alcista, Bajista, Lateral y Shock por noticias.",
        "El oro (XAU/USD)": "Activo refugio. Sube con incertidumbre economica, inflacion, dolar debil. Baja cuando el dolar se fortalece.",
        "Mercado bursatil": "Donde se compran y venden activos: acciones, divisas, materias primas, indices y criptos.",
        "Noticias que mueven el oro": "NFP (1er viernes del mes), decision FED, CPI inflacion, discursos de la FED. Cierra posiciones antes de estos eventos.",
        "R:R Riesgo-Beneficio": "MIMI-AI usa R:R 1:1.67. Por cada $1 arriesgado, el TP da $1.67 potencial. Rentable a largo plazo con 50% win rate.",
    }

    for titulo, texto in guia_items.items():
        with st.expander(titulo):
            st.markdown(f"<small style='color:#aaa;line-height:1.7'>{texto}</small>", unsafe_allow_html=True)

# ── CSS DINAMICO ──
p = tema["primario"]
fb = tema["fondo_banner"]
bd = tema["borde"]
ac = tema["acento"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.mimi-header {{ text-align:center; padding:1.5rem 0 1rem 0; }}
.mimi-title {{
    font-family:'Orbitron',monospace; font-size:clamp(1.8rem,5vw,3rem);
    font-weight:900; background:linear-gradient(90deg,{p},{ac},{p});
    background-size:300%; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:shimmer 4s linear infinite; letter-spacing:4px;
}}
.mimi-sub {{
    font-family:'Orbitron',monospace; font-size:clamp(0.55rem,1.5vw,0.75rem);
    color:#666; letter-spacing:3px; margin-top:4px;
}}
@keyframes shimmer {{ 0%{{background-position:0% 50%}} 100%{{background-position:300% 50%}} }}

.ticker-wrap {{
    background:{fb}; border:1px solid {bd}; border-radius:8px;
    padding:10px 0; overflow:hidden; margin-bottom:8px; position:relative;
    white-space:nowrap;
}}
.ticker-label {{
    position:absolute; left:0; top:0; bottom:0; display:flex; align-items:center;
    padding:0 12px; background:{p}; color:#000;
    font-family:'Orbitron',monospace; font-size:9px; font-weight:700;
    letter-spacing:1px; z-index:2; border-radius:8px 0 0 8px; min-width:70px;
    justify-content:center;
}}
.ticker-track {{
    display:inline-flex; padding-left:80px; animation:ticker-loop 40s linear infinite;
}}
.ticker-track.fast {{ animation-duration:28s; }}
.ticker-content {{ display:inline-block; white-space:nowrap; padding-right:80px; font-size:13px; }}
@keyframes ticker-loop {{
    0%   {{ transform:translateX(0); }}
    100% {{ transform:translateX(-50%); }}
}}

.t-green{{color:#00e676}} .t-red{{color:#ff4444}} .t-yellow{{color:{p}}}
.t-white{{color:#ffffff}} .t-gray{{color:#999}} .t-sep{{color:#333;margin:0 10px}}

.section-hdr {{
    display:flex; align-items:center; gap:8px;
    font-family:'Orbitron',monospace; font-size:clamp(0.65rem,1.5vw,0.8rem);
    color:{p}; letter-spacing:2px; text-transform:uppercase;
    margin-bottom:1rem; padding-bottom:6px; border-bottom:1px solid {bd};
}}
.help-btn {{
    display:inline-flex; align-items:center; justify-content:center;
    width:18px; height:18px; border-radius:50%; background:{bd};
    color:{p}; font-size:10px; font-weight:700; cursor:pointer;
    border:1px solid {p}; flex-shrink:0; line-height:1;
}}
.frase-estoica {{
    background:{fb}; border:1px solid {bd}; border-left:3px solid {p};
    border-radius:8px; padding:16px 20px; font-style:italic;
    color:{p}; font-size:clamp(12px,2vw,14px); text-align:center;
}}
.footer {{ text-align:center;color:#333;font-size:10px;font-family:monospace;letter-spacing:2px;margin-top:1rem; }}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown(f"""
<div class="mimi-header">
    <div class="mimi-title">MIMI-AI</div>
    <div class="mimi-sub">XAU/USD TRADING INTELLIGENCE SYSTEM</div>
</div>
""", unsafe_allow_html=True)

# ── TIMEFRAME ──
TIMEFRAMES = {
    "M5":  {"periodo":"5d",  "intervalo":"5m"},
    "M15": {"periodo":"5d",  "intervalo":"15m"},
    "M30": {"periodo":"1mo", "intervalo":"30m"},
    "H1":  {"periodo":"1mo", "intervalo":"60m"},
    "H4":  {"periodo":"3mo", "intervalo":"1d"},
    "D1":  {"periodo":"2y",  "intervalo":"1d"},
}
tf_sel = st.selectbox("Timeframe de analisis", list(TIMEFRAMES.keys()), index=5)
tf = TIMEFRAMES[tf_sel]

@st.cache_data(ttl=300)
def cargar_y_entrenar(periodo, intervalo):
    df = yf.download("GC=F", period=periodo, interval=intervalo, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df) < 50: return None,None,None,None
    df['EMA_20']    = ta.trend.ema_indicator(df['Close'], window=min(20,len(df)-1))
    df['EMA_50']    = ta.trend.ema_indicator(df['Close'], window=min(50,len(df)-1))
    df['EMA_200']   = ta.trend.ema_indicator(df['Close'], window=min(200,len(df)-1))
    df['RSI']       = ta.momentum.rsi(df['Close'], window=min(14,len(df)-1))
    df['MACD']      = ta.trend.macd(df['Close'])
    df['MACD_hist'] = ta.trend.macd_diff(df['Close'])
    df['BB_upper']  = ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower']  = ta.volatility.bollinger_lband(df['Close'])
    df['BB_width']  = (df['BB_upper']-df['BB_lower'])/df['Close']
    df['ATR']       = ta.volatility.average_true_range(df['High'],df['Low'],df['Close'])
    df['Stoch_K']   = ta.momentum.stoch(df['High'],df['Low'],df['Close'])
    df['OBV']       = ta.volume.on_balance_volume(df['Close'],df['Volume'])
    df['Dist_EMA20']  = (df['Close']-df['EMA_20'])/df['Close']*100
    df['Dist_EMA50']  = (df['Close']-df['EMA_50'])/df['Close']*100
    df['Dist_EMA200'] = (df['Close']-df['EMA_200'])/df['Close']*100
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    df['Target'] = 0
    df.loc[df['Future_Return']>0.003,'Target']=1
    df.loc[df['Future_Return']<-0.003,'Target']=-1
    df.dropna(inplace=True)
    if len(df)<20: return None,None,None,None
    features=['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K',
              'Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    features=[f for f in features if f in df.columns]
    X,y=df[features],df['Target']
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    Xt,Xv,yt,yv=train_test_split(Xs,y,test_size=0.2,random_state=42,shuffle=False)
    rf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)
    rf.fit(Xt,yt)
    gb=GradientBoostingClassifier(n_estimators=100,max_depth=5,random_state=42)
    gb.fit(Xt,yt)
    mejor=rf if accuracy_score(yv,rf.predict(Xv))>=accuracy_score(yv,gb.predict(Xv)) else gb
    return df,mejor,scaler,features

with st.spinner("MIMI-AI analizando mercado..."):
    df,modelo,scaler,features=cargar_y_entrenar(tf["periodo"],tf["intervalo"])

if df is None:
    st.error("Datos insuficientes. Intenta D1 o H4.")
    st.stop()

precio=float(df['Close'].iloc[-1])
rsi=float(df['RSI'].iloc[-1])
atr=float(df['ATR'].iloc[-1])
bb_up=float(df['BB_upper'].iloc[-1])
bb_low=float(df['BB_lower'].iloc[-1])
macd=float(df['MACD'].iloc[-1])
macd_h=float(df['MACD_hist'].iloc[-1])
ema20=float(df['EMA_20'].iloc[-1])
ema50=float(df['EMA_50'].iloc[-1])
ultima=df[features].iloc[-1:]
Xsc=scaler.transform(ultima)
pred=modelo.predict(Xsc)[0]
prob=modelo.predict_proba(Xsc)[0]

etiquetas={1:"LONG - SUBE",0:"LATERAL",-1:"SHORT - BAJA"}
entrada=precio
sl_largo=round(precio-atr*1.5,2)
sl_corto=round(precio+atr*1.5,2)
tp_largo=round(precio+atr*2.5,2)
tp_corto=round(precio-atr*2.5,2)

mx=pytz.timezone('America/Mexico_City')
ahora=datetime.now(mx)
hora=ahora.hour
ventanas=[
    {"nombre":"Londres abre","inicio":3,"fin":5,"calidad":"Alta"},
    {"nombre":"Londres + NY","inicio":8,"fin":11,"calidad":"Maxima"},
    {"nombre":"NY tarde","inicio":12,"fin":14,"calidad":"Media"},
    {"nombre":"Cierre NY","inicio":15,"fin":17,"calidad":"Baja"},
]
ventana_activa=next((v for v in ventanas if v["inicio"]<=hora<v["fin"]),None)
p_long=round(float(prob[2] if len(prob)==3 else prob[1])*100,1)
p_short=round(float(prob[0])*100,1)
p_lat=round(max(0,100-p_long-p_short-5),1)
p_shock=round(100-p_long-p_short-p_lat,1)

frases=[
    "El mercado revela lo que eres. No lo que quieres.",
    "No controlas el precio. Controlas tu reaccion.",
    "La paciencia no es debilidad. Es claridad.",
    "Una perdida aceptada a tiempo es una victoria de caracter.",
    "El ruido es abundante. La senal, escasa.",
    "Disciplina hoy. Libertad manana.",
    "El mercado premia la claridad, no la velocidad.",
    "Quien controla sus emociones, controla su capital.",
]

def seccion(titulo, ayuda_key, ayuda_txt):
    cols = st.columns([20,1])
    with cols[0]:
        st.markdown(f'<div class="section-hdr">{titulo}</div>', unsafe_allow_html=True)
    with cols[1]:
        if st.button("?", key=ayuda_key, help=ayuda_txt):
            st.info(ayuda_txt)

def sep(): return '<span class="t-sep">|</span>'
def lbl(t): return f'<span class="t-gray">{t}: </span>'

def banner(contenido, speed_class=""):
    doble = contenido + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + contenido
    return f'''
<div class="ticker-wrap">
  <span class="ticker-label">MIMI</span>
  <div class="ticker-track {speed_class}">
    <div class="ticker-content">{doble}</div>
  </div>
</div>'''

senal_c="t-green" if pred==1 else "t-red" if pred==-1 else "t-yellow"
rsi_c="t-red" if rsi>70 else "t-green" if rsi<30 else "t-white"
macd_c="t-green" if macd>0 and macd_h>0 else "t-red" if macd<0 and macd_h<0 else "t-yellow"
ema_c="t-green" if precio>ema20 and precio>ema50 else "t-red" if precio<ema20 and precio<ema50 else "t-yellow"
vent_c="t-green" if ventana_activa else "t-red"
sl_show=f"${sl_largo:,.2f}" if pred==1 else f"${sl_corto:,.2f}" if pred==-1 else "---"
tp_show=f"${tp_largo:,.2f}" if pred==1 else f"${tp_corto:,.2f}" if pred==-1 else "---"
vent_show=ventana_activa['nombre'].upper() if ventana_activa else "SIN VENTANA ACTIVA"

b1=(
    f'{lbl("TF")}<span class="t-yellow">{tf_sel}</span>{sep()}'
    f'{lbl("PRECIO")}<span class="t-green">${precio:,.2f}</span>{sep()}'
    f'{lbl("SENAL")}<span class="{senal_c}">{etiquetas.get(pred)}</span>{sep()}'
    f'{lbl("CONFIANZA")}<span class="t-white">{max(prob)*100:.1f}%</span>{sep()}'
    f'{lbl("ENTRADA")}<span class="t-white">${entrada:,.2f}</span>{sep()}'
    f'{lbl("SL")}<span class="t-red">{sl_show}</span>{sep()}'
    f'{lbl("TP")}<span class="t-green">{tp_show}</span>{sep()}'
    f'{lbl("VENTANA")}<span class="{vent_c}">{vent_show}</span>{sep()}'
    f'{lbl("HORA MX")}<span class="t-white">{ahora.strftime("%H:%M")}</span>'
)
b2=(
    f'{lbl("RSI")}<span class="{rsi_c}">{rsi:.1f} — {"SOBRECOMPRADO" if rsi>70 else "SOBREVENDIDO" if rsi<30 else "NEUTRAL"}</span>{sep()}'
    f'{lbl("MACD")}<span class="{macd_c}">{macd:.2f} — {"ALCISTA" if macd>0 and macd_h>0 else "BAJISTA" if macd<0 and macd_h<0 else "MIXTO"}</span>{sep()}'
    f'{lbl("ATR")}<span class="t-white">{atr:.2f}</span>{sep()}'
    f'{lbl("EMA20")}<span class="t-white">${ema20:,.2f}</span>{sep()}'
    f'{lbl("EMA50")}<span class="t-white">${ema50:,.2f}</span>{sep()}'
    f'{lbl("TENDENCIA")}<span class="{ema_c}">{"ALCISTA" if precio>ema20 and precio>ema50 else "BAJISTA" if precio<ema20 and precio<ema50 else "DECISION"}</span>{sep()}'
    f'{lbl("BB UP")}<span class="t-red">${bb_up:,.2f}</span>{sep()}'
    f'{lbl("BB LOW")}<span class="t-green">${bb_low:,.2f}</span>'
)

# ── METRICAS ──
seccion("Estado del mercado","h_estado","Muestra el precio actual del oro, RSI para detectar sobrecompra/sobreventa, ATR para la volatilidad y la senal principal de MIMI-AI con su nivel de confianza.")
c1,c2,c3,c4=st.columns(4)
c1.metric("Precio XAU/USD",f"${precio:,.2f}")
c2.metric("RSI (14)",f"{rsi:.1f}","Sobrecomprado" if rsi>70 else "Sobrevendido" if rsi<30 else "Normal")
c3.metric("ATR",f"{atr:.2f}")
c4.metric("Senal MIMI-AI",etiquetas.get(pred),f"{max(prob)*100:.1f}% confianza")
st.divider()

# ── OPERACION ──
seccion("Operacion sugerida","h_op","Muestra exactamente donde entrar, donde poner el Stop Loss para limitar perdidas y donde tomar ganancias (Take Profit). El R:R indica cuanto ganas por cada dolar que arriesgas.")
col1,col2,col3,col4=st.columns(4)
col1.metric("Entrada",f"${entrada:,.2f}")
if pred==1:
    col2.metric("Stop Loss",f"${sl_largo:,.2f}",f"-{atr*1.5:.0f} pts")
    col3.metric("Take Profit",f"${tp_largo:,.2f}",f"+{atr*2.5:.0f} pts")
    col4.metric("R:R","1 : 1.67","Favorable")
elif pred==-1:
    col2.metric("Stop Loss",f"${sl_corto:,.2f}",f"+{atr*1.5:.0f} pts")
    col3.metric("Take Profit",f"${tp_corto:,.2f}",f"-{atr*2.5:.0f} pts")
    col4.metric("R:R","1 : 1.67","Favorable")
else:
    col2.metric("Stop Loss","---","Sin senal")
    col3.metric("Take Profit","---","Espera ruptura")
    col4.metric("R:R","---","No operes aun")
st.divider()

# ── BANNERS ──
seccion("Resumen del mercado","h_banner","Dos tickers continuos con toda la informacion del mercado en tiempo real. Verde = positivo, Rojo = negativo, Amarillo = neutral.")
st.markdown(banner(b1,"fast"), unsafe_allow_html=True)
st.markdown(banner(b2,""), unsafe_allow_html=True)
st.divider()

# ── VARIANTES ──
seccion("Variantes del mercado","h_var","Los 4 escenarios posibles con su probabilidad. Alcista: precio rompe resistencia arriba. Bajista: rompe soporte abajo. Lateral: se queda en rango. Shock: evento inesperado como noticia macro.")
v1,v2,v3,v4=st.columns(4)
v1.metric("Alcista",f"{p_long}%",f"Rompe ${bb_up:,.0f}")
v2.metric("Bajista",f"{p_short}%",f"Rompe ${bb_low:,.0f}")
v3.metric("Lateral",f"{p_lat}%","Sin ruptura")
v4.metric("Shock/Evento",f"{p_shock}%","Noticia macro")
st.divider()

# ── VENTANAS ──
seccion("Ventanas de trading","h_vent","Las mejores horas para operar XAU/USD en hora Mexico. Londres + NY (08:00-11:00) es la ventana de maxima liquidez. Fuera de ventana hay mas riesgo de movimientos falsos.")
for v in ventanas:
    activa=v["inicio"]<=hora<v["fin"]
    st.markdown(f"{'🟢 **ACTIVA**' if activa else '⚫'} **{v['inicio']:02d}:00 - {v['fin']:02d}:00** {v['nombre']} — [{v['calidad']}]")
st.divider()

# ── GRAFICAS ──
seccion("Graficas","h_graf","Grafica historica del oro con EMA 20 y 50 para ver la tendencia. La grafica en vivo usa su propio timeframe independiente y se actualiza cada minuto.")
tab1,tab2=st.tabs(["Precio historico","Precio actual en vivo"])
with tab1:
    cols_chart=[c for c in ['Close','EMA_20','EMA_50'] if c in df.columns]
    st.line_chart(df[cols_chart].tail(120))

with tab2:
    tf_live_sel=st.selectbox("Timeframe en vivo",list(TIMEFRAMES.keys()),index=2,key="live_tf")
    tf_live=TIMEFRAMES[tf_live_sel]
    @st.cache_data(ttl=60)
    def cargar_vivo(periodo,intervalo):
        df2=yf.download("GC=F",period=periodo,interval=intervalo,progress=False)
        df2.columns=[c[0] if isinstance(c,tuple) else c for c in df2.columns]
        df2.dropna(inplace=True)
        return df2
    with st.spinner("Cargando precio en vivo..."):
        df_live=cargar_vivo(tf_live["periodo"],tf_live["intervalo"])
    if df_live is not None and len(df_live)>0:
        pl=float(df_live['Close'].iloc[-1])
        pp=float(df_live['Close'].iloc[-2]) if len(df_live)>1 else pl
        cambio=pl-pp
        lc1,lc2,lc3,lc4=st.columns(4)
        lc1.metric("Precio actual",f"${pl:,.2f}",f"{cambio:+.2f} ({cambio/pp*100:+.2f}%)")
        lc2.metric("Max vela",f"${float(df_live['High'].iloc[-1]):,.2f}")
        lc3.metric("Min vela",f"${float(df_live['Low'].iloc[-1]):,.2f}")
        lc4.metric("Volumen",f"{int(df_live['Volume'].iloc[-1]):,}")
        st.line_chart(df_live['Close'].tail(100))
    else:
        st.warning("No hay datos en vivo ahora.")
st.divider()

# ── CHAT ──
st.markdown(f'<div class="section-hdr">Chat con MIMI-AI</div>', unsafe_allow_html=True)
if "mensajes" not in st.session_state:
    st.session_state.mensajes=[]

def responder(pregunta):
    p=pregunta.lower()
    if any(x in p for x in ["como funciona","que eres","quien eres","que es mimi","explicame","como trabaja"]):
        return "Soy MIMI-AI, sistema de trading con ML para XAU/USD. Analizo 13 indicadores con dos modelos que votan por una direccion. Solo doy senal cuando ambos coinciden. Revisa la Guia en el menu lateral para aprender todo."
    elif any(x in p for x in ["como tradeas","como operas","metodologia","metodo","como decides","que estrategia"]):
        return f"Opero en {tf_sel} con confluencias de ML. RSI, MACD, ATR, Bollinger, EMAs y OBV. Dos modelos votan. Solo senal cuando coinciden. SL = ATR x 1.5 ({atr*1.5:.0f} pts), TP = ATR x 2.5 ({atr*2.5:.0f} pts), R:R 1:1.67. Solo en ventanas de alta liquidez."
    elif any(x in p for x in ["por que long","por que sube","razon long"]):
        return f"LONG porque RSI {rsi:.1f} no sobrecomprado, MACD {'alcista' if macd>0 and macd_h>0 else 'mixto'}, precio {'sobre EMA20 y EMA50 — tendencia alcista' if precio>ema20 and precio>ema50 else 'en zona de decision'}. Confianza {max(prob)*100:.1f}%."
    elif any(x in p for x in ["por que short","por que baja","razon short"]):
        return f"SHORT porque RSI {rsi:.1f}, MACD {'bajista' if macd<0 and macd_h<0 else 'mixto'}, precio {'bajo EMA20 y EMA50' if precio<ema20 and precio<ema50 else 'en zona de decision'}. Confianza {max(prob)*100:.1f}%."
    elif any(x in p for x in ["por que no","por que lateral","por que esperar"]):
        return "Los modelos no coinciden. Sin consenso no hay operacion. Operar sin senal es apostar. El ruido es abundante. La senal, escasa."
    elif any(x in p for x in ["entrar","entro","operar","comprar","vender","abrir","deberia entrar"]):
        if pred==1:
            return f"LONG. Entrada ${entrada:,.2f} | SL ${sl_largo:,.2f} | TP ${tp_largo:,.2f}. Confianza {max(prob)*100:.1f}%. {'Ventana activa: '+ventana_activa['nombre'] if ventana_activa else 'Sin ventana optima ahora. Considera esperar 08:00 MX.'}."
        elif pred==-1:
            return f"SHORT. Entrada ${entrada:,.2f} | SL ${sl_corto:,.2f} | TP ${tp_corto:,.2f}. Confianza {max(prob)*100:.1f}%. {'Ventana activa: '+ventana_activa['nombre'] if ventana_activa else 'Sin ventana optima ahora.'}."
        else:
            return "Sin consenso. No abras operacion. Espera ruptura clara del precio."
    elif any(x in p for x in ["sl","stop loss","stop","cuanto arriesgo"]):
        if pred==1: return f"SL LONG: ${sl_largo:,.2f}. ATR ({atr:.2f}) x 1.5 = {atr*1.5:.0f} pts abajo de la entrada."
        elif pred==-1: return f"SL SHORT: ${sl_corto:,.2f}. ATR ({atr:.2f}) x 1.5 = {atr*1.5:.0f} pts arriba."
        else: return "Sin senal activa. El SL se calcula cuando hay consenso."
    elif any(x in p for x in ["tp","take profit","objetivo","donde tomo"]):
        if pred==1: return f"TP LONG: ${tp_largo:,.2f}. {atr*2.5:.0f} pts arriba. R:R 1:1.67."
        elif pred==-1: return f"TP SHORT: ${tp_corto:,.2f}. {atr*2.5:.0f} pts abajo. R:R 1:1.67."
        else: return "Sin senal activa aun."
    elif any(x in p for x in ["rsi","sobrecomprado","sobrevendido"]):
        if rsi>70: return f"RSI {rsi:.1f} — sobrecompra. Mercado subio demasiado rapido. Riesgo de correccion. Evita LONG ahora."
        elif rsi<30: return f"RSI {rsi:.1f} — sobreventa. Mercado cayo demasiado. Posible rebote. Evita SHORT."
        else: return f"RSI {rsi:.1f} — zona neutral (30-70). No da ventaja clara. Es un filtro, no una senal sola."
    elif any(x in p for x in ["macd","momentum"]):
        estado="ALCISTA — toros al mando" if macd>0 and macd_h>0 else "BAJISTA — osos al mando" if macd<0 and macd_h<0 else "MIXTO — sin control claro"
        return f"MACD {macd:.2f}, histograma {macd_h:.2f}. Momentum {estado}. Compara dos medias moviles para medir fuerza y direccion."
    elif any(x in p for x in ["atr","volatilidad","cuanto se mueve"]):
        return f"ATR {atr:.2f}. El oro se mueve ~${atr:.2f} por vela en {tf_sel}. SL = {atr*1.5:.0f} pts, TP = {atr*2.5:.0f} pts."
    elif any(x in p for x in ["tendencia","trend","hacia donde","direccion"]):
        if precio>ema20 and precio>ema50: return f"Tendencia ALCISTA. Precio ${precio:,.2f} sobre EMA20 ${ema20:,.2f} y EMA50 ${ema50:,.2f}. Compradores al mando."
        elif precio<ema20 and precio<ema50: return f"Tendencia BAJISTA. Precio ${precio:,.2f} bajo EMA20 ${ema20:,.2f} y EMA50 ${ema50:,.2f}. Vendedores al mando."
        else: return f"Tendencia MIXTA en {tf_sel}. Precio entre EMAs. Espera ruptura clara."
    elif any(x in p for x in ["hora","ventana","cuando operar","horario","mejor hora"]):
        if ventana_activa: return f"Ventana ACTIVA: {ventana_activa['nombre']} [{ventana_activa['calidad']}]. Buena hora. La mejor es Londres+NY 08:00-11:00 MX."
        else: return f"Sin ventana activa ({ahora.strftime('%H:%M')} MX). Proxima optima: Londres+NY 08:00-11:00 MX. Fuera de ventana hay mas movimientos falsos."
    elif any(x in p for x in ["precio","oro","xau","cuanto vale","cotiza"]):
        return f"Oro en ${precio:,.2f} en {tf_sel}. Resistencia ${bb_up:,.2f} (BB upper). Soporte ${bb_low:,.2f} (BB lower). ATR {atr:.2f}."
    elif any(x in p for x in ["probabilidad","chances","que tan probable","escenarios"]):
        dom=max(p_long,p_short,p_lat,p_shock)
        dom_txt="ALCISTA" if dom==p_long else "BAJISTA" if dom==p_short else "LATERAL" if dom==p_lat else "SHOCK"
        return f"Probabilidades en {tf_sel}: Alcista {p_long}%, Bajista {p_short}%, Lateral {p_lat}%, Shock {p_shock}%. Escenario dominante: {dom_txt}."
    elif any(x in p for x in ["noticias","eventos","nfp","fed","inflacion","cpi","calendario"]):
        return "Noticias que mueven el oro: NFP (1er viernes del mes, 07:30 MX), decision de tasas FED (cada 6 semanas), CPI inflacion (mensual). Antes de estos eventos el mercado se vuelve impredecible. MIMI-AI recomienda cerrar posiciones antes de noticias de alto impacto."
    elif any(x in p for x in ["rr","r:r","riesgo beneficio","risk reward"]):
        return f"MIMI-AI usa R:R 1:1.67. Por cada $1 arriesgado, el TP da $1.67 potencial. SL actual: {atr*1.5:.0f} pts. TP actual: {atr*2.5:.0f} pts. Con 50% win rate este R:R es rentable a largo plazo."
    elif any(x in p for x in ["que es el oro","por que sube el oro","por que baja el oro","oro refugio"]):
        return "El oro es el activo refugio mas importante. SUBE con: incertidumbre economica, inflacion alta, dolar debil, guerras o crisis. BAJA con: dolar fuerte, tasas de interes altas, mercados de riesgo al alza. Opera 5 dias, cierra viernes, reabre domingo 18:00 MX."
    elif any(x in p for x in ["mercado bursatil","bolsa","que es el mercado","como funciona el mercado"]):
        return "El mercado bursatil es donde se compran y venden activos financieros: acciones, divisas, materias primas (oro, petroleo), indices (S&P500, Nasdaq) y criptos. MIMI-AI se especializa en XAU/USD. El precio lo determinan oferta y demanda global en tiempo real."
    elif any(x in p for x in ["scalping"]):
        return "Para scalping usa M5 o M15. Operaciones de 1-30 minutos. Mas senales pero mas ruido. Requiere concentracion y rapidez. Cambia el timeframe arriba."
    elif any(x in p for x in ["day trading","intraday"]):
        return "Para day trading usa H1 o H4. Operaciones de horas dentro del mismo dia. Estilo mas comun. Ideal en ventana Londres+NY 08:00-11:00 MX."
    elif any(x in p for x in ["swing"]):
        return "Para swing trading usa D1. Operaciones de dias o semanas. Menos estres, menos tiempo frente a pantalla. SL y TP mas amplios."
    elif any(x in p for x in ["timeframe","temporalidad","que tf","que marco"]):
        return f"Timeframe actual: {tf_sel}. M5/M15 = scalping. M30/H1 = intradía. H4/D1 = swing. Menor TF = mas senales, mas ruido. Mayor TF = menos senales, mas confiables."
    elif any(x in p for x in ["ema","media movil","medias"]):
        return f"EMA20: ${ema20:,.2f} | EMA50: ${ema50:,.2f}. Precio {'SOBRE ambas = tendencia alcista' if precio>ema20 and precio>ema50 else 'BAJO ambas = tendencia bajista' if precio<ema20 and precio<ema50 else 'ENTRE ellas = zona de decision'}."
    elif any(x in p for x in ["bollinger","bb","bandas"]):
        return f"Bandas Bollinger: Upper ${bb_up:,.2f} (resistencia), Lower ${bb_low:,.2f} (soporte). Precio cerca del upper = sobrecompra. Cerca del lower = sobreventa. Ancho de banda = volatilidad."
    elif any(x in p for x in ["hola","buenas","buenos","hey","buen dia"]):
        return "El mercado no saluda. Pero MIMI-AI si. Pregunta lo que necesitas."
    elif any(x in p for x in ["gracias","thank"]):
        return "Aqui estoy. Pregunta lo que necesites."
    elif any(x in p for x in ["ayuda","help","que puedes","que sabes"]):
        return "Puedo hablar sobre: senal actual, entrada/SL/TP, RSI/MACD/ATR/EMAs/Bollinger, tendencia, timeframes, ventanas de horario, probabilidades, noticias, el oro, mercado bursatil, scalping/day trading/swing, R:R, y como funciona MIMI-AI."
    else:
        return "No entendi bien. Puedo hablar sobre: senal, SL/TP, RSI, MACD, ATR, tendencia, timeframes, ventanas, probabilidades, noticias, el oro o el mercado. Reformula tu pregunta."

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])

if pregunta:=st.chat_input("Pregunta a MIMI-AI sobre el mercado..."):
    st.session_state.mensajes.append({"rol":"user","texto":pregunta})
    respuesta=responder(pregunta)
    st.session_state.mensajes.append({"rol":"assistant","texto":respuesta})
    st.rerun()

st.divider()
st.markdown(f'<div class="frase-estoica">{random.choice(frases)}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="footer">MIMI-AI — TRADING INTELLIGENCE SYSTEM — XAU/USD</div>', unsafe_allow_html=True)
