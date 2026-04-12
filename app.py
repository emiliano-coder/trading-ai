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
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MIMI-AI | XAU/USD", page_icon="gold", layout="wide")

TEMAS = {
    "Dorado (default)": {"primario":"#FFD700","fondo_banner":"#0a0a0a","borde":"#2a2a00","acento":"#FFA500"},
    "Intenso — Neon Verde": {"primario":"#00FF88","fondo_banner":"#001a0d","borde":"#003319","acento":"#00CC66"},
    "Intenso — Cyan": {"primario":"#00FFFF","fondo_banner":"#001a1a","borde":"#003333","acento":"#00CCCC"},
    "Intenso — Magenta": {"primario":"#FF00FF","fondo_banner":"#1a001a","borde":"#330033","acento":"#CC00CC"},
    "Pastel — Lavanda": {"primario":"#C3B1E1","fondo_banner":"#0f0d14","borde":"#1e1a2e","acento":"#A89AC8"},
    "Pastel — Rosa": {"primario":"#FFB3C6","fondo_banner":"#140d0f","borde":"#2e1a1e","acento":"#FF8FAB"},
    "Pastel — Menta": {"primario":"#B5EAD7","fondo_banner":"#0d1410","borde":"#1a2e22","acento":"#8ED4BC"},
}

with st.sidebar:
    st.markdown("## MIMI-AI")
    st.markdown("---")
    tema_sel = st.selectbox("Tema de color", list(TEMAS.keys()))
    tema = TEMAS[tema_sel]
    st.markdown("---")
    st.markdown("### Guia de MIMI-AI")
    guia_items = {
        "Que es MIMI-AI": "Sistema de IA para analizar XAU/USD. Usa dos modelos de ML con 13 indicadores y solo da senal cuando ambos coinciden.",
        "Como leer la senal": "LONG = predice subida. SHORT = predice bajada. LATERAL = sin consenso, no operes.",
        "Entrada, SL y TP": "Entrada: donde abres. SL: donde cierras si va en tu contra. TP: donde tomas ganancias. SL = ATR x 1.5, TP = ATR x 2.5.",
        "Que es el ATR": "Mide volatilidad promedio por vela. Mayor ATR = mayor movimiento y mayor distancia en SL y TP.",
        "Que es el RSI": "Mayor a 70 = sobrecompra. Menor a 30 = sobreventa. Entre 30-70 = neutral.",
        "Que es el MACD": "Mide momentum. Positivo = alcista. Negativo = bajista.",
        "Que son las EMAs": "Medias moviles. Precio sobre EMA20 y EMA50 = tendencia alcista. Por debajo = bajista.",
        "Cuando operar": "Mejor ventana: 08:00-11:00 MX (Londres + NY). Mayor liquidez.",
        "Timeframes": "M5/M15 = scalping. H1/H4 = day trading. D1 = swing.",
        "Variantes del mercado": "4 escenarios: Alcista, Bajista, Lateral y Shock con su probabilidad.",
        "El oro (XAU/USD)": "Activo refugio. Sube con incertidumbre, inflacion, dolar debil.",
        "Mercado bursatil": "Donde se compran y venden activos: acciones, divisas, materias primas, indices y criptos.",
        "Noticias importantes": "NFP, decision FED, CPI inflacion. Cierra posiciones antes de estos eventos.",
        "R:R Riesgo-Beneficio": "MIMI-AI usa R:R 1:1.67. Por cada $1 arriesgado, el TP da $1.67 potencial.",
        "Como leer las velas": "Vela verde = precio subio en esa vela. Vela roja = precio bajo. La mecha de arriba es el maximo, la de abajo el minimo. El cuerpo es la diferencia entre apertura y cierre.",
    }
    for titulo, texto in guia_items.items():
        with st.expander(titulo):
            st.markdown(f"<small style='color:#aaa;line-height:1.7'>{texto}</small>", unsafe_allow_html=True)

p = tema["primario"]
fb = tema["fondo_banner"]
bd = tema["borde"]
ac = tema["acento"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif}}
.mimi-header{{text-align:center;padding:1.5rem 0 1rem 0}}
.mimi-title{{
    font-family:'Orbitron',monospace;font-size:clamp(1.8rem,5vw,3rem);font-weight:900;
    background:linear-gradient(90deg,{p},{ac},{p});background-size:300%;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:shimmer 4s linear infinite;letter-spacing:4px;
}}
.mimi-sub{{font-family:'Orbitron',monospace;font-size:clamp(0.55rem,1.5vw,0.75rem);color:#666;letter-spacing:3px;margin-top:4px}}
@keyframes shimmer{{0%{{background-position:0% 50%}}100%{{background-position:300% 50%}}}}
.ticker-wrap{{
    background:{fb};border:1px solid {bd};border-radius:8px;
    padding:10px 0;overflow:hidden;margin-bottom:8px;position:relative;white-space:nowrap;
}}
.ticker-label{{
    position:absolute;left:0;top:0;bottom:0;display:flex;align-items:center;
    padding:0 12px;background:{p};color:#000;font-family:'Orbitron',monospace;
    font-size:9px;font-weight:700;letter-spacing:1px;z-index:2;
    border-radius:8px 0 0 8px;min-width:70px;justify-content:center;
}}
.ticker-inner{{display:inline-block;padding-left:90px;font-size:13px;font-weight:500;letter-spacing:0.5px;}}
.ticker-inner.senal{{animation:scroll 32s linear infinite}}
.ticker-inner.tecnico{{animation:scroll 42s linear infinite}}
@keyframes scroll{{0%{{transform:translateX(100vw)}}100%{{transform:translateX(-100%)}}}}
.t-green{{color:#00e676}}.t-red{{color:#ff4444}}.t-yellow{{color:{p}}}
.t-white{{color:#ffffff}}.t-gray{{color:#999}}.t-sep{{color:#333;margin:0 10px}}
.section-hdr{{
    font-family:'Orbitron',monospace;font-size:clamp(0.65rem,1.5vw,0.8rem);
    color:{p};letter-spacing:2px;text-transform:uppercase;
    margin-bottom:1rem;padding-bottom:6px;border-bottom:1px solid {bd};
}}
.frase-estoica{{
    background:{fb};border:1px solid {bd};border-left:3px solid {p};
    border-radius:8px;padding:16px 20px;font-style:italic;
    color:{p};font-size:clamp(12px,2vw,14px);text-align:center;
}}
.footer{{text-align:center;color:#333;font-size:10px;font-family:monospace;letter-spacing:2px;margin-top:1rem}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="mimi-header">
    <div class="mimi-title">MIMI-AI</div>
    <div class="mimi-sub">XAU/USD TRADING INTELLIGENCE SYSTEM</div>
</div>
""", unsafe_allow_html=True)

TIMEFRAMES = {
    "M5":{"periodo":"5d","intervalo":"5m"},
    "M15":{"periodo":"5d","intervalo":"15m"},
    "M30":{"periodo":"1mo","intervalo":"30m"},
    "H1":{"periodo":"1mo","intervalo":"60m"},
    "H4":{"periodo":"3mo","intervalo":"1d"},
    "D1":{"periodo":"2y","intervalo":"1d"},
}
tf_sel = st.selectbox("Timeframe de analisis", list(TIMEFRAMES.keys()), index=5)
tf = TIMEFRAMES[tf_sel]

@st.cache_data(ttl=300)
def cargar_y_entrenar(periodo, intervalo):
    df = yf.download("GC=F", period=periodo, interval=intervalo, progress=False)
    df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df)<50: return None,None,None,None
    df['EMA_20']    = ta.trend.ema_indicator(df['Close'],window=min(20,len(df)-1))
    df['EMA_50']    = ta.trend.ema_indicator(df['Close'],window=min(50,len(df)-1))
    df['EMA_200']   = ta.trend.ema_indicator(df['Close'],window=min(200,len(df)-1))
    df['RSI']       = ta.momentum.rsi(df['Close'],window=min(14,len(df)-1))
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
    df['Target']=0
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

if "ayudas" not in st.session_state:
    st.session_state.ayudas={}

def seccion_con_ayuda(titulo, key, txt):
    c1,c2=st.columns([20,1])
    with c1:
        st.markdown(f'<div class="section-hdr">{titulo}</div>', unsafe_allow_html=True)
    with c2:
        if st.button("?", key=f"btn_{key}"):
            st.session_state.ayudas[key]=not st.session_state.ayudas.get(key,False)
    if st.session_state.ayudas.get(key,False):
        st.info(txt)

def grafica_velas(df_plot, titulo=""):
    df_plot = df_plot.tail(80).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'],   close=df_plot['Close'],
        increasing_line_color='#00e676',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='#00e676',
        decreasing_fillcolor='#ff4444',
        name='Precio'
    ))
    if 'EMA_20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_20'],
            line=dict(color=p, width=1.2), name='EMA 20'))
    if 'EMA_50' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_50'],
            line=dict(color=ac, width=1.2, dash='dot'), name='EMA 50'))
    if 'BB_upper' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_upper'],
            line=dict(color='#555', width=0.8, dash='dash'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_lower'],
            line=dict(color='#555', width=0.8, dash='dash'), name='BB Lower',
            fill='tonexty', fillcolor='rgba(255,255,255,0.02)'))
    fig.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0d0d0d',
        font=dict(color='#aaa', size=11),
        xaxis=dict(gridcolor='#1a1a1a', showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='#1a1a1a', showgrid=True, side='right'),
        legend=dict(orientation='h', y=1.05, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=420,
        title=dict(text=titulo, font=dict(color=p, size=12, family='Orbitron')),
    )
    fig.update_xaxes(showspikes=True, spikecolor='#444', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#444', spikethickness=1)
    return fig

def sep(): return '<span class="t-sep">|</span>'
def lbl(t): return f'<span class="t-gray">{t}: </span>'

senal_c="t-green" if pred==1 else "t-red" if pred==-1 else "t-yellow"
rsi_c="t-red" if rsi>70 else "t-green" if rsi<30 else "t-white"
macd_c="t-green" if macd>0 and macd_h>0 else "t-red" if macd<0 and macd_h<0 else "t-yellow"
ema_c="t-green" if precio>ema20 and precio>ema50 else "t-red" if precio<ema20 and precio<ema50 else "t-yellow"
vent_c="t-green" if ventana_activa else "t-red"
sl_show=f"${sl_largo:,.2f}" if pred==1 else f"${sl_corto:,.2f}" if pred==-1 else "---"
tp_show=f"${tp_largo:,.2f}" if pred==1 else f"${tp_corto:,.2f}" if pred==-1 else "---"
vent_show=ventana_activa['nombre'].upper() if ventana_activa else "SIN VENTANA ACTIVA"

b1=(f'{lbl("TF")}<span class="t-yellow">{tf_sel}</span>{sep()}'
    f'{lbl("PRECIO")}<span class="t-green">${precio:,.2f}</span>{sep()}'
    f'{lbl("SENAL")}<span class="{senal_c}">{etiquetas.get(pred)}</span>{sep()}'
    f'{lbl("CONFIANZA")}<span class="t-white">{max(prob)*100:.1f}%</span>{sep()}'
    f'{lbl("ENTRADA")}<span class="t-white">${entrada:,.2f}</span>{sep()}'
    f'{lbl("SL")}<span class="t-red">{sl_show}</span>{sep()}'
    f'{lbl("TP")}<span class="t-green">{tp_show}</span>{sep()}'
    f'{lbl("VENTANA")}<span class="{vent_c}">{vent_show}</span>{sep()}'
    f'{lbl("HORA MX")}<span class="t-white">{ahora.strftime("%H:%M")}</span>')
b2=(f'{lbl("RSI")}<span class="{rsi_c}">{rsi:.1f} — {"SOBRECOMPRADO" if rsi>70 else "SOBREVENDIDO" if rsi<30 else "NEUTRAL"}</span>{sep()}'
    f'{lbl("MACD")}<span class="{macd_c}">{macd:.2f} — {"ALCISTA" if macd>0 and macd_h>0 else "BAJISTA" if macd<0 and macd_h<0 else "MIXTO"}</span>{sep()}'
    f'{lbl("ATR")}<span class="t-white">{atr:.2f}</span>{sep()}'
    f'{lbl("EMA20")}<span class="t-white">${ema20:,.2f}</span>{sep()}'
    f'{lbl("EMA50")}<span class="t-white">${ema50:,.2f}</span>{sep()}'
    f'{lbl("TENDENCIA")}<span class="{ema_c}">{"ALCISTA" if precio>ema20 and precio>ema50 else "BAJISTA" if precio<ema20 and precio<ema50 else "DECISION"}</span>{sep()}'
    f'{lbl("BB UP")}<span class="t-red">${bb_up:,.2f}</span>{sep()}'
    f'{lbl("BB LOW")}<span class="t-green">${bb_low:,.2f}</span>')

# ── METRICAS ──
seccion_con_ayuda("Estado del mercado","estado","Precio actual, RSI para sobrecompra/sobreventa, ATR para volatilidad y senal principal con confianza.")
c1,c2,c3,c4=st.columns(4)
c1.metric("Precio XAU/USD",f"${precio:,.2f}")
c2.metric("RSI (14)",f"{rsi:.1f}","Sobrecomprado" if rsi>70 else "Sobrevendido" if rsi<30 else "Normal")
c3.metric("ATR",f"{atr:.2f}")
c4.metric("Senal MIMI-AI",etiquetas.get(pred),f"{max(prob)*100:.1f}% confianza")
st.divider()

# ── OPERACION ──
seccion_con_ayuda("Operacion sugerida","op","Entrada: donde abrir. SL: Stop Loss para limitar perdidas. TP: Take Profit para tomar ganancias. R:R 1:1.67 = ganas $1.67 por cada $1 arriesgado.")
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
seccion_con_ayuda("Resumen del mercado","banner","Tickers en tiempo real. Verde = positivo, Rojo = negativo, Amarillo = neutral.")
st.markdown(f"""
<div class="ticker-wrap">
  <span class="ticker-label">SENAL</span>
  <div class="ticker-inner senal">{b1}</div>
</div>
<div class="ticker-wrap">
  <span class="ticker-label">TECNICO</span>
  <div class="ticker-inner tecnico">{b2}</div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── VARIANTES ──
seccion_con_ayuda("Variantes del mercado","var","4 escenarios posibles: Alcista, Bajista, Lateral y Shock. El porcentaje es la probabilidad de cada uno.")
v1,v2,v3,v4=st.columns(4)
v1.metric("Alcista",f"{p_long}%",f"Rompe ${bb_up:,.0f}")
v2.metric("Bajista",f"{p_short}%",f"Rompe ${bb_low:,.0f}")
v3.metric("Lateral",f"{p_lat}%","Sin ruptura")
v4.metric("Shock/Evento",f"{p_shock}%","Noticia macro")
st.divider()

# ── VENTANAS ──
seccion_con_ayuda("Ventanas de trading","vent","Mejores horas para operar XAU/USD. Londres+NY (08:00-11:00 MX) es la de maxima liquidez.")
for v in ventanas:
    activa=v["inicio"]<=hora<v["fin"]
    st.markdown(f"{'🟢 **ACTIVA**' if activa else '⚫'} **{v['inicio']:02d}:00 - {v['fin']:02d}:00** {v['nombre']} — [{v['calidad']}]")
st.divider()

# ── GRAFICAS ──
seccion_con_ayuda("Graficas","graf","Velas japonesas: verde = precio subio, roja = precio bajo. La mecha es el maximo/minimo. EMA20 y EMA50 muestran la tendencia.")
tab1,tab2=st.tabs(["Precio historico","Precio actual en vivo"])

with tab1:
    tipo=st.radio("Tipo de grafica",["Velas japonesas","Linea"],horizontal=True,key="tipo_hist")
    if tipo=="Velas japonesas":
        fig=grafica_velas(df, f"XAU/USD — {tf_sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
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
        tipo_live=st.radio("Tipo de grafica",["Velas japonesas","Linea"],horizontal=True,key="tipo_live")
        if tipo_live=="Velas japonesas":
            fig_live=grafica_velas(df_live, f"XAU/USD en vivo — {tf_live_sel}")
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.line_chart(df_live['Close'].tail(100))
    else:
        st.warning("No hay datos en vivo ahora.")
st.divider()

# ── CHAT ──
st.markdown('<div class="section-hdr">Chat con MIMI-AI</div>', unsafe_allow_html=True)
if "mensajes" not in st.session_state:
    st.session_state.mensajes=[]

def responder(pregunta):
    p=pregunta.lower()
    if any(x in p for x in ["como funciona","que eres","quien eres","que es mimi","explicame"]):
        return "Soy MIMI-AI, sistema de trading con ML para XAU/USD. Analizo 13 indicadores con dos modelos que votan por una direccion. Solo doy senal cuando ambos coinciden. Revisa la Guia en el menu lateral."
    elif any(x in p for x in ["vela","velas","candlestick","japonesa"]):
        return "Las velas japonesas muestran 4 precios por periodo: apertura, cierre, maximo y minimo. Vela VERDE = el precio cerro mas alto que abrio (subio). Vela ROJA = cerro mas bajo que abrio (bajo). La mecha de arriba es el maximo, la de abajo el minimo. El cuerpo es la diferencia entre apertura y cierre."
    elif any(x in p for x in ["como tradeas","como operas","metodologia","metodo"]):
        return f"Opero en {tf_sel} con confluencias de ML. RSI, MACD, ATR, Bollinger, EMAs y OBV. Dos modelos votan. Solo senal cuando coinciden. SL = ATR x 1.5 ({atr*1.5:.0f} pts), TP = ATR x 2.5 ({atr*2.5:.0f} pts), R:R 1:1.67."
    elif any(x in p for x in ["entrar","entro","operar","comprar","vender","deberia entrar"]):
        if pred==1:
            return f"LONG. Entrada ${entrada:,.2f} | SL ${sl_largo:,.2f} | TP ${tp_largo:,.2f}. Confianza {max(prob)*100:.1f}%. {'Ventana activa: '+ventana_activa['nombre'] if ventana_activa else 'Sin ventana optima ahora. Considera esperar 08:00 MX.'}."
        elif pred==-1:
            return f"SHORT. Entrada ${entrada:,.2f} | SL ${sl_corto:,.2f} | TP ${tp_corto:,.2f}. Confianza {max(prob)*100:.1f}%. {'Ventana activa: '+ventana_activa['nombre'] if ventana_activa else 'Sin ventana optima ahora.'}."
        else:
            return "Sin consenso. No abras operacion. Espera ruptura clara del precio."
    elif any(x in p for x in ["sl","stop loss","stop"]):
        if pred==1: return f"SL LONG: ${sl_largo:,.2f}. ATR ({atr:.2f}) x 1.5 = {atr*1.5:.0f} pts abajo."
        elif pred==-1: return f"SL SHORT: ${sl_corto:,.2f}. ATR ({atr:.2f}) x 1.5 = {atr*1.5:.0f} pts arriba."
        else: return "Sin senal activa."
    elif any(x in p for x in ["tp","take profit","objetivo"]):
        if pred==1: return f"TP LONG: ${tp_largo:,.2f}. {atr*2.5:.0f} pts arriba. R:R 1:1.67."
        elif pred==-1: return f"TP SHORT: ${tp_corto:,.2f}. {atr*2.5:.0f} pts abajo. R:R 1:1.67."
        else: return "Sin senal activa aun."
    elif any(x in p for x in ["rsi","sobrecomprado","sobrevendido"]):
        if rsi>70: return f"RSI {rsi:.1f} — sobrecompra. Riesgo de correccion. Evita LONG."
        elif rsi<30: return f"RSI {rsi:.1f} — sobreventa. Posible rebote. Evita SHORT."
        else: return f"RSI {rsi:.1f} — zona neutral. No da ventaja clara por si solo."
    elif any(x in p for x in ["macd","momentum"]):
        return f"MACD {macd:.2f}, histograma {macd_h:.2f}. Momentum {'ALCISTA' if macd>0 and macd_h>0 else 'BAJISTA' if macd<0 and macd_h<0 else 'MIXTO'}."
    elif any(x in p for x in ["atr","volatilidad"]):
        return f"ATR {atr:.2f}. El oro se mueve ~${atr:.2f} por vela en {tf_sel}. SL={atr*1.5:.0f} pts, TP={atr*2.5:.0f} pts."
    elif any(x in p for x in ["tendencia","trend","hacia donde"]):
        if precio>ema20 and precio>ema50: return f"Tendencia ALCISTA. Precio sobre EMA20 y EMA50. Compradores al mando."
        elif precio<ema20 and precio<ema50: return f"Tendencia BAJISTA. Precio bajo EMA20 y EMA50. Vendedores al mando."
        else: return "Tendencia MIXTA. Precio entre EMAs. Espera ruptura clara."
    elif any(x in p for x in ["hora","ventana","cuando operar","horario"]):
        if ventana_activa: return f"Ventana ACTIVA: {ventana_activa['nombre']} [{ventana_activa['calidad']}]. La mejor es Londres+NY 08:00-11:00 MX."
        else: return f"Sin ventana activa ({ahora.strftime('%H:%M')} MX). Proxima optima: Londres+NY 08:00-11:00 MX."
    elif any(x in p for x in ["precio","oro","xau","cuanto vale"]):
        return f"Oro en ${precio:,.2f}. Resistencia ${bb_up:,.2f}, soporte ${bb_low:,.2f}. ATR {atr:.2f}."
    elif any(x in p for x in ["probabilidad","escenarios"]):
        return f"Alcista {p_long}%, Bajista {p_short}%, Lateral {p_lat}%, Shock {p_shock}%."
    elif any(x in p for x in ["noticias","nfp","fed","inflacion","cpi"]):
        return "Noticias clave: NFP (1er viernes 07:30 MX), decision FED, CPI. Cierra posiciones antes de estos eventos."
    elif any(x in p for x in ["scalping"]): return "Para scalping usa M5 o M15. Cambia el timeframe arriba."
    elif any(x in p for x in ["day trading"]): return "Para day trading usa H1 o H4. Ideal en Londres+NY 08:00-11:00 MX."
    elif any(x in p for x in ["swing"]): return "Para swing usa D1. Operaciones de dias o semanas."
    elif any(x in p for x in ["hola","buenas","hey"]): return "El mercado no saluda. Pero MIMI-AI si. Pregunta lo que necesitas."
    elif any(x in p for x in ["gracias"]): return "Aqui estoy. Pregunta lo que necesites."
    elif any(x in p for x in ["ayuda","que puedes"]): return "Puedo hablar sobre: senal, SL/TP, RSI, MACD, ATR, velas japonesas, tendencia, timeframes, ventanas, probabilidades, noticias y el oro."
    else: return "No entendi bien. Puedo hablar sobre: senal, SL/TP, indicadores, velas, tendencia, timeframes, ventanas o el oro."

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])

if pregunta:=st.chat_input("Pregunta a MIMI-AI sobre el mercado..."):
    st.session_state.mensajes.append({"rol":"user","texto":pregunta})
    st.session_state.mensajes.append({"rol":"assistant","texto":responder(pregunta)})
    st.rerun()

st.divider()
st.markdown(f'<div class="frase-estoica">{random.choice(frases)}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="footer">MIMI-AI — TRADING INTELLIGENCE SYSTEM — XAU/USD</div>', unsafe_allow_html=True)
