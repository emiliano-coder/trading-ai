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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.mimi-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.mimi-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #FFD700, #FFA500, #FF6B00, #FFD700);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    letter-spacing: 6px;
    text-shadow: none;
}
.mimi-sub {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    color: #888;
    letter-spacing: 4px;
    margin-top: 4px;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}

.ticker-wrap {
    background: #0a0a0a;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 10px 0;
    overflow: hidden;
    margin-bottom: 8px;
    position: relative;
}
.ticker-label {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    padding: 0 14px;
    background: #FFD700;
    color: #000;
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    z-index: 2;
    border-radius: 8px 0 0 8px;
}
.ticker-inner {
    display: inline-block;
    white-space: nowrap;
    padding-left: 120px;
    animation: ticker 25s linear infinite;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.ticker-inner.slow {
    animation-duration: 35s;
}
@keyframes ticker {
    0%   { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}
.t-green  { color: #00e676; }
.t-red    { color: #ff4444; }
.t-yellow { color: #FFD700; }
.t-white  { color: #ffffff; }
.t-gray   { color: #aaaaaa; }
.t-sep    { color: #444; margin: 0 10px; }

.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    color: #FFD700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #222;
}

.guide-card {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-left: 3px solid #FFD700;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.guide-title {
    color: #FFD700;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
    font-family: 'Orbitron', monospace;
    letter-spacing: 1px;
}
.guide-text {
    color: #ccc;
    font-size: 13px;
    line-height: 1.7;
}

.frase-estoica {
    background: #0a0a0a;
    border: 1px solid #333;
    border-left: 3px solid #FFD700;
    border-radius: 8px;
    padding: 16px 20px;
    font-style: italic;
    color: #FFD700;
    font-size: 14px;
    text-align: center;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="mimi-header">
    <div class="mimi-title">MIMI-AI</div>
    <div class="mimi-sub">XAU/USD TRADING INTELLIGENCE SYSTEM</div>
</div>
""", unsafe_allow_html=True)

TIMEFRAMES = {
    "M5":  {"periodo": "5d",  "intervalo": "5m"},
    "M15": {"periodo": "5d",  "intervalo": "15m"},
    "M30": {"periodo": "1mo", "intervalo": "30m"},
    "H1":  {"periodo": "1mo", "intervalo": "60m"},
    "H4":  {"periodo": "3mo", "intervalo": "1d"},
    "D1":  {"periodo": "2y",  "intervalo": "1d"},
}

col_tf, col_chart = st.columns([1, 3])
with col_tf:
    tf_sel = st.selectbox("Timeframe de analisis", list(TIMEFRAMES.keys()), index=5)
tf = TIMEFRAMES[tf_sel]

@st.cache_data(ttl=300)
def cargar_y_entrenar(periodo, intervalo):
    df = yf.download("GC=F", period=periodo, interval=intervalo, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    if len(df) < 50:
        return None, None, None, None
    df['EMA_20']    = ta.trend.ema_indicator(df['Close'], window=min(20, len(df)-1))
    df['EMA_50']    = ta.trend.ema_indicator(df['Close'], window=min(50, len(df)-1))
    df['EMA_200']   = ta.trend.ema_indicator(df['Close'], window=min(200, len(df)-1))
    df['RSI']       = ta.momentum.rsi(df['Close'], window=min(14, len(df)-1))
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
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    df['Target'] = 0
    df.loc[df['Future_Return'] >  0.003, 'Target'] =  1
    df.loc[df['Future_Return'] < -0.003, 'Target'] = -1
    df.dropna(inplace=True)
    if len(df) < 20:
        return None, None, None, None
    features = ['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K',
                'Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    features = [f for f in features if f in df.columns]
    X, y = df[features], df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    mejor = rf if accuracy_score(y_test, rf.predict(X_test)) >= accuracy_score(y_test, gb.predict(X_test)) else gb
    return df, mejor, scaler, features

with st.spinner("MIMI-AI analizando mercado..."):
    df, modelo, scaler, features = cargar_y_entrenar(tf["periodo"], tf["intervalo"])

if df is None:
    st.error("Datos insuficientes para este timeframe. Intenta D1 o H4.")
    st.stop()

precio  = float(df['Close'].iloc[-1])
rsi     = float(df['RSI'].iloc[-1])
atr     = float(df['ATR'].iloc[-1])
bb_up   = float(df['BB_upper'].iloc[-1])
bb_low  = float(df['BB_lower'].iloc[-1])
macd    = float(df['MACD'].iloc[-1])
macd_h  = float(df['MACD_hist'].iloc[-1])
ema20   = float(df['EMA_20'].iloc[-1])
ema50   = float(df['EMA_50'].iloc[-1])
ultima  = df[features].iloc[-1:]
X_sc    = scaler.transform(ultima)
pred    = modelo.predict(X_sc)[0]
prob    = modelo.predict_proba(X_sc)[0]

etiquetas = {1: "LONG - SUBE", 0: "LATERAL", -1: "SHORT - BAJA"}
entrada  = precio
sl_largo = round(precio - atr * 1.5, 2)
sl_corto = round(precio + atr * 1.5, 2)
tp_largo = round(precio + atr * 2.5, 2)
tp_corto = round(precio - atr * 2.5, 2)

mx   = pytz.timezone('America/Mexico_City')
ahora = datetime.now(mx)
hora = ahora.hour
ventanas = [
    {"nombre": "Londres abre", "inicio": 3,  "fin": 5,  "calidad": "Alta"},
    {"nombre": "Londres + NY", "inicio": 8,  "fin": 11, "calidad": "Maxima"},
    {"nombre": "NY tarde",     "inicio": 12, "fin": 14, "calidad": "Media"},
    {"nombre": "Cierre NY",    "inicio": 15, "fin": 17, "calidad": "Baja"},
]
ventana_activa = next((v for v in ventanas if v["inicio"] <= hora < v["fin"]), None)

frases = [
    "El mercado revela lo que eres. No lo que quieres.",
    "No controlas el precio. Controlas tu reaccion.",
    "La paciencia no es debilidad. Es claridad.",
    "Una perdida aceptada a tiempo es una victoria de caracter.",
    "El ruido es abundante. La senal, escasa.",
    "El que espera el momento perfecto no opera. Opera cuando la razon lo indica.",
    "Disciplina hoy. Libertad manana.",
    "El mercado premia la claridad, no la velocidad.",
    "Quien controla sus emociones, controla su capital.",
]

# ── METRICAS ──
st.markdown('<div class="section-title">Estado del mercado</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio XAU/USD", f"${precio:,.2f}")
c2.metric("RSI (14)", f"{rsi:.1f}", "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Normal")
c3.metric("ATR", f"{atr:.2f}")
c4.metric("Senal MIMI-AI", etiquetas.get(pred), f"{max(prob)*100:.1f}% confianza")

st.divider()

# ── OPERACION ──
st.markdown('<div class="section-title">Operacion sugerida</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Entrada", f"${entrada:,.2f}")
if pred == 1:
    col2.metric("Stop Loss", f"${sl_largo:,.2f}", f"-{atr*1.5:.0f} pts")
    col3.metric("Take Profit", f"${tp_largo:,.2f}", f"+{atr*2.5:.0f} pts")
    col4.metric("R:R", "1 : 1.67", "Favorable")
elif pred == -1:
    col2.metric("Stop Loss", f"${sl_corto:,.2f}", f"+{atr*1.5:.0f} pts")
    col3.metric("Take Profit", f"${tp_corto:,.2f}", f"-{atr*2.5:.0f} pts")
    col4.metric("R:R", "1 : 1.67", "Favorable")
else:
    col2.metric("Stop Loss", "---", "Sin senal")
    col3.metric("Take Profit", "---", "Espera ruptura")
    col4.metric("R:R", "---", "No operes aun")

st.divider()

# ── BANNERS ──
st.markdown('<div class="section-title">Resumen del mercado</div>', unsafe_allow_html=True)

senal_c = "t-green" if pred == 1 else "t-red" if pred == -1 else "t-yellow"
rsi_c   = "t-red" if rsi > 70 else "t-green" if rsi < 30 else "t-white"
macd_c  = "t-green" if macd > 0 and macd_h > 0 else "t-red" if macd < 0 and macd_h < 0 else "t-yellow"
ema_c   = "t-green" if precio > ema20 and precio > ema50 else "t-red" if precio < ema20 and precio < ema50 else "t-yellow"
vent_c  = "t-green" if ventana_activa else "t-red"
conf_c  = "t-green" if max(prob)*100 >= 50 else "t-yellow"
sl_show = f"${sl_largo:,.2f}" if pred == 1 else f"${sl_corto:,.2f}" if pred == -1 else "---"
tp_show = f"${tp_largo:,.2f}" if pred == 1 else f"${tp_corto:,.2f}" if pred == -1 else "---"
vent_show = ventana_activa['nombre'].upper() if ventana_activa else "SIN VENTANA ACTIVA"

def sep(): return '<span class="t-sep">|</span>'
def lbl(txt): return f'<span class="t-gray">{txt}: </span>'

b1 = (
    f'{lbl("TF")}<span class="t-yellow">{tf_sel}</span>{sep()}'
    f'{lbl("PRECIO")}<span class="t-green">${precio:,.2f}</span>{sep()}'
    f'{lbl("SENAL")}<span class="{senal_c}">{etiquetas.get(pred)}</span>{sep()}'
    f'{lbl("CONFIANZA")}<span class="{conf_c}">{max(prob)*100:.1f}%</span>{sep()}'
    f'{lbl("ENTRADA")}<span class="t-white">${entrada:,.2f}</span>{sep()}'
    f'{lbl("SL")}<span class="t-red">{sl_show}</span>{sep()}'
    f'{lbl("TP")}<span class="t-green">{tp_show}</span>{sep()}'
    f'{lbl("VENTANA")}<span class="{vent_c}">{vent_show}</span>{sep()}'
    f'{lbl("HORA MX")}<span class="t-white">{ahora.strftime("%H:%M")}</span>'
)

b2 = (
    f'{lbl("RSI")}<span class="{rsi_c}">{rsi:.1f} {"SOBRECOMPRADO" if rsi > 70 else "SOBREVENDIDO" if rsi < 30 else "NEUTRAL"}</span>{sep()}'
    f'{lbl("MACD")}<span class="{macd_c}">{macd:.2f} — {"ALCISTA" if macd > 0 and macd_h > 0 else "BAJISTA" if macd < 0 and macd_h < 0 else "MIXTO"}</span>{sep()}'
    f'{lbl("ATR")}<span class="t-white">{atr:.2f}</span>{sep()}'
    f'{lbl("EMA 20")}<span class="t-white">${ema20:,.2f}</span>{sep()}'
    f'{lbl("EMA 50")}<span class="t-white">${ema50:,.2f}</span>{sep()}'
    f'{lbl("TENDENCIA")}<span class="{ema_c}">{"ALCISTA" if precio > ema20 and precio > ema50 else "BAJISTA" if precio < ema20 and precio < ema50 else "ZONA DE DECISION"}</span>{sep()}'
    f'{lbl("BB UPPER")}<span class="t-red">${bb_up:,.2f}</span>{sep()}'
    f'{lbl("BB LOWER")}<span class="t-green">${bb_low:,.2f}</span>'
)

st.markdown(f'''
<div class="ticker-wrap">
  <span class="ticker-label">SENAL</span>
  <div class="ticker-inner">{b1}</div>
</div>
<div class="ticker-wrap">
  <span class="ticker-label">TECNICO</span>
  <div class="ticker-inner slow">{b2}</div>
</div>
''', unsafe_allow_html=True)

st.divider()

# ── VARIANTES ──
st.markdown('<div class="section-title">Variantes del mercado</div>', unsafe_allow_html=True)
p_long  = round(float(prob[2] if len(prob) == 3 else prob[1]) * 100, 1)
p_short = round(float(prob[0]) * 100, 1)
p_lat   = round(max(0, 100 - p_long - p_short - 5), 1)
p_shock = round(100 - p_long - p_short - p_lat, 1)
v1, v2, v3, v4 = st.columns(4)
v1.metric("Alcista",      f"{p_long}%",  f"Rompe ${bb_up:,.0f}")
v2.metric("Bajista",      f"{p_short}%", f"Rompe ${bb_low:,.0f}")
v3.metric("Lateral",      f"{p_lat}%",   "Sin ruptura")
v4.metric("Shock/Evento", f"{p_shock}%", "Noticia macro")

st.divider()

# ── VENTANAS ──
st.markdown('<div class="section-title">Ventanas de trading — hora Mexico</div>', unsafe_allow_html=True)
for v in ventanas:
    activa = v["inicio"] <= hora < v["fin"]
    st.markdown(f"{'🟢 **ACTIVA**' if activa else '⚫'} **{v['inicio']:02d}:00 - {v['fin']:02d}:00** {v['nombre']} — [{v['calidad']}]")

st.divider()

# ── GRAFICAS ──
st.markdown('<div class="section-title">Graficas</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Precio historico", "Precio actual en vivo"])

with tab1:
    tipo_grafica = st.radio("Tipo de grafica", ["Linea", "Velas (OHLC)"], horizontal=True, key="tipo_hist")
    if tipo_grafica == "Linea":
        cols_chart = [c for c in ['Close','EMA_20','EMA_50'] if c in df.columns]
        st.line_chart(df[cols_chart].tail(120))
    else:
        df_velas = df[['Open','High','Low','Close']].tail(80).copy()
        st.bar_chart(df_velas['Close'].tail(80))
        st.caption("Vista simplificada de precios de cierre por vela")

with tab2:
    tf_live_sel = st.selectbox("Timeframe en vivo", list(TIMEFRAMES.keys()), index=2, key="live_tf")
    tipo_live = st.radio("Tipo de grafica", ["Linea", "Velas (OHLC)"], horizontal=True, key="tipo_live")
    tf_live = TIMEFRAMES[tf_live_sel]

    @st.cache_data(ttl=60)
    def cargar_vivo(periodo, intervalo):
        df2 = yf.download("GC=F", period=periodo, interval=intervalo, progress=False)
        df2.columns = [c[0] if isinstance(c, tuple) else c for c in df2.columns]
        df2.dropna(inplace=True)
        return df2

    with st.spinner("Cargando precio en vivo..."):
        df_live = cargar_vivo(tf_live["periodo"], tf_live["intervalo"])

    if df_live is not None and len(df_live) > 0:
        precio_live = float(df_live['Close'].iloc[-1])
        precio_prev = float(df_live['Close'].iloc[-2]) if len(df_live) > 1 else precio_live
        cambio = precio_live - precio_prev
        cambio_pct = cambio / precio_prev * 100
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Precio actual", f"${precio_live:,.2f}", f"{cambio:+.2f} ({cambio_pct:+.2f}%)")
        lc2.metric("Max vela", f"${float(df_live['High'].iloc[-1]):,.2f}")
        lc3.metric("Min vela", f"${float(df_live['Low'].iloc[-1]):,.2f}")
        lc4.metric("Volumen", f"{int(df_live['Volume'].iloc[-1]):,}")
        if tipo_live == "Linea":
            st.line_chart(df_live['Close'].tail(100))
        else:
            st.bar_chart(df_live['Close'].tail(80))
            st.caption("Vista simplificada de precios de cierre por vela")
    else:
        st.warning("No hay datos en vivo disponibles ahora.")

st.divider()

# ── GUIA ──
st.markdown('<div class="section-title">Guia de MIMI-AI</div>', unsafe_allow_html=True)

guia = [
    ("Que es MIMI-AI",
     "MIMI-AI es un sistema de inteligencia artificial disenado para analizar el mercado del oro (XAU/USD). Usa dos modelos de machine learning que analizan 13 indicadores tecnicos y votan por una direccion. Solo da senal cuando ambos modelos coinciden, lo que reduce falsas senales."),
    ("Como leer la senal principal",
     "La senal puede ser LONG (el modelo predice subida), SHORT (predice bajada) o LATERAL (sin consenso, no operes). El porcentaje de confianza indica que tan seguro esta el modelo. Menos del 50% = senal debil, mas del 65% = senal fuerte."),
    ("Que es el SL y el TP",
     "SL (Stop Loss) es el precio donde cierras la operacion si el mercado va en tu contra, para limitar perdidas. TP (Take Profit) es donde tomas tus ganancias. MIMI-AI calcula el SL con ATR x 1.5 y el TP con ATR x 2.5, dando una relacion riesgo-beneficio de 1:1.67."),
    ("Que es el ATR",
     "ATR (Average True Range) mide la volatilidad promedio del oro por vela. Si el ATR es 156, significa que el oro se mueve en promedio $156 por dia. A mayor ATR, mayor volatilidad y mayor distancia en SL y TP."),
    ("Que es el RSI",
     "RSI mide si el mercado esta sobrecomprado (arriba de 70, posible caida) o sobrevendido (abajo de 30, posible rebote). Entre 30 y 70 es zona neutral. MIMI-AI lo usa como filtro para evitar entrar en extremos."),
    ("Que es el MACD",
     "MACD mide el momentum del mercado. Cuando el MACD y su histograma son positivos, el momentum es alcista. Cuando son negativos, es bajista. MIMI-AI lo combina con RSI y EMAs para tomar decisiones."),
    ("Que son las EMAs",
     "EMA 20 y EMA 50 son medias moviles que muestran la tendencia. Si el precio esta por encima de ambas, la tendencia es alcista. Si esta por debajo, es bajista. Si esta entre ellas, el mercado esta en zona de decision."),
    ("Cuando operar",
     "Las mejores horas para operar XAU/USD son de 08:00 a 11:00 hora Mexico (sesion Londres + Nueva York). Es cuando hay mayor liquidez y movimiento. MIMI-AI te muestra cual ventana esta activa ahora."),
    ("Que son los timeframes",
     "M5 y M15 son para scalping (operaciones de minutos). H1 y H4 son para day trading (operaciones de horas). D1 es para swing trading (operaciones de dias). Cambia el timeframe segun tu estilo de trading."),
    ("Que son las variantes del mercado",
     "MIMI-AI calcula 4 posibles escenarios: Alcista (precio sube y rompe resistencia), Bajista (precio cae y rompe soporte), Lateral (precio se mueve en rango sin direccion) y Shock (evento inesperado como noticias macro). El porcentaje es la probabilidad de cada escenario."),
    ("Como funciona el oro",
     "El oro (XAU/USD) es el activo refugio mas importante del mundo. Sube cuando hay incertidumbre economica, inflacion o debilidad del dolar americano. Baja cuando el dolar se fortalece o cuando los mercados de riesgo suben. El oro opera 5 dias a la semana, cierra los viernes y reabre los domingos."),
    ("Que es el mercado bursatil",
     "El mercado bursatil es donde se compran y venden activos financieros: acciones, divisas, materias primas como el oro, indices y criptomonedas. MIMI-AI se especializa en XAU/USD, que es el par oro contra el dolar estadounidense."),
]

for titulo, texto in guia:
    with st.expander(titulo):
        st.markdown(f'<div class="guide-text">{texto}</div>', unsafe_allow_html=True)

st.divider()

# ── CHAT ──
st.markdown('<div class="section-title">Chat con MIMI-AI</div>', unsafe_allow_html=True)

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

def responder(pregunta):
    p = pregunta.lower()

    # Como funciona MIMI-AI
    if any(x in p for x in ["como funciona", "que eres", "quien eres", "que es mimi", "explicame", "como trabaja"]):
        return "Soy MIMI-AI, un sistema de trading basado en machine learning. Analizo 13 indicadores tecnicos del oro con dos modelos (Random Forest y Gradient Boosting). Solo doy senal cuando ambos coinciden. Calculo entrada, SL y TP automaticamente con base en el ATR. Revisa la seccion Guia de MIMI-AI para aprender todo sobre como usar este sistema."

    # Metodologia
    elif any(x in p for x in ["como tradeas", "como operas", "que estrategia", "metodologia", "metodo", "como decides"]):
        return f"Opero en {tf_sel} con confluencias de ML. Calculo RSI, MACD, ATR, Bollinger, Stochastico, EMAs y OBV. Dos modelos votan por una direccion. Solo doy senal cuando ambos coinciden. SL = ATR x 1.5 ({atr*1.5:.0f} pts), TP = ATR x 2.5 ({atr*2.5:.0f} pts), R:R de 1:1.67. Solo opero en ventanas de alta liquidez para evitar movimientos falsos."

    # Por que long
    elif any(x in p for x in ["por que long", "por que sube", "por que comprarias", "razon long"]):
        return f"El modelo dice LONG porque: RSI en {rsi:.1f} (no sobrecomprado), MACD {'positivo con histograma positivo — momentum alcista' if macd > 0 and macd_h > 0 else 'con senales mixtas'}, precio {'por encima de EMA 20 y 50 — tendencia alcista confirmada' if precio > ema20 and precio > ema50 else 'en zona de decision entre EMAs'}. Ambos modelos coincidieron con {max(prob)*100:.1f}% de confianza."

    # Por que short
    elif any(x in p for x in ["por que short", "por que baja", "por que venderia", "razon short"]):
        return f"El modelo dice SHORT porque: RSI en {rsi:.1f}, MACD {'negativo con histograma negativo — momentum bajista' if macd < 0 and macd_h < 0 else 'mixto'}, precio {'por debajo de EMA 20 y 50 — tendencia bajista' if precio < ema20 and precio < ema50 else 'entre EMAs'}. Ambos modelos coincidieron con {max(prob)*100:.1f}% de confianza."

    # Por que no operar
    elif any(x in p for x in ["por que no", "por que lateral", "por que esperar", "por que no operas"]):
        return "Los dos modelos ML no coinciden en direccion. Cuando no hay consenso, no hay operacion. Operar sin consenso es apostar, no tradear. El ruido es abundante. La senal, escasa. Espera a que se alineen los indicadores."

    # Entrar
    elif any(x in p for x in ["entrar", "entro", "operar", "comprar", "vender", "abrir operacion", "deberia entrar"]):
        if pred == 1:
            return f"LONG en {tf_sel}. Precio actual ${entrada:,.2f}. SL en ${sl_largo:,.2f} (abajo {atr*1.5:.0f} pts). TP en ${tp_largo:,.2f} (arriba {atr*2.5:.0f} pts). Confianza {max(prob)*100:.1f}%. {'Ventana activa ahora: ' + ventana_activa['nombre'] if ventana_activa else 'No hay ventana optima activa ahora. Considera esperar a las 08:00 MX.'}."
        elif pred == -1:
            return f"SHORT en {tf_sel}. Precio actual ${entrada:,.2f}. SL en ${sl_corto:,.2f} (arriba {atr*1.5:.0f} pts). TP en ${tp_corto:,.2f} (abajo {atr*2.5:.0f} pts). Confianza {max(prob)*100:.1f}%. {'Ventana activa ahora: ' + ventana_activa['nombre'] if ventana_activa else 'No hay ventana optima activa ahora.'}."
        else:
            return "Los modelos no tienen consenso ahora. Sin senal clara no abras operacion. La paciencia no es debilidad. Es claridad. Espera ruptura del precio en cualquier direccion."

    # SL
    elif any(x in p for x in ["sl", "stop loss", "stop", "donde pongo el stop", "cuanto arriesgo"]):
        if pred == 1:
            return f"Stop Loss para LONG: ${sl_largo:,.2f}. Es ATR ({atr:.2f}) x 1.5 = {atr*1.5:.0f} puntos por debajo de la entrada ${entrada:,.2f}. Si el precio cae a ${sl_largo:,.2f}, sal inmediatamente sin dudarlo."
        elif pred == -1:
            return f"Stop Loss para SHORT: ${sl_corto:,.2f}. Es ATR ({atr:.2f}) x 1.5 = {atr*1.5:.0f} puntos por encima de la entrada ${entrada:,.2f}. Si el precio sube a ${sl_corto:,.2f}, sal inmediatamente."
        else:
            return "Sin senal activa no hay SL calculado. Espera a que MIMI-AI de una senal clara."

    # TP
    elif any(x in p for x in ["tp", "take profit", "objetivo", "target", "donde tomo ganancias"]):
        if pred == 1:
            return f"Take Profit para LONG: ${tp_largo:,.2f}. Es ATR ({atr:.2f}) x 2.5 = {atr*2.5:.0f} puntos arriba de la entrada. R:R de 1:1.67 — por cada $1 que arriesgas, puedes ganar $1.67."
        elif pred == -1:
            return f"Take Profit para SHORT: ${tp_corto:,.2f}. Es ATR ({atr:.2f}) x 2.5 = {atr*2.5:.0f} puntos abajo de la entrada. R:R de 1:1.67."
        else:
            return "Sin senal activa aun. El TP se calcula automaticamente cuando hay consenso."

    # RSI
    elif any(x in p for x in ["rsi", "sobrecomprado", "sobrevendido"]):
        if rsi > 70:
            return f"RSI en {rsi:.1f} — zona de sobrecompra. El mercado ha subido demasiado rapido. Hay riesgo de correccion a la baja. Evita abrir LONG ahora. Si ya estas en LONG, considera asegurar ganancias."
        elif rsi < 30:
            return f"RSI en {rsi:.1f} — zona de sobreventa. El mercado ha caido demasiado. Puede haber rebote. Evita abrir SHORT ahora."
        else:
            return f"RSI en {rsi:.1f} — zona neutral entre 30 y 70. No da ventaja ni en largo ni en corto. El RSI solo es un filtro, no una senal por si solo."

    # MACD
    elif any(x in p for x in ["macd", "momentum"]):
        estado = "ALCISTA — los toros tienen el control" if macd > 0 and macd_h > 0 else "BAJISTA — los osos tienen el control" if macd < 0 and macd_h < 0 else "MIXTO — sin direccion dominante"
        return f"MACD: {macd:.2f}, histograma: {macd_h:.2f}. Momentum {estado}. El MACD compara dos medias moviles para medir la fuerza y direccion del movimiento."

    # ATR
    elif any(x in p for x in ["atr", "volatilidad", "cuanto se mueve"]):
        return f"ATR actual: {atr:.2f}. El oro se mueve en promedio ${atr:.2f} por vela en {tf_sel}. A mayor ATR, mayor volatilidad. MIMI-AI usa ATR x 1.5 = ${atr*1.5:.0f} para el SL y ATR x 2.5 = ${atr*2.5:.0f} para el TP."

    # Tendencia
    elif any(x in p for x in ["tendencia", "trend", "direccion del mercado", "hacia donde va"]):
        if precio > ema20 and precio > ema50:
            return f"Tendencia ALCISTA en {tf_sel}. El precio ${precio:,.2f} esta por encima de EMA 20 (${ema20:,.2f}) y EMA 50 (${ema50:,.2f}). Los compradores tienen el control. Favorece operaciones LONG."
        elif precio < ema20 and precio < ema50:
            return f"Tendencia BAJISTA en {tf_sel}. El precio ${precio:,.2f} esta por debajo de EMA 20 (${ema20:,.2f}) y EMA 50 (${ema50:,.2f}). Los vendedores tienen el control. Favorece operaciones SHORT."
        else:
            return f"Tendencia MIXTA en {tf_sel}. El precio esta entre las EMAs. Es una zona de decision — espera que el precio rompa claramente en una direccion antes de operar."

    # Ventanas horario
    elif any(x in p for x in ["hora", "ventana", "cuando operar", "horario", "mejor hora"]):
        if ventana_activa:
            return f"Ventana activa AHORA: {ventana_activa['nombre']} [{ventana_activa['calidad']}]. Buena hora para operar. La mejor ventana es Londres + NY de 08:00 a 11:00 MX — maxima liquidez y movimiento."
        else:
            return f"Sin ventana activa ahora (son las {ahora.strftime('%H:%M')} MX). La proxima ventana optima es Londres + NY de 08:00 a 11:00 MX. Operar fuera de ventana aumenta el riesgo de movimientos falsos."

    # Precio oro
    elif any(x in p for x in ["precio", "oro", "xau", "cuanto vale", "cotiza"]):
        return f"El oro (XAU/USD) cotiza en ${precio:,.2f} ahora en timeframe {tf_sel}. ATR de {atr:.2f} indica la volatilidad actual. Resistencia en ${bb_up:,.2f} (BB upper) y soporte en ${bb_low:,.2f} (BB lower)."

    # Timeframes
    elif any(x in p for x in ["timeframe", "temporalidad", "marco temporal", "que timeframe"]):
        return f"Timeframe actual: {tf_sel}. M5/M15 = scalping (segundos a minutos). M30/H1 = intradía (horas). H4/D1 = swing trading (dias). A menor timeframe, mas senales pero mas ruido. A mayor timeframe, menos senales pero mas confiables."

    # Scalping
    elif any(x in p for x in ["scalping"]):
        return "Para scalping usa M5 o M15. Son operaciones rapidas de 1 a 30 minutos. Requieren mayor concentracion y rapidez. El SL es mas pequeno pero tambien el TP. Cambia el timeframe arriba del todo."

    # Day trading
    elif any(x in p for x in ["day trading", "intradía", "intraday"]):
        return "Para day trading usa H1 o H4. Operaciones que duran horas dentro del mismo dia. Es el estilo mas comun. Las ventanas de Londres + NY son las ideales para este estilo."

    # Swing
    elif any(x in p for x in ["swing"]):
        return "Para swing trading usa D1. Operaciones que pueden durar dias o semanas. Menor estres, menos tiempo frente a la pantalla. El SL y TP son mas amplios porque el mercado tiene mas espacio para moverse."

    # El oro como activo
    elif any(x in p for x in ["que es el oro", "por que sube el oro", "por que baja el oro", "oro refugio"]):
        return "El oro es el activo refugio mas importante del mundo. SUBE cuando: hay incertidumbre economica, inflacion alta, el dolar se debilita, hay guerras o crisis geopoliticas. BAJA cuando: el dolar se fortalece, suben las tasas de interes, los mercados de riesgo van bien. Opera 5 dias a la semana, cierra viernes y reabre el domingo a las 18:00 MX."

    # Mercado bursatil
    elif any(x in p for x in ["mercado bursatil", "bolsa", "que es el mercado", "como funciona el mercado"]):
        return "El mercado bursatil es donde se compran y venden activos financieros. Incluye: acciones (empresas), divisas (forex), materias primas (oro, petroleo), indices (S&P500, Nasdaq) y criptomonedas. MIMI-AI se especializa en XAU/USD — el par mas operado de materias primas. El precio lo determinan la oferta y demanda global en tiempo real."

    # Probabilidades
    elif any(x in p for x in ["probabilidad", "probabilidades", "que tan probable", "chances"]):
        return f"Probabilidades actuales en {tf_sel}: Alcista {p_long}% (rompe ${bb_up:,.0f}), Bajista {p_short}% (rompe ${bb_low:,.0f}), Lateral {p_lat}% (se queda en rango), Shock {p_shock}% (evento inesperado). Escenario dominante: {'ALCISTA' if p_long == max(p_long, p_short, p_lat, p_shock) else 'BAJISTA' if p_short == max(p_long, p_short, p_lat, p_shock) else 'LATERAL'}."

    # Noticias
    elif any(x in p for x in ["noticias", "eventos", "calendario", "nfp", "fed", "inflacion", "cpi"]):
        return "Las noticias mas importantes para el oro son: NFP (primer viernes de cada mes, 07:30 MX), decision de tasas de la FED (cada 6 semanas), CPI inflacion (mensual), discursos del presidente de la FED. Antes de estos eventos, el mercado puede moverse violentamente. MIMI-AI recomienda cerrar posiciones antes de noticias de alto impacto."

    # R:R
    elif any(x in p for x in ["rr", "r:r", "riesgo beneficio", "risk reward"]):
        return f"MIMI-AI usa un R:R de 1:1.67. Esto significa que por cada $1 que arriesgas en el SL, el TP te da $1.67 de ganancia potencial. SL actual: {atr*1.5:.0f} pts. TP actual: {atr*2.5:.0f} pts. Con un win rate del 50%, este R:R es rentable a largo plazo."

    elif any(x in p for x in ["hola", "buenas", "buenos", "hey", "buen dia"]):
        return "El mercado no saluda. Pero MIMI-AI si. Pregunta lo que necesitas saber sobre el oro o el sistema."

    elif any(x in p for x in ["gracias", "thanks"]):
        return "El mercado no da las gracias. Pero aqui estoy. Pregunta lo que necesites."

    elif any(x in p for x in ["ayuda", "help", "que puedes hacer", "que sabes"]):
        return "Puedo responder sobre: como funciono, entrada/SL/TP, RSI/MACD/ATR/EMAs, tendencia, timeframes, ventanas de horario, precio del oro, probabilidades, noticias importantes, mercado bursatil, scalping/day trading/swing, R:R, y por que di la senal actual. Pregunta lo que quieras."

    else:
        return "No entendi bien esa pregunta. Puedo hablar sobre: senal actual, entrada/SL/TP, indicadores (RSI, MACD, ATR), tendencia, timeframes, ventanas de horario, el oro, el mercado bursatil, probabilidades o como funciona MIMI-AI. Reformula tu pregunta."

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])

if pregunta := st.chat_input("Pregunta a MIMI-AI sobre el mercado..."):
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    respuesta = responder(pregunta)
    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta})
    st.rerun()

st.divider()
st.markdown(f'<div class="frase-estoica">{random.choice(frases)}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#333;font-size:11px;font-family:monospace;letter-spacing:2px;">MIMI-AI — TRADING INTELLIGENCE SYSTEM — XAU/USD</div>', unsafe_allow_html=True)
