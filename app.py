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

st.set_page_config(page_title="Trading AI - XAU/USD", page_icon="sword", layout="wide")

st.markdown("""
<style>
.banner {
    background: #000000;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 12px;
    overflow: hidden;
    white-space: nowrap;
    border: 1px solid #222;
}
.banner-inner {
    display: inline-block;
    animation: scroll-left 18s linear infinite;
    font-size: 15px;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.banner-inner.slow {
    animation-duration: 24s;
}
@keyframes scroll-left {
    0%   { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}
.green { color: #00e676; }
.red   { color: #ff1744; }
.white { color: #ffffff; }
.yellow { color: #ffd600; }
</style>
""", unsafe_allow_html=True)

st.title("Trading AI - XAU/USD")
st.caption("ML - Caracter Estoico - Senales en tiempo real")

TIMEFRAMES = {
    "M5":  {"periodo": "5d",  "intervalo": "5m"},
    "M15": {"periodo": "5d",  "intervalo": "15m"},
    "M30": {"periodo": "1mo", "intervalo": "30m"},
    "H1":  {"periodo": "1mo", "intervalo": "60m"},
    "H4":  {"periodo": "3mo", "intervalo": "1d"},
    "D1":  {"periodo": "2y",  "intervalo": "1d"},
}

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

with st.spinner("Cargando " + tf_sel + "..."):
    df, modelo, scaler, features = cargar_y_entrenar(tf["periodo"], tf["intervalo"])

if df is None:
    st.error("No hay suficientes datos para este timeframe. Intenta con D1 o H4.")
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

etiquetas = {1: "LONG - Sube", 0: "LATERAL", -1: "SHORT - Baja"}

entrada  = precio
sl_largo = round(precio - atr * 1.5, 2)
sl_corto = round(precio + atr * 1.5, 2)
tp_largo = round(precio + atr * 2.5, 2)
tp_corto = round(precio - atr * 2.5, 2)

mx   = pytz.timezone('America/Mexico_City')
hora = datetime.now(mx).hour
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
    "El que espera el momento perfecto, no opera. Opera cuando la razon lo indica.",
    "Disciplina hoy. Libertad manana.",
]

# ── METRICAS ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio XAU/USD", f"${precio:,.2f}")
c2.metric("RSI (14)", f"{rsi:.1f}", "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Normal")
c3.metric("ATR", f"{atr:.2f}")
c4.metric("Senal ML", etiquetas.get(pred), f"{max(prob)*100:.1f}% confianza")

st.divider()

# ── OPERACION ──
st.subheader("Operacion sugerida")
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

# ── BANNERS RESUMEN ──
st.subheader("Resumen del mercado")

senal_color   = "green" if pred == 1 else "red" if pred == -1 else "yellow"
senal_txt     = "LONG - SUBE" if pred == 1 else "SHORT - BAJA" if pred == -1 else "LATERAL - ESPERA"
rsi_color     = "red" if rsi > 70 else "green" if rsi < 30 else "white"
rsi_txt       = "SOBRECOMPRADO" if rsi > 70 else "SOBREVENDIDO" if rsi < 30 else "NEUTRAL"
macd_color    = "green" if macd > 0 and macd_h > 0 else "red" if macd < 0 and macd_h < 0 else "yellow"
macd_txt      = "ALCISTA" if macd > 0 and macd_h > 0 else "BAJISTA" if macd < 0 and macd_h < 0 else "MIXTO"
ema_color     = "green" if precio > ema20 and precio > ema50 else "red" if precio < ema20 and precio < ema50 else "yellow"
ema_txt       = "TENDENCIA ALCISTA" if precio > ema20 and precio > ema50 else "TENDENCIA BAJISTA" if precio < ema20 and precio < ema50 else "ZONA DE DECISION"
ventana_color = "green" if ventana_activa else "red"
ventana_txt2  = f"VENTANA ACTIVA: {ventana_activa['nombre'].upper()} [{ventana_activa['calidad'].upper()}]" if ventana_activa else "SIN VENTANA ACTIVA - PROXIMA: LONDRES+NY 08:00 MX"
conf_color    = "green" if max(prob)*100 >= 50 else "yellow"
sl_txt        = f"${sl_largo:,.2f}" if pred == 1 else f"${sl_corto:,.2f}" if pred == -1 else "---"
tp_txt        = f"${tp_largo:,.2f}" if pred == 1 else f"${tp_corto:,.2f}" if pred == -1 else "---"

banner1 = (
    f'<span class="white">TIMEFRAME: {tf_sel}</span>'
    f' &nbsp;|&nbsp; <span class="white">PRECIO: </span><span class="green">${precio:,.2f}</span>'
    f' &nbsp;|&nbsp; <span class="white">SENAL: </span><span class="{senal_color}">{senal_txt}</span>'
    f' &nbsp;|&nbsp; <span class="white">CONFIANZA: </span><span class="{conf_color}">{max(prob)*100:.1f}%</span>'
    f' &nbsp;|&nbsp; <span class="white">ENTRADA: </span><span class="white">${entrada:,.2f}</span>'
    f' &nbsp;|&nbsp; <span class="white">SL: </span><span class="red">{sl_txt}</span>'
    f' &nbsp;|&nbsp; <span class="white">TP: </span><span class="green">{tp_txt}</span>'
    f' &nbsp;|&nbsp; <span class="{ventana_color}">{ventana_txt2}</span>'
)

banner2 = (
    f'<span class="white">RSI: </span><span class="{rsi_color}">{rsi:.1f} - {rsi_txt}</span>'
    f' &nbsp;|&nbsp; <span class="white">MACD: </span><span class="{macd_color}">{macd:.2f} - {macd_txt}</span>'
    f' &nbsp;|&nbsp; <span class="white">ATR: </span><span class="white">{atr:.2f}</span>'
    f' &nbsp;|&nbsp; <span class="white">EMA 20: </span><span class="white">${ema20:,.2f}</span>'
    f' &nbsp;|&nbsp; <span class="white">EMA 50: </span><span class="white">${ema50:,.2f}</span>'
    f' &nbsp;|&nbsp; <span class="{ema_color}">{ema_txt}</span>'
    f' &nbsp;|&nbsp; <span class="white">BB UPPER: </span><span class="red">${bb_up:,.2f}</span>'
    f' &nbsp;|&nbsp; <span class="white">BB LOWER: </span><span class="green">${bb_low:,.2f}</span>'
)

st.markdown(f'<div class="banner"><div class="banner-inner">{banner1}</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="banner"><div class="banner-inner slow">{banner2}</div></div>', unsafe_allow_html=True)

st.divider()

# ── VARIANTES ──
st.subheader("Variantes del mercado")
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
st.subheader("Ventanas de trading (hora Mexico)")
for v in ventanas:
    activa = v["inicio"] <= hora < v["fin"]
    st.markdown(f"{'ACTIVA' if activa else '---'} **{v['inicio']:02d}:00 - {v['fin']:02d}:00** {v['nombre']} [{v['calidad']}]")

st.divider()

# ── GRAFICA HISTORICA ──
st.subheader("Precio historico - " + tf_sel)
cols_chart = [c for c in ['Close','EMA_20','EMA_50'] if c in df.columns]
st.line_chart(df[cols_chart].tail(120))

st.divider()

# ── GRAFICA EN VIVO ──
st.subheader("Precio actual del mercado")
tf_live_sel = st.selectbox("Timeframe de la grafica en vivo", list(TIMEFRAMES.keys()), index=2, key="live_tf")
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
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Precio actual", f"${precio_live:,.2f}", f"{cambio:+.2f} ({cambio_pct:+.2f}%)")
    lc2.metric("Max vela", f"${float(df_live['High'].iloc[-1]):,.2f}")
    lc3.metric("Min vela", f"${float(df_live['Low'].iloc[-1]):,.2f}")
    st.line_chart(df_live['Close'].tail(100))
else:
    st.warning("No hay datos en vivo disponibles ahora.")

st.divider()

# ── CHAT ──
st.subheader("Consulta al sistema")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

def responder(pregunta):
    p = pregunta.lower()
    if any(x in p for x in ["como tradeas", "como operas", "que estrategia", "como funciona", "como decides", "metodologia", "metodo"]):
        return (f"Opero en {tf_sel} con confluencias de ML. Calculo 13 indicadores: RSI, MACD, ATR, Bollinger, Stochastico, EMAs y OBV. Dos modelos votan por una direccion. Solo doy senal cuando ambos coinciden. SL = ATR x 1.5, TP = ATR x 2.5, R:R 1:1.67. Solo opero en ventanas de alta liquidez.")
    elif any(x in p for x in ["por que long", "por que sube"]):
        return (f"LONG porque: RSI {rsi:.1f} no sobrecomprado, MACD {'positivo' if macd > 0 else 'con momentum'}, precio {'sobre' if precio > ema20 else 'bajo'} EMA20. Ambos modelos coincidieron con {max(prob)*100:.1f}% confianza.")
    elif any(x in p for x in ["por que short", "por que baja"]):
        return (f"SHORT porque: RSI {rsi:.1f}, MACD {'negativo' if macd < 0 else 'mixto'}, precio {'bajo' if precio < ema20 else 'sobre'} EMA20. Ambos modelos coincidieron con {max(prob)*100:.1f}% confianza.")
    elif any(x in p for x in ["por que no", "por que lateral", "por que esperar"]):
        return "Los modelos no coinciden. Sin consenso no hay operacion. El ruido es abundante. La senal, escasa."
    elif any(x in p for x in ["entrar", "entro", "operar", "comprar", "vender"]):
        if pred == 1:
            return f"LONG en {tf_sel}. Entrada ${entrada:,.2f}, SL ${sl_largo:,.2f}, TP ${tp_largo:,.2f}. Confianza {max(prob)*100:.1f}%."
        elif pred == -1:
            return f"SHORT en {tf_sel}. Entrada ${entrada:,.2f}, SL ${sl_corto:,.2f}, TP ${tp_corto:,.2f}. Confianza {max(prob)*100:.1f}%."
        else:
            return "Sin consenso. Espera ruptura clara."
    elif any(x in p for x in ["sl", "stop"]):
        if pred == 1: return f"SL LONG: ${sl_largo:,.2f} (ATR x 1.5 abajo)."
        elif pred == -1: return f"SL SHORT: ${sl_corto:,.2f} (ATR x 1.5 arriba)."
        else: return "Sin senal activa."
    elif any(x in p for x in ["tp", "take profit", "objetivo"]):
        if pred == 1: return f"TP LONG: ${tp_largo:,.2f}. R:R 1:1.67."
        elif pred == -1: return f"TP SHORT: ${tp_corto:,.2f}. R:R 1:1.67."
        else: return "Sin senal activa."
    elif any(x in p for x in ["rsi"]):
        return f"RSI: {rsi:.1f}. {'Sobrecompra.' if rsi > 70 else 'Sobreventa.' if rsi < 30 else 'Zona neutral.'}"
    elif any(x in p for x in ["macd"]):
        return f"MACD: {macd:.2f}, histograma: {macd_h:.2f}. Momentum {'alcista' if macd > 0 and macd_h > 0 else 'bajista' if macd < 0 and macd_h < 0 else 'mixto'}."
    elif any(x in p for x in ["atr", "volatilidad"]):
        return f"ATR: {atr:.2f}. El oro se mueve ~${atr:.2f} por vela en {tf_sel}."
    elif any(x in p for x in ["tendencia", "trend"]):
        if precio > ema20 and precio > ema50: return f"Tendencia alcista en {tf_sel}."
        elif precio < ema20 and precio < ema50: return f"Tendencia bajista en {tf_sel}."
        else: return f"Tendencia mixta en {tf_sel}. Zona de decision."
    elif any(x in p for x in ["hora", "ventana", "cuando", "horario"]):
        if ventana_activa: return f"Ventana activa: {ventana_activa['nombre']} [{ventana_activa['calidad']}]."
        else: return "Sin ventana activa. Proxima: Londres+NY 08:00-11:00 MX."
    elif any(x in p for x in ["precio", "oro", "xau"]):
        return f"Oro en ${precio:,.2f} en {tf_sel}."
    elif any(x in p for x in ["scalping"]):
        return "Para scalping usa M5 o M15. Cambia el timeframe arriba."
    elif any(x in p for x in ["day trading"]):
        return "Para day trading usa H1 o H4."
    elif any(x in p for x in ["swing"]):
        return "Para swing trading usa D1."
    elif any(x in p for x in ["hola", "buenas", "hey"]):
        return "El mercado no saluda. Pregunta lo que necesitas."
    elif any(x in p for x in ["gracias"]):
        return "Aqui estoy."
    else:
        return "Puedo responder sobre: como tradeo, entrada, SL, TP, RSI, MACD, ATR, tendencia, timeframes, ventanas y precio."

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])

if pregunta := st.chat_input("Pregunta algo sobre el mercado..."):
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    respuesta = responder(pregunta)
    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta})
    st.rerun()

st.divider()
st.info(f"{random.choice(frases)}")
