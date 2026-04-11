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
st.title("Trading AI - XAU/USD")
st.caption("ML - Caracter Estoico - Senales en tiempo real")

@st.cache_data(ttl=3600)
def cargar_y_entrenar():
    df = yf.download("GC=F", period="2y", interval="1d", progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    df['EMA_20']  = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50']  = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['RSI']     = ta.momentum.rsi(df['Close'], window=14)
    df['MACD']    = ta.trend.macd(df['Close'])
    df['MACD_hist'] = ta.trend.macd_diff(df['Close'])
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
    df['ATR']      = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Stoch_K']  = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['OBV']      = ta.volume.on_balance_volume(df['Close'], df['Volume'])
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
    features = ['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K',
                'Dist_EMA20','Dist_EMA50','Dist_EMA200','Return_1d','Return_3d','Return_5d','OBV']
    features = [f for f in features if f in df.columns]
    X, y = df[features], df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    mejor = rf if accuracy_score(y_test, rf.predict(X_test)) >= accuracy_score(y_test, gb.predict(X_test)) else gb
    return df, mejor, scaler, features

with st.spinner("Entrenando modelos... un momento."):
    df, modelo, scaler, features = cargar_y_entrenar()

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
X_scaled = scaler.transform(ultima)
pred    = modelo.predict(X_scaled)[0]
prob    = modelo.predict_proba(X_scaled)[0]

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio XAU/USD", f"${precio:,.2f}")
c2.metric("RSI (14)", f"{rsi:.1f}", "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Normal")
c3.metric("ATR (14)", f"{atr:.2f}")
c4.metric("Senal ML", etiquetas.get(pred), f"{max(prob)*100:.1f}% confianza")

st.divider()

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

st.subheader("Resumen del mercado")
cond_rsi  = "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "neutral"
cond_macd = "alcista" if macd > 0 and macd_h > 0 else "bajista" if macd < 0 and macd_h < 0 else "mixto"
cond_ema  = "por encima de EMA 20 y 50 (tendencia alcista)" if precio > ema20 and precio > ema50 else "por debajo de EMA 20 y 50 (tendencia bajista)" if precio < ema20 and precio < ema50 else "entre EMAs (zona de decision)"
ventana_txt = f"Ventana activa: {ventana_activa['nombre']} [{ventana_activa['calidad']}]" if ventana_activa else "Sin ventana activa ahora. Proxima: Londres + NY a las 08:00 MX"

resumen = f"El oro cotiza en **${precio:,.2f}**. RSI en **{rsi:.1f}** ({cond_rsi}), momentum MACD **{cond_macd}**. Precio {cond_ema}. Senal del modelo: **{etiquetas.get(pred)}** con {max(prob)*100:.1f}% de confianza. {ventana_txt}."
if pred == 1:
    resumen += f" Entrada sugerida: **${entrada:,.2f}** | SL: **${sl_largo:,.2f}** | TP: **${tp_largo:,.2f}**"
elif pred == -1:
    resumen += f" Entrada sugerida: **${entrada:,.2f}** | SL: **${sl_corto:,.2f}** | TP: **${tp_corto:,.2f}**"
else:
    resumen += " El modelo no encuentra consenso. Espera una ruptura clara antes de entrar."

st.info(resumen)

st.divider()

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

st.subheader("Ventanas de trading (hora Mexico)")
for v in ventanas:
    activa = v["inicio"] <= hora < v["fin"]
    st.markdown(f"{'ACTIVA' if activa else '---'} **{v['inicio']:02d}:00 - {v['fin']:02d}:00** {v['nombre']} [{v['calidad']}]")

st.divider()

st.subheader("Precio historico")
st.line_chart(df[['Close', 'EMA_20', 'EMA_50']].tail(120))

st.divider()

st.subheader("Consulta al sistema")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

def responder(pregunta):
    p = pregunta.lower()
    if any(x in p for x in ["entrar", "entro", "operar", "comprar", "vender"]):
        if pred == 1:
            return f"El modelo indica LONG. Entrada en ${entrada:,.2f}, SL en ${sl_largo:,.2f}, TP en ${tp_largo:,.2f}. Confianza: {max(prob)*100:.1f}%. Solo operas en ventana activa."
        elif pred == -1:
            return f"El modelo indica SHORT. Entrada en ${entrada:,.2f}, SL en ${sl_corto:,.2f}, TP en ${tp_corto:,.2f}. Confianza: {max(prob)*100:.1f}%. Solo operas en ventana activa."
        else:
            return "El modelo no tiene consenso ahora. La paciencia no es debilidad. Espera una ruptura clara."
    elif any(x in p for x in ["sl", "stop", "stop loss"]):
        if pred == 1:
            return f"Stop Loss para LONG: ${sl_largo:,.2f}. Es ATR x 1.5 por debajo de la entrada."
        elif pred == -1:
            return f"Stop Loss para SHORT: ${sl_corto:,.2f}. Es ATR x 1.5 por encima de la entrada."
        else:
            return "No hay senal activa. El SL se calcula cuando hay una direccion clara."
    elif any(x in p for x in ["tp", "take profit", "objetivo", "target"]):
        if pred == 1:
            return f"Take Profit para LONG: ${tp_largo:,.2f}. R:R de 1:1.67."
        elif pred == -1:
            return f"Take Profit para SHORT: ${tp_corto:,.2f}. R:R de 1:1.67."
        else:
            return "No hay senal activa aun. Espera consenso del modelo."
    elif any(x in p for x in ["rsi"]):
        return f"RSI actual: {rsi:.1f}. {'Zona de sobrecompra, cuidado con longs.' if rsi > 70 else 'Zona de sobreventa, posible rebote.' if rsi < 30 else 'Zona neutral, el RSI no da ventaja clara ahora.'}"
    elif any(x in p for x in ["tendencia", "trend", "direccion"]):
        if precio > ema20 and precio > ema50:
            return f"Tendencia alcista. Precio ${precio:,.2f} por encima de EMA 20 (${ema20:,.2f}) y EMA 50 (${ema50:,.2f})."
        elif precio < ema20 and precio < ema50:
            return f"Tendencia bajista. Precio ${precio:,.2f} por debajo de EMA 20 (${ema20:,.2f}) y EMA 50 (${ema50:,.2f})."
        else:
            return "Tendencia mixta. Precio entre EMAs. Zona de decision, espera ruptura."
    elif any(x in p for x in ["hora", "ventana", "cuando", "horario"]):
        if ventana_activa:
            return f"Ventana activa: {ventana_activa['nombre']} [{ventana_activa['calidad']}]. Buena hora para operar."
        else:
            return "No hay ventana activa ahora. La proxima optima es Londres + NY de 08:00 a 11:00 hora Mexico."
    elif any(x in p for x in ["precio", "oro", "xau"]):
        return f"El oro cotiza en ${precio:,.2f}. ATR de {atr:.2f} indica la volatilidad promedio diaria."
    elif any(x in p for x in ["macd"]):
        return f"MACD: {macd:.2f}, histograma: {macd_h:.2f}. Momentum {'alcista' if macd > 0 and macd_h > 0 else 'bajista' if macd < 0 and macd_h < 0 else 'mixto'}."
    elif any(x in p for x in ["atr", "volatilidad"]):
        return f"ATR: {atr:.2f}. El oro se mueve en promedio ${atr:.2f} por dia. SL usa ATR x 1.5, TP usa ATR x 2.5."
    elif any(x in p for x in ["hola", "buenas", "buenos", "hey"]):
        return "El mercado no saluda. Pregunta lo que necesitas saber."
    elif any(x in p for x in ["gracias"]):
        return "Aqui estoy. Pregunta lo que necesites."
    else:
        return "Puedo responder sobre: entrada, SL, TP, RSI, MACD, ATR, tendencia, ventanas de horario y precio del oro."

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
