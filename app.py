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
ultima  = df[features].iloc[-1:]
X_scaled = scaler.transform(ultima)
pred    = modelo.predict(X_scaled)[0]
prob    = modelo.predict_proba(X_scaled)[0]

etiquetas = {1: "LONG - Sube", 0: "LATERAL", -1: "SHORT - Baja"}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio XAU/USD", f"${precio:,.2f}")
c2.metric("RSI (14)", f"{rsi:.1f}", "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Normal")
c3.metric("ATR (14)", f"{atr:.2f}")
c4.metric("Senal", etiquetas.get(pred), f"{max(prob)*100:.1f}% confianza")

st.divider()

col_h, col_s = st.columns(2)
with col_h:
    st.subheader("Ventanas de trading")
    mx   = pytz.timezone('America/Mexico_City')
    hora = datetime.now(mx).hour
    ventanas = [
        {"nombre": "Londres abre", "inicio": 3,  "fin": 5,  "calidad": "Alta"},
        {"nombre": "Londres + NY", "inicio": 8,  "fin": 11, "calidad": "Maxima"},
        {"nombre": "NY tarde",     "inicio": 12, "fin": 14, "calidad": "Media"},
        {"nombre": "Cierre NY",    "inicio": 15, "fin": 17, "calidad": "Baja"},
    ]
    for v in ventanas:
        activa = v["inicio"] <= hora < v["fin"]
        st.markdown(f"{'ACTIVA' if activa else '---'} {v['inicio']:02d}:00-{v['fin']:02d}:00 {v['nombre']} [{v['calidad']}]")

with col_s:
    st.subheader("Senal principal")
    st.markdown(f"**Precio:** ${precio:,.2f}")
    st.markdown(f"**Direccion:** {etiquetas.get(pred)} ({max(prob)*100:.1f}%)")
    st.markdown(f"**SL largo:** ${precio - atr*1.5:,.2f}")
    st.markdown(f"**SL corto:** ${precio + atr*1.5:,.2f}")
    st.markdown(f"**EMA 20:** ${float(df['EMA_20'].iloc[-1]):,.2f}")
    st.markdown(f"**EMA 50:** ${float(df['EMA_50'].iloc[-1]):,.2f}")

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

st.subheader("Precio historico")
st.line_chart(df[['Close', 'EMA_20', 'EMA_50']].tail(120))

frases = [
    "El mercado revela lo que eres. No lo que quieres.",
    "No controlas el precio. Controlas tu reaccion.",
    "La paciencia no es debilidad. Es claridad.",
    "Una perdida aceptada a tiempo es una victoria de caracter.",
    "El ruido es abundante. La senal, escasa.",
]
st.info(f"{random.choice(frases)}")
