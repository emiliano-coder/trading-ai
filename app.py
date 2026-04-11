import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pytz
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trading AI — XAU/USD", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
.metric-box {background:#1a1a2e;border-radius:10px;padding:16px;text-align:center}
.metric-label {color:#888;font-size:13px}
.metric-value {color:#fff;font-size:24px;font-weight:600}
.long {color:#00c851}
.short {color:#ff4444}
.neutral {color:#ffbb33}
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Trading AI — XAU/USD")
st.caption("ML + LSTM · Carácter Estoico · Señales en tiempo real")

@st.cache_data(ttl=3600)
def cargar_y_entrenar():
    df = yf.download("GC=F", period="2y", interval="1d", progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)

    df['EMA_20']  = ta.ema(df['Close'], length=20)
    df['EMA_50']  = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI']     = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD']      = macd.iloc[:,0]
    df['MACD_hist'] = macd.iloc[:,1]
    bb = ta.bbands(df['Close'], length=20)
    bb_upper = [c for c in bb.columns if 'BBU' in c][0]
    bb_lower = [c for c in bb.columns if 'BBL' in c][0]
    df['BB_upper'] = bb[bb_upper]
    df['BB_lower'] = bb[bb_lower]
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
    df['ATR']      = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['OBV']      = ta.obv(df['Close'], df['Volume'])
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_K']     = stoch.iloc[:,0]
    df['Dist_EMA20']  = (df['Close'] - df['EMA_20'])  / df['Close'] * 100
    df['Dist_EMA50']  = (df['Close'] - df['EMA_50'])  / df['Close'] * 100
    df['Dist_EMA200'] = (df['Close'] - df['EMA_200']) / df['Close'] * 100
    df['Return_1d']   = df['Close'].pct_change(1)
    df['Return_3d']   = df['Close'].pct_change(3)
    df['Return_5d']   = df['Close'].pct_change(5)

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

with st.spinner("Entrenando modelos... esto toma un momento."):
    df, modelo, scaler, features = cargar_y_entrenar()

precio   = float(df['Close'].iloc[-1])
rsi      = float(df['RSI'].iloc[-1])
atr      = float(df['ATR'].iloc[-1])
macd_val = float(df['MACD'].iloc[-1])
macd_h   = float(df['MACD_hist'].iloc[-1])
bb_up    = float(df['BB_upper'].iloc[-1])
bb_low   = float(df['BB_lower'].iloc[-1])

ultima   = df[features].iloc[-1:]
X_scaled = scaler.transform(ultima)
pred     = modelo.predict(X_scaled)[0]
prob     = modelo.predict_proba(X_scaled)[0]

etiquetas = {1: "LONG — Sube", 0: "LATERAL", -1: "SHORT — Baja"}
colores   = {1: "long", 0: "neutral", -1: "short"}

# ── MÉTRICAS ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio XAU/USD", f"${precio:,.2f}")
c2.metric("RSI (14)", f"{rsi:.1f}", "Sobrecomprado" if rsi>70 else "Sobrevendido" if rsi<30 else "Normal")
c3.metric("ATR (14)", f"{atr:.2f}")
c4.metric("Señal", etiquetas.get(pred), f"{max(prob)*100:.1f}% confianza")

st.divider()

# ── HORARIO ──
col_h, col_s = st.columns(2)
with col_h:
    st.subheader("🕐 Ventanas de trading")
    mx   = pytz.timezone('America/Mexico_City')
    hora = datetime.now(mx).hour
    ventanas = [
        {"nombre": "Londres abre",  "inicio": 3,  "fin": 5,  "calidad": "Alta"},
        {"nombre": "Londres + NY",  "inicio": 8,  "fin": 11, "calidad": "Máxima ⭐"},
        {"nombre": "NY tarde",      "inicio": 12, "fin": 14, "calidad": "Media"},
        {"nombre": "Cierre NY",     "inicio": 15, "fin": 17, "calidad": "Baja"},
    ]
    for v in ventanas:
        activa = v["inicio"] <= hora < v["fin"]
        st.markdown(f"{'🟢' if activa else '⚫'} **{v['inicio']:02d}:00–{v['fin']:02d}:00** {v['nombre']} [{v['calidad']}]")

with col_s:
    st.subheader("🎯 Señal principal")
    st.markdown(f"**Precio:** ${precio:,.2f}")
    st.markdown(f"**ML:** {etiquetas.get(pred)} ({max(prob)*100:.1f}%)")
    st.markdown(f"**SL largo:** ${precio - atr*1.5:,.2f}")
    st.markdown(f"**SL corto:** ${precio + atr*1.5:,.2f}")
    st.markdown(f"**EMA 20:** ${float(df['EMA_20'].iloc[-1]):,.2f}")
    st.markdown(f"**EMA 50:** ${float(df['EMA_50'].iloc[-1]):,.2f}")

st.divider()

# ── VARIANTES ──
st.subheader("🔭 Variantes del mercado")
p_long  = round(float(prob[2] if len(prob)==3 else prob[1]) * 100, 1)
p_short = round(float(prob[0]) * 100, 1)
p_lat   = round(max(0, 100 - p_long - p_short - 5), 1)
p_shock = round(100 - p_long - p_short - p_lat, 1)

v1, v2, v3, v4 = st.columns(4)
v1.metric("📈 Alcista",     f"{p_long}%",  f"Rompe {bb_up:,.0f}")
v2.metric("📉 Bajista",     f"{p_short}%", f"Rompe {bb_low:,.0f}")
v3.metric("➡️  Lateral",    f"{p_lat}%",   "Sin ruptura")
v4.metric("⚡ Shock/Evento", f"{p_shock}%", "Noticia macro")

st.divider()

# ── GRÁFICA ──
st.subheader("📊 Precio histórico")
st.line_chart(df[['Close','EMA_20','EMA_50']].tail(120))

# ── FRASE ──
import random
frases = [
    "El mercado revela lo que eres. No lo que quieres.",
    "No controlas el precio. Controlas tu reacción.",
    "La paciencia no es debilidad. Es claridad.",
    "Una pérdida aceptada a tiempo es una victoria de carácter.",
    "El ruido es abundante. La señal, escasa.",
]
st.info(f"🪨 *{random.choice(frases)}*")
