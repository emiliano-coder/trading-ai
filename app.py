import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pytz
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trading AI — XAU/USD", page_icon="⚔️", layout="wide")

st.title("⚔️ Trading AI — XAU/USD")
st.caption("ML · Carácter Estoico · Señales en tiempo real")

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

with st.spinner("Entrenando modelos... un momento."):
    df, modelo, scaler, features = cargar_y_entrenar()

precio   = float(df['Close'].iloc[-1])
rsi      = float(df['RSI'].iloc[-1])
atr      = float(df['ATR'].iloc[-1])
macd_val = float(df['MACD'].iloc[-1])
macd_h   = float(df['MACD_hist'].iloc[-1])
bb_up    = float(d
