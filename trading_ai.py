# ═══════════════════════════════════════════════════════════════════
#  TRADING AI v2 — XAU/USD
#  Carácter: Socrático · Estoico · Preciso
# ═══════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pandas_ta as ta

def voz(tipo, mensaje):
    prefijos = {
        'entrada':  "⚔️  ENTRA",
        'salida':   "🛡️  SAL",
        'espera':   "🪨  ESPERA",
        'alerta':   "⚠️  ATENCIÓN",
        'variante': "🔭 ESCENARIO",
        'monitor':  "👁️  MONITOR",
    }
    print(f"\n{prefijos.get(tipo,'📡')} — {mensaje}")

def sabiduria():
    frases = [
        "El mercado revela lo que eres. No lo que quieres.",
        "No controlas el precio. Controlas tu reacción.",
        "La paciencia no es debilidad. Es claridad.",
        "Una pérdida aceptada a tiempo es una victoria de carácter.",
        "El ruido es abundante. La señal, escasa.",
    ]
    import random
    print(f"\n   ── {random.choice(frases)}")

def obtener_datos(simbolo="GC=F", periodo="2y", intervalo="1d"):
    df = yf.download(simbolo, period=periodo, interval=intervalo, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df

def precio_actual():
    df = yf.download("GC=F", period="1d", interval="5m", progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return float(df['Close'].iloc[-1])

def agregar_indicadores(df):
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
    df.dropna(inplace=True)
    return df

def preparar_target(df, horizonte=5, umbral=0.003):
    df['Future_Return'] = df['Close'].pct_change(horizonte).shift(-horizonte)
    df['Target'] = 0
    df.loc[df['Future_Return'] >  umbral, 'Target'] =  1
    df.loc[df['Future_Return'] < -umbral, 'Target'] = -1
    df.dropna(inplace=True)
    return df

def entrenar_ml(df):
    features = ['RSI','MACD','MACD_hist','BB_width','ATR','Stoch_K',
                 'Dist_EMA20','Dist_EMA50','Dist_EMA200',
                 'Return_1d','Return_3d','Return_5d','OBV']
    features = [f for f in features if f in df.columns]
    X, y = df[features], df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_gb = accuracy_score(y_test, gb.predict(X_test))
    mejor = rf if acc_rf >= acc_gb else gb
    joblib.dump({'modelo': mejor, 'scaler': scaler, 'features': features}, 'modelo_ml.pkl')
    return mejor, scaler, features

def entrenar_lstm(df):
    features = ['Close','RSI','MACD','ATR','BB_width','Return_1d']
    features = [f for f in features if f in df.columns]
    scaler_lstm = StandardScaler()
    X = scaler_lstm.fit_transform(df[features])
    y = (df['Target'].values + 1).astype(int)
    PASOS = 30
    Xs, ys = [], []
    for i in range(PASOS, len(X)):
        Xs.append(X[i-PASOS:i])
        ys.append(y[i])
    X_seq, y_seq = np.array(Xs), np.array(ys)
    split = int(len(X_seq) * 0.8)
    y_train_cat = tf.keras.utils.to_categorical(y_seq[:split], 3)
    y_test_cat  = tf.keras.utils.to_categorical(y_seq[split:], 3)
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(PASOS, len(features))),
        Dropout(0.3), LSTM(32), Dropout(0.2),
        Dense(16, activation='relu'), Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_seq[:split], y_train_cat, epochs=50, batch_size=32,
              validation_data=(X_seq[split:], y_test_cat),
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
    model.save('modelo_lstm.h5')
    joblib.dump({'scaler': scaler_lstm, 'features': features, 'pasos': PASOS}, 'lstm_config.pkl')
    return model, scaler_lstm, features, PASOS

def horario_optimo():
    mx = pytz.timezone('America/Mexico_City')
    ahora = datetime.now(mx)
    hora  = ahora.hour
    ventanas = [
        {"nombre": "Londres abre",  "inicio": 3,  "fin": 5,  "calidad": "Alta"},
        {"nombre": "Londres + NY",  "inicio": 8,  "fin": 11, "calidad": "Máxima"},
        {"nombre": "NY tarde",      "inicio": 12, "fin": 14, "calidad": "Media"},
        {"nombre": "Cierre NY",     "inicio": 15, "fin": 17, "calidad": "Baja"},
    ]
    actual = None
    proxima = None
    for v in ventanas:
        if v["inicio"] <= hora < v["fin"]:
            actual = v
        elif hora < v["inicio"] and proxima is None:
            proxima = v
    print("\n╔══════════════════════════════════╗")
    print("║        VENTANAS DE TRADING        ║")
    print("╚══════════════════════════════════╝")
    print(f"   Hora actual (MX): {ahora.strftime('%H:%M')}\n")
    for v in ventanas:
        marca = " ◄ AHORA" if actual and v["nombre"] == actual["nombre"] else ""
        print(f"   {v['inicio']:02d}:00–{v['fin']:02d}:00  {v['nombre']:<20} [{v['calidad']}]{marca}")
    if actual:
        voz('entrada', f"Estás en ventana activa: {actual['nombre']} [{actual['calidad']}]")
    elif proxima:
        voz('espera', f"Próxima ventana: {proxima['nombre']} a las {proxima['inicio']:02d}:00 MX")
    else:
        voz('espera', "Sin ventanas activas. Mercado en reposo.")
    return actual

def variantes_mercado(df, prob_ml):
    precio  = float(df['Close'].iloc[-1])
    atr     = float(df['ATR'].iloc[-1])
    rsi     = float(df['RSI'].iloc[-1])
    bb_up   = float(df['BB_upper'].iloc[-1])
    bb_low  = float(df['BB_lower'].iloc[-1])
    p_long  = float(prob_ml[2]) if len(prob_ml) == 3 else float(prob_ml[1])
    p_short = float(prob_ml[0])
    p_lat   = float(prob_ml[1]) if len(prob_ml) == 3 else 0.15
    p_shock = max(0.05, round(1 - p_long - p_short - p_lat, 2))
    total = p_long + p_short + p_lat + p_shock
    p_long  = round(p_long  / total * 100, 1)
    p_short = round(p_short / total * 100, 1)
    p_lat   = round(p_lat   / total * 100, 1)
    p_shock = round(100 - p_long - p_short - p_lat, 1)
    escenarios = [
        {"nombre": "ALCISTA",      "prob": p_long,  "condicion": f"Precio rompe {bb_up:.2f}",              "objetivo": f"{precio + atr*2:.2f}", "sl": f"{precio - atr*1.5:.2f}", "accion": "Mantén o escala si RSI < 70", "icono": "📈"},
        {"nombre": "BAJISTA",      "prob": p_short, "condicion": f"Precio rompe {bb_low:.2f}",             "objetivo": f"{precio - atr*2:.2f}", "sl": f"{precio + atr*1.5:.2f}", "accion": "Sal si no tienes cobertura",  "icono": "📉"},
        {"nombre": "LATERAL",      "prob": p_lat,   "condicion": f"Precio entre {bb_low:.2f}–{bb_up:.2f}","objetivo": f"{precio:.2f} ± {atr*0.5:.2f}", "sl": f"Ambos lados ±{atr*1.2:.2f}", "accion": "No operes. Espera ruptura.", "icono": "➡️ "},
        {"nombre": "SHOCK/EVENTO", "prob": p_shock, "condicion": "Noticia macro inesperada",               "objetivo": "Indefinido",            "sl": "Activa SL inmediato",     "accion": "Cierra antes de noticias clave", "icono": "⚡"},
    ]
    escenarios.sort(key=lambda x: x["prob"], reverse=True)
    print("\n╔══════════════════════════════════╗")
    print("║      VARIANTES DEL MERCADO        ║")
    print("╚══════════════════════════════════╝")
    for e in escenarios:
        print(f"\n   {e['icono']} {e['nombre']} — {e['prob']}%")
        print(f"      Condición : {e['condicion']}")
        print(f"      Objetivo   : {e['objetivo']}")
        print(f"      Stop Loss  : {e['sl']}")
        print(f"      Acción     : {e['accion']}")
    return escenarios

def señal_completa(df, modelo_ml, scaler_ml, features_ml,
                   modelo_lstm=None, scaler_lstm=None, features_lstm=None, pasos=30):
    ultima   = df[features_ml].iloc[-1:]
    X_scaled = scaler_ml.transform(ultima)
    pred_ml  = modelo_ml.predict(X_scaled)[0]
    prob_ml  = modelo_ml.predict_proba(X_scaled)[0]
    pred_lstm = None
    if modelo_lstm and scaler_lstm:
        X_seq = scaler_lstm.transform(df[features_lstm].tail(pasos))
        X_seq = X_seq.reshape(1, pasos, len(features_lstm))
        prob_lstm  = modelo_lstm.predict(X_seq, verbose=0)[0]
        pred_lstm  = int(np.argmax(prob_lstm)) - 1
    consenso = pred_ml if pred_lstm is None else (pred_ml if pred_ml == pred_lstm else 0)
    precio = float(df['Close'].iloc[-1])
    rsi    = float(df['RSI'].iloc[-1])
    atr    = float(df['ATR'].iloc[-1])
    macd   = float(df['MACD'].iloc[-1])
    mh     = float(df['MACD_hist'].iloc[-1])
    condicion_mercado = []
    if rsi > 70:   condicion_mercado.append("Sobrecomprado")
    elif rsi < 30: condicion_mercado.append("Sobrevendido")
    else:          condicion_mercado.append("RSI neutral")
    if macd > 0 and mh > 0:   condicion_mercado.append("Momentum alcista")
    elif macd < 0 and mh < 0: condicion_mercado.append("Momentum bajista")
    else:                      condicion_mercado.append("Momentum mixto")
    condicion_str = " · ".join(condicion_mercado)
    etiquetas = {1: "LONG — Sube", 0: "LATERAL", -1: "SHORT — Baja"}
    print("\n╔══════════════════════════════════╗")
    print("║          SEÑAL PRINCIPAL          ║")
    print("╚══════════════════════════════════╝")
    print(f"   Precio actual : ${precio:,.2f}")
    print(f"   Condición     : {condicion_str}")
    print(f"   ML            : {etiquetas.get(pred_ml)} ({max(prob_ml)*100:.1f}%)")
    if pred_lstm is not None:
        print(f"   LSTM          : {etiquetas.get(pred_lstm)}")
    print(f"   CONSENSO      : {etiquetas.get(consenso)}")
    print(f"   SL largo      : ${precio - atr*1.5:,.2f}")
    print(f"   SL corto      : ${precio + atr*1.5:,.2f}")
    print(f"   ATR           : {atr:.2f}  |  RSI: {rsi:.1f}")
    if consenso == 1:
        voz('entrada', f"Entrada LONG — Precio ${precio:,.2f} | {condicion_str}")
    elif consenso == -1:
        voz('entrada', f"Entrada SHORT — Precio ${precio:,.2f} | {condicion_str}")
    else:
        voz('espera', f"Sin consenso claro — {condicion_str}. No operes aún.")
    return consenso, prob_ml

def monitor_posicion(precio_entrada, direccion, atr, n_checks=6, intervalo_min=10):
    import time
    print("\n╔══════════════════════════════════╗")
    print("║        MONITOR DE POSICIÓN        ║")
    print("╚══════════════════════════════════╝")
    print(f"   Entrada: ${precio_entrada:,.2f}  |  Dirección: {'LONG' if direccion==1 else 'SHORT'}")
    print(f"   SL: ${precio_entrada - atr*1.5:,.2f} / TP: ${precio_entrada + atr*2:,.2f}")
    print(f"   Revisando cada {intervalo_min} minutos...\n")
    sl = precio_entrada - atr*1.5 if direccion == 1 else precio_entrada + atr*1.5
    tp = precio_entrada + atr*2.0 if direccion == 1 else precio_entrada - atr*2.0
    for i in range(n_checks):
        time.sleep(intervalo_min * 60)
        precio_v = precio_actual()
        pnl = (precio_v - precio_entrada) * direccion
        pnl_pct = pnl / precio_entrada * 100
        mx = pytz.timezone('America/Mexico_City')
        hora = datetime.now(mx).strftime('%H:%M')
        if (direccion == 1 and precio_v <= sl) or (direccion == -1 and precio_v >= sl):
            voz('salida', f"[{hora}] SL alcanzado — ${precio_v:,.2f} | P&L: {pnl_pct:.2f}%")
            print("   Sal ahora. Sin debate.")
            break
        elif (direccion == 1 and precio_v >= tp) or (direccion == -1 and precio_v <= tp):
            voz('salida', f"[{hora}] TP alcanzado — ${precio_v:,.2f} | P&L: {pnl_pct:.2f}%")
            print("   Toma la ganancia. El mercado puede revertir.")
            break
        else:
            estado = "MANTÉN" if pnl > 0 else "PRECAUCIÓN"
            voz('monitor', f"[{hora}] ${precio_v:,.2f} | P&L: {pnl_pct:+.2f}% → {estado}")

# ═══════════════════════════════════════════════════════════════
#  CORRER TODO
# ═══════════════════════════════════════════════════════════════
print("=" * 50)
print("   TRADING AI v2 — XAU/USD")
print("=" * 50)

df = obtener_datos("GC=F", periodo="2y", intervalo="1d")
df = agregar_indicadores(df)
df = preparar_target(df)

print("🤖 Entrenando modelos...")
modelo_ml, scaler_ml, features_ml = entrenar_ml(df)
modelo_lstm, scaler_lstm, features_lstm, pasos = entrenar_lstm(df)

ventana_activa = horario_optimo()
consenso, prob_ml = señal_completa(df, modelo_ml, scaler_ml, features_ml,
                                    modelo_lstm, scaler_lstm, features_lstm, pasos)
variantes_mercado(df, prob_ml)
sabiduria()

print("\n" + "=" * 50)
print("✅ SISTEMA LISTO")
print("=" * 50)
print("\n   Para activar monitor de posición, corre:")
print("   monitor_posicion(precio_entrada=XXXX, direccion=1, atr=float(df['ATR'].iloc[-1]))")
print("   direccion: 1=LONG  |  -1=SHORT")
