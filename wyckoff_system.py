# ═══════════════════════════════════════════════════════════════════
#  WYCKOFF + PRICE ACTION TRADING SYSTEM — OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import ta

# ───────────────────────────────────────────────────────
#  SWING POINTS
# ───────────────────────────────────────────────────────

def detect_swing_points(df, strength=3):
    highs, lows = [], []
    for i in range(strength, len(df) - strength):
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1,strength+1)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1,strength+1)):
            highs.append({'idx':i,'price':df['High'].iloc[i]})
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1,strength+1)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1,strength+1)):
            lows.append({'idx':i,'price':df['Low'].iloc[i]})
    return highs, lows

# ───────────────────────────────────────────────────────
#  ACUMULACIÓN
# ───────────────────────────────────────────────────────

def detect_accumulation(df, window=20):
    zones = []
    for i in range(window, len(df)-window):
        high = df['High'].iloc[i-window:i].max()
        low  = df['Low'].iloc[i-window:i].min()
        if (high - low) / df['Close'].iloc[i] < 0.03:
            zones.append({'idx':i,'support':low,'resistance':high})
    return zones

# ───────────────────────────────────────────────────────
#  MANIPULACIÓN (MEJORADA)
# ───────────────────────────────────────────────────────

def detect_manipulation(df, acc_zones):
    manipulations = []
    
    for zone in acc_zones:
        i = zone['idx']
        if i >= len(df): continue
        
        c = df['Close'].iloc[i]
        h = df['High'].iloc[i]
        l = df['Low'].iloc[i]
        o = df['Open'].iloc[i]
        
        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        
        volatility_filter = abs(h - l) > c * 0.0015
        
        # BULLISH GRAB (mejorado)
        if lower_wick > body * 2 and l < zone['support'] and c > zone['support'] and volatility_filter:
            manipulations.append({'idx':i,'tipo':'BULL_GRAB','nivel':zone['support'],'zona':zone,'confianza':'ALTA'})
        
        # BEARISH GRAB (mejorado)
        elif upper_wick > body * 2 and h > zone['resistance'] and c < zone['resistance'] and volatility_filter:
            manipulations.append({'idx':i,'tipo':'BEAR_GRAB','nivel':zone['resistance'],'zona':zone,'confianza':'ALTA'})
    
    return manipulations

# ───────────────────────────────────────────────────────
#  DISTRIBUCIÓN (OPTIMIZADA)
# ───────────────────────────────────────────────────────

def detect_distribution(df, manipulations):
    signals = []
    
    macd_hist = ta.trend.macd_diff(df['Close'])
    rsi = ta.momentum.rsi(df['Close'])
    ema20 = ta.trend.ema_indicator(df['Close'], window=20)
    
    for manip in manipulations:
        i = manip['idx']
        if i+3 >= len(df): continue
        
        for j in range(i+1, i+4):
            c = df['Close'].iloc[j]
            macd = macd_hist.iloc[j]
            rsi_v = rsi.iloc[j]
            
            # filtro rango muerto
            recent_range = df['High'].iloc[j-10:j].max() - df['Low'].iloc[j-10:j].min()
            if recent_range / c < 0.002:
                continue
            
            # LONG
            if manip['tipo']=='BULL_GRAB':
                trend_ok = c > ema20.iloc[j]
                
                if c > manip['zona']['resistance'] and macd > 0 and rsi_v > 50 and trend_ok:
                    atr = ta.volatility.average_true_range(df['High'],df['Low'],df['Close']).iloc[j]
                    
                    sl = min(manip['nivel'], manip['zona']['support']) - atr * 1.2
                    rr = 2.5
                    tp = c + (c - sl) * rr
                    
                    signals.append({'tipo':'LONG','entrada':c,'sl':sl,'tp':tp})
                    break
            
            # SHORT
            elif manip['tipo']=='BEAR_GRAB':
                trend_ok = c < ema20.iloc[j]
                
                if c < manip['zona']['support'] and macd < 0 and rsi_v < 50 and trend_ok:
                    atr = ta.volatility.average_true_range(df['High'],df['Low'],df['Close']).iloc[j]
                    
                    sl = max(manip['nivel'], manip['zona']['resistance']) + atr * 1.2
                    rr = 2.5
                    tp = c - (sl - c) * rr
                    
                    signals.append({'tipo':'SHORT','entrada':c,'sl':sl,'tp':tp})
                    break
    
    return signals
