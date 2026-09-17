"""
core/market_structure.py
─────────────────────────────────────────────────────────────────────
Motor de Market Structure (estilo SMC/ICT) — independiente y reutilizable.

Detecta, sobre un DataFrame OHLCV:
  - Swing High / Swing Low
  - Higher High (HH), Higher Low (HL), Lower High (LH), Lower Low (LL)
  - Break of Structure alcista/bajista (BOS)
  - Change of Character alcista/bajista (CHoCH)
  - Tendencia estructural actual (bullish / bearish / ranging)
  - Fuerza/desplazamiento de cada ruptura

No depende de EMA, RSI, MACD ni de ningún otro indicador — solo de la
estructura de precio (OHLC). No depende de Streamlit ni de nada del
resto del proyecto: se puede importar y usar en cualquier script,
notebook o backtest.

─────────────────────────────────────────────────────────────────────
SOBRE EL LOOK-AHEAD BIAS (léase con cuidado, es el corazón del diseño)
─────────────────────────────────────────────────────────────────────
Un swing high/low se detecta con una ventana SIMÉTRICA de `swing_order`
velas: una vela `i` es swing high si su máximo es el más alto entre
`swing_order` velas antes Y `swing_order` velas después. Esto es
estándar en todo el análisis de estructura de mercado, pero tiene una
consecuencia inevitable: un swing en la vela `i` solo puede saberse con
certeza hasta la vela `i + swing_order` — porque hasta entonces no
existen las velas futuras necesarias para confirmarlo.

Este motor maneja esa realidad explícitamente en vez de ignorarla:

  - Cada swing tiene un `index` (dónde ocurrió) y un `confirmed_at`
    (en qué vela se supo con certeza que era swing).
  - Las columnas `swing_high` / `swing_low` marcan el swing en SU
    propia vela (para que se vea bien en una gráfica), pero toda la
    lógica de clasificación (HH/HL/LH/LL) y de rupturas (BOS/CHoCH)
    SOLO usa swings ya confirmados en el momento de cada vela —
    nunca se usa información de velas futuras para decidir una
    ruptura en el pasado.
  - Por lo mismo, si usas este motor en un backtest recorriendo vela
    por vela, la estructura que ves en la vela `i` es exactamente la
    que un trader real habría visto en ese momento (con el retraso
    natural de `swing_order` velas que tiene cualquier swing por
    ventana simétrica — eso no es look-ahead, es la latencia real de
    confirmar un swing; ignorar esa latencia sí sería look-ahead).

La ruptura (BOS/CHoCH) en sí se prueba con el precio de LA MISMA vela
que rompe (por defecto el cierre, `break_using="close"`), nunca con
precios futuros a esa vela.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Trend = Literal["bullish", "bearish", "ranging"]
SwingKind = Literal["high", "low"]
BreakKind = Literal["BOS", "CHoCH"]
BreakUsing = Literal["close", "wick"]


# ─────────────────────────────────────────────────────────────────────
#  Estructuras de datos internas
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SwingPoint:
    index: int                 # posición (iloc) de la vela donde ocurrió el swing
    price: float
    kind: SwingKind
    confirmed_at: int          # posición (iloc) donde se confirma (index + swing_order)
    label: Optional[str] = None    # "Swing High"/"HH"/"LH" o "Swing Low"/"HL"/"LL"
    broken: bool = False           # ya se usó para disparar una ruptura


@dataclass
class BreakEvent:
    kind: BreakKind
    direction: Literal["bullish", "bearish"]
    index: int          # vela donde ocurrió la ruptura
    price: float         # precio de ruptura (close o wick, según break_using)
    level: float          # nivel del swing roto
    displacement: float       # price - level (en precio, siempre positivo)
    displacement_pct: float   # displacement como % del nivel roto

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "index": self.index,
            "price": self.price,
            "level": self.level,
            "displacement": self.displacement,
            "displacement_pct": self.displacement_pct,
        }


# ─────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────
def _validar_columnas(df: pd.DataFrame) -> None:
    cols_lower = {c.lower() for c in df.columns}
    requeridas = {"open", "high", "low", "close"}
    faltantes = requeridas - cols_lower
    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias en el DataFrame: {sorted(faltantes)}. "
            f"Se requieren al menos: open, high, low, close (volume es opcional)."
        )


def _detectar_swings_crudos(high: np.ndarray, low: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Marca swing highs/lows con ventana simétrica de `order` velas a cada lado.
    NO tiene en cuenta la latencia de confirmación — eso se maneja después,
    en `analyze_market_structure`, vía `confirmed_at = index + order`.
    """
    n = len(high)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if order < 1:
        raise ValueError("swing_order debe ser >= 1")
    for i in range(order, n - order):
        ventana_h = high[i - order : i + order + 1]
        ventana_l = low[i - order : i + order + 1]
        if high[i] == ventana_h.max():
            swing_high[i] = True
        if low[i] == ventana_l.min():
            swing_low[i] = True
    return swing_high, swing_low


def _swing_a_dict(sp: Optional[SwingPoint]) -> Optional[dict]:
    if sp is None:
        return None
    return {
        "index": sp.index,
        "price": sp.price,
        "label": sp.label,
        "confirmed_at": sp.confirmed_at,
        "broken": sp.broken,
    }


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def analyze_market_structure(
    df: pd.DataFrame,
    swing_order: int = 5,
    break_using: BreakUsing = "close",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analiza la estructura de mercado de un DataFrame OHLCV, sin look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Debe tener columnas open, high, low, close (case-insensitive).
        `volume` es opcional — si no está, el motor funciona igual.
        El índice puede ser DatetimeIndex o cualquier otro; no se usa
        para la lógica, solo se conserva en la salida.
    swing_order : int, default 5
        Sensibilidad del swing: cuántas velas a cada lado se exigen
        para confirmar un swing high/low. Más alto → menos swings,
        más significativos. Más bajo → más swings, más ruido.
    break_using : "close" | "wick", default "close"
        Con qué precio se prueba si un nivel fue roto. "close" es más
        conservador (evita mechas falsas); "wick" (high/low) es más
        sensible y detecta la ruptura más temprano.

    Returns
    -------
    (df_out, summary)
        df_out : copia del DataFrame original + columnas nuevas:
            swing_high, swing_low, structure_label,
            bos_bullish, bos_bearish, choch_bullish, choch_bearish,
            structure_trend, displacement
        summary : dict con el estado estructural más reciente:
            trend, last_structure, last_swing_high, last_swing_low,
            bos, choch, displacement
    """
    _validar_columnas(df)
    if break_using not in ("close", "wick"):
        raise ValueError("break_using debe ser 'close' o 'wick'")

    df_out = df.copy()
    df_out.columns = [c.lower() for c in df_out.columns]
    n = len(df_out)

    high = df_out["high"].to_numpy(dtype=float)
    low = df_out["low"].to_numpy(dtype=float)
    close = df_out["close"].to_numpy(dtype=float)

    swing_high_raw, swing_low_raw = _detectar_swings_crudos(high, low, swing_order)

    # columnas de salida (se llenan vela por vela, en orden temporal real)
    col_swing_high = np.zeros(n, dtype=bool)
    col_swing_low = np.zeros(n, dtype=bool)
    col_label: List[Optional[str]] = [None] * n
    col_bos_bull = np.zeros(n, dtype=bool)
    col_bos_bear = np.zeros(n, dtype=bool)
    col_choch_bull = np.zeros(n, dtype=bool)
    col_choch_bear = np.zeros(n, dtype=bool)
    col_trend: List[str] = ["ranging"] * n
    col_displacement = np.full(n, np.nan)

    # agenda: en qué vela se confirma cada swing crudo detectado
    confirm_en: Dict[int, List[Tuple[int, SwingKind]]] = {}
    for i in range(n):
        if swing_high_raw[i]:
            confirm_en.setdefault(i + swing_order, []).append((i, "high"))
        if swing_low_raw[i]:
            confirm_en.setdefault(i + swing_order, []).append((i, "low"))

    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    trend: Trend = "ranging"
    last_bos: Optional[BreakEvent] = None
    last_choch: Optional[BreakEvent] = None
    last_structure_label: Optional[str] = None

    def _clasificar_y_registrar(sp: SwingPoint) -> None:
        nonlocal last_swing_high, last_swing_low, last_structure_label
        if sp.kind == "high":
            sp.label = "Swing High" if last_swing_high is None else (
                "HH" if sp.price > last_swing_high.price else "LH"
            )
            last_swing_high = sp
            col_swing_high[sp.index] = True
        else:
            sp.label = "Swing Low" if last_swing_low is None else (
                "HL" if sp.price > last_swing_low.price else "LL"
            )
            last_swing_low = sp
            col_swing_low[sp.index] = True
        last_structure_label = sp.label
        col_label[sp.confirmed_at] = sp.label

    for i in range(n):
        # 1) confirmar los swings que "maduran" justo en esta vela
        for (s_idx, kind) in confirm_en.get(i, []):
            precio_swing = high[s_idx] if kind == "high" else low[s_idx]
            sp = SwingPoint(index=s_idx, price=precio_swing, kind=kind, confirmed_at=i)
            _clasificar_y_registrar(sp)

        # 2) probar rupturas usando SOLO el precio de esta vela contra
        #    swings ya confirmados — nunca información futura
        precio_prueba = close[i] if break_using == "close" else high[i]
        precio_prueba_baja = close[i] if break_using == "close" else low[i]

        if last_swing_high is not None and not last_swing_high.broken and precio_prueba > last_swing_high.price:
            nivel = last_swing_high.price
            disp = precio_prueba - nivel
            disp_pct = (disp / nivel * 100) if nivel else 0.0
            last_swing_high.broken = True
            if trend in ("bullish", "ranging"):
                ev = BreakEvent("BOS", "bullish", i, precio_prueba, nivel, disp, disp_pct)
                col_bos_bull[i] = True
            else:
                ev = BreakEvent("CHoCH", "bullish", i, precio_prueba, nivel, disp, disp_pct)
                col_choch_bull[i] = True
                last_choch = ev
            if ev.kind == "BOS":
                last_bos = ev
            trend = "bullish"
            col_displacement[i] = disp

        if last_swing_low is not None and not last_swing_low.broken and precio_prueba_baja < last_swing_low.price:
            nivel = last_swing_low.price
            disp = nivel - precio_prueba_baja
            disp_pct = (disp / nivel * 100) if nivel else 0.0
            last_swing_low.broken = True
            if trend in ("bearish", "ranging"):
                ev = BreakEvent("BOS", "bearish", i, precio_prueba_baja, nivel, disp, disp_pct)
                col_bos_bear[i] = True
            else:
                ev = BreakEvent("CHoCH", "bearish", i, precio_prueba_baja, nivel, disp, disp_pct)
                col_choch_bear[i] = True
                last_choch = ev
            if ev.kind == "BOS":
                last_bos = ev
            trend = "bearish"
            col_displacement[i] = disp

        col_trend[i] = trend

    df_out["swing_high"] = col_swing_high
    df_out["swing_low"] = col_swing_low
    df_out["structure_label"] = col_label
    df_out["bos_bullish"] = col_bos_bull
    df_out["bos_bearish"] = col_bos_bear
    df_out["choch_bullish"] = col_choch_bull
    df_out["choch_bearish"] = col_choch_bear
    df_out["structure_trend"] = col_trend
    df_out["displacement"] = col_displacement

    ultimo_evento = None
    if last_bos and last_choch:
        ultimo_evento = last_bos if last_bos.index >= last_choch.index else last_choch
    else:
        ultimo_evento = last_bos or last_choch

    summary = {
        "trend": trend,
        "last_structure": last_structure_label,
        "last_swing_high": _swing_a_dict(last_swing_high),
        "last_swing_low": _swing_a_dict(last_swing_low),
        "bos": last_bos.to_dict() if last_bos else None,
        "choch": last_choch.to_dict() if last_choch else None,
        "displacement": ultimo_evento.displacement if ultimo_evento else None,
    }
    return df_out, summary


# ─────────────────────────────────────────────────────────────────────
#  Prueba con datos OHLCV simulados
# ─────────────────────────────────────────────────────────────────────
def _generar_datos_simulados(n: int = 200, semilla: int = 42) -> pd.DataFrame:
    """Genera un OHLCV sintético con una tendencia alcista y luego una
    bajista, solo para probar el motor sin depender de ninguna API."""
    rng = np.random.default_rng(semilla)
    precio = 100.0
    cierres = [precio]
    for i in range(n - 1):
        deriva = 0.06 if i < n * 0.55 else -0.08
        precio += rng.normal(0, 0.9) + deriva
        cierres.append(precio)
    cierres = np.array(cierres)
    highs = cierres + rng.uniform(0.2, 1.1, n)
    lows = cierres - rng.uniform(0.2, 1.1, n)
    opens = np.concatenate([[cierres[0]], cierres[:-1]]) + rng.normal(0, 0.15, n)
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": cierres, "volume": volume},
        index=idx,
    )


def _demo() -> None:
    """Corre `python -m core.market_structure` para probar el módulo solo."""
    df = _generar_datos_simulados()
    df_sin_volumen = df.drop(columns=["volume"])

    print("=" * 60)
    print("PRUEBA 1 — con volumen")
    print("=" * 60)
    df_out, summary = analyze_market_structure(df, swing_order=5)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nÚltimas 12 velas (columnas de estructura):")
    cols = ["close", "swing_high", "swing_low", "structure_label",
            "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
            "structure_trend", "displacement"]
    print(df_out.tail(12)[cols].to_string())

    print("\n" + "=" * 60)
    print("PRUEBA 2 — sin columna volume (debe funcionar igual)")
    print("=" * 60)
    df_out2, summary2 = analyze_market_structure(df_sin_volumen, swing_order=5)
    assert "volume" not in df_out2.columns
    print("  OK — funcionó sin 'volume'. Trend:", summary2["trend"])

    print("\n" + "=" * 60)
    print("PRUEBA 3 — swing_order más sensible (3) vs. más conservador (10)")
    print("=" * 60)
    _, s_sens = analyze_market_structure(df, swing_order=3)
    _, s_cons = analyze_market_structure(df, swing_order=10)
    print(f"  swing_order=3  -> trend={s_sens['trend']}, last_structure={s_sens['last_structure']}")
    print(f"  swing_order=10 -> trend={s_cons['trend']}, last_structure={s_cons['last_structure']}")

    print("\nTodas las pruebas corrieron sin errores.")


if __name__ == "__main__":
    _demo()
