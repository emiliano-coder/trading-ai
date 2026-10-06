"""
core/fvg.py
─────────────────────────────────────────────────────────────────────
Fair Value Gap (FVG) Engine — independiente y reutilizable.

Detecta, sobre un DataFrame OHLC de 3 velas en 3 velas:
  - Bullish FVG / Bearish FVG
  - Tamaño del gap (top - bottom)
  - Estado de cada FVG a lo largo del tiempo: activo, parcialmente
    mitigado, completamente mitigado (según un criterio configurable:
    touch / 50% / full), o invalidado.

No depende de EMA/RSI/MACD. No decide señales BUY/SELL/WAIT — es un
motor puramente informativo, pensado para correr en paralelo a la
estrategia existente (igual que market_structure.py y liquidity.py).

─────────────────────────────────────────────────────────────────────
DEFINICIÓN EXACTA
─────────────────────────────────────────────────────────────────────
Para tres velas consecutivas 1, 2, 3 (posiciones i-2, i-1, i):

  Bullish FVG:  High(vela 1) < Low(vela 3)
                → rango del gap = [High(vela 1), Low(vela 3)]
                  bottom = High(vela 1), top = Low(vela 3)

  Bearish FVG:  Low(vela 1) > High(vela 3)
                → rango del gap = [High(vela 3), Low(vela 1)]
                  bottom = High(vela 3), top = Low(vela 1)

La vela del medio (vela 2) es la que "salta" el hueco — no participa
en los límites del gap, solo confirma que hubo un movimiento fuerte
entre la vela 1 y la vela 3.

─────────────────────────────────────────────────────────────────────
SOBRE EL LOOK-AHEAD BIAS
─────────────────────────────────────────────────────────────────────
A diferencia de un swing (que necesita velas FUTURAS para confirmarse),
un FVG solo necesita velas PASADAS respecto al punto de confirmación:
para confirmar un FVG en la vela `i`, se usan las velas `i-2`, `i-1` e
`i` — todas ya conocidas en el momento en que la vela `i` cierra. Por
eso un FVG:

  - NUNCA se marca antes de que la vela 3 (la vela `i`) haya cerrado.
    El motor evalúa el patrón exactamente en la iteración `i`, nunca
    antes — no hay ninguna ventana hacia adelante involucrada.
  - Una vez creado, su estado (activo → parcialmente mitigado →
    mitigado / invalidado) se actualiza ÚNICAMENTE con velas
    POSTERIORES a su creación (`j > created_at`), nunca con la vela
    de creación ni con nada anterior.
  - Los estados "mitigado" e "invalidado" son terminales: una vez
    alcanzados, no se reevalúan con velas futuras (evita que un FVG
    "reviva" de forma ambigua).

Esto es válido para backtesting: el estado de un FVG en la vela `i` es
exactamente el que un trader real habría podido observar en ese momento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from core.market_structure import _validar_columnas

FVGKind = Literal["bullish", "bearish"]
FVGStatus = Literal["active", "partially_mitigated", "mitigated", "invalidated"]
MitigationCriterion = Literal["touch", "50%", "full"]

CRITERIOS_MITIGACION: Dict[str, float] = {"touch": 0.0, "50%": 50.0, "full": 100.0}
CRITERIO_DEFAULT: MitigationCriterion = "50%"


# ─────────────────────────────────────────────────────────────────────
#  Estructura de datos
# ─────────────────────────────────────────────────────────────────────
@dataclass
class FVGZone:
    kind: FVGKind
    created_at: int          # posición (iloc) de la vela 3 — donde se confirma el FVG
    top: float
    bottom: float
    size: float                # top - bottom, siempre > 0
    status: FVGStatus = "active"
    max_fill_pct: float = 0.0   # qué tanto del gap se ha llenado, lo más profundo alcanzado
    mitigated_at: Optional[int] = None
    invalidated_at: Optional[int] = None
    mitigation_criterion: str = CRITERIO_DEFAULT

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "created_at": self.created_at,
            "top": self.top,
            "bottom": self.bottom,
            "size": self.size,
            "status": self.status,
            "max_fill_pct": round(self.max_fill_pct, 2),
            "mitigated_at": self.mitigated_at,
            "invalidated_at": self.invalidated_at,
            "mitigation_criterion": self.mitigation_criterion,
        }


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def analyze_fvg(
    df: pd.DataFrame,
    mitigation_criterion: MitigationCriterion = CRITERIO_DEFAULT,
    min_gap_pct: float = 0.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detecta Fair Value Gaps (bullish/bearish) sobre un DataFrame OHLC,
    sin look-ahead bias, y sigue su estado (activo/parcial/mitigado/
    invalidado) vela por vela.

    Parameters
    ----------
    df : DataFrame con columnas open, high, low, close (volume opcional).
    mitigation_criterion : "touch" | "50%" | "full"
        - "touch": se considera mitigado en cuanto el precio toca el
          gap, aunque sea con una mecha mínima.
        - "50%"  : se considera mitigado cuando el precio ha rellenado
          al menos la mitad del gap (por defecto).
        - "full" : se considera mitigado solo cuando el precio rellena
          el gap por completo.
    min_gap_pct : tamaño mínimo del gap, como % del precio, para que
        se registre (0 = sin filtro, se detecta cualquier gap genuino).

    Returns
    -------
    (df_out, summary)
        df_out : copia de `df` + columnas nuevas:
            fvg_bullish, fvg_bearish (evento: se creó un FVG en esta vela),
            fvg_top, fvg_bottom, fvg_size (solo llenas en la vela de creación),
            fvg_active (¿hay algún FVG vivo —activo o parcial— en esta vela?),
            fvg_mitigated (evento: algún FVG se volvió "mitigated" en esta vela),
            fvg_invalidated (evento: algún FVG se invalidó en esta vela)
        summary : dict con:
            active_bullish_fvg, active_bearish_fvg,
            last_bullish_fvg, last_bearish_fvg,
            last_mitigated_fvg, last_invalidated_fvg,
            all_fvg (lista completa, para auditar/backtest)
    """
    if df is None or len(df) == 0:
        raise ValueError("El DataFrame está vacío o es None — no hay nada que analizar.")
    _validar_columnas(df)
    if mitigation_criterion not in CRITERIOS_MITIGACION:
        raise ValueError(
            f"mitigation_criterion debe ser uno de {list(CRITERIOS_MITIGACION)}, "
            f"se recibió: {mitigation_criterion!r}"
        )
    threshold = CRITERIOS_MITIGACION[mitigation_criterion]

    df_out = df.copy()
    df_out.columns = [c.lower() for c in df_out.columns]
    n = len(df_out)

    h = df_out["high"].to_numpy(dtype=float)
    l = df_out["low"].to_numpy(dtype=float)
    c = df_out["close"].to_numpy(dtype=float)

    col_fvg_bull = np.zeros(n, dtype=bool)
    col_fvg_bear = np.zeros(n, dtype=bool)
    col_top = np.full(n, np.nan)
    col_bottom = np.full(n, np.nan)
    col_size = np.full(n, np.nan)
    col_active = np.zeros(n, dtype=bool)
    col_mitigated_evt = np.zeros(n, dtype=bool)
    col_invalidated_evt = np.zeros(n, dtype=bool)

    zonas: List[FVGZone] = []

    for i in range(n):
        # 1) ¿se confirma un FVG nuevo en esta vela? (usa SOLO i-2, i-1, i)
        if i >= 2:
            if np.isfinite(h[i - 2]) and np.isfinite(l[i]) and h[i - 2] < l[i]:
                top, bottom = float(l[i]), float(h[i - 2])
                size = top - bottom
                if size > 0 and (min_gap_pct <= 0 or (bottom != 0 and size / abs(bottom) * 100 >= min_gap_pct)):
                    zonas.append(FVGZone("bullish", i, top, bottom, size, mitigation_criterion=mitigation_criterion))
                    col_fvg_bull[i] = True
                    col_top[i], col_bottom[i], col_size[i] = top, bottom, size

            if np.isfinite(l[i - 2]) and np.isfinite(h[i]) and l[i - 2] > h[i]:
                top, bottom = float(l[i - 2]), float(h[i])
                size = top - bottom
                if size > 0 and (min_gap_pct <= 0 or (bottom != 0 and size / abs(bottom) * 100 >= min_gap_pct)):
                    zonas.append(FVGZone("bearish", i, top, bottom, size, mitigation_criterion=mitigation_criterion))
                    col_fvg_bear[i] = True
                    col_top[i], col_bottom[i], col_size[i] = top, bottom, size

        # 2) actualizar el estado de TODAS las zonas vivas, usando SOLO
        #    la vela actual `i` (nunca una vela futura, y nunca la
        #    propia vela de creación de cada zona)
        for z in zonas:
            if z.status in ("mitigated", "invalidated"):
                continue  # estado terminal, no se reevalúa
            if i <= z.created_at:
                continue  # todavía no hay vela posterior a su creación

            if z.kind == "bullish":
                intrusion = max(0.0, z.top - l[i])
                intrusion = min(intrusion, z.size)
                fill_pct = (intrusion / z.size * 100) if z.size else 0.0
                z.max_fill_pct = max(z.max_fill_pct, fill_pct)
                if c[i] < z.bottom:
                    z.status, z.invalidated_at = "invalidated", i
                    col_invalidated_evt[i] = True
                    continue
            else:  # bearish
                intrusion = max(0.0, h[i] - z.bottom)
                intrusion = min(intrusion, z.size)
                fill_pct = (intrusion / z.size * 100) if z.size else 0.0
                z.max_fill_pct = max(z.max_fill_pct, fill_pct)
                if c[i] > z.top:
                    z.status, z.invalidated_at = "invalidated", i
                    col_invalidated_evt[i] = True
                    continue

            if z.max_fill_pct > 0 and z.max_fill_pct >= threshold:
                z.status, z.mitigated_at = "mitigated", i
                col_mitigated_evt[i] = True
            elif z.max_fill_pct > 0:
                z.status = "partially_mitigated"

        col_active[i] = any(z.created_at <= i and z.status in ("active", "partially_mitigated") for z in zonas)

    df_out["fvg_bullish"] = col_fvg_bull
    df_out["fvg_bearish"] = col_fvg_bear
    df_out["fvg_top"] = col_top
    df_out["fvg_bottom"] = col_bottom
    df_out["fvg_size"] = col_size
    df_out["fvg_active"] = col_active
    df_out["fvg_mitigated"] = col_mitigated_evt
    df_out["fvg_invalidated"] = col_invalidated_evt

    bullish_zonas = [z for z in zonas if z.kind == "bullish"]
    bearish_zonas = [z for z in zonas if z.kind == "bearish"]
    activos_bull = [z for z in bullish_zonas if z.status in ("active", "partially_mitigated")]
    activos_bear = [z for z in bearish_zonas if z.status in ("active", "partially_mitigated")]

    mitigados = [z for z in zonas if z.status == "mitigated"]
    invalidados = [z for z in zonas if z.status == "invalidated"]
    last_mitigado = max(mitigados, key=lambda z: z.mitigated_at) if mitigados else None
    last_invalidado = max(invalidados, key=lambda z: z.invalidated_at) if invalidados else None

    summary = {
        "active_bullish_fvg": [z.to_dict() for z in activos_bull],
        "active_bearish_fvg": [z.to_dict() for z in activos_bear],
        "last_bullish_fvg": bullish_zonas[-1].to_dict() if bullish_zonas else None,
        "last_bearish_fvg": bearish_zonas[-1].to_dict() if bearish_zonas else None,
        "last_mitigated_fvg": last_mitigado.to_dict() if last_mitigado else None,
        "last_invalidated_fvg": last_invalidado.to_dict() if last_invalidado else None,
        "all_fvg": [z.to_dict() for z in zonas],
    }
    return df_out, summary


# ─────────────────────────────────────────────────────────────────────
#  Datos simulados + pruebas deliberadas (8 casos del spec)
# ─────────────────────────────────────────────────────────────────────
def _generar_datos_fvg_bullish() -> pd.DataFrame:
    """OHLC a mano: zona plana (0-4, sin FVG), un bullish FVG limpio
    confirmado en la vela 7 (top=100.50, bottom=100.00), y velas de
    seguimiento (8-11) que lo llevan de activo -> parcial -> mitigado."""
    filas = [
        (99.90, 99.95, 99.85, 99.92),    # 0
        (99.92, 99.97, 99.87, 99.93),    # 1
        (99.93, 99.98, 99.88, 99.91),    # 2
        (99.91, 99.96, 99.86, 99.94),    # 3
        (99.94, 99.99, 99.89, 99.93),    # 4   zona plana, sin FVG
        (99.93, 100.00, 99.90, 99.95),   # 5   vela 1 (high=100.00 -> bottom del gap)
        (99.95, 100.90, 99.92, 100.80),  # 6   vela 2 — impulso (low=99.92 no limpia high[4])
        (100.80, 101.00, 100.50, 100.95),# 7   vela 3 (low=100.50 > high[5] -> FVG top=100.50 bottom=100.00)
        (100.95, 101.00, 100.80, 100.90),# 8   sigue arriba del gap, fill=0
        (100.90, 100.95, 100.375, 100.60),# 9  low=100.375 -> intrusión 0.125 -> fill=25% (parcial)
        (100.60, 100.70, 100.25, 100.40),# 10  low=100.25 -> intrusión 0.25 -> fill=50% (mitigado con criterio 50%)
        (100.40, 100.45, 99.95, 100.05), # 11  low=99.95 (<bottom) -> fill=100%; close=100.05(>=bottom) -> NO invalida
    ]
    idx = pd.date_range("2025-03-01", periods=len(filas), freq="1h")
    return pd.DataFrame(filas, columns=["open", "high", "low", "close"], index=idx)


def _generar_datos_fvg_bearish_invalidado() -> pd.DataFrame:
    """OHLC a mano: un bearish FVG limpio confirmado en la vela 7
    (top=100.00, bottom=99.60), invalidado en la vela 8 (close cierra
    por encima del top)."""
    filas = [
        (99.90, 99.95, 99.85, 99.92),    # 0
        (99.92, 99.97, 99.87, 99.93),    # 1
        (99.93, 99.98, 99.88, 99.91),    # 2
        (99.91, 99.96, 99.86, 99.89),    # 3
        (99.89, 99.95, 99.84, 99.90),    # 4   zona plana, sin FVG
        (99.98, 100.05, 100.00, 99.99),  # 5   vela 1 (low=100.00 -> top del gap)
        (99.95, 100.08, 99.10, 99.20),   # 6   vela 2 — impulso bajista
        (99.20, 99.55, 99.00, 99.10),    # 7   vela 3 (high=99.60... ver nota abajo)
        (99.10, 100.50, 99.05, 100.20),  # 8   high=100.50 (toca y pasa el top) y close=100.20>top -> INVALIDADO
    ]
    idx = pd.date_range("2025-03-01", periods=len(filas), freq="1h")
    return pd.DataFrame(filas, columns=["open", "high", "low", "close"], index=idx)


def _demo() -> None:
    """Corre `python -m core.fvg` — ejecuta los 8 casos de prueba del spec."""
    dfA = _generar_datos_fvg_bullish()
    dfB = _generar_datos_fvg_bearish_invalidado()

    print("=" * 70)
    print("CASO 1 — Bullish FVG detectado")
    print("=" * 70)
    df_out, summary = analyze_fvg(dfA, mitigation_criterion="50%")
    assert df_out["fvg_bullish"].iloc[7] == True, "Se esperaba un bullish FVG confirmado en la vela 7"
    assert abs(df_out["fvg_top"].iloc[7] - 100.50) < 1e-6
    assert abs(df_out["fvg_bottom"].iloc[7] - 100.00) < 1e-6
    print(f"  OK — bullish FVG en vela 7: top={df_out['fvg_top'].iloc[7]}, bottom={df_out['fvg_bottom'].iloc[7]}")

    print("\n" + "=" * 70)
    print("CASO 2 — Bearish FVG detectado")
    print("=" * 70)
    dfB_out, summaryB = analyze_fvg(dfB, mitigation_criterion="50%")
    assert dfB_out["fvg_bearish"].iloc[7] == True, "Se esperaba un bearish FVG confirmado en la vela 7"
    print(f"  OK — bearish FVG en vela 7: top={dfB_out['fvg_top'].iloc[7]}, bottom={dfB_out['fvg_bottom'].iloc[7]}")

    print("\n" + "=" * 70)
    print("CASO 3 — No se detecta FVG donde no existe gap")
    print("=" * 70)
    assert not df_out["fvg_bullish"].iloc[4] and not df_out["fvg_bearish"].iloc[4]
    assert not dfB_out["fvg_bullish"].iloc[4] and not dfB_out["fvg_bearish"].iloc[4]
    print("  OK — las zonas planas (vela 4 en ambos datasets) no generan ningún FVG.")

    print("\n" + "=" * 70)
    print("CASO 4 — El FVG solo existe después de cerrar la vela 3")
    print("=" * 70)
    assert df_out["fvg_bullish"].iloc[6] == False, "No debía existir FVG todavía en la vela 6"
    assert df_out["fvg_bullish"].iloc[7] == True
    print("  OK — en la vela 6 no hay FVG; aparece exactamente en la vela 7 (cuando cierra la vela 3).")

    print("\n" + "=" * 70)
    print("CASO 5 — FVG parcialmente mitigado")
    print("=" * 70)
    _, summary_hasta_9 = analyze_fvg(dfA.iloc[:10], mitigation_criterion="50%")
    estado_en_9 = [z for z in summary_hasta_9["all_fvg"] if z["created_at"] == 7][0]
    assert estado_en_9["status"] == "partially_mitigated", f"Se esperaba partially_mitigated, se obtuvo {estado_en_9['status']}"
    print(f"  OK — con datos hasta la vela 9, el FVG está 'partially_mitigated' (fill={estado_en_9['max_fill_pct']}%).")

    print("\n" + "=" * 70)
    print("CASO 6 — FVG completamente mitigado")
    print("=" * 70)
    fvg1_final = [z for z in summary["all_fvg"] if z["created_at"] == 7][0]
    assert fvg1_final["status"] == "mitigated", f"Se esperaba mitigated, se obtuvo {fvg1_final['status']}"
    print(f"  OK — con criterio '50%', el FVG queda 'mitigated' en la vela {fvg1_final['mitigated_at']} (fill={fvg1_final['max_fill_pct']}%).")
    # con criterio "full" necesita llegar a fill=100% (vela 11) para mitigarse
    _, summary_full = analyze_fvg(dfA, mitigation_criterion="full")
    fvg1_full = [z for z in summary_full["all_fvg"] if z["created_at"] == 7][0]
    assert fvg1_full["status"] == "mitigated" and fvg1_full["mitigated_at"] == 11 and fvg1_full["max_fill_pct"] == 100.0
    print(f"  OK — con criterio 'full', el MISMO FVG necesita llegar a 100% (vela {fvg1_full['mitigated_at']}) para mitigarse.")

    print("\n" + "=" * 70)
    print("CASO 7 — FVG invalidado correctamente")
    print("=" * 70)
    fvgB = [z for z in summaryB["all_fvg"] if z["created_at"] == 7][0]
    assert fvgB["status"] == "invalidated", f"Se esperaba invalidated, se obtuvo {fvgB['status']}"
    assert fvgB["invalidated_at"] == 8
    print(f"  OK — el FVG bajista queda 'invalidated' en la vela {fvgB['invalidated_at']} (close cerró por encima del top).")

    print("\n" + "=" * 70)
    print("CASO 8 — No existe look-ahead bias")
    print("=" * 70)
    assert all(z["created_at"] >= 2 for z in summary["all_fvg"]), "Ningún FVG puede confirmarse con menos de 3 velas"
    # El estado del FVG justo al crearse debe ser IDÉNTICO corriendo el motor
    # con datos truncados justo ahí, o con todo el dataset completo —
    # si alguna vela futura influyera en la creación, no coincidirían.
    _, summary_truncado = analyze_fvg(dfA.iloc[:8], mitigation_criterion="50%")
    fvg1_truncado = [z for z in summary_truncado["all_fvg"] if z["created_at"] == 7][0]
    assert fvg1_truncado["status"] == "active" and fvg1_truncado["max_fill_pct"] == 0.0
    assert fvg1_truncado["top"] == fvg1_final["top"] and fvg1_truncado["bottom"] == fvg1_final["bottom"]
    print("  OK — el estado del FVG en su vela de creación es idéntico con datos truncados o completos (no usa el futuro).")

    print("\n" + "=" * 70)
    print("ROBUSTEZ — vacío, pocas velas, columnas faltantes, NaN, sin volume")
    print("=" * 70)
    try:
        analyze_fvg(pd.DataFrame({"open": [], "high": [], "low": [], "close": []}))
        print("  ERROR: debía lanzar ValueError con DataFrame vacío")
    except ValueError:
        print("  OK — DataFrame vacío -> ValueError controlado.")

    try:
        analyze_fvg(pd.DataFrame({"open": [1, 2], "high": [1, 2]}))
        print("  ERROR: debía lanzar ValueError con columnas faltantes")
    except ValueError:
        print("  OK — columnas OHLC faltantes -> ValueError controlado.")

    df_pocas = dfA.head(2)
    _, s_pocas = analyze_fvg(df_pocas)
    print(f"  OK — con 2 velas no truena (no puede haber FVG todavía). FVGs: {len(s_pocas['all_fvg'])}")

    df_nan = dfA.copy()
    df_nan.iloc[3, df_nan.columns.get_loc("high")] = np.nan
    _, s_nan = analyze_fvg(df_nan)
    print(f"  OK — con un NaN en 'high' no truena. FVGs detectados: {len(s_nan['all_fvg'])}")

    df_sin_vol = dfA.copy()
    assert "volume" not in df_sin_vol.columns
    _, s_sin_vol = analyze_fvg(df_sin_vol)
    print(f"  OK — sin columna 'volume' funciona igual. FVGs: {len(s_sin_vol['all_fvg'])}")

    print("\nTodas las pruebas de core/fvg.py corrieron sin errores.")


if __name__ == "__main__":
    _demo()
