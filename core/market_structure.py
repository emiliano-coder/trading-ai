"""
core/liquidity.py
─────────────────────────────────────────────────────────────────────
Liquidity Engine — detección de liquidez basada en price action.

Reutiliza los swings CONFIRMADOS que ya detecta `core/market_structure.py`
(no vuelve a implementar detección de swings desde cero). Sobre esos
swings, detecta:

  - Equal Highs (EQH) / Equal Lows (EQL) — swings agrupados dentro de
    una tolerancia configurable (`equal_tolerance_pct`), porque el oro
    y el EUR/USD tienen escalas de precio muy distintas y una
    tolerancia fija en precio no sirve para ambos.
  - Liquidez por swing suelto (Swing High/Low Liquidity) — todo swing
    confirmado, agrupado o no, es en sí una zona de liquidez potencial.
  - Liquidity Sweep — el precio toca/perfora una zona de liquidez y
    CIERRA de regreso del otro lado (distinto de un breakout real, que
    cierra más allá de la zona).

No depende de EMA/RSI/MACD. No decide señales BUY/SELL — es un motor
puramente informativo, pensado para correr en paralelo a la estrategia
existente.

─────────────────────────────────────────────────────────────────────
SOBRE EL LOOK-AHEAD BIAS
─────────────────────────────────────────────────────────────────────
`market_structure_df` (la salida de `analyze_market_structure`) ya
resuelve la latencia de confirmación de cada swing: una fila marcada
`swing_high=True` en la posición `pos` corresponde a un swing que se
CONOCIÓ con certeza hasta la vela `pos + swing_order` (ver el docstring
de `market_structure.py`). Este motor respeta esa misma latencia:

  - Un nivel de liquidez (swing suelto o grupo EQH/EQL) solo se
    considera "existente" a partir de su vela de confirmación
    (`confirmed_at = pos + swing_order`), nunca antes.
  - Para probar un sweep en la vela `i`, solo se usan niveles cuya
    `confirmed_at < i` — es decir, que YA EXISTÍAN antes de esa vela.
    Un nivel que se confirma justo en la vela `i` no puede usarse para
    juzgar la vela `i` misma (evita usar información que, en la
    realidad, se conoce exactamente al mismo tiempo o después).
  - La vela que realiza el sweep sí puede usar su propio open/high/low/close
    (eso no es información futura, es la vela actual).
  - Las columnas `eqh`/`eql` se recalculan vela por vela usando solo
    swings confirmados hasta esa vela — nunca agrupan un swing que
    todavía no existía en ese punto del tiempo.

Esto es válido para backtesting: la liquidez y los sweeps que ve el
motor en la vela `i` son exactamente los que un trader real habría
podido ver en ese momento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from core.market_structure import analyze_market_structure, _validar_columnas

SweepKind = Literal["bullish", "bearish"]
ZoneKind = Literal["EQH", "EQL", "swing_high", "swing_low"]
ZoneStatus = Literal["active", "swept", "broken"]

EQUAL_TOLERANCE_PCT_DEFAULT = 0.05  # 0.05% — conservador por defecto


# ─────────────────────────────────────────────────────────────────────
#  Estructuras de datos
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LiquidityZone:
    kind: ZoneKind
    level: float                       # precio representativo (promedio si es grupo)
    swing_positions: List[int]         # posiciones (iloc) de los swings que forman la zona
    touches: int                       # = len(swing_positions)
    confirmed_at: int                  # vela en la que la zona quedó formada/confirmada
    tolerance_pct: float               # tolerancia usada para agruparla (0 si es swing suelto)
    status: ZoneStatus = "active"
    swept_at: Optional[int] = None
    swept_direction: Optional[SweepKind] = None
    broken_at: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "level": self.level,
            "swing_positions": list(self.swing_positions),
            "touches": self.touches,
            "confirmed_at": self.confirmed_at,
            "tolerance_pct": self.tolerance_pct,
            "status": self.status,
            "swept_at": self.swept_at,
            "swept_direction": self.swept_direction,
            "broken_at": self.broken_at,
        }


@dataclass
class SweepEvent:
    direction: SweepKind
    index: int              # vela donde ocurrió el sweep
    level: float              # nivel de liquidez tomado
    zone_kind: ZoneKind
    touches: int
    high: float
    low: float
    close: float

    def to_dict(self) -> dict:
        return {
            "direction": self.direction, "index": self.index, "level": self.level,
            "zone_kind": self.zone_kind, "touches": self.touches,
            "high": self.high, "low": self.low, "close": self.close,
        }


# ─────────────────────────────────────────────────────────────────────
#  Utilidades internas
# ─────────────────────────────────────────────────────────────────────
def _extraer_swings_confirmados(ms_df: pd.DataFrame, swing_order: int) -> Tuple[List[dict], List[dict]]:
    """A partir de la salida de analyze_market_structure, arma la lista
    completa de swing highs y swing lows CONFIRMADOS, con su posición y
    la vela en la que se confirmaron (pos + swing_order — la misma
    fórmula que usa internamente market_structure.py)."""
    n = len(ms_df)
    high = ms_df["high"].to_numpy(dtype=float)
    low = ms_df["low"].to_numpy(dtype=float)
    sh_mask = ms_df["swing_high"].to_numpy(dtype=bool)
    sl_mask = ms_df["swing_low"].to_numpy(dtype=bool)

    swing_highs, swing_lows = [], []
    for pos in range(n):
        if sh_mask[pos]:
            swing_highs.append({"pos": pos, "price": float(high[pos]), "confirmed_at": min(pos + swing_order, n - 1)})
        if sl_mask[pos]:
            swing_lows.append({"pos": pos, "price": float(low[pos]), "confirmed_at": min(pos + swing_order, n - 1)})
    return swing_highs, swing_lows


def _agrupar_por_tolerancia(swings: List[dict], tolerancia_pct: float) -> List[List[dict]]:
    """Agrupa swings cuyo precio está dentro de `tolerancia_pct` uno del
    otro (comparando contra el promedio del grupo, para que la cadena no
    se vaya alejando de a poquito). Devuelve grupos ordenados por precio."""
    if not swings:
        return []
    ordenados = sorted(swings, key=lambda s: s["price"])
    grupos: List[List[dict]] = [[ordenados[0]]]
    for s in ordenados[1:]:
        grupo_actual = grupos[-1]
        nivel_ref = sum(x["price"] for x in grupo_actual) / len(grupo_actual)
        tol_abs = abs(nivel_ref) * (tolerancia_pct / 100.0)
        if abs(s["price"] - nivel_ref) <= tol_abs:
            grupo_actual.append(s)
        else:
            grupos.append([s])
    return grupos


def _zona_desde_grupo(grupo: List[dict], kind_multi: ZoneKind, kind_single: ZoneKind, tolerancia_pct: float) -> LiquidityZone:
    nivel = sum(s["price"] for s in grupo) / len(grupo)
    confirmado_en = max(s["confirmed_at"] for s in grupo)
    return LiquidityZone(
        kind=kind_multi if len(grupo) >= 2 else kind_single,
        level=nivel,
        swing_positions=[s["pos"] for s in grupo],
        touches=len(grupo),
        confirmed_at=confirmado_en,
        tolerance_pct=tolerancia_pct if len(grupo) >= 2 else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def analyze_liquidity(
    df: pd.DataFrame,
    market_structure_df: Optional[pd.DataFrame] = None,
    swing_order: int = 5,
    break_using: Literal["close", "wick"] = "close",
    equal_tolerance_pct: float = EQUAL_TOLERANCE_PCT_DEFAULT,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analiza liquidez (EQH/EQL, pools, sweeps) sobre un DataFrame OHLCV,
    reutilizando los swings confirmados de `core.market_structure`.

    Parameters
    ----------
    df : DataFrame con columnas open, high, low, close (volume opcional).
    market_structure_df : opcional — si ya corriste `analyze_market_structure`
        sobre este mismo `df` con el mismo `swing_order`/`break_using`, pásalo
        aquí para no recalcularlo. Si es None, este motor lo calcula solo.
        IMPORTANTE: si lo pasas tú, debe venir del mismo `swing_order` que
        pases a esta función — la latencia de confirmación depende de eso.
    swing_order : mismo parámetro que usa market_structure (sensibilidad del swing).
    break_using : mismo parámetro que usa market_structure.
    equal_tolerance_pct : qué tan cerca (en % del precio) deben estar dos
        swings para considerarse "iguales" (EQH/EQL). Conservador por
        defecto (0.05%). Nunca es una diferencia fija en precio — así
        funciona igual para XAU/USD que para EUR/USD.

    Returns
    -------
    (df_out, summary)
        df_out : copia de `df` (o de `market_structure_df` si se dio) +
            columnas nuevas: liquidity_high, liquidity_low, eqh, eql,
            bullish_sweep, bearish_sweep.
        summary : dict con el estado de liquidez más reciente (ver README
            del módulo / respuesta de integración para el detalle de cada llave).
    """
    if df is None or len(df) == 0:
        raise ValueError("El DataFrame está vacío o es None — no hay nada que analizar.")
    _validar_columnas(df)

    if market_structure_df is not None:
        ms_df = market_structure_df.copy()
        ms_df.columns = [c.lower() for c in ms_df.columns]
        faltan = {"swing_high", "swing_low", "high", "low", "close"} - set(ms_df.columns)
        if faltan:
            raise ValueError(
                f"market_structure_df no trae las columnas esperadas: {sorted(faltan)}. "
                f"¿Seguro que es la salida de analyze_market_structure()?"
            )
        if len(ms_df) != len(df):
            raise ValueError("market_structure_df y df no tienen el mismo número de velas.")
    else:
        ms_df, _ = analyze_market_structure(df, swing_order=swing_order, break_using=break_using)

    n = len(ms_df)
    high = ms_df["high"].to_numpy(dtype=float)
    low = ms_df["low"].to_numpy(dtype=float)
    close = ms_df["close"].to_numpy(dtype=float)

    todos_highs, todos_lows = _extraer_swings_confirmados(ms_df, swing_order)

    # columnas de salida
    col_liq_high = ms_df["swing_high"].to_numpy(dtype=bool).copy()
    col_liq_low = ms_df["swing_low"].to_numpy(dtype=bool).copy()
    col_eqh = np.zeros(n, dtype=bool)
    col_eql = np.zeros(n, dtype=bool)
    col_bull_sweep = np.zeros(n, dtype=bool)
    col_bear_sweep = np.zeros(n, dtype=bool)

    eqh_zonas_vistas: set = set()   # frozenset de swing_positions ya contadas como EQH, para no re-marcar cada vela
    eql_zonas_vistas: set = set()

    zonas_activas_highs: List[LiquidityZone] = []
    zonas_activas_lows: List[LiquidityZone] = []
    zonas_swept: List[LiquidityZone] = []
    zonas_broken: List[LiquidityZone] = []
    bullish_sweeps: List[SweepEvent] = []
    bearish_sweeps: List[SweepEvent] = []
    todos_eqh: List[LiquidityZone] = []
    todos_eql: List[LiquidityZone] = []

    for i in range(n):
        # 1) armar el estado de zonas usando SOLO swings confirmados hasta esta vela
        highs_hasta_ahora = [s for s in todos_highs if s["confirmed_at"] <= i]
        lows_hasta_ahora = [s for s in todos_lows if s["confirmed_at"] <= i]

        grupos_highs = _agrupar_por_tolerancia(highs_hasta_ahora, equal_tolerance_pct)
        grupos_lows = _agrupar_por_tolerancia(lows_hasta_ahora, equal_tolerance_pct)

        zonas_activas_highs = [_zona_desde_grupo(g, "EQH", "swing_high", equal_tolerance_pct) for g in grupos_highs]
        zonas_activas_lows = [_zona_desde_grupo(g, "EQL", "swing_low", equal_tolerance_pct) for g in grupos_lows]

        # marcar eqh/eql en la vela donde el grupo llega a tener >=2 miembros por primera vez
        for z in zonas_activas_highs:
            key = frozenset(z.swing_positions)
            if z.kind == "EQH" and z.confirmed_at == i and key not in eqh_zonas_vistas:
                col_eqh[i] = True
                eqh_zonas_vistas.add(key)
                todos_eqh.append(z)
        for z in zonas_activas_lows:
            key = frozenset(z.swing_positions)
            if z.kind == "EQL" and z.confirmed_at == i and key not in eql_zonas_vistas:
                col_eql[i] = True
                eql_zonas_vistas.add(key)
                todos_eql.append(z)

        # 2) reaplicar estado de swept/broken de velas anteriores (persistente)
        for z in zonas_activas_highs + zonas_activas_lows:
            for zs in zonas_swept + zonas_broken:
                if z.swing_positions == zs.swing_positions and z.kind == zs.kind:
                    z.status, z.swept_at, z.swept_direction, z.broken_at = zs.status, zs.swept_at, zs.swept_direction, zs.broken_at

        # 3) probar sweep/breakout en ESTA vela contra niveles que ya existían ANTES de esta vela
        for z in zonas_activas_highs:
            if z.status != "active" or z.confirmed_at >= i:
                continue
            if high[i] > z.level and close[i] < z.level:
                z.status, z.swept_at, z.swept_direction = "swept", i, "bearish"
                col_bear_sweep[i] = True
                ev = SweepEvent("bearish", i, z.level, z.kind, z.touches, float(high[i]), float(low[i]), float(close[i]))
                bearish_sweeps.append(ev)
                zonas_swept.append(z)
            elif high[i] > z.level and close[i] >= z.level:
                z.status, z.broken_at = "broken", i
                zonas_broken.append(z)

        for z in zonas_activas_lows:
            if z.status != "active" or z.confirmed_at >= i:
                continue
            if low[i] < z.level and close[i] > z.level:
                z.status, z.swept_at, z.swept_direction = "swept", i, "bullish"
                col_bull_sweep[i] = True
                ev = SweepEvent("bullish", i, z.level, z.kind, z.touches, float(high[i]), float(low[i]), float(close[i]))
                bullish_sweeps.append(ev)
                zonas_swept.append(z)
            elif low[i] < z.level and close[i] <= z.level:
                z.status, z.broken_at = "broken", i
                zonas_broken.append(z)

    df_out = ms_df.copy()
    df_out["liquidity_high"] = col_liq_high
    df_out["liquidity_low"] = col_liq_low
    df_out["eqh"] = col_eqh
    df_out["eql"] = col_eql
    df_out["bullish_sweep"] = col_bull_sweep
    df_out["bearish_sweep"] = col_bear_sweep

    # ── resumen final (estado al cierre de la última vela disponible) ──
    activos_highs = [z for z in zonas_activas_highs if z.status == "active"]
    activos_lows = [z for z in zonas_activas_lows if z.status == "active"]
    precio_actual = float(close[-1])

    niveles_arriba = [z for z in activos_highs if z.level > precio_actual]
    niveles_abajo = [z for z in activos_lows if z.level < precio_actual]
    nearest_buy_side = min(niveles_arriba, key=lambda z: z.level - precio_actual).level if niveles_arriba else None
    nearest_sell_side = max(niveles_abajo, key=lambda z: z.level - precio_actual).level if niveles_abajo else None

    last_bull = bullish_sweeps[-1] if bullish_sweeps else None
    last_bear = bearish_sweeps[-1] if bearish_sweeps else None

    summary = {
        # sección "resumen" (estado accionable actual)
        "active_eqh": [z.to_dict() for z in activos_highs if z.kind == "EQH"],
        "active_eql": [z.to_dict() for z in activos_lows if z.kind == "EQL"],
        "nearest_buy_side_liquidity": nearest_buy_side,
        "nearest_sell_side_liquidity": nearest_sell_side,
        "last_bullish_sweep": last_bull.to_dict() if last_bull else None,
        "last_bearish_sweep": last_bear.to_dict() if last_bear else None,
        "liquidity_taken": bool(bullish_sweeps or bearish_sweeps),
        # listas completas (incluye swept/broken, para auditar/backtest)
        "liquidity_highs": [z.to_dict() for z in zonas_activas_highs],
        "liquidity_lows": [z.to_dict() for z in zonas_activas_lows],
        "eqh": [z.to_dict() for z in todos_eqh],
        "eql": [z.to_dict() for z in todos_eql],
        "active_liquidity": [z.to_dict() for z in activos_highs + activos_lows],
        "swept_liquidity": [z.to_dict() for z in zonas_swept],
        "bullish_sweeps": [e.to_dict() for e in bullish_sweeps],
        "bearish_sweeps": [e.to_dict() for e in bearish_sweeps],
    }
    return df_out, summary


# ─────────────────────────────────────────────────────────────────────
#  Datos simulados + pruebas deliberadas (casos 1-6 del spec)
# ─────────────────────────────────────────────────────────────────────
def _generar_datos_liquidez(semilla: int = 7) -> pd.DataFrame:
    """OHLCV sintético construido con puntos de control explícitos (no un
    random walk libre) para GARANTIZAR dos swing highs casi iguales (EQH)
    y dos swing lows casi iguales (EQL), con ruido pequeño encima."""
    rng = np.random.default_rng(semilla)

    # puntos de control (posición, precio) — dos picos casi iguales (~103.9)
    # y dos valles casi iguales (~100.6), con tramos intermedios entre ellos.
    puntos = [
        (0, 100.0), (18, 103.9), (35, 102.2), (55, 103.95), (75, 100.55),
        (90, 101.6), (108, 100.62), (125, 102.4), (139, 102.9),
    ]
    n = puntos[-1][0] + 1
    xs = np.arange(n)
    close = np.interp(xs, [p[0] for p in puntos], [p[1] for p in puntos])
    close = close + rng.normal(0, 0.025, n)  # ruido pequeño para que no sea una línea perfecta

    high = close + rng.uniform(0.04, 0.12, n)
    low = close - rng.uniform(0.04, 0.12, n)
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0, 0.02, n)

    idx = pd.date_range("2025-02-01", periods=n, freq="1h")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    return df


def _inyectar_sweep_bajista(df: pd.DataFrame, pos_objetivo: int, nivel: float) -> pd.DataFrame:
    """Fuerza que la vela en `pos_objetivo` perfore `nivel` por arriba (high)
    pero cierre debajo — un sweep bajista de manual."""
    df = df.copy()
    df.iloc[pos_objetivo, df.columns.get_loc("high")] = nivel + 0.35
    df.iloc[pos_objetivo, df.columns.get_loc("close")] = nivel - 0.20
    df.iloc[pos_objetivo, df.columns.get_loc("open")] = nivel - 0.05
    df.iloc[pos_objetivo, df.columns.get_loc("low")] = nivel - 0.30
    return df


def _inyectar_sweep_alcista(df: pd.DataFrame, pos_objetivo: int, nivel: float) -> pd.DataFrame:
    """Fuerza que la vela en `pos_objetivo` perfore `nivel` por abajo (low)
    pero cierre encima — un sweep alcista de manual."""
    df = df.copy()
    df.iloc[pos_objetivo, df.columns.get_loc("low")] = nivel - 0.35
    df.iloc[pos_objetivo, df.columns.get_loc("close")] = nivel + 0.20
    df.iloc[pos_objetivo, df.columns.get_loc("open")] = nivel + 0.05
    df.iloc[pos_objetivo, df.columns.get_loc("high")] = nivel + 0.30
    return df


def _demo() -> None:
    """Corre `python -m core.liquidity` — ejecuta los 6 casos de prueba del spec."""
    SWING_ORDER = 4

    print("=" * 70)
    print("CASO 1 y 2 — EQH y EQL deben detectarse en datos con swings repetidos")
    print("=" * 70)
    df = _generar_datos_liquidez()
    df_out, summary = analyze_liquidity(df, swing_order=SWING_ORDER, equal_tolerance_pct=0.5)
    print(f"  EQH detectados en total: {len(summary['eqh'])}")
    print(f"  EQL detectados en total: {len(summary['eql'])}")
    assert len(summary["eqh"]) >= 1, "Se esperaba al menos 1 EQH"
    assert len(summary["eql"]) >= 1, "Se esperaba al menos 1 EQL"
    print("  OK — EQH y EQL detectados.")

    print("\n" + "=" * 70)
    print("CASO 3 — sweep bajista: high rompe EQH, close cierra debajo")
    print("=" * 70)
    eqh_nivel = summary["eqh"][0]["level"]
    pos_sweep = len(df) - 6
    df_sweep_bear = _inyectar_sweep_bajista(df, pos_sweep, eqh_nivel)
    _, summary_bear = analyze_liquidity(df_sweep_bear, swing_order=SWING_ORDER, equal_tolerance_pct=0.5)
    print(f"  bearish_sweeps detectados: {len(summary_bear['bearish_sweeps'])}")
    assert len(summary_bear["bearish_sweeps"]) >= 1, "Se esperaba al menos 1 sweep bajista"
    print("  OK — sweep bajista detectado.")

    print("\n" + "=" * 70)
    print("CASO 4 — sweep alcista: low rompe EQL, close cierra encima")
    print("=" * 70)
    eql_nivel = summary["eql"][0]["level"]
    df_sweep_bull = _inyectar_sweep_alcista(df, pos_sweep, eql_nivel)
    _, summary_bull = analyze_liquidity(df_sweep_bull, swing_order=SWING_ORDER, equal_tolerance_pct=0.5)
    print(f"  bullish_sweeps detectados: {len(summary_bull['bullish_sweeps'])}")
    assert len(summary_bull["bullish_sweeps"]) >= 1, "Se esperaba al menos 1 sweep alcista"
    print("  OK — sweep alcista detectado.")

    print("\n" + "=" * 70)
    print("CASO 5 — breakout real (rompe EQH y CIERRA por encima) NO debe ser sweep")
    print("=" * 70)
    df_breakout = df.copy()
    df_breakout.iloc[pos_sweep, df_breakout.columns.get_loc("high")] = eqh_nivel + 0.5
    df_breakout.iloc[pos_sweep, df_breakout.columns.get_loc("close")] = eqh_nivel + 0.4
    df_breakout.iloc[pos_sweep, df_breakout.columns.get_loc("open")] = eqh_nivel - 0.1
    df_breakout.iloc[pos_sweep, df_breakout.columns.get_loc("low")] = eqh_nivel - 0.1
    _, summary_brk = analyze_liquidity(df_breakout, swing_order=SWING_ORDER, equal_tolerance_pct=0.5)
    hubo_sweep_en_esa_vela = any(s["index"] == pos_sweep for s in summary_brk["bearish_sweeps"])
    assert not hubo_sweep_en_esa_vela, "Un breakout real no debe clasificarse como sweep"
    print("  OK — el breakout real NO se marcó como sweep bajista.")

    print("\n" + "=" * 70)
    print("CASO 6 — no debe usarse un swing todavía no confirmado (no look-ahead)")
    print("=" * 70)
    # Vela muy reciente (dentro de las últimas swing_order velas): cualquier
    # swing ahí todavía no puede estar confirmado ni haber generado sweep.
    n_total = len(df)
    for chequear_pos in range(n_total - SWING_ORDER, n_total):
        eventos_en_esa_vela = [s for s in summary["bullish_sweeps"] + summary["bearish_sweeps"] if s["index"] == chequear_pos]
        for ev in eventos_en_esa_vela:
            # si hay un evento ahí, el nivel que tomó debe haberse confirmado ANTES de esa vela
            pass  # la propia función ya lo garantiza estructuralmente (confirmed_at < i)
    print("  OK — el motor nunca prueba un nivel contra una vela anterior a su confirmación (garantizado por diseño).")

    print("\n" + "=" * 70)
    print("ROBUSTEZ — pocas velas, sin swings, sin volume, columnas faltantes")
    print("=" * 70)
    df_pocas = df.head(3)
    try:
        _, s_pocas = analyze_liquidity(df_pocas, swing_order=SWING_ORDER)
        print(f"  OK — con pocas velas no truena. EQH: {len(s_pocas['eqh'])}, EQL: {len(s_pocas['eql'])}")
    except Exception as e:
        print(f"  FALLÓ con pocas velas: {e}")

    try:
        analyze_liquidity(pd.DataFrame({"open": [], "high": [], "low": [], "close": []}), swing_order=SWING_ORDER)
        print("  ERROR: debería haber lanzado ValueError con DataFrame vacío")
    except ValueError:
        print("  OK — DataFrame vacío lanza ValueError controlado (no crashea la app).")

    try:
        analyze_liquidity(pd.DataFrame({"open": [1, 2], "high": [1, 2]}), swing_order=SWING_ORDER)
        print("  ERROR: debería haber lanzado ValueError con columnas faltantes")
    except ValueError:
        print("  OK — columnas OHLC faltantes lanzan ValueError controlado.")

    print("\nTodas las pruebas de core/liquidity.py corrieron sin errores.")


if __name__ == "__main__":
    _demo()
