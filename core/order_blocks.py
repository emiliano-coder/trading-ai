"""
core/order_blocks.py
─────────────────────────────────────────────────────────────────────
Order Blocks Engine — independiente y reutilizable.

Detecta Bullish y Bearish Order Blocks de forma CONSERVADORA: nunca
etiqueta un OB solo porque una vela sea grande o de cierto color. Un
OB requiere contexto — una vela de origen (la última vela "opuesta"
antes del impulso) MÁS desplazamiento posterior significativo (medido
en ATR, calculado aquí mismo sin librerías externas), y opcionalmente
una ruptura de estructura (BOS) real si se le da la salida de
`market_structure.py`.

Puede aprovechar (de forma opcional, nunca obligatoria):
  - market_structure_df  (de core.market_structure.analyze_market_structure)
    → para exigir BOS real como confirmación, en vez de solo desplazamiento.
  - liquidity_df          (de core.liquidity.analyze_liquidity)
    → para marcar si el OB nace justo después de un liquidity sweep
      (una confluencia típica que suma a su "strength").
  - fvg_df                (de core.fvg.analyze_fvg)
    → para marcar si el OB se solapa con un FVG activo (otra confluencia).

El módulo funciona completo sin ninguno de los tres (son opcionales) —
no depende de ellos para operar, solo los aprovecha si están disponibles.

No depende de EMA/RSI/MACD. No decide señales BUY/SELL/WAIT — es
puramente informativo, pensado para correr en paralelo a la estrategia
existente (igual que los otros motores de `core/`).

─────────────────────────────────────────────────────────────────────
DEFINICIÓN
─────────────────────────────────────────────────────────────────────
Bullish Order Block — zona de demanda:
  1. Se localiza la ÚLTIMA vela bajista (close < open) inmediatamente
     antes de que empiece un movimiento alcista — es decir, una vela
     bajista en la posición `k` cuya vela siguiente `k+1` YA NO es
     bajista (si lo fuera, `k` no sería "la última", lo sería `k+1`
     o una posterior).
  2. Se exige desplazamiento alcista significativo después: dentro de
     una ventana configurable (`displacement_window` velas), el precio
     debe moverse al alza al menos `displacement_atr_mult` veces el
     ATR de la vela `k` (o, alternativamente, `min_displacement_pct`
     % del precio — lo que se le dé).
  3. Opcionalmente (`require_bos=True`), además del desplazamiento, se
     exige que ocurra un BOS alcista real (de `market_structure_df`)
     dentro de esa misma ventana — si no aparece, NO se confirma el OB.

Bearish Order Block — zona de oferta: exactamente lo mismo, en espejo
(última vela alcista antes de desplazamiento bajista, BOS bajista).

Esto evita la regla simplista "vela roja = OB alcista" — una vela
bajista sin desplazamiento real después NUNCA se marca como OB.

─────────────────────────────────────────────────────────────────────
SOBRE candle_time / created_at / confirmed_at
─────────────────────────────────────────────────────────────────────
  - candle_time   : la vela de origen (`k`) — la última vela opuesta
                     antes del impulso. Es la ubicación histórica de
                     la zona, pero en la vela `k` TODAVÍA NO SABEMOS
                     si va a ser un OB válido.
  - created_at    : `k + 1` — la primera vela del posible impulso.
                     Desde aquí se empieza a medir el desplazamiento.
  - confirmed_at  : la vela en la que el desplazamiento (y el BOS, si
                     se exige) ya cumplió el umbral — este es el
                     momento en el que, en la realidad, un trader
                     habría podido saber que ese `k` era un OB válido.

Las columnas `ob_bullish`/`ob_bearish`/`ob_top`/`ob_bottom`/etc. se
escriben en la posición `k` (para poder dibujar la zona en su lugar
histórico correcto en una gráfica), pero **ningún estado de mitigación
se evalúa con velas anteriores a `confirmed_at`** — el seguimiento de
activo/tocado/mitigado/invalidado arranca estrictamente después de
`confirmed_at`, nunca antes. Esto es la misma filosofía que usa
`swing_high`/`swing_low` en `market_structure.py`: la columna marca el
origen para visualización, pero la lógica de validez respeta cuándo se
confirmó de verdad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from core.market_structure import _validar_columnas

OBKind = Literal["bullish", "bearish"]
OBStatus = Literal["active", "touched", "partially_mitigated", "fully_mitigated", "invalidated"]
MitigationCriterion = Literal["touch", "50%", "full"]
ZoneMethod = Literal["high_low", "open_close", "body_wick"]

CRITERIOS_MITIGACION: Dict[str, float] = {"touch": 0.0, "50%": 50.0, "full": 100.0}
CRITERIO_DEFAULT: MitigationCriterion = "50%"
ZONA_DEFAULT: ZoneMethod = "high_low"

_STATUS_RANK = {"active": 0, "touched": 1, "partially_mitigated": 2, "fully_mitigated": 3}


# ─────────────────────────────────────────────────────────────────────
#  Estructura de datos
# ─────────────────────────────────────────────────────────────────────
@dataclass
class OrderBlockZone:
    kind: OBKind
    candle_time: int
    created_at: int
    confirmed_at: int
    top: float
    bottom: float
    zone_method: str
    status: OBStatus = "active"
    max_fill_pct: float = 0.0
    touched_at: Optional[int] = None
    mitigated_at: Optional[int] = None
    invalidated_at: Optional[int] = None
    touches: int = 0
    mitigation_criterion: str = CRITERIO_DEFAULT
    # features de calidad (se calculan una vez, al confirmarse)
    displacement_size: float = 0.0
    displacement_atr: float = 0.0
    bos_confirmed: bool = False
    distance_to_bos: Optional[int] = None
    volume_confirmation: Optional[bool] = None
    near_liquidity_sweep: bool = False
    overlap_with_fvg: bool = False
    strength: float = 0.0

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    def freshness(self) -> str:
        if self.status == "invalidated":
            return "invalidated"
        if self.status == "fully_mitigated":
            return "mitigated"
        if self.status == "active":
            return "fresh"
        return "touched"

    def to_dict(self, as_of: Optional[int] = None) -> dict:
        age = (as_of - self.confirmed_at) if as_of is not None else None
        return {
            "kind": self.kind,
            "candle_time": self.candle_time,
            "created_at": self.created_at,
            "confirmed_at": self.confirmed_at,
            "top": self.top,
            "bottom": self.bottom,
            "midpoint": self.midpoint,
            "zone_method": self.zone_method,
            "status": self.status,
            "freshness": self.freshness(),
            "max_fill_pct": round(self.max_fill_pct, 2),
            "touched_at": self.touched_at,
            "mitigated_at": self.mitigated_at,
            "invalidated_at": self.invalidated_at,
            "number_of_touches": self.touches,
            "age": age,
            "mitigation_criterion": self.mitigation_criterion,
            "displacement_size": round(self.displacement_size, 6),
            "displacement_atr": round(self.displacement_atr, 3),
            "bos_confirmed": self.bos_confirmed,
            "distance_to_bos": self.distance_to_bos,
            "volume_confirmation": self.volume_confirmation,
            "near_liquidity_sweep": self.near_liquidity_sweep,
            "overlap_with_fvg": self.overlap_with_fvg,
            "strength": round(self.strength, 1),
        }


# ─────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────
def _atr_serie(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR calculado a mano (sin librerías de indicadores), suavizado
    estilo Wilder. Causal: atr[i] solo usa información hasta la vela i."""
    n = len(h)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0] if n else 0
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.zeros(n)
    if n == 0:
        return atr
    atr[0] = tr[0]
    for i in range(1, n):
        if i < period:
            atr[i] = tr[: i + 1].mean()
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _zona_desde_vela(kind: OBKind, o: float, h: float, l: float, c: float, metodo: ZoneMethod) -> Tuple[float, float]:
    if metodo == "high_low":
        return float(h), float(l)
    if metodo == "open_close":
        return float(max(o, c)), float(min(o, c))
    if metodo == "body_wick":
        if kind == "bullish":  # vela bajista de origen: top = open (cuerpo), bottom = low (mecha completa)
            return float(o), float(l)
        else:  # vela alcista de origen: top = high (mecha completa), bottom = open (cuerpo)
            return float(h), float(o)
    raise ValueError(f"zone_method desconocido: {metodo}")


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def analyze_order_blocks(
    df: pd.DataFrame,
    market_structure_df: Optional[pd.DataFrame] = None,
    liquidity_df: Optional[pd.DataFrame] = None,
    fvg_df: Optional[pd.DataFrame] = None,
    zone_method: ZoneMethod = ZONA_DEFAULT,
    displacement_window: int = 3,
    displacement_atr_mult: float = 1.5,
    atr_period: int = 14,
    require_bos: bool = False,
    mitigation_criterion: MitigationCriterion = CRITERIO_DEFAULT,
    min_displacement_pct: float = 0.0,
    liquidity_lookback: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detecta Order Blocks (bullish/bearish) de forma conservadora, sin
    look-ahead bias, y sigue su estado a lo largo del tiempo.

    Parameters
    ----------
    df : DataFrame con columnas open, high, low, close (volume opcional).
    market_structure_df : opcional — salida de analyze_market_structure()
        sobre este mismo df. Si se da y `require_bos=True`, se exige un
        BOS real (no solo desplazamiento) para confirmar el OB.
    liquidity_df : opcional — salida de analyze_liquidity(). Si se da,
        se marca si el OB nace cerca de un liquidity sweep reciente
        (confluencia que suma a `strength`).
    fvg_df : opcional — salida de analyze_fvg(). Si se da, se marca si
        la zona del OB se solapa con un FVG activo al momento de
        confirmarse (otra confluencia).
    zone_method : "high_low" (vela completa, con mechas) | "open_close"
        (solo el cuerpo) | "body_wick" (cuerpo por un lado, mecha
        completa por el otro — ver docstring del módulo).
    displacement_window : cuántas velas después de la vela de origen se
        revisan buscando desplazamiento suficiente.
    displacement_atr_mult : cuántos ATR de movimiento se exigen como
        desplazamiento mínimo (ATR calculado internamente, sin `ta`).
    atr_period : período para el cálculo del ATR.
    require_bos : si True, exige además un BOS real de `market_structure_df`
        en la ventana de desplazamiento. Si no se da `market_structure_df`,
        esta exigencia se ignora silenciosamente (el módulo sigue
        funcionando solo con desplazamiento) — documentado así a propósito.
    mitigation_criterion : "touch" | "50%" | "full" (igual que en liquidity/fvg).
    min_displacement_pct : umbral alternativo/adicional de desplazamiento
        como % del precio (0 = no se usa, solo se exige el de ATR).
    liquidity_lookback : cuántas velas antes de la vela de origen se
        revisan buscando un sweep reciente (para `near_liquidity_sweep`).

    Returns
    -------
    (df_out, summary)
    """
    if df is None or len(df) == 0:
        raise ValueError("El DataFrame está vacío o es None — no hay nada que analizar.")
    _validar_columnas(df)
    if mitigation_criterion not in CRITERIOS_MITIGACION:
        raise ValueError(f"mitigation_criterion debe ser uno de {list(CRITERIOS_MITIGACION)}")
    if zone_method not in ("high_low", "open_close", "body_wick"):
        raise ValueError("zone_method debe ser 'high_low', 'open_close' o 'body_wick'")
    threshold = CRITERIOS_MITIGACION[mitigation_criterion]
    half_threshold = threshold / 2.0

    df_out = df.copy()
    df_out.columns = [c.lower() for c in df_out.columns]
    n = len(df_out)

    o = df_out["open"].to_numpy(dtype=float)
    h = df_out["high"].to_numpy(dtype=float)
    l = df_out["low"].to_numpy(dtype=float)
    c = df_out["close"].to_numpy(dtype=float)
    vol = df_out["volume"].to_numpy(dtype=float) if "volume" in df_out.columns else None

    ms = None
    if market_structure_df is not None:
        ms = market_structure_df.copy()
        ms.columns = [x.lower() for x in ms.columns]

    liq = None
    if liquidity_df is not None:
        liq = liquidity_df.copy()
        liq.columns = [x.lower() for x in liq.columns]

    fvg = None
    if fvg_df is not None:
        fvg = fvg_df.copy()
        fvg.columns = [x.lower() for x in fvg.columns]

    atr = _atr_serie(h, l, c, period=atr_period)

    col_ob_bull = np.zeros(n, dtype=bool)
    col_ob_bear = np.zeros(n, dtype=bool)
    col_top = np.full(n, np.nan)
    col_bottom = np.full(n, np.nan)
    col_mid = np.full(n, np.nan)
    col_created_at = np.full(n, np.nan)
    col_confirmed_at = np.full(n, np.nan)
    col_strength = np.full(n, np.nan)
    col_active = np.zeros(n, dtype=bool)
    col_touched = np.zeros(n, dtype=bool)
    col_mitigated = np.zeros(n, dtype=bool)
    col_invalidated = np.zeros(n, dtype=bool)
    col_fresh = np.zeros(n, dtype=bool)

    zonas: List[OrderBlockZone] = []

    def _es_bajista(idx: int) -> bool:
        return c[idx] < o[idx]

    def _es_alcista(idx: int) -> bool:
        return c[idx] > o[idx]

    def _buscar_confirmacion(k: int, kind: OBKind) -> Optional[Tuple[int, float, float, bool, Optional[int]]]:
        """Busca, desde k+1 hasta k+displacement_window, el primer punto
        donde el desplazamiento (y el BOS si se exige) cumple el umbral.
        Devuelve (confirmed_at, displacement_size, displacement_atr,
        bos_confirmed, distance_to_bos) o None si no se confirma."""
        window_end = min(k + displacement_window, n - 1)
        if window_end <= k:
            return None
        atr_k = atr[k] if atr[k] > 0 else np.nan
        ref = c[k]
        extremo = -np.inf if kind == "bullish" else np.inf
        confirmed_idx = None
        disp_size = 0.0
        for j in range(k + 1, window_end + 1):
            extremo = max(extremo, h[j]) if kind == "bullish" else min(extremo, l[j])
            move = (extremo - ref) if kind == "bullish" else (ref - extremo)
            disp_atr = (move / atr_k) if (atr_k and not np.isnan(atr_k) and atr_k > 0) else 0.0
            disp_pct = (move / abs(ref) * 100) if ref else 0.0
            if disp_atr >= displacement_atr_mult or (min_displacement_pct > 0 and disp_pct >= min_displacement_pct):
                confirmed_idx = j
                disp_size = move
                break
        if confirmed_idx is None:
            return None

        disp_atr_final = (disp_size / atr_k) if (atr_k and not np.isnan(atr_k) and atr_k > 0) else 0.0
        bos_confirmed = False
        distancia_bos = None

        if require_bos:
            if ms is None:
                pass  # no hay market_structure_df -> se ignora la exigencia, el módulo sigue funcionando
            else:
                col_bos = "bos_bullish" if kind == "bullish" else "bos_bearish"
                if col_bos not in ms.columns:
                    return None
                sub = ms[col_bos].to_numpy(dtype=bool)[k + 1 : window_end + 1]
                if not sub.any():
                    return None  # se exige BOS y no aparece en la ventana -> no se confirma
                primer_bos = k + 1 + int(np.argmax(sub))
                confirmed_idx = max(confirmed_idx, primer_bos)
                bos_confirmed = True
                distancia_bos = primer_bos - k
        elif ms is not None:
            # no se exige, pero si está disponible lo registramos como dato informativo
            col_bos = "bos_bullish" if kind == "bullish" else "bos_bearish"
            if col_bos in ms.columns:
                sub = ms[col_bos].to_numpy(dtype=bool)[k + 1 : window_end + 1]
                if sub.any():
                    primer_bos = k + 1 + int(np.argmax(sub))
                    bos_confirmed = True
                    distancia_bos = primer_bos - k

        return confirmed_idx, disp_size, disp_atr_final, bos_confirmed, distancia_bos

    def _volumen_confirma(k: int, window_end: int) -> Optional[bool]:
        if vol is None:
            return None
        prev_start = max(0, k - 20)
        prev_avg = np.nanmean(vol[prev_start:k]) if k > prev_start else np.nan
        disp_avg = np.nanmean(vol[k + 1 : window_end + 1])
        if not np.isfinite(prev_avg) or prev_avg <= 0:
            return None
        return bool(disp_avg > prev_avg * 1.1)

    def _cerca_de_sweep(k: int, kind: OBKind) -> bool:
        if liq is None:
            return False
        col_sweep = "bullish_sweep" if kind == "bullish" else "bearish_sweep"
        if col_sweep not in liq.columns:
            return False
        ini = max(0, k - liquidity_lookback)
        return bool(liq[col_sweep].to_numpy(dtype=bool)[ini : k + 1].any())

    def _overlap_fvg(top: float, bottom: float, hasta: int) -> bool:
        if fvg is None:
            return False
        tcol, bcol = "fvg_top", "fvg_bottom"
        if tcol not in fvg.columns or bcol not in fvg.columns:
            return False
        sub = fvg.iloc[: hasta + 1]
        ftop = sub[tcol].to_numpy(dtype=float)
        fbot = sub[bcol].to_numpy(dtype=float)
        valido = ~np.isnan(ftop) & ~np.isnan(fbot)
        if not valido.any():
            return False
        ftop, fbot = ftop[valido], fbot[valido]
        return bool(np.any((ftop >= bottom) & (fbot <= top)))

    # ── 1) detectar y confirmar Order Blocks (causal, de izquierda a derecha) ──
    for k in range(n - 1):
        if _es_bajista(k) and not _es_bajista(k + 1):
            res = _buscar_confirmacion(k, "bullish")
            if res:
                confirmed_idx, disp_size, disp_atr_final, bos_conf, dist_bos = res
                top, bottom = _zona_desde_vela("bullish", o[k], h[k], l[k], c[k], zone_method)
                z = OrderBlockZone(
                    kind="bullish", candle_time=k, created_at=k + 1, confirmed_at=confirmed_idx,
                    top=top, bottom=bottom, zone_method=zone_method, mitigation_criterion=mitigation_criterion,
                    displacement_size=disp_size, displacement_atr=disp_atr_final,
                    bos_confirmed=bos_conf, distance_to_bos=dist_bos,
                )
                z.volume_confirmation = _volumen_confirma(k, min(k + displacement_window, n - 1))
                z.near_liquidity_sweep = _cerca_de_sweep(k, "bullish")
                z.overlap_with_fvg = _overlap_fvg(top, bottom, confirmed_idx)
                z.strength = _calcular_strength(z)
                zonas.append(z)
                col_ob_bull[k] = True
                col_top[k], col_bottom[k], col_mid[k] = top, bottom, z.midpoint
                col_created_at[k], col_confirmed_at[k] = z.created_at, z.confirmed_at
                col_strength[k] = z.strength

        if _es_alcista(k) and not _es_alcista(k + 1):
            res = _buscar_confirmacion(k, "bearish")
            if res:
                confirmed_idx, disp_size, disp_atr_final, bos_conf, dist_bos = res
                top, bottom = _zona_desde_vela("bearish", o[k], h[k], l[k], c[k], zone_method)
                z = OrderBlockZone(
                    kind="bearish", candle_time=k, created_at=k + 1, confirmed_at=confirmed_idx,
                    top=top, bottom=bottom, zone_method=zone_method, mitigation_criterion=mitigation_criterion,
                    displacement_size=disp_size, displacement_atr=disp_atr_final,
                    bos_confirmed=bos_conf, distance_to_bos=dist_bos,
                )
                z.volume_confirmation = _volumen_confirma(k, min(k + displacement_window, n - 1))
                z.near_liquidity_sweep = _cerca_de_sweep(k, "bearish")
                z.overlap_with_fvg = _overlap_fvg(top, bottom, confirmed_idx)
                z.strength = _calcular_strength(z)
                zonas.append(z)
                col_ob_bear[k] = True
                col_top[k], col_bottom[k], col_mid[k] = top, bottom, z.midpoint
                col_created_at[k], col_confirmed_at[k] = z.created_at, z.confirmed_at
                col_strength[k] = z.strength

    # ── 2) seguir el estado de cada zona, vela por vela, SOLO desde confirmed_at+1 ──
    for i in range(n):
        for z in zonas:
            if z.confirmed_at > i:
                continue  # todavía no existe de verdad en la vela i
            if z.status == "invalidated":
                continue  # terminal
            if i == z.confirmed_at:
                continue  # la vela de confirmación no cuenta como "revisita"

            if z.kind == "bullish":
                intrusion = max(0.0, z.top - l[i])
                intrusion = min(intrusion, z.top - z.bottom)
                size = z.top - z.bottom
                fill_pct = (intrusion / size * 100) if size else 0.0
                if fill_pct > z.max_fill_pct:
                    z.touches += 1
                z.max_fill_pct = max(z.max_fill_pct, fill_pct)
                if c[i] < z.bottom:
                    z.status, z.invalidated_at = "invalidated", i
                    col_invalidated[i] = True
                    continue
            else:
                intrusion = max(0.0, h[i] - z.bottom)
                intrusion = min(intrusion, z.top - z.bottom)
                size = z.top - z.bottom
                fill_pct = (intrusion / size * 100) if size else 0.0
                if fill_pct > z.max_fill_pct:
                    z.touches += 1
                z.max_fill_pct = max(z.max_fill_pct, fill_pct)
                if c[i] > z.top:
                    z.status, z.invalidated_at = "invalidated", i
                    col_invalidated[i] = True
                    continue

            if z.status != "fully_mitigated" and z.max_fill_pct > 0:
                if z.touched_at is None:
                    z.touched_at = i
                    col_touched[i] = True
                nuevo_status = z.status
                if z.max_fill_pct >= threshold and threshold >= 0 and z.max_fill_pct > 0:
                    nuevo_status = "fully_mitigated"
                elif z.max_fill_pct >= half_threshold and half_threshold > 0:
                    nuevo_status = "partially_mitigated"
                else:
                    nuevo_status = "touched"
                if _STATUS_RANK.get(nuevo_status, 0) > _STATUS_RANK.get(z.status, 0):
                    z.status = nuevo_status
                    if z.status == "fully_mitigated":
                        z.mitigated_at = i
                        col_mitigated[i] = True

        col_active[i] = any(z.confirmed_at <= i and z.status not in ("fully_mitigated", "invalidated") for z in zonas)
        col_fresh[i] = any(z.confirmed_at <= i and z.status == "active" for z in zonas)

    df_out["ob_bullish"] = col_ob_bull
    df_out["ob_bearish"] = col_ob_bear
    df_out["ob_top"] = col_top
    df_out["ob_bottom"] = col_bottom
    df_out["ob_midpoint"] = col_mid
    df_out["ob_created_at"] = col_created_at
    df_out["ob_confirmed_at"] = col_confirmed_at
    df_out["ob_strength"] = col_strength
    df_out["ob_active"] = col_active
    df_out["ob_touched"] = col_touched
    df_out["ob_mitigated"] = col_mitigated
    df_out["ob_invalidated"] = col_invalidated
    df_out["ob_fresh"] = col_fresh

    ultimo_idx = n - 1
    bullish_z = [z for z in zonas if z.kind == "bullish"]
    bearish_z = [z for z in zonas if z.kind == "bearish"]
    activos_bull = [z for z in bullish_z if z.status not in ("fully_mitigated", "invalidated")]
    activos_bear = [z for z in bearish_z if z.status not in ("fully_mitigated", "invalidated")]
    fresh_bull = [z for z in bullish_z if z.status == "active"]
    fresh_bear = [z for z in bearish_z if z.status == "active"]

    precio_actual = float(c[-1]) if n else None
    nearest_bull = None
    nearest_bear = None
    if precio_actual is not None:
        candidatos_bull = [z for z in activos_bull if z.top <= precio_actual]  # demanda, suele estar debajo
        candidatos_bear = [z for z in activos_bear if z.bottom >= precio_actual]  # oferta, suele estar arriba
        if candidatos_bull:
            nearest_bull = max(candidatos_bull, key=lambda z: z.top)
        if candidatos_bear:
            nearest_bear = min(candidatos_bear, key=lambda z: z.bottom)

    summary = {
        "active_bullish_obs": [z.to_dict(as_of=ultimo_idx) for z in activos_bull],
        "active_bearish_obs": [z.to_dict(as_of=ultimo_idx) for z in activos_bear],
        "fresh_bullish_obs": [z.to_dict(as_of=ultimo_idx) for z in fresh_bull],
        "fresh_bearish_obs": [z.to_dict(as_of=ultimo_idx) for z in fresh_bear],
        "last_bullish_ob": bullish_z[-1].to_dict(as_of=ultimo_idx) if bullish_z else None,
        "last_bearish_ob": bearish_z[-1].to_dict(as_of=ultimo_idx) if bearish_z else None,
        "nearest_bullish_ob": nearest_bull.to_dict(as_of=ultimo_idx) if nearest_bull else None,
        "nearest_bearish_ob": nearest_bear.to_dict(as_of=ultimo_idx) if nearest_bear else None,
        "all_order_blocks": [z.to_dict(as_of=ultimo_idx) for z in zonas],
    }
    return df_out, summary


def _calcular_strength(z: OrderBlockZone) -> float:
    """Puntuación informativa simple (0-100) de qué tan 'de calidad' se
    ve el OB en el momento de confirmarse. NO es una señal de trading —
    es un insumo pensado para un futuro Setup Engine, nada más."""
    parte_desplazamiento = 40.0 * min(max(z.displacement_atr, 0) / 2.0, 1.0)
    parte_bos = 20.0 if z.bos_confirmed else 0.0
    if z.volume_confirmation is True:
        parte_volumen = 15.0
    elif z.volume_confirmation is None:
        parte_volumen = 7.5  # sin dato de volumen, ni suma ni resta del todo
    else:
        parte_volumen = 0.0
    parte_fvg = 15.0 if z.overlap_with_fvg else 0.0
    parte_liquidez = 10.0 if z.near_liquidity_sweep else 0.0
    total = parte_desplazamiento + parte_bos + parte_volumen + parte_fvg + parte_liquidez
    return float(max(0.0, min(100.0, total)))


# ─────────────────────────────────────────────────────────────────────
#  Datos simulados + pruebas deliberadas (12 casos del spec)
# ─────────────────────────────────────────────────────────────────────
def _generar_datos_ob_bullish() -> pd.DataFrame:
    """OHLC a mano: zona plana sin desplazamiento (para el caso 3),
    luego una vela bajista de origen (k=5) seguida de un impulso alcista
    con desplazamiento real (confirmación en k=6 o k=7), y velas de
    seguimiento que llevan la zona de activa -> touched -> parcial -> mitigada."""
    filas = [
        (100.00, 100.05, 99.95, 100.02),  # 0
        (100.02, 100.08, 99.97, 100.03),  # 1  vela bajista chica, SIN desplazamiento después (caso 3)
        (100.03, 100.06, 99.96, 99.98),   # 2  (sigue plano, no debe generar OB)
        (99.98, 100.04, 99.95, 100.00),   # 3
        (100.00, 100.06, 99.97, 100.02),  # 4
        (100.02, 100.05, 99.80, 99.85),   # 5  vela de ORIGEN (bajista) — k=5
        (99.85, 100.60, 99.83, 100.55),   # 6  impulso alcista fuerte (desplazamiento)
        (100.55, 100.90, 100.50, 100.85), # 7  confirma aún más
        (100.85, 100.95, 100.70, 100.90), # 8  fill=0 (sigue arriba de la zona)
        (100.90, 100.95, 100.00, 100.10), # 9  entra a la zona (low=100.00) -> touched/parcial
        (100.10, 100.20, 99.90, 100.05),  # 10 entra más profundo -> fill mayor -> mitigado (según criterio)
    ]
    idx = pd.date_range("2025-04-01", periods=len(filas), freq="1h")
    return pd.DataFrame(filas, columns=["open", "high", "low", "close"], index=idx)


def _generar_datos_ob_bearish_invalidado() -> pd.DataFrame:
    """OHLC a mano: vela alcista de origen seguida de desplazamiento
    bajista real, y luego una vela que invalida la zona (cierre por
    encima del top)."""
    filas = [
        (100.00, 100.05, 99.95, 100.02),  # 0
        (100.02, 100.08, 99.97, 100.05),  # 1
        (100.05, 100.09, 99.98, 100.03),  # 2
        (100.03, 100.07, 99.96, 100.01),  # 3
        (100.01, 100.08, 99.97, 100.04),  # 4
        (100.04, 100.25, 100.02, 100.20), # 5  vela de ORIGEN (alcista) — k=5
        (100.20, 100.22, 99.50, 99.60),   # 6  impulso bajista fuerte
        (99.60, 99.65, 99.10, 99.20),     # 7  confirma más
        (99.20, 99.30, 99.00, 99.10),     # 8  fill=0
        (99.10, 100.40, 99.05, 100.30),   # 9  high rebasa el top (100.25) y CIERRA por encima -> invalidado
    ]
    idx = pd.date_range("2025-04-01", periods=len(filas), freq="1h")
    return pd.DataFrame(filas, columns=["open", "high", "low", "close"], index=idx)


def _demo() -> None:
    """Corre `python -m core.order_blocks` — ejecuta los 12 casos del spec."""
    dfA = _generar_datos_ob_bullish()
    dfB = _generar_datos_ob_bearish_invalidado()

    print("=" * 70)
    print("CASO 1 — Bullish OB válido")
    print("=" * 70)
    df_out, summary = analyze_order_blocks(dfA, displacement_atr_mult=1.2, mitigation_criterion="50%")
    assert df_out["ob_bullish"].iloc[5] == True, "Se esperaba un bullish OB en la vela 5"
    ob1 = [z for z in summary["all_order_blocks"] if z["candle_time"] == 5][0]
    print(f"  OK — OB bullish en vela 5: top={ob1['top']}, bottom={ob1['bottom']}, confirmado en vela {ob1['confirmed_at']}")

    print("\n" + "=" * 70)
    print("CASO 2 — Bearish OB válido")
    print("=" * 70)
    dfB_out, summaryB = analyze_order_blocks(dfB, displacement_atr_mult=1.2, mitigation_criterion="50%")
    assert dfB_out["ob_bearish"].iloc[5] == True, "Se esperaba un bearish OB en la vela 5"
    obB = [z for z in summaryB["all_order_blocks"] if z["candle_time"] == 5][0]
    print(f"  OK — OB bearish en vela 5: top={obB['top']}, bottom={obB['bottom']}, confirmado en vela {obB['confirmed_at']}")

    print("\n" + "=" * 70)
    print("CASO 3 — No se crea OB solo por el color de la vela")
    print("=" * 70)
    assert df_out["ob_bullish"].iloc[1] == False, "La vela bajista chica sin desplazamiento NO debe marcarse como OB"
    assert df_out["ob_bullish"].iloc[2] == False
    print("  OK — la vela bajista en la posición 1 (sin desplazamiento después) no genera ningún OB.")

    print("\n" + "=" * 70)
    print("CASO 4 — El OB requiere confirmación (desplazamiento)")
    print("=" * 70)
    assert ob1["confirmed_at"] > ob1["candle_time"], "confirmed_at debe ser posterior a candle_time"
    assert ob1["created_at"] == ob1["candle_time"] + 1
    print(f"  OK — candle_time={ob1['candle_time']}, created_at={ob1['created_at']}, confirmed_at={ob1['confirmed_at']} (confirmed_at siempre >= created_at > candle_time).")

    print("\n" + "=" * 70)
    print("CASO 5 — No existe look-ahead bias")
    print("=" * 70)
    _, summary_truncado = analyze_order_blocks(dfA.iloc[: ob1["confirmed_at"] + 1], displacement_atr_mult=1.2, mitigation_criterion="50%")
    ob1_truncado = [z for z in summary_truncado["all_order_blocks"] if z["candle_time"] == 5][0]
    assert ob1_truncado["top"] == ob1["top"] and ob1_truncado["bottom"] == ob1["bottom"]
    assert ob1_truncado["confirmed_at"] == ob1["confirmed_at"]
    assert ob1_truncado["status"] == "active" and ob1_truncado["max_fill_pct"] == 0.0
    print("  OK — truncando el dataset justo en confirmed_at, el OB queda idéntico y 'active' (no usa velas futuras).")

    print("\n" + "=" * 70)
    print("CASO 6 — OB fresh")
    print("=" * 70)
    _, s_fresh = analyze_order_blocks(dfA.iloc[: ob1["confirmed_at"] + 2], displacement_atr_mult=1.2, mitigation_criterion="50%")
    ob1_fresh = [z for z in s_fresh["all_order_blocks"] if z["candle_time"] == 5][0]
    assert ob1_fresh["freshness"] == "fresh"
    print(f"  OK — justo después de confirmarse (antes de ser tocado), freshness='{ob1_fresh['freshness']}'.")

    print("\n" + "=" * 70)
    print("CASO 7 — OB touched")
    print("=" * 70)
    ob1_final = [z for z in summary["all_order_blocks"] if z["candle_time"] == 5][0]
    assert ob1_final["touched_at"] is not None
    print(f"  OK — el OB fue tocado por primera vez en la vela {ob1_final['touched_at']}.")

    print("\n" + "=" * 70)
    print("CASO 8 — OB parcialmente mitigado")
    print("=" * 70)
    _, s_parcial = analyze_order_blocks(dfA.iloc[:10], displacement_atr_mult=1.2, mitigation_criterion="full")
    ob1_parcial = [z for z in s_parcial["all_order_blocks"] if z["candle_time"] == 5][0]
    assert ob1_parcial["status"] in ("touched", "partially_mitigated"), f"Se esperaba touched/partially_mitigated, se obtuvo {ob1_parcial['status']}"
    print(f"  OK — con criterio 'full' y datos hasta la vela 9, el OB está en estado '{ob1_parcial['status']}' (fill={ob1_parcial['max_fill_pct']}%).")

    print("\n" + "=" * 70)
    print("CASO 9 — OB completamente mitigado")
    print("=" * 70)
    assert ob1_final["status"] == "fully_mitigated", f"Se esperaba fully_mitigated, se obtuvo {ob1_final['status']}"
    print(f"  OK — con criterio '50%' y todos los datos, el OB queda 'fully_mitigated' en la vela {ob1_final['mitigated_at']}.")

    print("\n" + "=" * 70)
    print("CASO 10 — OB invalidado")
    print("=" * 70)
    assert obB["status"] == "invalidated", f"Se esperaba invalidated, se obtuvo {obB['status']}"
    print(f"  OK — el OB bajista queda 'invalidated' en la vela {obB['invalidated_at']} (close cerró por encima del top).")

    print("\n" + "=" * 70)
    print("CASO 11 — Dataset pequeño no rompe")
    print("=" * 70)
    _, s_pocas = analyze_order_blocks(dfA.head(3))
    print(f"  OK — con 3 velas no truena. OBs detectados: {len(s_pocas['all_order_blocks'])}")

    print("\n" + "=" * 70)
    print("CASO 12 — Dataset sin volume no rompe")
    print("=" * 70)
    assert "volume" not in dfA.columns
    _, s_sin_vol = analyze_order_blocks(dfA, displacement_atr_mult=1.2)
    print(f"  OK — sin columna 'volume' funciona igual (volume_confirmation queda en None). OBs: {len(s_sin_vol['all_order_blocks'])}")

    print("\n" + "=" * 70)
    print("ROBUSTEZ EXTRA — vacío, columnas faltantes, NaN")
    print("=" * 70)
    try:
        analyze_order_blocks(pd.DataFrame({"open": [], "high": [], "low": [], "close": []}))
        print("  ERROR: debía lanzar ValueError con DataFrame vacío")
    except ValueError:
        print("  OK — DataFrame vacío -> ValueError controlado.")
    try:
        analyze_order_blocks(pd.DataFrame({"open": [1, 2], "high": [1, 2]}))
        print("  ERROR: debía lanzar ValueError con columnas faltantes")
    except ValueError:
        print("  OK — columnas OHLC faltantes -> ValueError controlado.")
    df_nan = dfA.copy()
    df_nan.iloc[2, df_nan.columns.get_loc("high")] = np.nan
    _, s_nan = analyze_order_blocks(df_nan, displacement_atr_mult=1.2)
    print(f"  OK — con un NaN en 'high' no truena. OBs: {len(s_nan['all_order_blocks'])}")

    print("\n" + "=" * 70)
    print("EXTRA — integración opcional con market_structure.py (require_bos)")
    print("=" * 70)
    from core.market_structure import analyze_market_structure
    ms_df, _ = analyze_market_structure(dfA, swing_order=2)
    _, s_bos = analyze_order_blocks(dfA, market_structure_df=ms_df, require_bos=True, displacement_atr_mult=1.2)
    print(f"  OK — corrió con require_bos=True y market_structure_df sin tronar. OBs confirmados: {len(s_bos['all_order_blocks'])}")
    _, s_bos_sin_ms = analyze_order_blocks(dfA, market_structure_df=None, require_bos=True, displacement_atr_mult=1.2)
    print(f"  OK — con require_bos=True pero SIN market_structure_df, no truena (cae a solo desplazamiento). OBs: {len(s_bos_sin_ms['all_order_blocks'])}")

    print("\nTodas las pruebas de core/order_blocks.py corrieron sin errores.")


if __name__ == "__main__":
    _demo()
