"""
core/setup_engine.py
─────────────────────────────────────────────────────────────────────
Setup Engine — identifica y clasifica CONFIGURACIONES potenciales
combinando los módulos existentes de MIMI-AI:

  - core.market_structure  → BOS / CHoCH y swings confirmados
  - core.liquidity         → sweeps de liquidez
  - core.fvg               → FVG activos
  - core.order_blocks      → Order Blocks válidos
  - core.multi_timeframe   → contexto HTF/MTF (ya sin velas sin cerrar)

NO ejecuta operaciones, NO cambia BUY/SELL/WAIT, NO toca el scoring y NO
está conectado al sistema real de decisiones. Los precios de entrada /
stop / objetivo son REFERENCIAS ANALÍTICAS, no órdenes. Tampoco inventa
probabilidades de éxito: un setup `confirmed` solo significa que cumplió
las reglas de abajo, no que sea rentable (eso exigiría validación
estadística que este módulo no hace).

─────────────────────────────────────────────────────────────────────
REGLAS DE UN SETUP (bullish; el bearish es el espejo exacto)
─────────────────────────────────────────────────────────────────────
Ancla — un setup NACE con un sweep alcista de liquidez (el precio perfora
un mínimo relevante y CIERRA de regreso arriba). `created_at` = momento en
que cierra la vela del sweep. `invalidation_price` = el mínimo de esa vela.

Condiciones (códigos que verás en conditions_met / conditions_missing):
  context          HTF/MTF no contradictorios (ninguno bajista) y CON datos.
                   Datos ausentes -> NO cuenta como cumplida (se lista como
                   faltante). Opcional: exigir contexto alcista de verdad.
  liquidity_sweep  el ancla (siempre cumplida al nacer).
  structure_shift  BOS o CHoCH alcista en las `structure_window` velas
                   posteriores al sweep (nunca en la misma vela del sweep).
  poi_zone         existe un FVG alcista activo u Order Block alcista
                   válido, ya conocido en la vela anterior.
  price_near_zone  el cierre está dentro de la zona o a <= zone_proximity_pct.
  zone_reaction    EVENTO de confirmación: una vela posterior al candidato
                   toca la zona y cierra de regreso por encima de su borde
                   (o de su punto medio), con vela alcista.

─────────────────────────────────────────────────────────────────────
ESTADOS Y TRANSICIONES (nunca retroceden; invalidated/expired son finales)
─────────────────────────────────────────────────────────────────────
  forming      nació el sweep pero aún no se cumplen las condiciones
               mínimas de `candidate_requires`.
  candidate    se cumplen las condiciones mínimas (por defecto: context +
               liquidity_sweep + structure_shift). Es SETUP PARCIAL: todavía
               falta la reacción en zona.
  confirmed    ocurrió el evento `zone_reaction` estando ya en candidate.
               NO basta con acumular condiciones: la confirmación es un
               evento de precio concreto, en una vela posterior al candidato.
  invalidated  (antes de cualquier otro paso en cada vela)
               · un cierre perfora `invalidation_price`, o
               · el contexto HTF/MTF pasa a contradictorio (solo mientras es
                 forming/candidate), o
               · nace con contexto contradictorio (queda registrado como
                 descartado, con sus contradicciones).
  expired      · forming sin structure_shift tras `structure_window` velas;
               · forming/candidate sin confirmar tras `max_age_bars` velas;
               · confirmed tras `confirmed_ttl_bars` velas de vigencia.

Clasificación: completo = confirmed · parcial = forming/candidate ·
descartado = invalidated/expired.

─────────────────────────────────────────────────────────────────────
LOOK-AHEAD BIAS
─────────────────────────────────────────────────────────────────────
El motor recorre las velas de izquierda a derecha y en la vela `t` solo
usa información ya disponible en `t`:
  - sweeps: se conocen al cerrar su vela (el motor de liquidez solo usa
    niveles confirmados ANTES de esa vela);
  - BOS/CHoCH: eventos de su vela, calculados con swings ya confirmados;
  - zonas FVG/OB: se usan desde su `created_at` / `confirmed_at` y solo si
    seguían activas en la vela ANTERIOR (nunca se usa la vela actual para
    decidir que una zona existía);
  - contexto HTF/MTF: viene de `multi_timeframe`, que solo expone velas
    mayores ya cerradas (`*_bar_time` + duración <= `timestamp`);
  - objetivos: solo swings ya confirmados (posición + swing_order <= t).
Cada setup guarda un `history` de snapshots con su timestamp; `state_as_of`
reconstruye su estado en cualquier momento pasado sin reescribirlo.
`timestamp` = momento (cierre de vela) del último cambio de estado;
`created_at` / `confirmed_at` = cierre de la vela del sweep / de la reacción.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.fvg import analyze_fvg
from core.liquidity import analyze_liquidity
from core.market_structure import _validar_columnas, analyze_market_structure
from core.multi_timeframe import _asegurar_datetimeindex, _norm_ts, analyze_multi_timeframe
from core.order_blocks import analyze_order_blocks

DIRECCIONES = ("bullish", "bearish")
ESTADOS = ("forming", "candidate", "confirmed", "invalidated", "expired")
ACTIVOS = ("forming", "candidate", "confirmed")
CONDICIONES = ("context", "liquidity_sweep", "structure_shift", "poi_zone", "price_near_zone", "zone_reaction")
_MODULO_DE = {
    "context": "multi_timeframe", "liquidity_sweep": "liquidity", "structure_shift": "market_structure",
    "price_near_zone": "price_action", "zone_reaction": "price_action",
}


# ─────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SetupConfig:
    # motores
    swing_order: int = 5
    equal_tolerance_pct: float = 0.05
    mitigation_criterion: str = "50%"
    displacement_atr_mult: float = 1.5
    # timeframes del contexto
    htf_freq: str = "4h"
    mtf_freq: str = "1h"
    ltf_freq: str = "15min"
    context_timeframes: Tuple[str, ...] = ("htf", "mtf")
    # Velas mínimas ya cerradas que debe tener un timeframe de contexto para que su lectura cuente.
    # None = automático: 2*swing_order+3 si el contexto se calcula aquí; sin chequeo si lo inyectas tú.
    min_context_bars: Optional[int] = None
    require_supportive_context: bool = False   # True: exige contexto alcista/bajista real, no solo "no contradictorio"
    # reglas del setup
    min_sweep_touches: int = 1                 # 2 = solo sweeps de EQH/EQL (swings iguales)
    structure_events: Tuple[str, ...] = ("bos", "choch")
    candidate_requires: Tuple[str, ...] = ("context", "liquidity_sweep", "structure_shift")
    zone_types: Tuple[str, ...] = ("fvg", "order_block")
    poi_lookback_bars: int = 5                 # zonas nacidas hasta N velas antes del sweep también cuentan
    zone_proximity_pct: float = 0.1
    confirm_close_level: str = "edge"          # "edge" (borde de la zona) | "mid" (punto medio)
    # vigencia
    structure_window: int = 12
    max_age_bars: int = 60
    confirmed_ttl_bars: int = 30
    # varios
    include_discarded: bool = True             # registrar también los nacidos con contexto contradictorio
    min_bars: Optional[int] = None


def _resolver_config(config: Any) -> SetupConfig:
    if config is None:
        return SetupConfig()
    if isinstance(config, SetupConfig):
        return config
    if isinstance(config, dict):
        return replace(SetupConfig(), **config)
    raise ValueError("config debe ser None, un SetupConfig o un dict")


# ─────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────
def _limpio(v: Any) -> Optional[Any]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _resumen_vacio(cfg: SetupConfig, symbol: str, motivo: str, errores: Optional[List[str]] = None) -> Dict:
    return {
        "symbol": symbol, "total_setups": 0, "by_status": {e: 0 for e in ESTADOS},
        "complete": 0, "partial": 0, "discarded": 0,
        "by_direction": {d: 0 for d in DIRECCIONES}, "active_setup_ids": [],
        "last_timestamp": None, "insufficient_data": True, "reason": motivo,
        "modules_used": [], "modules_unavailable": [], "context_available": False,
        "errors": errores or [], "config": cfg.__dict__.copy(),
    }


def state_as_of(setup: Dict, timestamp: pd.Timestamp) -> Optional[Dict]:
    """Estado del setup tal como era conocido en `timestamp` (o None si todavía
    no existía). Se reconstruye desde `history`; nunca usa snapshots posteriores."""
    previo = None
    for snap in setup["history"]:
        if snap["timestamp"] <= timestamp:
            previo = snap
        else:
            break
    return previo


def setups_to_dataframe(setups: List[Dict]) -> pd.DataFrame:
    cols = ["setup_id", "symbol", "direction", "setup_type", "status", "timestamp", "created_at", "confirmed_at",
            "invalidation_price", "entry_zone_low", "entry_zone_high", "stop_reference", "target_reference",
            "conditions_met", "conditions_missing", "contradictions", "invalidation_reason", "expiry_reason", "status_reason"]
    if not setups:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{c: s.get(c) for c in cols} for s in setups])


# ─────────────────────────────────────────────────────────────────────
#  Estado interno de un setup durante el recorrido
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _Setup:
    direction: str
    sweep: dict
    idx_sweep: int
    invalidation_price: float
    created_at: pd.Timestamp
    setup_id: str
    status: str = "forming"
    shift_idx: Optional[int] = None
    shift_type: Optional[str] = None
    candidate_idx: Optional[int] = None
    confirmed_idx: Optional[int] = None
    confirmed_at: Optional[pd.Timestamp] = None
    zone: Optional[dict] = None
    target_frozen: Optional[float] = None
    target_is_frozen: bool = False
    invalidation_reason: Optional[str] = None
    expiry_reason: Optional[str] = None
    terminal_idx: Optional[int] = None
    history: List[dict] = field(default_factory=list)
    contradictions_seen: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def detect_setups(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    config: Any = None,
    context: Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Detecta setups alcistas y bajistas (sweep → cambio de estructura → zona →
    reacción) sin look-ahead bias. Solo informativo.

    Parameters
    ----------
    df : OHLC(V) del timeframe más fino (LTF), con índice de fechas.
    symbol : etiqueta del instrumento (XAUUSD, EURUSD...).
    config : SetupConfig o dict con cualquier subconjunto de sus campos.
    context : opcional — salida de `analyze_multi_timeframe` para este mismo
        `df` (mismo número de filas). Si no se da, se calcula internamente.

    Returns
    -------
    (setups, summary) — `setups` es una lista de dicts (uno por sweep),
    ordenada por `created_at`; ver el docstring del módulo para los campos.
    Nunca lanza por datos insuficientes o faltantes: devuelve ([], resumen
    con insufficient_data=True y el motivo).
    """
    cfg = _resolver_config(config)
    errores: List[str] = []

    if df is None or len(df) == 0:
        return [], _resumen_vacio(cfg, symbol, "DataFrame vacío")
    try:
        _validar_columnas(df)
        base = _asegurar_datetimeindex(df)
    except ValueError as e:
        return [], _resumen_vacio(cfg, symbol, f"datos inválidos: {e}")
    n = len(base)
    min_bars = cfg.min_bars if cfg.min_bars is not None else 3 * cfg.swing_order + 5
    if n < min_bars:
        return [], _resumen_vacio(cfg, symbol, f"pocas velas ({n} < {min_bars})")

    o = base["open"].to_numpy(dtype=float)
    h = base["high"].to_numpy(dtype=float)
    l = base["low"].to_numpy(dtype=float)
    c = base["close"].to_numpy(dtype=float)
    ts_arr = list(_norm_ts(base.index) + pd.Timedelta(cfg.ltf_freq))  # momento en que cierra cada vela

    # ── motores sobre el LTF (cada uno aislado: si falla, se registra) ──
    ms_df = liq_df = fvg_df = None
    liq_sum = fvg_sum = ob_sum = None
    usados, no_disponibles = [], []
    try:
        ms_df, _ = analyze_market_structure(base, swing_order=cfg.swing_order, break_using="close")
        usados.append("market_structure")
    except Exception as e:
        errores.append(f"market_structure: {e}"); no_disponibles.append("market_structure")
    try:
        liq_df, liq_sum = analyze_liquidity(base, market_structure_df=ms_df, swing_order=cfg.swing_order,
                                            equal_tolerance_pct=cfg.equal_tolerance_pct)
        usados.append("liquidity")
    except Exception as e:
        errores.append(f"liquidity: {e}"); no_disponibles.append("liquidity")
    try:
        fvg_df, fvg_sum = analyze_fvg(base, mitigation_criterion=cfg.mitigation_criterion)
        usados.append("fvg")
    except Exception as e:
        errores.append(f"fvg: {e}"); no_disponibles.append("fvg")
    try:
        _, ob_sum = analyze_order_blocks(base, market_structure_df=ms_df, liquidity_df=liq_df, fvg_df=fvg_df,
                                         displacement_atr_mult=cfg.displacement_atr_mult,
                                         mitigation_criterion=cfg.mitigation_criterion)
        usados.append("order_blocks")
    except Exception as e:
        errores.append(f"order_blocks: {e}"); no_disponibles.append("order_blocks")

    # ── contexto HTF/MTF ────────────────────────────────────────────────
    ctx = None
    if context is not None:
        if len(context) != n:
            errores.append(f"context: tiene {len(context)} filas y df tiene {n}; se ignora el contexto")
        else:
            ctx = context.reset_index(drop=True)
    else:
        try:
            ctx_df, _ = analyze_multi_timeframe(
                df=base, htf_freq=cfg.htf_freq, mtf_freq=cfg.mtf_freq, ltf_freq=cfg.ltf_freq,
                swing_order=cfg.swing_order, equal_tolerance_pct=cfg.equal_tolerance_pct,
                mitigation_criterion=cfg.mitigation_criterion, displacement_atr_mult=cfg.displacement_atr_mult)
            if len(ctx_df) == n:
                ctx = ctx_df.reset_index(drop=True)
        except Exception as e:
            errores.append(f"multi_timeframe: {e}")
    if ctx is not None:
        usados.append("multi_timeframe")
    else:
        no_disponibles.append("multi_timeframe")

    def _col(nombre: str) -> np.ndarray:
        if ctx is not None and nombre in ctx.columns:
            return np.array([_limpio(v) for v in ctx[nombre].to_numpy(dtype=object)], dtype=object)
        return np.full(n, None, dtype=object)

    trend_ctx = {tf: _col(f"{tf}_trend") for tf in ("htf", "mtf", "ltf")}
    barra_ctx = {tf: _col(f"{tf}_bar_time") for tf in ("htf", "mtf")}

    # Un timeframe con muy pocas velas cerradas reporta "ranging" por defecto (no hay swings que medir):
    # eso NO es información. Hasta tener `min_ctx` velas cerradas, su lectura se trata como desconocida.
    min_ctx = cfg.min_context_bars if cfg.min_context_bars is not None else (2 * cfg.swing_order + 3 if context is None else 0)
    if min_ctx > 0:
        for tf in ("htf", "mtf"):
            vistas, depurado = set(), trend_ctx[tf].copy()
            for i in range(n):
                if barra_ctx[tf][i] is not None:
                    vistas.add(pd.Timestamp(barra_ctx[tf][i]))
                if len(vistas) < min_ctx:
                    depurado[i] = None
            trend_ctx[tf] = depurado

    # ── datos de los motores, ya en forma de arreglos/listas causales ──
    falsos = np.zeros(n, dtype=bool)

    def _evt(col: str) -> np.ndarray:
        return ms_df[col].to_numpy(dtype=bool) if ms_df is not None and col in ms_df.columns else falsos

    shift_evt = {
        "bullish": {"bos": _evt("bos_bullish"), "choch": _evt("choch_bullish")},
        "bearish": {"bos": _evt("bos_bearish"), "choch": _evt("choch_bearish")},
    }
    contra_evt = {"bullish": _evt("bos_bearish") | _evt("choch_bearish"),
                  "bearish": _evt("bos_bullish") | _evt("choch_bullish")}

    swing_hi, swing_lo = [], []   # (confirmed_at, precio)
    if ms_df is not None:
        for pos in np.flatnonzero(ms_df["swing_high"].to_numpy(dtype=bool)):
            swing_hi.append((int(pos) + cfg.swing_order, float(h[pos])))
        for pos in np.flatnonzero(ms_df["swing_low"].to_numpy(dtype=bool)):
            swing_lo.append((int(pos) + cfg.swing_order, float(l[pos])))

    zonas: List[dict] = []
    if fvg_sum is not None and "fvg" in cfg.zone_types:
        for z in fvg_sum["all_fvg"]:
            fin = min([v for v in (z.get("mitigated_at"), z.get("invalidated_at")) if v is not None] + [n + 1])
            zonas.append({"source": "fvg", "kind": z["kind"], "avail": z["created_at"], "end": fin,
                          "top": z["top"], "bottom": z["bottom"]})
    if ob_sum is not None and "order_block" in cfg.zone_types:
        for z in ob_sum["all_order_blocks"]:
            fin = min([v for v in (z.get("mitigated_at"), z.get("invalidated_at")) if v is not None] + [n + 1])
            zonas.append({"source": "order_block", "kind": z["kind"], "avail": z["confirmed_at"], "end": fin,
                          "top": z["top"], "bottom": z["bottom"]})

    sweeps_por_barra: Dict[int, List[Tuple[str, dict]]] = {}
    if liq_sum is not None:
        for e in liq_sum["bullish_sweeps"]:
            sweeps_por_barra.setdefault(int(e["index"]), []).append(("bullish", e))
        for e in liq_sum["bearish_sweeps"]:
            sweeps_por_barra.setdefault(int(e["index"]), []).append(("bearish", e))

    # ── funciones auxiliares del recorrido (todas causales en `t`) ─────
    def _contexto(direccion: str, t: int) -> Tuple[str, List[str]]:
        contrario = "bearish" if direccion == "bullish" else "bullish"
        contradicciones, desconocido, no_apoya = [], False, False
        for tf in cfg.context_timeframes:
            v = trend_ctx[tf][t]
            if v is None:
                desconocido = True
            elif v == contrario:
                contradicciones.append(f"{tf.upper()} {v} contradice el setup {'alcista' if direccion == 'bullish' else 'bajista'}")
            elif cfg.require_supportive_context and v != direccion:
                no_apoya = True
        if contradicciones:
            return "contradictory", contradicciones
        if desconocido:
            return "unknown", []
        if no_apoya:
            return "unsupportive", []
        return "ok", []

    def _zonas_elegibles(s: _Setup, t: int) -> List[dict]:
        """Zonas conocidas y todavía activas en la vela ANTERIOR (t-1)."""
        b = t - 1
        return [z for z in zonas if z["kind"] == s.direction and s.idx_sweep - cfg.poi_lookback_bars <= z["avail"] <= b < z["end"]]

    def _distancia(z: dict, precio: float) -> float:
        if z["bottom"] <= precio <= z["top"]:
            return 0.0
        return min(abs(precio - z["top"]), abs(precio - z["bottom"]))

    def _objetivo(s: _Setup, t: int, ref: float) -> Optional[float]:
        if s.direction == "bullish":
            cand = [p for (conf, p) in swing_hi if conf <= t and p > ref]
            return min(cand) if cand else None
        cand = [p for (conf, p) in swing_lo if conf <= t and p < ref]
        return max(cand) if cand else None

    def _objetivo_ahora(s: _Setup, t: int) -> Optional[float]:
        ref = c[t]
        if s.zone is not None:  # el objetivo debe quedar más allá tanto de la zona como del precio actual
            ref = max(s.zone["top"], c[t]) if s.direction == "bullish" else min(s.zone["bottom"], c[t])
        return _objetivo(s, t, ref)

    def _snapshot(s: _Setup, t: int, met: List[str], contradicciones: List[str], razon: str) -> dict:
        faltan = [x for x in CONDICIONES if x not in met]
        z = s.zone
        evidencia = ["liquidity"]
        for cond in met:
            m = _MODULO_DE.get(cond)
            if m and m not in evidencia:
                evidencia.append(m)
        if "poi_zone" in met and z is not None:
            evidencia.append(z["source"])
        return {
            "timestamp": ts_arr[t], "status": s.status, "conditions_met": list(met), "conditions_missing": faltan,
            "entry_zone_low": z["bottom"] if z else None, "entry_zone_high": z["top"] if z else None,
            "entry_zone_source": z["source"] if z else None,
            "target_reference": s.target_frozen if s.target_is_frozen else _objetivo_ahora(s, t),
            "contradictions": list(contradicciones), "evidence_modules": evidencia,
            "invalidation_reason": s.invalidation_reason, "expiry_reason": s.expiry_reason, "status_reason": razon,
            "context_htf_bar": barra_ctx["htf"][t], "context_mtf_bar": barra_ctx["mtf"][t],
        }

    def _registrar(s: _Setup, snap: dict) -> None:
        if not s.history:
            s.history.append(snap)
            return
        previo = {k: v for k, v in s.history[-1].items() if k != "timestamp"}
        nuevo = {k: v for k, v in snap.items() if k != "timestamp"}
        if previo != nuevo:
            s.history.append(snap)

    def _razon(s: _Setup, met: List[str]) -> str:
        dir_es = "alcista" if s.direction == "bullish" else "bajista"
        faltan = [x for x in CONDICIONES if x not in met]
        if s.status == "forming":
            return f"Sweep {dir_es} detectado; aún faltan condiciones mínimas ({', '.join(x for x in cfg.candidate_requires if x not in met) or 'ninguna'})."
        if s.status == "candidate":
            return f"Cumple las condiciones mínimas ({', '.join(cfg.candidate_requires)}); falta la reacción en zona para confirmar."
        if s.status == "confirmed":
            return f"Reacción confirmada en zona ({s.zone['source'] if s.zone else 'n/d'}) tras el cambio de estructura."
        if s.status == "invalidated":
            return s.invalidation_reason or "Invalidado."
        return s.expiry_reason or "Expirado."

    # ── recorrido causal vela por vela ──────────────────────────────────
    setups: List[_Setup] = []
    for t in range(n):
        # 1) nacimientos: sweeps cuya vela cierra en t
        for direccion, ev in sweeps_por_barra.get(t, []):
            if ev.get("touches", 1) < cfg.min_sweep_touches:
                continue
            inv = float(l[t]) if direccion == "bullish" else float(h[t])
            sid = f"{symbol}-{direccion}-{ts_arr[t].strftime('%Y%m%dT%H%M%S')}"
            s = _Setup(direction=direccion, sweep=ev, idx_sweep=t, invalidation_price=inv, created_at=ts_arr[t], setup_id=sid)
            estado_ctx, contra = _contexto(direccion, t)
            met = ["liquidity_sweep"]
            if estado_ctx == "ok":
                met.append("context")
            if estado_ctx == "contradictory":
                if not cfg.include_discarded:
                    continue
                s.status, s.terminal_idx = "invalidated", t
                s.invalidation_reason = "contexto HTF/MTF contradictorio al nacer el sweep: " + "; ".join(contra)
                s.contradictions_seen = list(contra)
            snap = _snapshot(s, t, met, contra, _razon(s, met))
            _registrar(s, snap)
            setups.append(s)

        # 2) evolución de los setups activos (solo con información de la vela t)
        for s in setups:
            if s.status not in ACTIVOS or s.idx_sweep >= t:
                continue
            dir_es = "alcista" if s.direction == "bullish" else "bajista"
            estado_ctx, contra = _contexto(s.direction, t)
            for x in contra:
                if x not in s.contradictions_seen:
                    s.contradictions_seen.append(x)
            if contra_evt[s.direction][t]:
                txt = f"BOS/CHoCH {'bajista' if s.direction == 'bullish' else 'alcista'} en LTF contra el setup {dir_es}"
                if txt not in s.contradictions_seen:
                    s.contradictions_seen.append(txt)

            # a) invalidación (tiene prioridad sobre todo lo demás en la vela)
            rompe = (c[t] < s.invalidation_price) if s.direction == "bullish" else (c[t] > s.invalidation_price)
            if rompe:
                s.status, s.terminal_idx = "invalidated", t
                s.invalidation_reason = (f"cierre {c[t]:.5f} {'por debajo' if s.direction == 'bullish' else 'por encima'} "
                                         f"del precio de invalidación {s.invalidation_price:.5f}")
            elif estado_ctx == "contradictory" and s.status in ("forming", "candidate"):
                s.status, s.terminal_idx = "invalidated", t
                s.invalidation_reason = "el contexto HTF/MTF pasó a contradictorio: " + "; ".join(contra)

            met_actual = ["liquidity_sweep"]
            if s.status in ACTIVOS:
                # b) progreso: structure_shift, candidate, confirmed
                if s.shift_idx is None and t - s.idx_sweep <= cfg.structure_window:
                    for tipo in cfg.structure_events:
                        if shift_evt[s.direction][tipo][t]:
                            s.shift_idx, s.shift_type = t, tipo
                            break
                if estado_ctx == "ok":
                    met_actual.append("context")
                if s.shift_idx is not None:
                    met_actual.append("structure_shift")
                elegibles = _zonas_elegibles(s, t)
                if elegibles:
                    met_actual.append("poi_zone")
                    mas_cerca = min(elegibles, key=lambda z: _distancia(z, c[t]))
                    if s.confirmed_idx is None:
                        s.zone = mas_cerca
                    prox = abs(c[t]) * cfg.zone_proximity_pct / 100.0
                    if _distancia(mas_cerca, c[t]) <= prox:
                        met_actual.append("price_near_zone")

                estaba_candidate = s.status == "candidate"
                if s.status == "forming" and all(x in met_actual for x in cfg.candidate_requires):
                    s.status, s.candidate_idx = "candidate", t

                # confirmación: evento de reacción, estrictamente POSTERIOR al candidato
                if estaba_candidate and s.candidate_idx is not None and t > s.candidate_idx:
                    for z in elegibles:
                        nivel = (z["top"] if s.direction == "bullish" else z["bottom"]) if cfg.confirm_close_level == "edge" \
                            else (z["top"] + z["bottom"]) / 2.0
                        if s.direction == "bullish":
                            reaccion = l[t] <= z["top"] and c[t] > nivel and c[t] > o[t]
                        else:
                            reaccion = h[t] >= z["bottom"] and c[t] < nivel and c[t] < o[t]
                        if reaccion:
                            s.status, s.confirmed_idx, s.confirmed_at, s.zone = "confirmed", t, ts_arr[t], z
                            s.target_frozen, s.target_is_frozen = _objetivo_ahora(s, t), True   # referencias fijas desde la confirmación
                            break
                if s.confirmed_idx is not None and "zone_reaction" not in met_actual:
                    met_actual.append("zone_reaction")

                # c) vigencia
                if s.status == "forming" and s.shift_idx is None and t - s.idx_sweep >= cfg.structure_window:
                    s.status, s.terminal_idx = "expired", t
                    s.expiry_reason = f"sin cambio de estructura (BOS/CHoCH) en {cfg.structure_window} velas tras el sweep"
                elif s.status in ("forming", "candidate") and t - s.idx_sweep >= cfg.max_age_bars:
                    s.status, s.terminal_idx = "expired", t
                    s.expiry_reason = f"sin confirmación en {cfg.max_age_bars} velas desde el sweep"
                elif s.status == "confirmed" and s.confirmed_idx is not None and t - s.confirmed_idx >= cfg.confirmed_ttl_bars:
                    s.status, s.terminal_idx = "expired", t
                    s.expiry_reason = f"vigencia posterior a la confirmación agotada ({cfg.confirmed_ttl_bars} velas)"
            else:
                met_actual = [x for x in ("liquidity_sweep", "context", "structure_shift", "poi_zone", "price_near_zone", "zone_reaction")
                              if x in s.history[-1]["conditions_met"]]

            snap = _snapshot(s, t, met_actual, s.contradictions_seen, _razon(s, met_actual))
            _registrar(s, snap)

    # ── armar la salida ─────────────────────────────────────────────────
    salida: List[Dict] = []
    for s in setups:
        ultimo = s.history[-1]
        z = ultimo
        salida.append({
            "setup_id": s.setup_id, "symbol": symbol, "direction": s.direction, "setup_type": "liquidity_sweep_structure_poi",
            "status": s.status, "timestamp": ultimo["timestamp"], "created_at": s.created_at, "confirmed_at": s.confirmed_at,
            "invalidation_price": s.invalidation_price, "entry_zone_low": z["entry_zone_low"], "entry_zone_high": z["entry_zone_high"],
            "entry_zone_source": z["entry_zone_source"], "stop_reference": s.invalidation_price,
            "target_reference": z["target_reference"],
            "conditions_met": z["conditions_met"], "conditions_missing": z["conditions_missing"],
            "contradictions": z["contradictions"], "evidence_modules": z["evidence_modules"],
            "invalidation_reason": s.invalidation_reason, "expiry_reason": s.expiry_reason, "status_reason": z["status_reason"],
            "sweep": {"index": s.idx_sweep, "level": s.sweep.get("level"), "zone_kind": s.sweep.get("zone_kind"), "touches": s.sweep.get("touches")},
            "structure_shift": {"index": s.shift_idx, "type": s.shift_type},
            "history": s.history,
        })

    por_estado = {e: sum(1 for x in salida if x["status"] == e) for e in ESTADOS}
    resumen = {
        "symbol": symbol, "total_setups": len(salida), "by_status": por_estado,
        "complete": por_estado["confirmed"], "partial": por_estado["forming"] + por_estado["candidate"],
        "discarded": por_estado["invalidated"] + por_estado["expired"],
        "by_direction": {d: sum(1 for x in salida if x["direction"] == d) for d in DIRECCIONES},
        "active_setup_ids": [x["setup_id"] for x in salida if x["status"] in ACTIVOS],
        "last_timestamp": ts_arr[-1], "insufficient_data": False, "reason": None,
        "modules_used": usados, "modules_unavailable": no_disponibles, "context_available": ctx is not None,
        "errors": errores, "config": cfg.__dict__.copy(),
    }
    return salida, resumen


# ─────────────────────────────────────────────────────────────────────
#  Datos simulados + pruebas (10 casos del spec)
# ─────────────────────────────────────────────────────────────────────
def _filas_alcista() -> List[Tuple[float, float, float, float]]:
    """OHLC a mano (open, high, low, close). Guion, con swing_order=2:
      0-8   caída hasta L1 (mínimo 106.00 en la vela 8)
      9-13  rebote a H1; 14-18 caída a L2 (106.05 en la 18) -> EQL con L1
      19-25 rebote a H2 (107.9 en la 23); 26-29 baja hacia el EQL sin romperlo
      30    SWEEP alcista: low 105.55 (< EQL ~106.025) y cierra 106.25 (encima)
      31-33 impulso: cierra sobre H2 en la 32 (BOS alcista) y deja un FVG
            alcista [106.9, 107.5] confirmado en la 33
      34-35 retroceso; 36 REACCIÓN: toca la zona (low 107.45) y cierra arriba
      37-45 continuación"""
    f: List[Tuple[float, float, float, float]] = []
    b = lambda o_, h_, l_, c_: f.append((o_, h_, l_, c_))
    b(109.5, 109.8, 109.3, 109.6); b(109.6, 109.9, 109.4, 109.5); b(109.5, 109.7, 109.2, 109.3); b(109.3, 109.5, 108.9, 109.0)
    b(109.0, 109.1, 108.3, 108.5); b(108.5, 108.6, 107.7, 108.0); b(108.0, 108.1, 106.9, 107.0); b(107.0, 107.1, 106.4, 106.6)
    b(106.6, 106.7, 106.00, 106.4)
    b(106.4, 107.0, 106.3, 106.9); b(106.9, 107.5, 106.8, 107.4); b(107.4, 107.9, 107.3, 107.8); b(107.8, 108.0, 107.6, 107.9)
    b(107.9, 108.2, 107.7, 108.0)
    b(108.0, 108.0, 107.4, 107.5); b(107.5, 107.6, 107.0, 107.1); b(107.1, 107.2, 106.6, 106.7); b(106.7, 106.8, 106.2, 106.4)
    b(106.4, 106.5, 106.05, 106.3)
    b(106.3, 106.9, 106.25, 106.8); b(106.8, 107.4, 106.7, 107.3); b(107.3, 107.6, 107.2, 107.5); b(107.5, 107.7, 107.4, 107.6)
    b(107.6, 107.9, 107.5, 107.7)
    b(107.7, 107.8, 107.2, 107.3); b(107.3, 107.4, 106.9, 107.0)
    b(107.0, 107.1, 106.6, 106.7); b(106.7, 106.8, 106.4, 106.5); b(106.5, 106.6, 106.3, 106.4); b(106.4, 106.5, 106.25, 106.3)
    b(106.3, 106.4, 105.55, 106.25)
    b(106.25, 106.9, 106.2, 106.8)
    b(106.8, 108.3, 106.75, 108.2)
    b(108.2, 108.6, 107.5, 108.4)
    b(108.4, 108.5, 107.9, 108.0); b(108.0, 108.1, 107.6, 107.7)
    b(107.55, 107.9, 107.45, 107.8)
    for k in range(9):
        base = 107.8 + 0.2 * (k + 1)
        b(base - 0.1, base + 0.2, base - 0.2, base)
    return f


def _a_df(filas, inicio: str = "2025-06-02 00:00", freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range(inicio, periods=len(filas), freq=freq)
    return pd.DataFrame(list(filas), columns=["open", "high", "low", "close"], index=idx)


def _espejo(filas, k: float = 216.0):
    """Refleja los precios (arriba <-> abajo): el guion alcista pasa a bajista."""
    return [(k - o_, k - l_, k - h_, k - c_) for (o_, h_, l_, c_) in filas]


def _ctx_constante(df: pd.DataFrame, htf: Optional[str], mtf: Optional[str], cambio: Optional[Tuple[int, str]] = None) -> pd.DataFrame:
    """Contexto multi-timeframe controlado a mano. `cambio=(i, valor)` pone el HTF
    en `valor` desde la vela i (para probar que el contexto cambia a mitad de camino)."""
    cierre = _norm_ts(df.index) + pd.Timedelta("15min")
    htf_vals = np.array([htf] * len(df), dtype=object)
    if cambio is not None:
        htf_vals[cambio[0]:] = cambio[1]
    return pd.DataFrame({
        "timestamp": cierre,
        "htf_trend": htf_vals, "mtf_trend": np.array([mtf] * len(df), dtype=object), "ltf_trend": np.array(["ranging"] * len(df), dtype=object),
        "htf_bar_time": cierre.floor("4h") - pd.Timedelta("4h"), "mtf_bar_time": cierre.floor("1h") - pd.Timedelta("1h"),
    }, index=df.index)


def _random_walk(n: int = 700, seed: int = 3) -> pd.DataFrame:
    """Serie M15 ruidosa (sin guion) para probar con contexto multi-timeframe REAL."""
    rng = np.random.default_rng(seed)
    cierre = 100 + np.cumsum(rng.normal(0, 0.35, n)) + 2.0 * np.sin(np.arange(n) / 40.0)
    abre = np.concatenate([[cierre[0]], cierre[:-1]])
    alto = np.maximum(abre, cierre) + np.abs(rng.normal(0, 0.18, n))
    bajo = np.minimum(abre, cierre) - np.abs(rng.normal(0, 0.18, n))
    idx = pd.date_range("2025-06-02 00:00", periods=n, freq="15min")
    return pd.DataFrame({"open": abre, "high": alto, "low": bajo, "close": cierre}, index=idx)


def _escenarios_demo() -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Escenarios sintéticos listos para la interfaz: nombre -> (OHLC, contexto inyectado).
    Cubren los estados principales del motor. Usar con swing_order=2 y equal_tolerance_pct=0.1."""
    A = _filas_alcista()
    dfA = _a_df(A)
    dfB = _a_df(_espejo(A))
    inv = A[:31] + [(106.25, 106.4, 105.9, 106.0), (106.0, 106.1, 105.2, 105.4), (105.4, 105.8, 105.3, 105.7), (105.7, 106.0, 105.6, 105.9)]
    dfI = _a_df(inv)
    lat = A[:31] + [(106.4, 106.7, 106.2, 106.5 + (0.1 if k % 2 else -0.1)) for k in range(16)]
    dfE = _a_df(lat)
    return {
        "Alcista completo (confirmed)": (dfA, _ctx_constante(dfA, "bullish", "bullish")),
        "Bajista completo (confirmed)": (dfB, _ctx_constante(dfB, "bearish", "bearish")),
        "Alcista parcial (candidate)": (dfA.iloc[:35], _ctx_constante(dfA, "bullish", "bullish").iloc[:35]),
        "Alcista en formación (forming)": (dfA.iloc[:32], _ctx_constante(dfA, "bullish", "bullish").iloc[:32]),
        "Contexto contradictorio (descartado)": (dfA, _ctx_constante(dfA, "bearish", "bullish")),
        "Alcista invalidado": (dfI, _ctx_constante(dfI, "bullish", "bullish")),
        "Alcista expirado": (dfE, _ctx_constante(dfE, "bullish", "bullish")),
    }


def _ultimo(setups: List[Dict], direccion: str) -> Dict:
    return [s for s in setups if s["direction"] == direccion][-1]


def _demo() -> None:
    """Corre `python -m core.setup_engine` — ejecuta los 10 casos del spec."""
    import contextlib, io
    CFG = dict(swing_order=2, equal_tolerance_pct=0.1)
    A = _filas_alcista()
    dfA = _a_df(A)
    ctxA = _ctx_constante(dfA, "bullish", "bullish")
    ts = lambda i: _norm_ts(dfA.index)[i] + pd.Timedelta("15min")

    print("=" * 70)
    print("CASO 1 — Bullish setup válido (sweep → BOS → FVG → reacción en zona)")
    print("=" * 70)
    sA, resA = detect_setups(dfA, "XAUUSD", CFG, context=ctxA)
    assert resA["total_setups"] == 1 and sA[0]["direction"] == "bullish", f"se esperaba 1 setup alcista, hubo {[(x['direction'], x['status']) for x in sA]}"
    s1 = sA[0]
    assert s1["status"] == "confirmed", s1["status"]
    assert s1["created_at"] == ts(30) and s1["confirmed_at"] == ts(36)
    for cond in ("context", "liquidity_sweep", "structure_shift", "poi_zone", "zone_reaction"):
        assert cond in s1["conditions_met"], f"falta {cond}: {s1['conditions_met']}"
    assert s1["invalidation_price"] == 105.55 and s1["stop_reference"] == 105.55
    assert abs(s1["entry_zone_low"] - 106.9) < 1e-9 and abs(s1["entry_zone_high"] - 107.5) < 1e-9
    assert s1["target_reference"] is not None and s1["target_reference"] > s1["entry_zone_high"]
    assert s1["structure_shift"]["type"] == "bos" and s1["structure_shift"]["index"] == 32
    print(f"  OK — {s1['setup_id']} confirmed; creado {s1['created_at']}, confirmado {s1['confirmed_at']}")
    print(f"       zona {s1['entry_zone_low']}-{s1['entry_zone_high']} ({s1['entry_zone_source']}), invalidación {s1['invalidation_price']}, objetivo {s1['target_reference']}")
    print(f"       módulos con evidencia: {s1['evidence_modules']}")
    print(f"       por qué: {s1['status_reason']}")

    print("\n" + "=" * 70)
    print("CASO 2 — Bearish setup válido (escenario espejo)")
    print("=" * 70)
    dfB = _a_df(_espejo(A))
    ctxB = _ctx_constante(dfB, "bearish", "bearish")
    sB, resB = detect_setups(dfB, "EURUSD", CFG, context=ctxB)
    assert resB["total_setups"] == 1 and sB[0]["direction"] == "bearish", [(x["direction"], x["status"]) for x in sB]
    s2 = sB[0]
    assert s2["status"] == "confirmed" and s2["confirmed_at"] == ts(36)
    assert abs(s2["invalidation_price"] - (216 - 105.55)) < 1e-9
    assert s2["target_reference"] is not None and s2["target_reference"] < s2["entry_zone_low"]
    print(f"  OK — {s2['setup_id']} confirmed; zona {s2['entry_zone_low']:.2f}-{s2['entry_zone_high']:.2f}, invalidación {s2['invalidation_price']:.2f}")

    print("\n" + "=" * 70)
    print("CASO 3 — Setup parcial (forming y candidate, sin confirmar)")
    print("=" * 70)
    sf, _ = detect_setups(dfA.iloc[:32], "XAUUSD", CFG, context=ctxA.iloc[:32])      # hasta la vela 31
    assert sf[0]["status"] == "forming" and "structure_shift" in sf[0]["conditions_missing"]
    sc, rc = detect_setups(dfA.iloc[:35], "XAUUSD", CFG, context=ctxA.iloc[:35])     # hasta la vela 34
    assert sc[0]["status"] == "candidate" and "zone_reaction" in sc[0]["conditions_missing"]
    assert rc["partial"] == 1 and rc["complete"] == 0
    print(f"  OK — hasta la vela 31: '{sf[0]['status']}' (faltan {sf[0]['conditions_missing']})")
    print(f"       hasta la vela 34: '{sc[0]['status']}' (faltan {sc[0]['conditions_missing']})")

    print("\n" + "=" * 70)
    print("CASO 4 — Contexto contradictorio")
    print("=" * 70)
    ctx_c = _ctx_constante(dfA, "bearish", "bullish")
    s4, r4 = detect_setups(dfA, "XAUUSD", CFG, context=ctx_c)
    assert s4[0]["status"] == "invalidated" and s4[0]["contradictions"]
    assert all("candidate" not in [h["status"] for h in x["history"]] for x in s4)
    print(f"  OK — HTF bajista desde el inicio: invalidado al nacer. {s4[0]['invalidation_reason']}")
    ctx_m = _ctx_constante(dfA, "bullish", "bullish", cambio=(34, "bearish"))
    s4b, _ = detect_setups(dfA.iloc[:40], "XAUUSD", CFG, context=ctx_m.iloc[:40])
    hist = [h["status"] for h in s4b[0]["history"]]
    assert s4b[0]["status"] == "invalidated" and "candidate" in hist and "confirmed" not in hist
    assert "contradictorio" in s4b[0]["invalidation_reason"]
    print(f"  OK — HTF cambia a bajista en la vela 34: historial {hist} -> {s4b[0]['invalidation_reason'][:60]}...")

    print("\n" + "=" * 70)
    print("CASO 5 — Datos insuficientes / ausentes (no inventa condiciones)")
    print("=" * 70)
    s5, r5 = detect_setups(dfA.head(5), "XAUUSD", CFG)
    assert s5 == [] and r5["insufficient_data"]
    print(f"  OK — 5 velas: sin setups, motivo: {r5['reason']}")
    s5, r5 = detect_setups(pd.DataFrame({"open": [], "high": [], "low": [], "close": []}), "XAUUSD", CFG)
    assert s5 == [] and r5["insufficient_data"]
    s5, r5 = detect_setups(pd.DataFrame({"open": [1, 2], "high": [1, 2]}), "XAUUSD", CFG)
    assert s5 == [] and r5["insufficient_data"]
    print("  OK — DataFrame vacío y columnas faltantes: sin excepción, resumen explica el motivo.")
    s5, r5 = detect_setups(dfA, "XAUUSD", CFG, context=_ctx_constante(dfA, None, None))
    assert s5[0]["status"] == "forming" and "context" in s5[0]["conditions_missing"] and "context" not in s5[0]["conditions_met"]
    assert all(h["status"] in ("forming",) for h in s5[0]["history"])
    print(f"  OK — contexto HTF/MTF sin datos: el setup se queda en '{s5[0]['status']}', 'context' figura como faltante (no se da por cumplido).")
    dfN = dfA.copy(); dfN.iloc[3, dfN.columns.get_loc("high")] = np.nan; dfN.iloc[12, dfN.columns.get_loc("low")] = np.nan
    s5, _ = detect_setups(dfN, "XAUUSD", CFG, context=ctxA)
    print(f"  OK — con NaN en el OHLC no truena (setups: {len(s5)}).")
    s5, r5 = detect_setups(dfA, "XAUUSD", CFG)   # contexto REAL calculado aquí: solo 1 vela H4 cerrada al nacer el sweep
    assert r5["context_available"] and s5[0]["status"] == "forming"
    assert "context" in s5[0]["conditions_missing"] and "context" not in s5[0]["conditions_met"]
    assert "candidate" not in [h["status"] for h in s5[0]["history"]]
    print(f"  OK — contexto real con pocas velas HTF (1 vela H4 cerrada): se trata como desconocido -> '{s5[0]['status']}', 'context' faltante.")
    s5b, _ = detect_setups(dfA, "XAUUSD", dict(CFG, min_context_bars=0))
    assert s5b[0]["status"] == "confirmed"
    print("  OK — (contraste) con min_context_bars=0 se aceptaría ese 'ranging' por defecto y confirmaría: por eso el chequeo está activo por defecto.")

    print("\n" + "=" * 70)
    print("CASO 6 — Setup invalidado (cierre bajo el mínimo del sweep)")
    print("=" * 70)
    inv = A[:31] + [(106.25, 106.4, 105.9, 106.0), (106.0, 106.1, 105.2, 105.4), (105.4, 105.8, 105.3, 105.7), (105.7, 106.0, 105.6, 105.9)]
    dfI = _a_df(inv)
    s6, _ = detect_setups(dfI, "XAUUSD", CFG, context=_ctx_constante(dfI, "bullish", "bullish"))
    assert s6[0]["status"] == "invalidated" and "invalidación" in s6[0]["invalidation_reason"]
    assert s6[0]["history"][-1]["timestamp"] == ts(32)
    print(f"  OK — invalidado en la vela 32: {s6[0]['invalidation_reason']}")

    print("\n" + "=" * 70)
    print("CASO 7 — Setup expirado")
    print("=" * 70)
    lat = A[:31]
    for k in range(15):
        lat.append((106.4, 106.7, 106.2, 106.5 + (0.1 if k % 2 else -0.1)))
    dfE = _a_df(lat)
    s7, _ = detect_setups(dfE, "XAUUSD", dict(CFG, structure_window=8), context=_ctx_constante(dfE, "bullish", "bullish"))
    assert s7[0]["status"] == "expired" and "cambio de estructura" in s7[0]["expiry_reason"]
    assert s7[0]["history"][-1]["timestamp"] == ts(38)
    print(f"  OK — expirado en la vela 38: {s7[0]['expiry_reason']}")
    s7b, _ = detect_setups(dfA, "XAUUSD", dict(CFG, confirmed_ttl_bars=3), context=ctxA)
    assert s7b[0]["status"] == "expired" and "confirmación" in s7b[0]["expiry_reason"] and s7b[0]["confirmed_at"] == ts(36)
    print(f"  OK — un setup confirmado también expira por vigencia: {s7b[0]['expiry_reason']}")

    print("\n" + "=" * 70)
    print("CASO 8 — Sin look-ahead (truncado vs. completo, mismo timestamp)")
    print("=" * 70)

    def _comparar(df_full, ctx_full, cfg, cortes, etiqueta):
        full, _ = detect_setups(df_full, "XAUUSD", cfg, context=ctx_full)
        comparados = 0
        for corte in cortes:
            tr, _ = detect_setups(df_full.iloc[:corte], "XAUUSD", cfg, context=None if ctx_full is None else ctx_full.iloc[:corte])
            t_corte = _norm_ts(df_full.index)[corte - 1] + pd.Timedelta("15min")
            ids_full = [s["setup_id"] for s in full if s["created_at"] <= t_corte]
            assert ids_full == [s["setup_id"] for s in tr], f"{etiqueta} corte={corte}: distintos setups {ids_full} vs {[s['setup_id'] for s in tr]}"
            for sf_ in full:
                if sf_["created_at"] > t_corte:
                    continue
                st = state_as_of(sf_, t_corte)
                st_tr = [x for x in tr if x["setup_id"] == sf_["setup_id"]][0]
                assert st == st_tr["history"][-1], f"{etiqueta} corte={corte}: el estado difiere para {sf_['setup_id']}"
                esperado_conf = sf_["confirmed_at"] if (sf_["confirmed_at"] is not None and sf_["confirmed_at"] <= t_corte) else None
                assert st_tr["confirmed_at"] == esperado_conf
                comparados += 1
        return comparados, len(full)

    comp, tot = _comparar(dfA, ctxA, CFG, [31, 32, 33, 34, 35, 36, 37, 40, 44], "guion")
    assert comp >= 8
    print(f"  OK — guion alcista: {comp} comparaciones (setup × corte) idénticas, con cortes antes/durante/después de cada transición.")
    compB, _ = _comparar(dfB, ctxB, CFG, [31, 33, 36, 40], "espejo")
    print(f"  OK — guion bajista: {compB} comparaciones idénticas.")
    cfg_real = dict(swing_order=3, equal_tolerance_pct=0.1)
    elegido = None
    for semilla in range(1, 60):
        dfR = _random_walk(700, semilla)
        sR, _ = detect_setups(dfR, "EURUSD", cfg_real)
        con_estado = [x for x in sR if len(x["history"]) >= 2]
        if len(sR) >= 3 and con_estado and any(x["context_htf_bar"] is not None for x in [h for s_ in sR for h in s_["history"]]) if sR else False:
            elegido = (semilla, dfR, sR)
            break
    assert elegido is not None, "No se encontró una serie realista con setups para el test"
    semilla, dfR, sR = elegido
    comp_r, tot_r = _comparar(dfR, None, cfg_real, [250, 331, 420, 520, 640], "real")
    print(f"  OK — serie ruidosa (semilla {semilla}, contexto HTF/MTF REAL): {tot_r} setups, {comp_r} comparaciones idénticas.")

    print("\n" + "=" * 70)
    print("CASO 9 — No usar velas HTF sin cerrar")
    print("=" * 70)
    revisadas = 0
    for corte in (333, 475, 700):   # cortes que terminan a mitad de una vela H4
        sx, _ = detect_setups(dfR.iloc[:corte], "EURUSD", cfg_real)
        t_fin = _norm_ts(dfR.index)[corte - 1] + pd.Timedelta("15min")
        for s_ in sx:
            for snap in s_["history"]:
                hb, mb = snap["context_htf_bar"], snap["context_mtf_bar"]
                if hb is not None:
                    assert pd.Timestamp(hb) + pd.Timedelta("4h") <= snap["timestamp"], "se usó una vela HTF sin cerrar"
                    revisadas += 1
                if mb is not None:
                    assert pd.Timestamp(mb) + pd.Timedelta("1h") <= snap["timestamp"], "se usó una vela MTF sin cerrar"
    assert revisadas > 0, "test vacío: ningún snapshot usó contexto HTF"
    print(f"  OK — en {revisadas} snapshots, la vela HTF/MTF usada ya había cerrado antes del timestamp del snapshot.")

    print("\n" + "=" * 70)
    print("CASO 10 — Los módulos anteriores siguen funcionando sin modificaciones")
    print("=" * 70)
    import core.fvg as _fvg, core.liquidity as _liq, core.market_structure as _ms, core.multi_timeframe as _mtf, core.order_blocks as _ob
    for nombre, mod in (("market_structure", _ms), ("liquidity", _liq), ("fvg", _fvg), ("order_blocks", _ob), ("multi_timeframe", _mtf)):
        with contextlib.redirect_stdout(io.StringIO()):
            mod._demo()
        print(f"  OK — core.{nombre}._demo() sigue pasando todas sus pruebas.")

    print("\nTodas las pruebas de core/setup_engine.py corrieron sin errores.")


if __name__ == "__main__":
    _demo()
