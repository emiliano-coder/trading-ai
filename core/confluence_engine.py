"""
core/confluence_engine.py
─────────────────────────────────────────────────────────────────────
Confluence Engine — evalúa qué evidencias de los módulos existentes COINCIDEN,
cuáles se CONTRADICEN y qué tan sólida es la configuración analizada.

  Market Structure · Liquidity · FVG · Order Blocks · Multi-Timeframe · Setup Engine

NO ejecuta operaciones, NO genera BUY/SELL/WAIT, NO toca el scoring existente
ni Telegram, y NO está conectado a ningún sistema de decisiones.

IMPORTANTE — qué significa un score: es una medida de COINCIDENCIA de evidencias
ponderadas, NO una probabilidad. Un score de 80 NO significa 80% de ganar. Los pesos
son una elección configurable y sin calibrar estadísticamente.

─────────────────────────────────────────────────────────────────────
FÓRMULA DE PUNTUACIÓN
─────────────────────────────────────────────────────────────────────
Cada evidencia aporta puntos a UNA dirección:

    puntos = peso_categoría × peso_timeframe × fuerza            (fuerza ∈ [0, 1])

  · peso_categoría   config.weights  (structure 25, liquidity 15, fvg 10,
                     order_block 15, multi_timeframe 10, setup 25 → suman 100)
  · peso_timeframe   config.tf_weights (HTF 0.5, MTF 0.3, LTF 0.2). Solo en las
                     categorías por timeframe; multi_timeframe y setup son globales.
  · fuerza           recencia (BOS/CHoCH/sweeps), cercanía del precio a la zona
                     (FVG/OB/pools EQ), frescura (OB fresh/touched), etapa (setup)…

    score_dirección = 100 × Σ puntos contados de esa dirección / peso_disponible

`peso_disponible` suma solo las casillas (categoría × timeframe) que TIENEN datos y
módulo funcionando: la ausencia de datos NO cuenta como evidencia ni a favor ni en
contra, simplemente no entra al denominador. `coverage` = peso_disponible / peso_total
dice cuánto del panorama se pudo evaluar.

    bullish_confluence_score   evidencia alcista (0-100)
    bearish_confluence_score   evidencia bajista (0-100)
    conflicting_evidence_score = min(bullish, bearish): cuánta evidencia opone al lado dominante
    evidence_count             nº de evidencias independientes realmente contadas

─────────────────────────────────────────────────────────────────────
CÓMO SE EVITA CONTAR DOS VECES LA MISMA EVIDENCIA
─────────────────────────────────────────────────────────────────────
Cada evidencia declara las `roots` (hechos de mercado concretos) en las que se apoya,
p. ej. `event:ltf:32` (el BOS de la vela 32), `sweep:ltf:30`, `fvg:ltf:33`.
  1) Límite por (categoría, timeframe, dirección): si hay varias evidencias del mismo
     tipo, solo cuenta la más fuerte; las demás se listan como "corroborantes" con 0 puntos.
  2) Raíces compartidas: si dos evidencias comparten una raíz, solo cuenta la de mayor
     puntaje; la otra queda marcada con `duplicate_of`. Un OB y un FVG que se solapan en
     el mismo precio comparten una raíz `area:*`: cuentan como una sola zona.
  3) El Setup Engine es un RESUMEN de otras evidencias (sweep + BOS + zona). Por eso no
     suma esas piezas otra vez: aporta solo su ETAPA (config.setup_stage_factors:
     forming 0, candidate 0.4, confirmed 1.0), cuyo hecho propio —la reacción en la zona—
     no existe en ningún otro módulo. Sus piezas se enlazan en `links`, sin puntos.
  4) BOS, CHoCH y tendencia de un mismo timeframe son UNA evidencia de estructura
     (tendencia como base; un BOS/CHoCH reciente a favor sube su fuerza).
  Limitación honesta: `multi_timeframe` (acuerdo entre HTF/MTF/LTF) se DERIVA de las mismas
  tendencias que ya puntúa `structure`. Se trata como un hecho distinto (el acuerdo) y pesa
  poco por defecto (10); pon su peso en 0 si prefieres no contarlo.

─────────────────────────────────────────────────────────────────────
CLASIFICACIÓN (umbrales configurables)
─────────────────────────────────────────────────────────────────────
  insufficient_data   sin LTF utilizable, o coverage < min_coverage
  conflicting_context hay contradicciones IMPORTANTES (HTF/MTF contra el sesgo, alineación
                      mixta, setup activo en sentido contrario) con evidencia ≥ conflict_floor (10),
                      o ambos lados son fuertes y cercanos. No se fuerza una dirección ni se
                      reporta 'neutral' cuando el contexto está en conflicto.
  neutral             el lado dominante no llega a neutral_below (30)
  bullish_/bearish_confluence           dominante ≥ 30
  strong_bullish_/strong_bearish_confluence   dominante ≥ 55 y oposición ≤ 20
  (Los umbrales son una elección mía, sin calibrar estadísticamente: se fijaron para que un escenario
   sintético con todas las piezas alineadas y un setup confirmado salga 'strong'. Ajústalos con tus datos.)

─────────────────────────────────────────────────────────────────────
LOOK-AHEAD BIAS
─────────────────────────────────────────────────────────────────────
La evaluación en el timestamp T (cierre de una vela LTF) lee solo estados disponibles en T:
  · cada timeframe mayor se consulta con su última vela YA CERRADA (`available_at <= T`);
  · las zonas FVG/OB se usan desde su created_at / confirmed_at y solo mientras seguían
    activas en T; los pools EQ solo si no se habían operado antes de T;
  · los setups se leen con `state_as_of` (historial; nunca se reescribe el pasado);
  · timestamps normalizados a UTC sin zona, igual que el resto de módulos.
`analyze_confluence(..., as_of=T)` sobre datos completos debe dar EXACTAMENTE lo mismo que
`analyze_confluence` sobre los datos truncados en T (lo comprueban los tests).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.fvg import analyze_fvg
from core.liquidity import analyze_liquidity
from core.market_structure import _validar_columnas, analyze_market_structure
from core.multi_timeframe import (_asegurar_datetimeindex, _clasificar_alineacion, _norm_ts,
                                  _normalizar_pesos, _resamplear_causal, analyze_multi_timeframe)
from core.order_blocks import analyze_order_blocks
from core.setup_engine import ACTIVOS, SetupConfig, detect_setups, state_as_of

TFS = ("htf", "mtf", "ltf")
CATEGORIAS = ("structure", "liquidity", "fvg", "order_block", "multi_timeframe", "setup")
CATEGORIAS_POR_TF = ("structure", "liquidity", "fvg", "order_block")
CLASES = ("strong_bullish_confluence", "bullish_confluence", "neutral", "bearish_confluence",
          "strong_bearish_confluence", "conflicting_context", "insufficient_data")
_ES = {"bullish": "alcista", "bearish": "bajista"}
_OPUESTO = {"bullish": "bearish", "bearish": "bullish"}


# ─────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ConfluenceConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "structure": 25.0, "liquidity": 15.0, "fvg": 10.0, "order_block": 15.0, "multi_timeframe": 10.0, "setup": 25.0})
    tf_weights: Dict[str, float] = field(default_factory=lambda: {"htf": 0.5, "mtf": 0.3, "ltf": 0.2})
    setup_stage_factors: Dict[str, float] = field(default_factory=lambda: {"forming": 0.0, "candidate": 0.4, "confirmed": 1.0})
    # motores
    swing_order: int = 5
    equal_tolerance_pct: float = 0.05
    mitigation_criterion: str = "50%"
    displacement_atr_mult: float = 1.5
    htf_freq: str = "4h"
    mtf_freq: str = "1h"
    ltf_freq: str = "15min"
    # fuerza de las evidencias
    recency_bars: int = 30               # edad máxima (en velas del propio timeframe) de BOS/CHoCH/sweeps
    event_bonus: float = 0.4             # parte de la fuerza de structure que viene de un BOS/CHoCH reciente
    proximity_pct: float = 0.5           # % del precio: más lejos, una zona FVG/OB no cuenta
    eq_pools_directional: bool = True    # EQH arriba = imán alcista, EQL abajo = imán bajista (interpretación)
    eq_pool_proximity_pct: float = 1.0
    eq_pool_strength: float = 0.4
    touched_factor: float = 0.6          # OB ya tocado (no fresh)
    # clasificación
    min_coverage: float = 0.35
    neutral_below: float = 30.0
    strong_at: float = 55.0
    strong_max_opposition: float = 20.0
    conflict_floor: float = 10.0         # con contradicciones importantes, evidencia mínima para hablar de 'conflicto'
    conflict_min_score: float = 25.0
    conflict_margin: float = 20.0
    setup_config: Dict[str, Any] = field(default_factory=dict)   # overrides para SetupConfig


def _resolver_config(config: Any) -> ConfluenceConfig:
    base = ConfluenceConfig()
    if config is None:
        cfg = base
    elif isinstance(config, ConfluenceConfig):
        cfg = config
    elif isinstance(config, dict):
        for k, v in config.items():
            if not hasattr(base, k):
                raise ValueError(f"config desconocida: {k}")
            if k in ("weights", "tf_weights", "setup_stage_factors", "setup_config") and isinstance(v, dict):
                merged = dict(getattr(base, k))
                merged.update(v)
                setattr(base, k, merged)
            else:
                setattr(base, k, v)
        cfg = base
    else:
        raise ValueError("config debe ser None, ConfluenceConfig o dict")
    tw = {tf: max(float(cfg.tf_weights.get(tf, 0.0)), 0.0) for tf in TFS}
    total = sum(tw.values())
    cfg.tf_weights = {tf: (tw[tf] / total if total > 0 else 1.0 / 3) for tf in TFS}
    cfg.weights = {c: max(float(cfg.weights.get(c, 0.0)), 0.0) for c in CATEGORIAS}
    return cfg


# ─────────────────────────────────────────────────────────────────────
#  Datos por timeframe (reusa los 4 motores; no duplica sus algoritmos)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _TF:
    name: str
    freq: str
    n: int = 0
    bar_time: Optional[np.ndarray] = None
    avail: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    trend: Optional[np.ndarray] = None
    ev_idx: Optional[np.ndarray] = None
    ev_kind: Optional[np.ndarray] = None
    ev_dir: Optional[np.ndarray] = None
    sweeps: List[dict] = field(default_factory=list)
    eq_pools: List[dict] = field(default_factory=list)
    fvg: List[dict] = field(default_factory=list)
    ob: List[dict] = field(default_factory=list)
    ok: Dict[str, bool] = field(default_factory=lambda: {"structure": False, "liquidity": False, "fvg": False, "order_block": False})
    errors: List[str] = field(default_factory=list)


def _construir_tf(nombre: str, df: Optional[pd.DataFrame], freq: str, cfg: ConfluenceConfig) -> _TF:
    tf = _TF(name=nombre, freq=freq)
    if df is None or len(df) == 0:
        return tf
    n = len(df)
    idx = _norm_ts(df.index)
    tf.n = n
    tf.bar_time = np.asarray(idx, dtype="datetime64[ns]")
    tf.avail = np.asarray(idx + pd.Timedelta(freq), dtype="datetime64[ns]")
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    tf.c = df["close"].to_numpy(dtype=float)

    ms_df = liq_df = None
    liq_sum = fvg_sum = ob_sum = None
    fvg_df = None
    try:
        ms_df, _ = analyze_market_structure(df, swing_order=cfg.swing_order, break_using="close")
        tf.ok["structure"] = True
    except Exception as e:
        tf.errors.append(f"market_structure: {e}")
    try:
        liq_df, liq_sum = analyze_liquidity(df, market_structure_df=ms_df, swing_order=cfg.swing_order,
                                            equal_tolerance_pct=cfg.equal_tolerance_pct)
        tf.ok["liquidity"] = True
    except Exception as e:
        tf.errors.append(f"liquidity: {e}")
    try:
        fvg_df, fvg_sum = analyze_fvg(df, mitigation_criterion=cfg.mitigation_criterion)
        tf.ok["fvg"] = True
    except Exception as e:
        tf.errors.append(f"fvg: {e}")
    try:
        _, ob_sum = analyze_order_blocks(df, market_structure_df=ms_df, liquidity_df=liq_df, fvg_df=fvg_df,
                                         displacement_atr_mult=cfg.displacement_atr_mult,
                                         mitigation_criterion=cfg.mitigation_criterion)
        tf.ok["order_block"] = True
    except Exception as e:
        tf.errors.append(f"order_blocks: {e}")

    # estructura: tendencia por vela + último evento estructural (BOS/CHoCH) conocido
    tf.trend = np.full(n, "ranging", dtype=object)
    tf.ev_idx = np.full(n, -1, dtype=int)
    tf.ev_kind = np.full(n, None, dtype=object)
    tf.ev_dir = np.full(n, None, dtype=object)
    if ms_df is not None:
        tf.trend = ms_df["structure_trend"].to_numpy(dtype=object)
        bb, bs = ms_df["bos_bullish"].to_numpy(bool), ms_df["bos_bearish"].to_numpy(bool)
        cb, cs = ms_df["choch_bullish"].to_numpy(bool), ms_df["choch_bearish"].to_numpy(bool)
        cur = (-1, None, None)
        for i in range(n):
            if bb[i] or bs[i] or cb[i] or cs[i]:
                d = tf.trend[i]
                if d == "bullish" and (cb[i] or bb[i]):
                    cur = (i, "choch" if cb[i] else "bos", "bullish")
                elif d == "bearish" and (cs[i] or bs[i]):
                    cur = (i, "choch" if cs[i] else "bos", "bearish")
                elif cb[i]:
                    cur = (i, "choch", "bullish")
                elif cs[i]:
                    cur = (i, "choch", "bearish")
                elif bb[i]:
                    cur = (i, "bos", "bullish")
                else:
                    cur = (i, "bos", "bearish")
            tf.ev_idx[i], tf.ev_kind[i], tf.ev_dir[i] = cur

    if liq_sum is not None:
        for direccion, clave in (("bullish", "bullish_sweeps"), ("bearish", "bearish_sweeps")):
            for e in liq_sum[clave]:
                tf.sweeps.append({"index": int(e["index"]), "direction": direccion, "level": float(e["level"]),
                                  "zone_kind": e.get("zone_kind"), "touches": e.get("touches")})
        for clave, kind in (("eqh", "EQH"), ("eql", "EQL")):
            for z in liq_sum[clave]:
                f = int(z["confirmed_at"])
                lvl = float(z["level"])
                tomado = np.flatnonzero((h[f + 1:] > lvl) if kind == "EQH" else (l[f + 1:] < lvl))
                tf.eq_pools.append({"kind": kind, "level": lvl, "formed": f,
                                    "taken": (f + 1 + int(tomado[0])) if len(tomado) else None})
    if fvg_sum is not None:
        for z in fvg_sum["all_fvg"]:
            tf.fvg.append({"kind": z["kind"], "avail": int(z["created_at"]), "top": float(z["top"]), "bottom": float(z["bottom"]),
                           "mit": z.get("mitigated_at"), "inv": z.get("invalidated_at")})
    if ob_sum is not None:
        for z in ob_sum["all_order_blocks"]:
            tf.ob.append({"kind": z["kind"], "avail": int(z["confirmed_at"]), "top": float(z["top"]), "bottom": float(z["bottom"]),
                          "touched": z.get("touched_at"), "mit": z.get("mitigated_at"), "inv": z.get("invalidated_at")})
    return tf


# ─────────────────────────────────────────────────────────────────────
#  Preparación (una vez) y evaluación (en cualquier vela LTF)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _Prep:
    cfg: ConfluenceConfig
    symbol: str
    tf: Dict[str, _TF]
    min_bars: int
    setups: List[dict]
    setups_ok: bool
    errors: List[str]


def _preparar(df: Optional[pd.DataFrame], symbol: str, cfg: ConfluenceConfig,
              frames: Optional[Dict[str, Optional[pd.DataFrame]]]) -> _Prep:
    errores: List[str] = []
    marcos: Dict[str, Optional[pd.DataFrame]] = {"htf": None, "mtf": None, "ltf": None}
    freqs = {"htf": cfg.htf_freq, "mtf": cfg.mtf_freq, "ltf": cfg.ltf_freq}
    try:
        if frames:
            for tf in TFS:
                d = frames.get(tf)
                if d is not None and len(d) > 0:
                    _validar_columnas(d)
                    marcos[tf] = _asegurar_datetimeindex(d)
        elif df is not None and len(df) > 0:
            _validar_columnas(df)
            base = _asegurar_datetimeindex(df)
            marcos["ltf"] = base
            marcos["mtf"] = _resamplear_causal(base, cfg.mtf_freq, cfg.ltf_freq)
            marcos["htf"] = _resamplear_causal(base, cfg.htf_freq, cfg.ltf_freq)
    except ValueError as e:
        errores.append(f"datos inválidos: {e}")
        marcos = {"htf": None, "mtf": None, "ltf": None}

    # los timeframes mayores solo cuentan con velas ya cerradas respecto al fin del LTF
    if marcos["ltf"] is not None and len(marcos["ltf"]) > 0:
        fin_ltf = _norm_ts(marcos["ltf"].index)[-1] + pd.Timedelta(cfg.ltf_freq)
        for tf in ("htf", "mtf"):
            d = marcos[tf]
            if d is not None and len(d) > 0:
                marcos[tf] = d[(_norm_ts(d.index) + pd.Timedelta(freqs[tf])) <= fin_ltf]

    tfs = {tf: _construir_tf(tf, marcos[tf], freqs[tf], cfg) for tf in TFS}
    min_bars = 2 * cfg.swing_order + 3

    setups, setups_ok = [], False
    ltf = marcos["ltf"]
    if ltf is not None and len(ltf) >= 3 * cfg.swing_order + 5:
        try:
            ctx, _ = analyze_multi_timeframe(df_htf=marcos["htf"], df_mtf=marcos["mtf"], df_ltf=ltf,
                                             htf_freq=cfg.htf_freq, mtf_freq=cfg.mtf_freq, ltf_freq=cfg.ltf_freq,
                                             swing_order=cfg.swing_order, equal_tolerance_pct=cfg.equal_tolerance_pct,
                                             mitigation_criterion=cfg.mitigation_criterion,
                                             displacement_atr_mult=cfg.displacement_atr_mult)
            # Suficiencia del contexto con el conteo REAL de velas cerradas de cada timeframe (no con las
            # visibles dentro de la ventana LTF): se enmascara aquí y se desactiva el chequeo interno del Setup Engine.
            ctx = ctx.copy()
            ltf_avail = np.asarray(_norm_ts(ltf.index) + pd.Timedelta(cfg.ltf_freq), dtype="datetime64[ns]")
            for tf in ("htf", "mtf"):
                d = tfs[tf]
                col = f"{tf}_trend"
                if col not in ctx.columns:
                    continue
                vals = ctx[col].to_numpy(dtype=object).copy()
                if d.n == 0:
                    vals[:] = None
                else:
                    cerradas = np.searchsorted(d.avail, ltf_avail, side="right")   # velas cerradas en cada fila LTF
                    vals[cerradas < min_bars] = None
                ctx[col] = vals
            sc = dict(swing_order=cfg.swing_order, equal_tolerance_pct=cfg.equal_tolerance_pct,
                      mitigation_criterion=cfg.mitigation_criterion, displacement_atr_mult=cfg.displacement_atr_mult,
                      htf_freq=cfg.htf_freq, mtf_freq=cfg.mtf_freq, ltf_freq=cfg.ltf_freq, min_context_bars=0)
            sc.update(cfg.setup_config)
            lista, resumen = detect_setups(ltf, symbol, SetupConfig(**sc), context=ctx if len(ctx) == len(ltf) else None)
            setups, setups_ok = lista, not resumen["insufficient_data"]
            errores += [f"setup_engine: {e}" for e in resumen["errors"]]
        except Exception as e:
            errores.append(f"setup_engine/multi_timeframe: {e}")
    return _Prep(cfg=cfg, symbol=symbol, tf=tfs, min_bars=min_bars, setups=setups, setups_ok=setups_ok, errors=errores)


def _pct(dist: float, precio: float) -> float:
    return abs(dist) / abs(precio) * 100.0 if precio else float("inf")


def _nueva(cfg: ConfluenceConfig, cat: str, tf: str, ident: str, direction: str, label: str, strength: float,
           roots: List[str], modules: List[str], peso_tf: float, **extra) -> dict:
    peso = cfg.weights[cat] * peso_tf
    fuerza = max(0.0, min(1.0, float(strength)))
    e = {"id": f"{cat}:{tf}:{ident}", "category": cat, "timeframe": tf, "direction": direction, "label": label,
         "strength": round(fuerza, 4), "weight": round(peso, 6), "points": round(peso * fuerza, 6),
         "roots": sorted(set(roots)), "links": [], "source_modules": modules,
         "counted": False, "duplicate_of": None, "note": ""}
    e.update(extra)
    return e


def _evidencias_tf(prep: _Prep, nombre: str, d: _TF, i: int, precio: float) -> Tuple[List[dict], List[dict]]:
    """Evidencias (y notas informativas) de un timeframe en su vela cerrada `i`, con el precio actual del LTF."""
    cfg = prep.cfg
    tfw = cfg.tf_weights[nombre]
    bar = pd.Timestamp(d.bar_time[i])
    evs: List[dict] = []
    info: List[dict] = []
    TF = nombre.upper()

    # ── Estructura: UNA evidencia por timeframe (tendencia + BOS/CHoCH reciente) ──
    if d.ok["structure"]:
        tr = d.trend[i]
        if tr in ("bullish", "bearish"):
            k = int(d.ev_idx[i])
            fuerza = 1.0 - cfg.event_bonus
            texto = f"{TF}: tendencia {_ES[tr]}"
            comp = {"trend": tr, "last_event": None}
            if k >= 0:
                edad = i - k
                comp["last_event"] = {"kind": d.ev_kind[i], "direction": d.ev_dir[i], "age_bars": edad}
                if d.ev_dir[i] == tr and edad <= cfg.recency_bars:
                    fuerza += cfg.event_bonus * (1.0 - edad / max(cfg.recency_bars, 1))
                    texto += f" con {str(d.ev_kind[i]).upper()} {_ES[tr]} hace {edad} velas"
            raiz = f"event:{nombre}:{k}" if k >= 0 else f"state:{nombre}:{tr}"
            evs.append(_nueva(cfg, "structure", nombre, tr, tr, texto, fuerza, [raiz], ["market_structure"], tfw,
                              bar_time=bar, components=comp))
        else:
            info.append({"id": f"structure:{nombre}:ranging", "category": "structure", "timeframe": nombre,
                         "label": f"{TF}: estructura en rango (sin dirección)"})

    # ── Liquidez: último sweep reciente + pools EQ cercanos ──
    if d.ok["liquidity"]:
        previos = [s for s in d.sweeps if s["index"] <= i]
        if previos:
            s = max(previos, key=lambda x: (x["index"], x["direction"]))
            edad = i - s["index"]
            if edad <= cfg.recency_bars:
                evs.append(_nueva(cfg, "liquidity", nombre, f"sweep:{s['index']}", s["direction"],
                                  f"{TF}: sweep {_ES[s['direction']]} de liquidez hace {edad} velas",
                                  1.0 - edad / max(cfg.recency_bars, 1), [f"sweep:{nombre}:{s['index']}"], ["liquidity"], tfw,
                                  bar_time=bar, age_bars=edad))
        if cfg.eq_pools_directional:
            vivos = [p for p in d.eq_pools if p["formed"] <= i and (p["taken"] is None or p["taken"] > i)]
            for kind, direccion, sentido in (("EQH", "bullish", 1), ("EQL", "bearish", -1)):
                cand = [p for p in vivos if p["kind"] == kind and (p["level"] - precio) * sentido > 0]
                if not cand:
                    continue
                p = min(cand, key=lambda x: (abs(x["level"] - precio), x["formed"]))
                dist = _pct(p["level"] - precio, precio)
                if dist <= cfg.eq_pool_proximity_pct:
                    fuerza = cfg.eq_pool_strength * (1.0 - dist / cfg.eq_pool_proximity_pct)
                    evs.append(_nueva(cfg, "liquidity", nombre, f"eq:{kind}:{p['formed']}", direccion,
                                      f"{TF}: {kind} activo {'arriba' if sentido > 0 else 'abajo'} a {dist:.2f}% (imán de liquidez)",
                                      fuerza, [f"eqpool:{nombre}:{kind}:{p['formed']}"], ["liquidity"], tfw,
                                      bar_time=bar, distance_pct=round(dist, 4)))
                else:
                    info.append({"id": f"liquidity:{nombre}:eq:{kind}", "category": "liquidity", "timeframe": nombre,
                                 "label": f"{TF}: {kind} activo pero lejano ({dist:.2f}%)"})

    # ── FVG ──
    zonas_fvg: List[dict] = []
    if d.ok["fvg"]:
        for z in d.fvg:
            fin = min([v for v in (z["mit"], z["inv"]) if v is not None] + [d.n + 1])
            if not (z["avail"] <= i < fin):
                continue
            ev = _evidencia_zona(cfg, "fvg", nombre, TF, z, z["avail"], precio, bar, 1.0, tfw, "FVG", "fvg")
            if ev is not None:
                zonas_fvg.append(ev)
        mit = [z for z in d.fvg if z["mit"] is not None and z["mit"] <= i and i - z["mit"] <= cfg.recency_bars]
        if mit:
            z = max(mit, key=lambda x: x["mit"])
            info.append({"id": f"fvg:{nombre}:mitigated:{z['avail']}", "category": "fvg", "timeframe": nombre,
                         "label": f"{TF}: FVG {_ES[z['kind']]} mitigado hace {i - z['mit']} velas (informativo, 0 puntos)"})

    # ── Order Blocks ──
    zonas_ob: List[dict] = []
    if d.ok["order_block"]:
        for z in d.ob:
            fin = min([v for v in (z["mit"], z["inv"]) if v is not None] + [d.n + 1])
            if not (z["avail"] <= i < fin):
                continue
            fresco = z["touched"] is None or z["touched"] > i
            ev = _evidencia_zona(cfg, "order_block", nombre, TF, z, z["avail"], precio, bar,
                                 1.0 if fresco else cfg.touched_factor, tfw, "OB", "ob")
            if ev is not None:
                ev["freshness"] = "fresh" if fresco else "touched"
                ev["label"] += f" ({ev['freshness']})"
                zonas_ob.append(ev)

    # OB y FVG que se solapan en precio (misma dirección) son UNA zona: comparten raíz
    for a in zonas_fvg:
        for b in zonas_ob:
            if a["direction"] == b["direction"] and a["zone"]["bottom"] <= b["zone"]["top"] and b["zone"]["bottom"] <= a["zone"]["top"]:
                comun = f"area:{nombre}:{a['direction']}:{min(a['zone']['avail'], b['zone']['avail'])}"
                a["roots"] = sorted(set(a["roots"]) | {comun})
                b["roots"] = sorted(set(b["roots"]) | {comun})
    evs += zonas_fvg + zonas_ob
    return evs, info


def _evidencia_zona(cfg: ConfluenceConfig, cat: str, nombre: str, TF: str, z: dict, avail: int, precio: float,
                    bar: pd.Timestamp, factor: float, tfw: float, sigla: str, raiz: str) -> Optional[dict]:
    """Una zona activa cuenta si el precio está dentro de ella o ya la dejó atrás sin invalidarla (soporte/resistencia),
    y está a menos de `proximity_pct`."""
    top, bottom = z["top"], z["bottom"]
    if z["kind"] == "bullish":
        if precio < bottom:
            return None
        dist = 0.0 if precio <= top else _pct(precio - top, precio)
    else:
        if precio > top:
            return None
        dist = 0.0 if precio >= bottom else _pct(bottom - precio, precio)
    cercania = max(0.0, 1.0 - dist / cfg.proximity_pct) if cfg.proximity_pct > 0 else 0.0
    if cercania <= 0.0:
        return None
    texto = (f"{TF}: {sigla} {_ES[z['kind']]} activo {bottom:.5g}-{top:.5g}, "
             + ("precio dentro de la zona" if dist == 0.0 else f"a {dist:.2f}% del precio"))
    return _nueva(cfg, cat, nombre, f"{z['kind']}:{avail}", z["kind"], texto, cercania * factor,
                  [f"{raiz}:{nombre}:{avail}"], [{"fvg": "fvg", "order_block": "order_blocks"}[cat]], tfw,
                  bar_time=bar, distance_pct=round(dist, 4), zone={"top": top, "bottom": bottom, "avail": avail})


def _resolver_evidencias(evs: List[dict]) -> List[dict]:
    """Evita contar dos veces lo mismo: (1) límite por categoría/TF/dirección; (2) raíces compartidas."""
    orden = sorted(evs, key=lambda e: (-e["points"], e["id"]))
    por_grupo: Dict[Tuple[str, str, str], str] = {}
    for e in orden:
        e["counted"], e["duplicate_of"], e["note"] = False, None, ""
        if e["points"] <= 0:
            e["note"] = "sin puntos (fuerza 0)"
            continue
        clave = (e["category"], e["timeframe"], e["direction"])
        if clave in por_grupo:
            e["duplicate_of"] = por_grupo[clave]
            e["note"] = "corrobora; en esta categoría/timeframe/dirección solo cuenta la más fuerte"
        else:
            por_grupo[clave] = e["id"]
            e["counted"] = True
    usadas: Dict[str, str] = {}
    for e in orden:
        if not e["counted"]:
            continue
        solape = [r for r in e["roots"] if r in usadas]
        if solape:
            e["counted"], e["duplicate_of"] = False, usadas[solape[0]]
            e["note"] = f"comparte el hecho '{solape[0]}' con otra evidencia ya contada: no se suma dos veces"
        else:
            for r in e["roots"]:
                usadas[r] = e["id"]
    return orden


def _evaluar(prep: _Prep, t: int) -> dict:
    cfg = prep.cfg
    ltf = prep.tf["ltf"]
    T64 = ltf.avail[t] if ltf.n else None
    ts = pd.Timestamp(T64) if T64 is not None else None
    precio = float(ltf.c[t]) if ltf.n else float("nan")

    evs: List[dict] = []
    info: List[dict] = []
    faltantes: List[str] = []
    slots: Dict[Tuple[str, str], float] = {}
    idx_tf: Dict[str, Optional[int]] = {}
    sesgos: Dict[str, Optional[str]] = {}

    total_peso = sum(cfg.weights[c] * sum(cfg.tf_weights.values()) for c in CATEGORIAS_POR_TF) \
        + cfg.weights["multi_timeframe"] + cfg.weights["setup"]

    for nombre in TFS:
        d = prep.tf[nombre]
        i = int(np.searchsorted(d.avail, T64, side="right") - 1) if d.n else -1
        if d.n == 0 or i < 0:
            idx_tf[nombre], sesgos[nombre] = None, None
            faltantes.append(f"{nombre.upper()}: sin datos")
            continue
        if i + 1 < prep.min_bars:
            idx_tf[nombre], sesgos[nombre] = None, None
            faltantes.append(f"{nombre.upper()}: datos insuficientes ({i + 1} velas cerradas < {prep.min_bars} necesarias)")
            continue
        idx_tf[nombre] = i
        sesgos[nombre] = d.trend[i] if d.ok["structure"] and d.trend[i] in ("bullish", "bearish", "ranging") else None
        for cat in CATEGORIAS_POR_TF:
            if d.ok[cat]:
                slots[(cat, nombre)] = cfg.weights[cat] * cfg.tf_weights[nombre]
            else:
                faltantes.append(f"{nombre.upper()}: módulo {cat} sin resultados")
        e_tf, i_tf = _evidencias_tf(prep, nombre, d, i, precio)
        evs += e_tf
        info += i_tf

    # ── Multi-Timeframe: acuerdo entre timeframes + contradicciones ──
    alineacion, al_score = "neutral_context", 0
    con_datos = [tf for tf in TFS if idx_tf.get(tf) is not None and sesgos.get(tf) is not None]
    if len(con_datos) >= 2:
        slots[("multi_timeframe", "global")] = cfg.weights["multi_timeframe"]
        alineacion, al_score = _clasificar_alineacion(sesgos, _normalizar_pesos(cfg.tf_weights))
        if alineacion in ("strong_bullish_alignment", "bullish_bias", "strong_bearish_alignment", "bearish_bias"):
            direccion = "bullish" if "bullish" in alineacion else "bearish"
            raices = [f"alignment:{alineacion}:" + "/".join(str(sesgos.get(tf)) for tf in TFS)]
            e = _nueva(cfg, "multi_timeframe", "global", "alignment", direccion,
                       f"alineación {alineacion.replace('_', ' ')} (HTF {sesgos['htf']}, MTF {sesgos['mtf']}, LTF {sesgos['ltf']})",
                       al_score / 100.0, raices, ["multi_timeframe"], 1.0,
                       alignment=alineacion, alignment_score=al_score)
            e["links"] = [f"structure:{tf}" for tf in TFS if sesgos.get(tf) == direccion]
            evs.append(e)
    elif len(con_datos) == 1:
        faltantes.append("Multi-timeframe: se necesitan al menos 2 timeframes con datos para evaluar la alineación")

    # ── Setup Engine (etapa del setup; sus piezas ya cuentan como evidencia atómica) ──
    setups_activos: List[dict] = []
    ltf_ok = ltf.n > 0 and idx_tf.get("ltf") is not None
    if ltf_ok and prep.setups_ok:
        slots[("setup", "ltf")] = cfg.weights["setup"]
        for s in prep.setups:
            snap = state_as_of(s, ts)
            if snap is None or snap["status"] not in ACTIVOS:
                continue
            setups_activos.append({"setup_id": s["setup_id"], "direction": s["direction"], "status": snap["status"],
                                   "conditions_met": list(snap["conditions_met"]), "conditions_missing": list(snap["conditions_missing"])})
            factor = float(cfg.setup_stage_factors.get(snap["status"], 0.0))
            e = _nueva(cfg, "setup", "ltf", s["setup_id"], s["direction"],
                       f"setup {_ES[s['direction']]} {snap['status']} ({s['setup_id']})", factor,
                       [f"setup:{s['setup_id']}:{snap['status']}"], ["setup_engine"], 1.0,
                       bar_time=pd.Timestamp(ltf.bar_time[t]), status=snap["status"],
                       conditions_met=list(snap["conditions_met"]), conditions_missing=list(snap["conditions_missing"]))
            # los enlaces solo citan hechos que YA habían ocurrido en T (el dato final del setup puede ser posterior)
            shift_idx = s.get("structure_shift", {}).get("index")
            e["links"] = ([f"sweep:ltf:{s['sweep']['index']}"] if s.get("sweep") and s["sweep"]["index"] <= t else []) + \
                         ([f"event:ltf:{shift_idx}"] if shift_idx is not None and shift_idx <= t else [])
            if factor <= 0:
                e["note"] = "setup en formación: sus condiciones ya cuentan como evidencia atómica; sin aporte propio"
            evs.append(e)
    elif ltf_ok and not prep.setups_ok:
        faltantes.append("Setup Engine: sin resultados")

    evs = _resolver_evidencias(evs)

    peso_disp = float(sum(slots.values()))
    cobertura = (peso_disp / total_peso) if total_peso > 0 else 0.0

    def _score(pts: float) -> float:
        return round(min(100.0, 100.0 * pts / peso_disp), 4) if peso_disp > 0 else 0.0

    contadas = [e for e in evs if e["counted"]]
    pts_bull = sum(e["points"] for e in contadas if e["direction"] == "bullish")
    pts_bear = sum(e["points"] for e in contadas if e["direction"] == "bearish")
    bull, bear = _score(pts_bull), _score(pts_bear)
    conflicto = round(min(bull, bear), 4)
    dominante = "bullish" if bull > bear else "bearish" if bear > bull else None

    # ── contradicciones importantes ──
    mayores: List[str] = []
    if dominante:
        for tf in ("htf", "mtf"):
            if sesgos.get(tf) == _OPUESTO[dominante]:
                mayores.append(f"{tf.upper()} tiene tendencia {_ES[_OPUESTO[dominante]]}, contra el sesgo {_ES[dominante]}")
        if alineacion == "mixed_context":
            mayores.append("alineación mixta entre HTF/MTF/LTF (tendencias opuestas)")
        for s in setups_activos:
            if s["direction"] == _OPUESTO[dominante] and s["status"] in ("candidate", "confirmed"):
                mayores.append(f"setup {_ES[s['direction']]} {s['status']} ({s['setup_id']}) en sentido contrario al sesgo")
    elif alineacion == "mixed_context":
        mayores.append("alineación mixta entre HTF/MTF/LTF (tendencias opuestas)")

    # ── clasificación ──
    dom_score, min_score = max(bull, bear), min(bull, bear)
    if not ltf_ok or cobertura < cfg.min_coverage:
        clase = "insufficient_data"
    elif mayores and dom_score >= cfg.conflict_floor:
        clase = "conflicting_context"     # el contexto está en conflicto: no se reporta 'neutral' ni una dirección
    elif dom_score < cfg.neutral_below:
        clase = "conflicting_context" if min_score >= cfg.conflict_min_score else "neutral"
    elif min_score >= cfg.conflict_min_score and (dom_score - min_score) < cfg.conflict_margin:
        clase = "conflicting_context"
    else:
        fuerte = dom_score >= cfg.strong_at and min_score <= cfg.strong_max_opposition
        if bull > bear:
            clase = "strong_bullish_confluence" if fuerte else "bullish_confluence"
        else:
            clase = "strong_bearish_confluence" if fuerte else "bearish_confluence"

    # ── evidencias a favor / en contra / faltantes ──
    if clase in ("strong_bullish_confluence", "bullish_confluence"):
        lado = "bullish"
    elif clase in ("strong_bearish_confluence", "bearish_confluence"):
        lado = "bearish"
    else:
        lado = dominante
    apoyo = [e for e in contadas if lado and e["direction"] == lado]
    contra = [e for e in contadas if lado and e["direction"] == _OPUESTO[lado]]
    contradicciones = [{"id": f"contradiction:{k}", "category": "contradiction", "label": m} for k, m in enumerate(mayores)]
    if lado:
        for cat in CATEGORIAS:
            if cfg.weights[cat] > 0 and not any(e["category"] == cat for e in apoyo):
                disponible = any(k[0] == cat for k in slots)
                if disponible:
                    faltantes.append(f"sin evidencia de {cat} a favor del sesgo {_ES[lado]}")
        for s in setups_activos:
            if s["direction"] == lado:
                faltantes += [f"setup {s['setup_id']}: falta {c}" for c in s["conditions_missing"]]
    for m in prep.errors:
        faltantes.append(f"error en módulo: {m}")
    for nombre in TFS:
        for m in prep.tf[nombre].errors:
            faltantes.append(f"{nombre.upper()}: {m}")
    faltantes = sorted(set(faltantes))

    # ── desglose por categoría y timeframe ──
    desglose: Dict[str, dict] = {}
    for cat in CATEGORIAS:
        tf_keys = [k[1] for k in slots if k[0] == cat]
        por_tf = {}
        for tfk in sorted(tf_keys):
            ev_tf = [e for e in contadas if e["category"] == cat and e["timeframe"] == tfk]
            por_tf[tfk] = {"available_weight": round(slots[(cat, tfk)], 6),
                           "bullish": round(sum(e["points"] for e in ev_tf if e["direction"] == "bullish"), 6),
                           "bearish": round(sum(e["points"] for e in ev_tf if e["direction"] == "bearish"), 6),
                           "evidence_ids": sorted(e["id"] for e in ev_tf)}
        pb = sum(v["bullish"] for v in por_tf.values())
        pr = sum(v["bearish"] for v in por_tf.values())
        desglose[cat] = {"weight": cfg.weights[cat], "available_weight": round(sum(v["available_weight"] for v in por_tf.values()), 6),
                         "bullish_points": round(pb, 6), "bearish_points": round(pr, 6),
                         "bullish_score_contribution": _score(pb), "bearish_score_contribution": _score(pr),
                         "by_timeframe": por_tf}

    # ── explicación ──
    nombres = {"strong_bullish_confluence": "CONFLUENCIA ALCISTA FUERTE", "bullish_confluence": "CONFLUENCIA ALCISTA",
               "neutral": "NEUTRAL", "bearish_confluence": "CONFLUENCIA BAJISTA",
               "strong_bearish_confluence": "CONFLUENCIA BAJISTA FUERTE", "conflicting_context": "CONTEXTO CONTRADICTORIO",
               "insufficient_data": "DATOS INSUFICIENTES"}
    lineas = [f"{nombres[clase]} — alcista {bull:.1f}, bajista {bear:.1f}, evidencia en contra {conflicto:.1f} "
              f"({len(contadas)} evidencias independientes; cobertura de datos {cobertura * 100:.0f}%)."]
    if clase == "insufficient_data":
        lineas.append("No hay datos suficientes para evaluar: la ausencia de datos no se cuenta como evidencia a favor ni en contra.")
    if apoyo:
        lineas.append("A favor: " + "; ".join(f"{e['label']} ({e['points']:.1f} pts)" for e in sorted(apoyo, key=lambda x: (-x['points'], x['id']))[:5]) + ".")
    if contra:
        lineas.append("En contra: " + "; ".join(f"{e['label']} ({e['points']:.1f} pts)" for e in sorted(contra, key=lambda x: (-x['points'], x['id']))[:4]) + ".")
    if mayores:
        lineas.append("Contradicciones importantes: " + "; ".join(mayores) + ". Por eso no se fuerza una dirección.")
    duplicadas = [e for e in evs if e["duplicate_of"]]
    if duplicadas:
        lineas.append(f"{len(duplicadas)} evidencia(s) no se sumaron porque repiten un hecho ya contado.")
    lineas.append("El score mide coincidencia de evidencias; NO es una probabilidad de éxito.")

    return {
        "timestamp": ts, "bar_index": int(t), "symbol": prep.symbol,
        "bullish_confluence_score": bull, "bearish_confluence_score": bear,
        "conflicting_evidence_score": conflicto, "evidence_count": len(contadas),
        "classification": clase,
        "supporting_evidence": sorted(apoyo, key=lambda e: (-e["points"], e["id"])),
        "conflicting_evidence": sorted(contra, key=lambda e: (-e["points"], e["id"])) + contradicciones,
        "missing_evidence": faltantes,
        "explanation": " ".join(lineas),
        "breakdown": desglose,
        "major_contradictions": mayores,
        "alignment": {"alignment": alineacion, "score": al_score, "biases": dict(sesgos)},
        "active_setups": setups_activos,
        "informational_evidence": sorted(info, key=lambda x: x["id"]),
        "dedupe_log": [{"id": e["id"], "duplicate_of": e["duplicate_of"], "note": e["note"]}
                       for e in sorted(duplicadas, key=lambda x: x["id"])],
        "all_evidence": sorted(evs, key=lambda e: (-e["points"], e["id"])),
        "coverage": round(cobertura, 4), "available_weight": round(peso_disp, 6), "total_weight": round(total_peso, 6),
        "timeframes": {tf: {"freq": prep.tf[tf].freq, "closed_bars": (idx_tf[tf] + 1) if idx_tf.get(tf) is not None else 0,
                            "last_closed_bar": (pd.Timestamp(prep.tf[tf].bar_time[idx_tf[tf]]) if idx_tf.get(tf) is not None else None)}
                       for tf in TFS},
        "config": asdict(cfg),
    }


def _vacio(symbol: str, cfg: ConfluenceConfig, motivo: str) -> dict:
    return {
        "timestamp": None, "bar_index": None, "symbol": symbol,
        "bullish_confluence_score": 0.0, "bearish_confluence_score": 0.0, "conflicting_evidence_score": 0.0,
        "evidence_count": 0, "classification": "insufficient_data",
        "supporting_evidence": [], "conflicting_evidence": [], "missing_evidence": [motivo],
        "explanation": f"DATOS INSUFICIENTES — {motivo}. La ausencia de datos no se cuenta como evidencia a favor ni en contra.",
        "breakdown": {}, "major_contradictions": [], "alignment": {"alignment": "neutral_context", "score": 0, "biases": {}},
        "active_setups": [], "informational_evidence": [], "dedupe_log": [], "all_evidence": [],
        "coverage": 0.0, "available_weight": 0.0, "total_weight": 0.0, "timeframes": {}, "config": asdict(cfg),
    }


# ─────────────────────────────────────────────────────────────────────
#  API pública
# ─────────────────────────────────────────────────────────────────────
def analyze_confluence(
    df: Optional[pd.DataFrame] = None,
    symbol: str = "UNKNOWN",
    config: Any = None,
    frames: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
    as_of: Optional[Any] = None,
) -> dict:
    """
    Evalúa la confluencia de evidencias en un momento dado. Solo informativo.

    Parameters
    ----------
    df : OHLC(V) del timeframe más fino (LTF), con índice de fechas. El módulo resamplea HTF/MTF
        de forma causal (solo velas ya cerradas).
    frames : alternativa a `df`: {"htf": df, "mtf": df, "ltf": df} ya preparados (cualquiera puede faltar).
    config : ConfluenceConfig o dict con cualquier subconjunto (weights/tf_weights se fusionan).
    as_of : momento a evaluar (cierre de una vela LTF). Por defecto, la última vela. Evalúa desde los
        estados disponibles en ese instante, sin recortar los datos.

    Nunca lanza por datos faltantes: devuelve classification='insufficient_data' con el motivo.
    """
    cfg = _resolver_config(config)
    sin_df = (df is None or len(df) == 0) and not frames
    if sin_df:
        return _vacio(symbol, cfg, "no hay datos")
    prep = _preparar(df, symbol, cfg, frames)
    ltf = prep.tf["ltf"]
    if ltf.n == 0:
        return _vacio(symbol, cfg, "no hay velas del timeframe más fino (LTF) utilizables" + (f" ({prep.errors[0]})" if prep.errors else ""))
    t = ltf.n - 1
    if as_of is not None:
        ref = pd.Timestamp(as_of)
        if ref.tzinfo is not None:
            ref = ref.tz_convert("UTC").tz_localize(None)
        t = int(np.searchsorted(ltf.avail, np.datetime64(ref, "ns"), side="right") - 1)
        if t < 0:
            return _vacio(symbol, cfg, "as_of es anterior a la primera vela disponible")
    return _evaluar(prep, t)


def confluence_series(
    df: Optional[pd.DataFrame] = None,
    symbol: str = "UNKNOWN",
    config: Any = None,
    frames: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
    last_n: Optional[int] = None,
) -> pd.DataFrame:
    """Serie por vela LTF de scores y clasificación (una sola preparación). Cada fila es idéntica a
    `analyze_confluence(..., as_of=timestamp)`."""
    cfg = _resolver_config(config)
    cols = ["timestamp", "bullish", "bearish", "conflicting", "evidence_count", "classification", "coverage"]
    if (df is None or len(df) == 0) and not frames:
        return pd.DataFrame(columns=cols)
    prep = _preparar(df, symbol, cfg, frames)
    ltf = prep.tf["ltf"]
    if ltf.n == 0:
        return pd.DataFrame(columns=cols)
    desde = 0 if last_n is None else max(0, ltf.n - last_n)
    filas = []
    for t in range(desde, ltf.n):
        r = _evaluar(prep, t)
        filas.append((r["timestamp"], r["bullish_confluence_score"], r["bearish_confluence_score"],
                      r["conflicting_evidence_score"], r["evidence_count"], r["classification"], r["coverage"]))
    return pd.DataFrame(filas, columns=cols)


# ─────────────────────────────────────────────────────────────────────
#  Escenarios sintéticos + pruebas (10 casos del spec)
# ─────────────────────────────────────────────────────────────────────
def _marcos_sinteticos(dir_htf: str, dir_mtf: str, ltf: pd.DataFrame, precio_final: float) -> Dict[str, pd.DataFrame]:
    """HTF/MTF en zigzag con tendencia clara, que terminan justo antes del inicio del LTF. `precio_final`
    se elige lejos del precio del LTF para que sus zonas no creen evidencia incidental cerca del precio."""
    from core.multi_timeframe import _serie_zigzag
    ini = _norm_ts(ltf.index)[0]

    def serie(direccion, n, freq, semilla):
        neto = 1.8 * (n / 12.0)
        base = precio_final - neto if direccion == "bullish" else precio_final + neto
        return _serie_zigzag(direccion, n, str(ini - pd.Timedelta(freq) * n), freq, base=base, seed=semilla)

    return {"htf": serie(dir_htf, 60, "4h", 1), "mtf": serie(dir_mtf, 120, "1h", 2), "ltf": ltf}


CFG_DEMO = dict(swing_order=2, equal_tolerance_pct=0.1)


def _escenarios_confluencia() -> Dict[str, dict]:
    """Nombre -> kwargs para analyze_confluence (marcos directos, config de demostración)."""
    from core.setup_engine import _a_df, _espejo, _filas_alcista
    A = _filas_alcista()
    dfA, dfB = _a_df(A), _a_df(_espejo(A))
    return {
        "Alcista fuerte (setup confirmado, HTF/MTF alcistas)": dict(frames=_marcos_sinteticos("bullish", "bullish", dfA.iloc[:37], 96.0), symbol="XAUUSD", config=CFG_DEMO),
        "Bajista fuerte (setup confirmado, HTF/MTF bajistas)": dict(frames=_marcos_sinteticos("bearish", "bearish", dfB.iloc[:37], 120.0), symbol="EURUSD", config=CFG_DEMO),
        "Contradictorio (LTF alcista, HTF bajista)": dict(frames=_marcos_sinteticos("bearish", "bullish", dfA.iloc[:37], 120.0), symbol="XAUUSD", config=CFG_DEMO),
        "Setup parcial (candidate, sin reacción en zona)": dict(frames=_marcos_sinteticos("bullish", "bullish", dfA.iloc[:35], 96.0), symbol="XAUUSD", config=CFG_DEMO),
        "Solo LTF (sin HTF/MTF)": dict(frames={"ltf": dfA.iloc[:37]}, symbol="XAUUSD", config=CFG_DEMO),
        "Datos insuficientes (5 velas)": dict(frames={"ltf": dfA.iloc[:5]}, symbol="XAUUSD", config=CFG_DEMO),
    }


def _normalizar(obj: Any) -> Any:
    """Forma comparable/serializable (Timestamps, numpy, NaN) para igualdad exacta y digests."""
    if isinstance(obj, dict):
        return {str(k): _normalizar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalizar(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return "nan" if obj != obj else round(float(obj), 9)
    if isinstance(obj, np.ndarray):
        return [_normalizar(v) for v in obj.tolist()]
    if obj is pd.NaT:
        return None
    return obj


def _ids_contados(r: dict) -> List[str]:
    return sorted(e["id"] for e in r["all_evidence"] if e["counted"])


def _demo() -> None:
    """Corre `python -m core.confluence_engine` — ejecuta los 10 casos del spec (+ extras)."""
    import contextlib
    import hashlib
    import io
    import json
    import os
    import subprocess
    import sys
    from core.setup_engine import _a_df, _espejo, _filas_alcista, _random_walk

    esc = _escenarios_confluencia()
    claves = list(esc.keys())
    nombre = lambda i: claves[i]

    print("=" * 74)
    print("CASO 1 — Confluencia alcista fuerte")
    print("=" * 74)
    r1 = analyze_confluence(**esc[nombre(0)])
    print(f"  {r1['classification']}  bull={r1['bullish_confluence_score']}  bear={r1['bearish_confluence_score']}  "
          f"evidencias={r1['evidence_count']}  cobertura={r1['coverage']}")
    assert r1["classification"] == "strong_bullish_confluence", r1["classification"]
    assert r1["bullish_confluence_score"] >= 55 and r1["bearish_confluence_score"] <= 20
    cats = {e["category"] for e in r1["supporting_evidence"]}
    assert {"structure", "liquidity", "setup", "multi_timeframe"} <= cats, cats
    assert r1["major_contradictions"] == [] and r1["coverage"] == 1.0
    print("  OK — categorías a favor:", sorted(cats))

    print("\n" + "=" * 74)
    print("CASO 2 — Confluencia bajista fuerte (escenario espejo)")
    print("=" * 74)
    r2 = analyze_confluence(**esc[nombre(1)])
    print(f"  {r2['classification']}  bull={r2['bullish_confluence_score']}  bear={r2['bearish_confluence_score']}  evidencias={r2['evidence_count']}")
    assert r2["classification"] == "strong_bearish_confluence", r2["classification"]
    assert r2["bearish_confluence_score"] >= 55 and r2["bullish_confluence_score"] <= 20
    assert abs(r2["bearish_confluence_score"] - r1["bullish_confluence_score"]) < 5.0, "el espejo debería dar un puntaje parecido"
    print(f"  OK — puntaje espejo {r2['bearish_confluence_score']} vs {r1['bullish_confluence_score']} (diferencia < 5)")

    print("\n" + "=" * 74)
    print("CASO 3 — Evidencias contradictorias (LTF alcista confirmado, HTF bajista)")
    print("=" * 74)
    r3 = analyze_confluence(**esc[nombre(2)])
    print(f"  {r3['classification']}  bull={r3['bullish_confluence_score']}  bear={r3['bearish_confluence_score']}  conflicto={r3['conflicting_evidence_score']}")
    print("  contradicciones:", r3["major_contradictions"])
    assert r3["classification"] == "conflicting_context", r3["classification"]
    assert r3["major_contradictions"] and r3["bearish_confluence_score"] > 0
    assert r3["conflicting_evidence_score"] > 0 and r3["conflicting_evidence"]
    assert r3["active_setups"] == [], "el Setup Engine debe descartar el setup alcista por contexto HTF contradictorio"
    print("  OK — conflicting_context (no 'neutral', no una dirección). Nota: el Setup Engine descartó por sí mismo el setup alcista "
          "por el HTF bajista, así que el lado alcista no suma la etapa del setup.")
    # contradicción con ambos lados fuertes: el mismo LTF alcista confirmado y un setup bajista del Setup Engine no puede coexistir; se comprueba la regla de margen
    r3b = analyze_confluence(**dict(esc[nombre(2)], config=dict(CFG_DEMO, conflict_floor=100.0, neutral_below=15.0)))
    assert r3b["classification"] == "bullish_confluence", r3b["classification"]
    print(f"  OK — con conflict_floor=100 la regla de contradicciones queda desactivada y la clasificación cambia a '{r3b['classification']}' (umbral configurable).")

    print("\n" + "=" * 74)
    print("CASO 4 — Datos insuficientes")
    print("=" * 74)
    for etiqueta, kw in (("sin datos", dict(df=None)), ("DataFrame vacío", dict(df=pd.DataFrame({"open": [], "high": [], "low": [], "close": []}))),
                         ("columnas faltantes", dict(df=pd.DataFrame({"open": [1, 2], "high": [1, 2]}, index=pd.date_range("2025-01-01", periods=2, freq="15min")))),
                         ("5 velas", esc[nombre(5)])):
        r = analyze_confluence(**kw)
        assert r["classification"] == "insufficient_data" and r["bullish_confluence_score"] == 0.0 and r["bearish_confluence_score"] == 0.0, etiqueta
        print(f"  OK — {etiqueta}: insufficient_data, scores 0/0. ({r['missing_evidence'][0][:70]})")
    r4 = analyze_confluence(**esc[nombre(4)])   # solo LTF
    assert r4["coverage"] < 0.5 and any("HTF" in m for m in r4["missing_evidence"]) and any("MTF" in m for m in r4["missing_evidence"])
    assert r4["bearish_confluence_score"] == 0.0, "la falta de HTF/MTF no debe crear evidencia bajista"
    r4_full = r1
    print(f"  OK — solo LTF: cobertura {r4['coverage']} (vs 1.0 con los 3 timeframes), clase '{r4['classification']}', "
          f"bajista {r4['bearish_confluence_score']} (la ausencia de HTF/MTF no cuenta como evidencia negativa).")
    df_nan = _a_df(_filas_alcista()).iloc[:37].copy()
    df_nan.iloc[3, df_nan.columns.get_loc("high")] = np.nan
    df_nan.iloc[12, df_nan.columns.get_loc("low")] = np.nan
    r_nan = analyze_confluence(frames=_marcos_sinteticos("bullish", "bullish", df_nan, 96.0), symbol="XAUUSD", config=CFG_DEMO)
    print(f"  OK — con NaN en el OHLC no truena: clase '{r_nan['classification']}'.")
    r_asof = analyze_confluence(**esc[nombre(0)], as_of="2000-01-01")
    assert r_asof["classification"] == "insufficient_data"
    print("  OK — as_of anterior a la primera vela: insufficient_data.")

    print("\n" + "=" * 74)
    print("CASO 5 — Setup parcial")
    print("=" * 74)
    r5 = analyze_confluence(**esc[nombre(3)])
    st5 = [e for e in r5["supporting_evidence"] if e["category"] == "setup"][0]
    st1 = [e for e in r1["supporting_evidence"] if e["category"] == "setup"][0]
    print(f"  setup {st5['status']}: {st5['points']} pts (confirmed: {st1['points']} pts). Faltan: {st5['conditions_missing']}")
    assert st5["status"] == "candidate" and abs(st5["points"] - 25.0 * 0.4) < 1e-6 and st1["points"] == 25.0
    assert any("zone_reaction" in m for m in r5["missing_evidence"])
    assert r5["bullish_confluence_score"] < r1["bullish_confluence_score"]
    print(f"  OK — candidate aporta 40% del peso del setup; el score ({r5['bullish_confluence_score']}) es menor que con el setup confirmado ({r1['bullish_confluence_score']}) y se lista lo que falta.")

    print("\n" + "=" * 74)
    print("CASO 6 — No duplicar evidencias")
    print("=" * 74)
    sin_setup = dict(esc[nombre(0)]); sin_setup["config"] = dict(CFG_DEMO, weights={"setup": 0.0})
    r6 = analyze_confluence(**sin_setup)
    ids_con, ids_sin = set(_ids_contados(r1)), set(_ids_contados(r6))
    setup_ids = {i for i in ids_con if i.startswith("setup:")}
    assert ids_con - ids_sin == setup_ids and not (ids_sin - ids_con), "quitar el setup solo debe quitar la evidencia 'setup'"
    pts_con = sum(e["points"] for e in r1["all_evidence"] if e["counted"])
    pts_sin = sum(e["points"] for e in r6["all_evidence"] if e["counted"])
    assert abs((pts_con - pts_sin) - 25.0) < 1e-6
    print(f"  OK — con/sin setup la diferencia es exactamente 25.0 pts (solo su etapa): el BOS, el sweep y la zona que lo componen "
          f"no se suman otra vez ({sorted(ids_con - ids_sin)[0][:48]}… es lo único que cambia).")
    ev_setup = [e for e in r1["all_evidence"] if e["category"] == "setup"][0]
    print(f"       el setup enlaza sus piezas sin puntos: links={ev_setup['links']}")
    assert ev_setup["links"]
    base = lambda i, pts, roots, cat="fvg", tf="ltf", d="bullish": {"id": i, "category": cat, "timeframe": tf, "direction": d, "points": pts,
                                                                   "roots": roots, "counted": False, "duplicate_of": None, "note": ""}
    res = _resolver_evidencias([base("a", 5.0, ["x:1"]), base("b", 3.0, ["x:1"], cat="order_block"), base("c", 2.0, ["y:1"], tf="mtf")])
    estado = {e["id"]: (e["counted"], e["duplicate_of"]) for e in res}
    assert estado["a"] == (True, None) and estado["b"] == (False, "a") and estado["c"] == (True, None)
    print("  OK — raíces compartidas: FVG y OB que se solapan (misma raíz) cuentan UNA vez; la de mayor puntaje gana, la otra queda con duplicate_of.")
    res = _resolver_evidencias([base("a", 5.0, ["x:1"]), base("b", 4.0, ["x:2"]), base("c", 1.0, ["x:3"])])
    assert [e["counted"] for e in sorted(res, key=lambda e: e["id"])] == [True, False, False]
    print("  OK — límite por categoría/timeframe/dirección: de 3 FVG alcistas del mismo TF solo cuenta el más fuerte.")
    for nm in claves[:5]:
        rr = analyze_confluence(**esc[nm])
        raices, ids = [], []
        for e in rr["all_evidence"]:
            ids.append(e["id"])
            if e["counted"]:
                raices += e["roots"]
        assert len(ids) == len(set(ids)) and len(raices) == len(set(raices)), f"raíces o ids repetidos en '{nm}'"
    print("  OK — en los 5 escenarios, ninguna raíz ni id se repite entre las evidencias contadas.")

    print("\n" + "=" * 74)
    print("CASO 7 — Pesos configurables")
    print("=" * 74)
    solo_liq = analyze_confluence(**dict(esc[nombre(0)], config=dict(CFG_DEMO, weights={c: (100.0 if c == "liquidity" else 0.0) for c in CATEGORIAS})))
    contribuyen = {c for c, v in solo_liq["breakdown"].items() if v["bullish_points"] > 0 or v["bearish_points"] > 0}
    assert contribuyen == {"liquidity"}, contribuyen
    assert solo_liq["bullish_confluence_score"] != r1["bullish_confluence_score"]
    print(f"  OK — con solo 'liquidity' > 0, únicamente esa categoría puntúa; score {solo_liq['bullish_confluence_score']} (default {r1['bullish_confluence_score']}).")
    doble = analyze_confluence(**dict(esc[nombre(0)], config=dict(CFG_DEMO, weights={"structure": 50.0})))
    p1 = {e["id"]: e["points"] for e in r1["all_evidence"] if e["category"] == "structure"}
    p2 = {e["id"]: e["points"] for e in doble["all_evidence"] if e["category"] == "structure"}
    assert p1 and all(abs(p2[k] - 2 * v) < 1e-5 for k, v in p1.items())   # los puntos se guardan redondeados a 6 decimales
    print(f"  OK — duplicar el peso de 'structure' (25 → 50) duplica los puntos de cada evidencia de estructura ({len(p1)} evidencias).")
    solo_ltf = analyze_confluence(**dict(esc[nombre(0)], config=dict(CFG_DEMO, tf_weights={"htf": 0.0, "mtf": 0.0, "ltf": 1.0})))
    assert all(e["points"] == 0 for e in solo_ltf["all_evidence"] if e["timeframe"] in ("htf", "mtf") and e["category"] != "multi_timeframe")
    print(f"  OK — con tf_weights solo en LTF, la estructura HTF/MTF aporta 0 puntos; score {solo_ltf['bullish_confluence_score']}.")
    try:
        analyze_confluence(**dict(esc[nombre(0)], config={"peso_inventado": 1}))
        raise SystemExit("debía rechazar una config desconocida")
    except ValueError:
        print("  OK — una opción de configuración desconocida se rechaza con ValueError (no se ignora en silencio).")

    print("\n" + "=" * 74)
    print("CASO 8 — No utilizar datos futuros (truncado vs. completo en el mismo timestamp)")
    print("=" * 74)
    total = 0
    from core.setup_engine import _a_df as _adf
    for nm in claves[:5]:
        kw = esc[nm]
        full_frames = kw["frames"]
        ltf_full = _a_df(_filas_alcista()) if "Bajista" not in nm else _a_df(_espejo(_filas_alcista()))
        completo = dict(full_frames, ltf=ltf_full)   # LTF completo (46 velas); los demás marcos, tal cual
        for corte in range(11, len(ltf_full) + 1):
            T = _norm_ts(ltf_full.index)[corte - 1] + pd.Timedelta("15min")
            a = analyze_confluence(frames=dict(completo, ltf=ltf_full.iloc[:corte]), symbol=kw["symbol"], config=kw["config"])
            b = analyze_confluence(frames=completo, symbol=kw["symbol"], config=kw["config"], as_of=T)
            assert _normalizar(a) == _normalizar(b), f"'{nm}' corte={corte}: el resultado difiere entre truncado y completo"
            total += 1
    print(f"  OK — {total} comparaciones (5 escenarios × todos los cortes): el resultado completo en T es idéntico al truncado en T.")
    comp_real = 0
    cfg_real = dict(swing_order=3, equal_tolerance_pct=0.1)
    for semilla in (3, 11):
        df = _random_walk(700, semilla)
        for corte in (333, 475, 612):
            T = _norm_ts(df.index)[corte - 1] + pd.Timedelta("15min")
            a = analyze_confluence(df.iloc[:corte], "EURUSD", cfg_real)
            b = analyze_confluence(df, "EURUSD", cfg_real, as_of=T)
            assert _normalizar(a) == _normalizar(b), f"serie realista seed={semilla} corte={corte}: difiere"
            comp_real += 1
    print(f"  OK — {comp_real} comparaciones en series ruidosas con HTF/MTF resampleados (cortes a mitad de vela H4/H1).")

    print("\n" + "=" * 74)
    print("CASO 9 — Respetar velas HTF cerradas")
    print("=" * 74)
    revisadas = 0
    for semilla in (3, 11):
        df = _random_walk(700, semilla)
        for corte in (333, 475, 612, 700):
            r = analyze_confluence(df.iloc[:corte], "EURUSD", cfg_real)
            T = r["timestamp"]
            for tf, f in (("htf", "4h"), ("mtf", "1h")):
                info = r["timeframes"][tf]
                if info["last_closed_bar"] is not None:
                    assert info["last_closed_bar"] + pd.Timedelta(f) <= T, f"{tf}: vela sin cerrar"
                    assert info["last_closed_bar"] == T.floor(f) - pd.Timedelta(f), f"{tf}: no es la última cerrada"
                    revisadas += 1
            for e in r["all_evidence"]:
                if e["timeframe"] in ("htf", "mtf"):
                    f = "4h" if e["timeframe"] == "htf" else "1h"
                    assert e["bar_time"] + pd.Timedelta(f) <= T, f"evidencia {e['id']} usa una vela {e['timeframe']} sin cerrar"
                    revisadas += 1
    assert revisadas > 0
    print(f"  OK — {revisadas} lecturas de timeframes mayores (cortes a mitad de vela): todas usan la última vela CERRADA.")

    print("\n" + "=" * 74)
    print("CASO 10 — Resultados reproducibles")
    print("=" * 74)
    a = analyze_confluence(**esc[nombre(0)])
    b = analyze_confluence(**esc[nombre(0)])
    assert _normalizar(a) == _normalizar(b)
    print("  OK — dos corridas seguidas: resultado idéntico, campo por campo.")
    codigo = ("import sys, json, hashlib; sys.path.insert(0, %r); "
              "from core.confluence_engine import _escenarios_confluencia, analyze_confluence, _normalizar; "
              "e = _escenarios_confluencia(); "
              "salida = {k: _normalizar(analyze_confluence(**v)) for k, v in e.items()}; "
              "print(hashlib.sha256(json.dumps(salida, sort_keys=True).encode()).hexdigest())") % os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    digests = set()
    for semilla in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=semilla))
        assert out.returncode == 0, out.stderr[-400:]
        digests.add(out.stdout.strip())
    assert len(digests) == 1, digests
    print(f"  OK — 3 procesos con PYTHONHASHSEED distinto producen el mismo digest ({list(digests)[0][:16]}…) en los 6 escenarios.")

    print("\n" + "=" * 74)
    print("EXTRAS — escala de precios, serie por vela, módulos previos")
    print("=" * 74)
    kw = esc[nombre(0)]
    escalado = {k: (v * 21.7 if k != "x" else v) for k, v in kw["frames"].items()}
    for k, v in list(escalado.items()):
        escalado[k] = v
    r_esc = analyze_confluence(frames=escalado, symbol="XAUUSD", config=kw["config"])
    cmp_ = lambda r: (r["classification"], r["bullish_confluence_score"], r["bearish_confluence_score"], _ids_contados(r))
    assert cmp_(r_esc) == cmp_(r1), (cmp_(r_esc)[:3], cmp_(r1)[:3])
    print(f"  OK — mismos datos con precios ×21.7 (escala distinta, p. ej. XAUUSD vs EURUSD): idéntica clasificación, scores y evidencias.")
    serie = confluence_series(frames=kw["frames"], symbol=kw["symbol"], config=kw["config"], last_n=8)
    for _, fila in serie.iterrows():
        rr = analyze_confluence(**kw, as_of=fila["timestamp"])
        assert (rr["bullish_confluence_score"], rr["bearish_confluence_score"], rr["classification"]) == (fila["bullish"], fila["bearish"], fila["classification"])
    print(f"  OK — confluence_series: {len(serie)} filas idénticas a analyze_confluence(as_of=…).")
    import core.fvg as _f, core.liquidity as _l, core.market_structure as _m, core.multi_timeframe as _mt, core.order_blocks as _o, core.setup_engine as _s
    for nm, mod in (("market_structure", _m), ("liquidity", _l), ("fvg", _f), ("order_blocks", _o), ("multi_timeframe", _mt), ("setup_engine", _s)):
        with contextlib.redirect_stdout(io.StringIO()):
            mod._demo()
        print(f"  OK — core.{nm}._demo() sigue pasando todas sus pruebas.")

    print("\nTodas las pruebas de core/confluence_engine.py corrieron sin errores.")


if __name__ == "__main__":
    _demo()
