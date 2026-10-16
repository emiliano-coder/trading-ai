"""
core/multi_timeframe.py
─────────────────────────────────────────────────────────────────────
Multi-Timeframe Context Engine — independiente y reutilizable.

Analiza el contexto de varios timeframes (por defecto HTF=H4, MTF=H1,
LTF=M15, todos configurables) y determina si hay ALINEACIÓN o CONFLICTO
entre ellos. En cada timeframe reutiliza, sin duplicar sus algoritmos:

  - core.market_structure  → tendencia, último BOS, último CHoCH
  - core.liquidity         → EQH/EQL, sweeps, liquidez activa
  - core.fvg               → FVG activos / mitigados
  - core.order_blocks      → OB activos, fresh / touched / mitigated

No decide señales BUY/SELL/WAIT. `alignment_score` describe el CONTEXTO,
nunca genera operaciones. Corre en paralelo a la estrategia existente.

─────────────────────────────────────────────────────────────────────
DOS MODOS DE ENTRADA
─────────────────────────────────────────────────────────────────────
  1) `df` (datos del timeframe más fino, LTF): el módulo resamplea él
     mismo hacia MTF y HTF, de forma causal (ver abajo).
  2) `df_htf`, `df_mtf`, `df_ltf`: tres DataFrames ya preparados (por
     ejemplo, descargados por separado). No se hace resampling.

─────────────────────────────────────────────────────────────────────
SINCRONIZACIÓN TEMPORAL Y LOOK-AHEAD BIAS
─────────────────────────────────────────────────────────────────────
Una vela etiquetada a las 12:00 en H4 abarca 12:00-16:00 y su OHLC final
solo se CONOCE a las 16:00. Por eso cada vela de cada timeframe tiene un
`available_at = etiqueta + duración del timeframe` (su hora de cierre).

  - Resampling causal: un bucket de HTF/MTF solo se acepta si su hora de
    cierre es <= la hora de cierre de la última vela base (LTF). Los
    buckets incompletos al final se descartan: nunca se trata una vela
    "en progreso" como si ya hubiera cerrado.
  - Alineación temporal: el estado de HTF/MTF se pega a cada vela LTF con
    `pd.merge_asof(..., direction="backward")` sobre `available_at`:
    cada vela LTF solo ve la última vela HTF/MTF que YA HABÍA CERRADO en
    ese momento. Nunca una que todavía estaba formándose.
  - Cada motor interno ya es causal (ver sus docstrings): su estado en la
    vela `i` solo depende de datos hasta `i`.

Para responder "¿qué sabía MIMI en este momento?": busca la fila por
`timestamp` (momento de cierre de la vela LTF evaluada). Las columnas
`htf_bar_time` / `mtf_bar_time` dicen QUÉ vela de HTF/MTF era la vigente.

Sobre created_at / confirmed_at: cada fila describe una lectura que ya es
final en su `timestamp` (los retrasos de confirmación de swings, OB, etc.
ya están absorbidos dentro de cada motor). En el summary, `created_at` y
`confirmed_at` son el `timestamp` de la última lectura.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.fvg import analyze_fvg
from core.liquidity import analyze_liquidity
from core.market_structure import _validar_columnas, analyze_market_structure
from core.order_blocks import analyze_order_blocks

TFS = ("htf", "mtf", "ltf")
PESOS_DEFAULT = {"htf": 0.5, "mtf": 0.3, "ltf": 0.2}
ALIGNMENTS = (
    "strong_bullish_alignment", "strong_bearish_alignment",
    "bullish_bias", "bearish_bias", "mixed_context", "neutral_context",
)


# ─────────────────────────────────────────────────────────────────────
#  Utilidades de tiempo / resampling
# ─────────────────────────────────────────────────────────────────────
def _asegurar_datetimeindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError("El índice debe ser de fechas (DatetimeIndex) para analizar múltiples timeframes.") from e
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _norm_ts(idx) -> pd.DatetimeIndex:
    """Normaliza a UTC sin zona, para que merge_asof compare peras con peras."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _resamplear_causal(df_base: pd.DataFrame, freq: str, base_freq: str) -> pd.DataFrame:
    """Resamplea OHLC(V) a un timeframe mayor, descartando los buckets que
    todavía no han cerrado respecto a la última vela base disponible."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df_base.columns:
        agg["volume"] = "sum"
    res = df_base.resample(freq).agg(agg)
    res = res.dropna(subset=["open", "high", "low", "close"])
    if res.empty:
        return res
    fin_base = _norm_ts(df_base.index)[-1] + pd.Timedelta(base_freq)
    cierre_bucket = _norm_ts(res.index) + pd.Timedelta(freq)
    return res[cierre_bucket <= fin_base]


# ─────────────────────────────────────────────────────────────────────
#  Estado por vela de un timeframe (reusa los 4 motores)
# ─────────────────────────────────────────────────────────────────────
def _rolling_last(bull: np.ndarray, bear: np.ndarray) -> np.ndarray:
    """Convierte eventos (bos_bullish/bos_bearish, etc.) en un estado
    acumulado: la dirección del último evento conocido hasta cada vela."""
    out = np.full(len(bull), None, dtype=object)
    actual = None
    for i in range(len(bull)):
        if bull[i]:
            actual = "bullish"
        elif bear[i]:
            actual = "bearish"
        out[i] = actual
    return out


def _dir_str(nb: int, ns: int) -> Optional[str]:
    if nb > 0 and ns > 0:
        return "both"
    if nb > 0:
        return "bullish"
    if ns > 0:
        return "bearish"
    return None


def _conteos_activos(n: int, zonas: List[dict], inicio: str, fin_keys: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Cuántas zonas bullish/bearish están vivas en CADA vela, de forma
    exacta y causal: viva desde `inicio` hasta (sin incluir) la primera
    vela en que se mitiga o invalida."""
    bull = np.zeros(n, dtype=int)
    bear = np.zeros(n, dtype=int)
    for z in zonas:
        ini = z[inicio]
        fin = n
        for k in fin_keys:
            v = z.get(k)
            if v is not None:
                fin = min(fin, v)
        if ini < fin:
            (bull if z["kind"] == "bullish" else bear)[ini:fin] += 1
    return bull, bear


def _analizar_tf(df: pd.DataFrame, freq: str, p: dict) -> dict:
    """Corre los 4 motores sobre un timeframe y arma su estado por vela."""
    n = len(df)
    errores: List[str] = []
    ms_df = liq_df = fvg_df = ob_df = None
    ms_sum = liq_sum = fvg_sum = ob_sum = None

    try:
        ms_df, ms_sum = analyze_market_structure(df, swing_order=p["swing_order"], break_using=p["break_using"])
    except Exception as e:  # un motor que falla no tumba el análisis
        errores.append(f"market_structure: {e}")
    try:
        liq_df, liq_sum = analyze_liquidity(df, market_structure_df=ms_df, swing_order=p["swing_order"],
                                            break_using=p["break_using"], equal_tolerance_pct=p["equal_tolerance_pct"])
    except Exception as e:
        errores.append(f"liquidity: {e}")
    try:
        fvg_df, fvg_sum = analyze_fvg(df, mitigation_criterion=p["mitigation_criterion"])
    except Exception as e:
        errores.append(f"fvg: {e}")
    try:
        ob_df, ob_sum = analyze_order_blocks(df, market_structure_df=ms_df, liquidity_df=liq_df, fvg_df=fvg_df,
                                             displacement_atr_mult=p["displacement_atr_mult"],
                                             mitigation_criterion=p["mitigation_criterion"])
    except Exception as e:
        errores.append(f"order_blocks: {e}")

    nada = np.full(n, None, dtype=object)
    trend = ms_df["structure_trend"].to_numpy(dtype=object) if ms_df is not None else np.full(n, "ranging", dtype=object)
    bos = _rolling_last(ms_df["bos_bullish"].to_numpy(bool), ms_df["bos_bearish"].to_numpy(bool)) if ms_df is not None else nada
    choch = _rolling_last(ms_df["choch_bullish"].to_numpy(bool), ms_df["choch_bearish"].to_numpy(bool)) if ms_df is not None else nada
    liquidity = _rolling_last(liq_df["bullish_sweep"].to_numpy(bool), liq_df["bearish_sweep"].to_numpy(bool)) if liq_df is not None else nada

    if fvg_sum is not None:
        fb, fs = _conteos_activos(n, fvg_sum["all_fvg"], "created_at", ("mitigated_at", "invalidated_at"))
        fvg_state = np.array([_dir_str(a, b) for a, b in zip(fb, fs)], dtype=object)
    else:
        fvg_state = nada
    if ob_sum is not None:
        ob_b, ob_s = _conteos_activos(n, ob_sum["all_order_blocks"], "confirmed_at", ("mitigated_at", "invalidated_at"))
        ob_state = np.array([_dir_str(a, b) for a, b in zip(ob_b, ob_s)], dtype=object)
    else:
        ob_state = nada

    idx_norm = _norm_ts(df.index)
    state = pd.DataFrame({
        "bar_time": idx_norm,
        "available_at": idx_norm + pd.Timedelta(freq),
        "trend": trend, "bos": bos, "choch": choch,
        "liquidity": liquidity, "fvg": fvg_state, "ob": ob_state,
    })

    ob_all = ob_sum["all_order_blocks"] if ob_sum else []
    fvg_all = fvg_sum["all_fvg"] if fvg_sum else []
    det = {
        "bars": n,
        "trend": ms_sum["trend"] if ms_sum else "ranging",
        "last_structure": ms_sum["last_structure"] if ms_sum else None,
        "last_bos": ms_sum["bos"] if ms_sum else None,
        "last_choch": ms_sum["choch"] if ms_sum else None,
        "active_eqh": len(liq_sum["active_eqh"]) if liq_sum else 0,
        "active_eql": len(liq_sum["active_eql"]) if liq_sum else 0,
        "last_bullish_sweep": liq_sum["last_bullish_sweep"] if liq_sum else None,
        "last_bearish_sweep": liq_sum["last_bearish_sweep"] if liq_sum else None,
        "nearest_buy_side_liquidity": liq_sum["nearest_buy_side_liquidity"] if liq_sum else None,
        "nearest_sell_side_liquidity": liq_sum["nearest_sell_side_liquidity"] if liq_sum else None,
        "fvg_active_bullish": len(fvg_sum["active_bullish_fvg"]) if fvg_sum else 0,
        "fvg_active_bearish": len(fvg_sum["active_bearish_fvg"]) if fvg_sum else 0,
        "fvg_mitigated": sum(1 for z in fvg_all if z["status"] == "mitigated"),
        "ob_active_bullish": len(ob_sum["active_bullish_obs"]) if ob_sum else 0,
        "ob_active_bearish": len(ob_sum["active_bearish_obs"]) if ob_sum else 0,
        "ob_fresh_bullish": len(ob_sum["fresh_bullish_obs"]) if ob_sum else 0,
        "ob_fresh_bearish": len(ob_sum["fresh_bearish_obs"]) if ob_sum else 0,
        "ob_touched": sum(1 for z in ob_all if z["status"] in ("touched", "partially_mitigated")),
        "ob_mitigated": sum(1 for z in ob_all if z["status"] == "fully_mitigated"),
        "errors": errores,
    }
    return {"state": state, "details": det}


# ─────────────────────────────────────────────────────────────────────
#  Alineación
# ─────────────────────────────────────────────────────────────────────
def _normalizar_pesos(pesos: Optional[Dict[str, float]]) -> Dict[str, float]:
    p = dict(PESOS_DEFAULT if pesos is None else pesos)
    for tf in TFS:
        p.setdefault(tf, 0.0)
    total = sum(max(p[tf], 0.0) for tf in TFS)
    if total <= 0:
        return dict(PESOS_DEFAULT)
    return {tf: max(p[tf], 0.0) / total for tf in TFS}


def _clasificar_alineacion(biases: Dict[str, Optional[str]], pesos: Dict[str, float]) -> Tuple[str, int]:
    """Clasifica la alineación entre HTF/MTF/LTF y calcula un score 0-100.

    Un timeframe sin dato (None) o 'ranging' no aporta peso a ninguna
    dirección. Con `bull`/`bear` = suma de pesos de los TF alcistas/bajistas:
      - bull>0 y bear>0       -> mixed_context, score = max(bull,bear)*50
                                  (el conflicto recorta el score a la mitad)
      - bull == 1 (los 3)     -> strong_bullish_alignment, score 100
      - bear == 1 (los 3)     -> strong_bearish_alignment, score 100
      - solo bull>0           -> bullish_bias,  score = bull*100
      - solo bear>0           -> bearish_bias,  score = bear*100
      - nada                  -> neutral_context, score 0
    """
    bull = sum(pesos[tf] for tf in TFS if biases.get(tf) == "bullish")
    bear = sum(pesos[tf] for tf in TFS if biases.get(tf) == "bearish")
    eps = 1e-9
    if bull > eps and bear > eps:
        return "mixed_context", int(round(max(bull, bear) * 50))
    if bull >= 1 - eps:
        return "strong_bullish_alignment", 100
    if bear >= 1 - eps:
        return "strong_bearish_alignment", 100
    if bull > eps:
        return "bullish_bias", int(round(bull * 100))
    if bear > eps:
        return "bearish_bias", int(round(bear * 100))
    return "neutral_context", 0


def _overall_bias(alignment: str) -> str:
    if alignment in ("strong_bullish_alignment", "bullish_bias"):
        return "bullish"
    if alignment in ("strong_bearish_alignment", "bearish_bias"):
        return "bearish"
    if alignment == "mixed_context":
        return "mixed"
    return "neutral"


def _confluencias(det: Dict[str, Optional[dict]], etiquetas: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Arma las listas de confluencias alcistas/bajistas y de conflictos,
    a partir del estado más reciente de cada timeframe."""
    bull, bear, conflictos = [], [], []

    def _later(a: Optional[dict], b: Optional[dict]) -> Optional[str]:
        if a and b:
            return "bullish" if a["index"] >= b["index"] else "bearish"
        if a:
            return "bullish"
        if b:
            return "bearish"
        return None

    for tf in TFS:
        d = det.get(tf)
        if not d:
            continue
        et = etiquetas[tf]
        if d["trend"] == "bullish":
            bull.append(f"{et}: tendencia alcista")
        elif d["trend"] == "bearish":
            bear.append(f"{et}: tendencia bajista")
        if d["last_bos"]:
            (bull if d["last_bos"]["direction"] == "bullish" else bear).append(f"{et}: último BOS {d['last_bos']['direction']}")
        if d["last_choch"]:
            (bull if d["last_choch"]["direction"] == "bullish" else bear).append(f"{et}: último CHoCH {d['last_choch']['direction']}")
        sw = _later(d["last_bullish_sweep"], d["last_bearish_sweep"])
        if sw == "bullish":
            bull.append(f"{et}: sweep alcista de liquidez (barrió mínimos y recuperó)")
        elif sw == "bearish":
            bear.append(f"{et}: sweep bajista de liquidez (barrió máximos y rechazó)")
        if d["fvg_active_bullish"] > 0:
            bull.append(f"{et}: {d['fvg_active_bullish']} FVG alcista(s) activo(s)")
        if d["fvg_active_bearish"] > 0:
            bear.append(f"{et}: {d['fvg_active_bearish']} FVG bajista(s) activo(s)")
        if d["ob_active_bullish"] > 0:
            bull.append(f"{et}: {d['ob_active_bullish']} OB alcista(s) activo(s) ({d['ob_fresh_bullish']} fresh)")
        if d["ob_active_bearish"] > 0:
            bear.append(f"{et}: {d['ob_active_bearish']} OB bajista(s) activo(s) ({d['ob_fresh_bearish']} fresh)")

    # conflictos: tendencias opuestas entre timeframes
    for i, a in enumerate(TFS):
        for b in TFS[i + 1:]:
            da, db = det.get(a), det.get(b)
            if da and db and {da["trend"], db["trend"]} == {"bullish", "bearish"}:
                conflictos.append(f"{etiquetas[a]} {da['trend']} vs {etiquetas[b]} {db['trend']}")
    # conflicto: un BOS reciente del TF menor va contra la tendencia del HTF
    dh = det.get("htf")
    if dh and dh["trend"] in ("bullish", "bearish"):
        for tf in ("mtf", "ltf"):
            d = det.get(tf)
            if d and d["last_bos"] and d["last_bos"]["direction"] != dh["trend"]:
                conflictos.append(f"{etiquetas[tf]} BOS {d['last_bos']['direction']} contra la tendencia {dh['trend']} de {etiquetas['htf']}")
    return bull, bear, conflictos


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def _salida_vacia(columnas: List[str], insuficientes: List[str], freqs: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    resumen = {
        "timestamp": None, "created_at": None, "confirmed_at": None, "timeframes": freqs,
        "htf_bias": None, "mtf_bias": None, "ltf_bias": None,
        "overall_bias": "neutral", "alignment": "neutral_context", "alignment_score": 0,
        "bullish_confluences": [], "bearish_confluences": [], "conflicts": [],
        "insufficient_timeframes": insuficientes, "per_timeframe": {},
    }
    return pd.DataFrame(columns=columnas), resumen


def analyze_multi_timeframe(
    df: Optional[pd.DataFrame] = None,
    df_htf: Optional[pd.DataFrame] = None,
    df_mtf: Optional[pd.DataFrame] = None,
    df_ltf: Optional[pd.DataFrame] = None,
    htf_freq: str = "4h",
    mtf_freq: str = "1h",
    ltf_freq: str = "15min",
    swing_order: int = 5,
    break_using: str = "close",
    equal_tolerance_pct: float = 0.05,
    mitigation_criterion: str = "50%",
    displacement_atr_mult: float = 1.5,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analiza el contexto multi-timeframe (HTF/MTF/LTF) sin look-ahead bias.

    Modo 1: pasa `df` (datos del LTF, a resolución `ltf_freq`) y el módulo
            resamplea hacia `mtf_freq` y `htf_freq` de forma causal.
    Modo 2: pasa `df_htf`, `df_mtf` y/o `df_ltf` ya preparados (sin resampling).
    Un timeframe sin datos (None/vacío) se reporta como insuficiente y no
    aporta dirección; el módulo no truena.

    Las frecuencias deben ser strings de pandas válidos para Timedelta y
    resample, en minúsculas: "4h", "1h", "15min", "1d"...

    Returns
    -------
    (df_out, summary)
        df_out : un renglón por vela del timeframe más fino disponible
            (índice = el de esas velas) con columnas: timestamp,
            htf_/mtf_/ltf_ + trend, bos, choch, liquidity, fvg, ob,
            htf_bar_time, mtf_bar_time, ltf_bar_time, alignment,
            alignment_score.
        summary : htf_bias, mtf_bias, ltf_bias, overall_bias, alignment,
            alignment_score, bullish_confluences, bearish_confluences,
            conflicts, insufficient_timeframes, per_timeframe, timestamp...
    """
    freqs = {"htf": htf_freq, "mtf": mtf_freq, "ltf": ltf_freq}
    pesos = _normalizar_pesos(weights)
    p = {"swing_order": swing_order, "break_using": break_using, "equal_tolerance_pct": equal_tolerance_pct,
         "mitigation_criterion": mitigation_criterion, "displacement_atr_mult": displacement_atr_mult}
    columnas_salida = ["timestamp"]
    for tf in TFS:
        columnas_salida += [f"{tf}_{c}" for c in ("trend", "bos", "choch", "liquidity", "fvg", "ob", "bar_time")]
    columnas_salida += ["alignment", "alignment_score"]

    # ── 1) obtener un DataFrame por timeframe ──────────────────────────
    frames: Dict[str, Optional[pd.DataFrame]] = {"htf": None, "mtf": None, "ltf": None}
    directos = {"htf": df_htf, "mtf": df_mtf, "ltf": df_ltf}
    if any(v is not None for v in directos.values()):
        for tf, d in directos.items():
            if d is not None and len(d) > 0:
                _validar_columnas(d)
                frames[tf] = _asegurar_datetimeindex(d)
    elif df is not None and len(df) > 0:
        _validar_columnas(df)
        base = _asegurar_datetimeindex(df)
        frames["ltf"] = base
        frames["mtf"] = _resamplear_causal(base, mtf_freq, ltf_freq)
        frames["htf"] = _resamplear_causal(base, htf_freq, ltf_freq)

    disponibles = [tf for tf in TFS if frames[tf] is not None and len(frames[tf]) > 0]
    insuficientes = [tf for tf in TFS if tf not in disponibles]
    if not disponibles:
        return _salida_vacia(columnas_salida, insuficientes, freqs)

    # ── 2) timeline base = timeframe más fino disponible ───────────────
    base_tf = next(tf for tf in ("ltf", "mtf", "htf") if tf in disponibles)
    base_df = frames[base_tf]
    base_fin = _norm_ts(base_df.index)[-1] + pd.Timedelta(freqs[base_tf])

    # velas de timeframes mayores que aún no han cerrado respecto a la base: fuera
    for tf in disponibles:
        if tf == base_tf:
            continue
        d = frames[tf]
        cierre = _norm_ts(d.index) + pd.Timedelta(freqs[tf])
        frames[tf] = d[cierre <= base_fin]
    disponibles = [tf for tf in TFS if frames[tf] is not None and len(frames[tf]) > 0]
    insuficientes = [tf for tf in TFS if tf not in disponibles]

    # ── 3) estado por vela de cada timeframe (reusa los 4 motores) ─────
    analisis = {tf: _analizar_tf(frames[tf], freqs[tf], p) for tf in disponibles}
    min_barras = 2 * swing_order + 3
    for tf in disponibles:
        if analisis[tf]["details"]["bars"] < min_barras and tf not in insuficientes:
            insuficientes.append(tf)  # se analiza igual, pero se avisa que hay pocas velas

    # ── 4) alinear en el tiempo (causal) sobre el timeline base ────────
    base_state = analisis[base_tf]["state"]
    out = pd.DataFrame(index=base_df.index)
    out["timestamp"] = base_state["available_at"].to_numpy()
    cols_estado = ["trend", "bos", "choch", "liquidity", "fvg", "ob"]
    for tf in TFS:
        if tf not in analisis:
            for c in cols_estado:
                out[f"{tf}_{c}"] = None if c != "trend" else None
            out[f"{tf}_bar_time"] = pd.NaT
            continue
        if tf == base_tf:
            for c in cols_estado:
                out[f"{tf}_{c}"] = base_state[c].to_numpy()
            out[f"{tf}_bar_time"] = base_state["bar_time"].to_numpy()
        else:
            izquierda = base_state[["available_at"]].copy()
            derecha = analisis[tf]["state"][["available_at", "bar_time"] + cols_estado].sort_values("available_at")
            fusion = pd.merge_asof(izquierda, derecha, on="available_at", direction="backward",
                                   suffixes=("", "_r"))
            for c in cols_estado:
                out[f"{tf}_{c}"] = fusion[c].to_numpy()
            out[f"{tf}_bar_time"] = fusion["bar_time"].to_numpy()

    # ── 5) alineación por vela ─────────────────────────────────────────
    alineaciones, scores = [], []
    for i in range(len(out)):
        biases = {tf: (out[f"{tf}_trend"].iat[i] if tf in analisis else None) for tf in TFS}
        al, sc = _clasificar_alineacion(biases, pesos)
        alineaciones.append(al)
        scores.append(sc)
    out["alignment"] = alineaciones
    out["alignment_score"] = scores
    out = out[columnas_salida]

    # ── 6) resumen (última lectura) ────────────────────────────────────
    ultimo = out.iloc[-1]
    biases_final = {tf: (ultimo[f"{tf}_trend"] if tf in analisis else None) for tf in TFS}
    al_final, sc_final = _clasificar_alineacion(biases_final, pesos)
    etiquetas = {tf: f"{tf.upper()} ({freqs[tf]})" for tf in TFS}
    det_por_tf = {tf: (analisis[tf]["details"] if tf in analisis else None) for tf in TFS}
    bull_c, bear_c, conf = _confluencias(det_por_tf, etiquetas)
    ts_final = ultimo["timestamp"]

    summary = {
        "timestamp": ts_final, "created_at": ts_final, "confirmed_at": ts_final,
        "timeframes": freqs,
        "htf_bias": biases_final["htf"], "mtf_bias": biases_final["mtf"], "ltf_bias": biases_final["ltf"],
        "overall_bias": _overall_bias(al_final),
        "alignment": al_final, "alignment_score": sc_final,
        "bullish_confluences": bull_c, "bearish_confluences": bear_c, "conflicts": conf,
        "insufficient_timeframes": insuficientes,
        "per_timeframe": det_por_tf,
    }
    return out, summary


# ─────────────────────────────────────────────────────────────────────
#  Datos simulados + pruebas (8 casos del spec)
# ─────────────────────────────────────────────────────────────────────
def _serie_zigzag(direction: str, n: int, start: str, freq: str, base: float = 100.0,
                  seg: int = 6, up: float = 3.0, down: float = 1.2, seed: int = 1) -> pd.DataFrame:
    """OHLC sintético en zigzag con tendencia neta clara (alcista o bajista):
    tramos de `seg` velas, avanzando `up` y retrocediendo `down` por ciclo."""
    rng = np.random.default_rng(seed)
    signo = 1.0 if direction == "bullish" else -1.0
    pts_x, pts_y = [0], [base]
    precio, x = base, 0
    while x < n - 1:
        x += seg
        precio += signo * up
        pts_x.append(min(x, n - 1)); pts_y.append(precio)
        x += seg
        precio -= signo * down
        pts_x.append(min(x, n - 1)); pts_y.append(precio)
    close = np.interp(np.arange(n), pts_x, pts_y) + rng.normal(0, 0.03, n)
    high = close + rng.uniform(0.04, 0.15, n)
    low = close - rng.uniform(0.04, 0.15, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _tres_timeframes(h: str, m: str, l: str, swing_seed: int = 1) -> Dict[str, pd.DataFrame]:
    """Tres series directas (sin resampling) que cubren el mismo tramo de tiempo."""
    inicio = "2025-05-05 00:00"
    return {
        "df_htf": _serie_zigzag(h, 48, inicio, "4h", seed=swing_seed),
        "df_mtf": _serie_zigzag(m, 192, inicio, "1h", seed=swing_seed + 1),
        "df_ltf": _serie_zigzag(l, 768, inicio, "15min", seed=swing_seed + 2),
    }


def _ltf_largo(n: int = 640, seed: int = 11) -> pd.DataFrame:
    """Serie M15 más 'real' (tendencia + ruido) para probar resampling/no-look-ahead."""
    rng = np.random.default_rng(seed)
    tendencia = np.cumsum(rng.normal(0.01, 0.12, n))
    onda = 1.5 * np.sin(np.arange(n) / 25.0)
    close = 100 + tendencia + onda
    high = close + rng.uniform(0.03, 0.2, n)
    low = close - rng.uniform(0.03, 0.2, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-05-05 00:00", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _demo() -> None:
    """Corre `python -m core.multi_timeframe` — ejecuta los 8 casos del spec."""
    P = dict(swing_order=3, equal_tolerance_pct=0.1)

    print("=" * 70)
    print("CASO 1 — HTF bullish + MTF bullish + LTF bullish")
    print("=" * 70)
    out1, s1 = analyze_multi_timeframe(**_tres_timeframes("bullish", "bullish", "bullish"), **P)
    print(f"  biases: {s1['htf_bias']}/{s1['mtf_bias']}/{s1['ltf_bias']} -> {s1['alignment']} (score {s1['alignment_score']})")
    assert (s1["htf_bias"], s1["mtf_bias"], s1["ltf_bias"]) == ("bullish", "bullish", "bullish")
    assert s1["alignment"] == "strong_bullish_alignment" and s1["alignment_score"] == 100
    assert s1["overall_bias"] == "bullish" and not s1["conflicts"]
    print("  OK — strong_bullish_alignment, sin conflictos.")

    print("\n" + "=" * 70)
    print("CASO 2 — HTF bearish + MTF bearish + LTF bearish")
    print("=" * 70)
    out2, s2 = analyze_multi_timeframe(**_tres_timeframes("bearish", "bearish", "bearish"), **P)
    print(f"  biases: {s2['htf_bias']}/{s2['mtf_bias']}/{s2['ltf_bias']} -> {s2['alignment']} (score {s2['alignment_score']})")
    assert s2["alignment"] == "strong_bearish_alignment" and s2["alignment_score"] == 100
    assert s2["overall_bias"] == "bearish" and not s2["conflicts"]
    print("  OK — strong_bearish_alignment, sin conflictos.")

    print("\n" + "=" * 70)
    print("CASO 3 — HTF bullish + MTF bearish + LTF bullish (conflicto)")
    print("=" * 70)
    out3, s3 = analyze_multi_timeframe(**_tres_timeframes("bullish", "bearish", "bullish"), **P)
    print(f"  biases: {s3['htf_bias']}/{s3['mtf_bias']}/{s3['ltf_bias']} -> {s3['alignment']} (score {s3['alignment_score']})")
    print(f"  conflictos: {s3['conflicts']}")
    assert s3["alignment"] == "mixed_context" and s3["overall_bias"] == "mixed"
    assert s3["alignment_score"] < 50 and len(s3["conflicts"]) >= 1
    print("  OK — mixed_context, score bajo y conflictos listados.")

    print("\n" + "=" * 70)
    print("CASO 4 — Timeframe con datos insuficientes")
    print("=" * 70)
    t4 = _tres_timeframes("bullish", "bullish", "bullish")
    t4["df_htf"] = t4["df_htf"].head(3)
    out4, s4 = analyze_multi_timeframe(**t4, **P)
    assert "htf" in s4["insufficient_timeframes"]
    assert s4["alignment"] != "strong_bullish_alignment", "Sin HTF suficiente no puede haber alineación fuerte"
    print(f"  OK — HTF con 3 velas: insuficiente={s4['insufficient_timeframes']}, alignment={s4['alignment']} (no truena).")
    t4b = _tres_timeframes("bullish", "bullish", "bullish")
    t4b["df_htf"] = None
    out4b, s4b = analyze_multi_timeframe(**t4b, **P)
    assert "htf" in s4b["insufficient_timeframes"] and s4b["htf_bias"] is None
    print(f"  OK — HTF ausente (None): htf_bias={s4b['htf_bias']}, alignment={s4b['alignment']}.")

    print("\n" + "=" * 70)
    print("CASO 5 — No existe look-ahead bias (invariancia al truncar el futuro)")
    print("=" * 70)
    ltf = _ltf_largo(640)
    full, _ = analyze_multi_timeframe(df=ltf, **P)
    cols_estado = [c for c in full.columns if c.endswith(("trend", "bos", "choch", "liquidity", "fvg", "ob"))]
    comprobadas = 0
    for corte in (200, 331, 480):
        trunc, _ = analyze_multi_timeframe(df=ltf.iloc[:corte], **P)
        comun = trunc.index
        a = full.loc[comun, cols_estado + ["alignment", "alignment_score"]].astype(object).where(full.loc[comun, cols_estado + ["alignment", "alignment_score"]].notna(), None)
        b = trunc[cols_estado + ["alignment", "alignment_score"]].astype(object).where(trunc[cols_estado + ["alignment", "alignment_score"]].notna(), None)
        assert a.equals(b), f"El resultado cambió al quitar velas futuras (corte={corte}) -> look-ahead"
        comprobadas += len(comun)
    print(f"  OK — con 3 cortes distintos, las {comprobadas} filas coinciden exactamente con/sin el futuro.")

    print("\n" + "=" * 70)
    print("CASO 6 — Correcta sincronización temporal")
    print("=" * 70)
    sync, _ = analyze_multi_timeframe(df=ltf, **P)
    for tf, freq in (("htf", "4h"), ("mtf", "1h")):
        barras = pd.to_datetime(sync[f"{tf}_bar_time"].dropna())
        ts = pd.to_datetime(sync.loc[barras.index, "timestamp"])
        assert (barras + pd.Timedelta(freq) <= ts).all(), f"{tf}: se usó una vela que todavía no había cerrado"
        siguiente = barras + pd.Timedelta(freq)  # la vela siguiente habría cerrado más tarde, no antes
        assert (siguiente + pd.Timedelta(freq) > ts).all(), f"{tf}: no es la vela cerrada más reciente"
    print("  OK — en TODAS las filas, la vela HTF/MTF usada ya había cerrado y es la más reciente disponible.")

    print("\n" + "=" * 70)
    print("CASO 7 — No mezclar datos futuros durante el resampling")
    print("=" * 70)
    base32 = _ltf_largo(32)
    r4 = _resamplear_causal(base32, "4h", "15min")
    assert len(r4) == 2
    primer = base32.iloc[:16]
    assert r4["open"].iloc[0] == primer["open"].iloc[0] and r4["close"].iloc[0] == primer["close"].iloc[-1]
    assert r4["high"].iloc[0] == primer["high"].max() and r4["low"].iloc[0] == primer["low"].min()
    base37 = _ltf_largo(37)
    r4b = _resamplear_causal(base37, "4h", "15min")
    r1b = _resamplear_causal(base37, "1h", "15min")
    assert len(r4b) == 2, "La vela H4 incompleta del final NO debe incluirse"
    assert len(r1b) == 9, "La vela H1 incompleta del final NO debe incluirse (9 completas de 10 buckets)"
    assert r4b["close"].iloc[-1] == r4["close"].iloc[-1], "Agregar velas parciales no debe alterar velas H4 ya cerradas"
    print("  OK — OHLC del H4 correcto; con 37 velas M15 quedan 2 H4 y 9 H1 (se descartan las incompletas).")

    print("\n" + "=" * 70)
    print("CASO 8 — Los módulos anteriores siguen funcionando sin cambios")
    print("=" * 70)
    import contextlib, io
    import core.fvg as _fvg, core.liquidity as _liq, core.market_structure as _ms, core.order_blocks as _ob
    for nombre, mod in (("market_structure", _ms), ("liquidity", _liq), ("fvg", _fvg), ("order_blocks", _ob)):
        with contextlib.redirect_stdout(io.StringIO()):
            mod._demo()
        print(f"  OK — core.{nombre}._demo() sigue pasando todas sus pruebas.")

    print("\n" + "=" * 70)
    print("ROBUSTEZ — vacío, pocas velas, sin volume, columnas faltantes")
    print("=" * 70)
    o, s = analyze_multi_timeframe(df=pd.DataFrame({"open": [], "high": [], "low": [], "close": []}))
    assert o.empty and s["alignment"] == "neutral_context"
    print("  OK — DataFrame vacío -> salida vacía y neutral_context (no truena).")
    o, s = analyze_multi_timeframe()
    assert o.empty
    print("  OK — sin ningún dato -> salida vacía.")
    o, s = analyze_multi_timeframe(df=ltf.head(5), **P)
    print(f"  OK — 5 velas: insuficientes={s['insufficient_timeframes']}, alignment={s['alignment']}.")
    assert "volume" not in ltf.columns
    print("  OK — todas las pruebas anteriores corrieron sin columna 'volume'.")
    try:
        analyze_multi_timeframe(df=pd.DataFrame({"open": [1, 2], "high": [1, 2]}, index=pd.date_range("2025-01-01", periods=2, freq="15min")))
        print("  ERROR: debía lanzar ValueError por columnas faltantes")
    except ValueError:
        print("  OK — columnas OHLC faltantes -> ValueError controlado.")

    print("\nTodas las pruebas de core/multi_timeframe.py corrieron sin errores.")


if __name__ == "__main__":
    _demo()
