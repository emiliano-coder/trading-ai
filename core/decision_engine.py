"""
core/decision_engine.py
─────────────────────────────────────────────────────────────────────
Decision Engine — decide si, EN ESTE MOMENTO, existe una oportunidad que cumpla
las condiciones configuradas, y explica por qué. Devuelve BUY, SELL o WAIT.

Es INFORMATIVO e INDEPENDIENTE:
  · NO ejecuta operaciones, NO se conecta a MT5/IBKR, NO toca el BUY/SELL/WAIT del
    sistema anterior (`generar_senal`), ni `calcular_score`, ni Telegram, ni la gestión
    de riesgo existente.
  · NO importa ni recalcula ningún otro módulo: es una función pura que lee el
    DICCIONARIO que devuelve `core.confluence_engine.analyze_confluence`.

Cómo consume los siete módulos (todos llegan a través del resultado de confluencia):
  Market Structure, Liquidity, FVG, Order Blocks → `breakdown` y evidencias por categoría
  Multi-Timeframe                               → `alignment` y `timeframes`
  Setup Engine                                  → `active_setups`
  Confluence Engine                             → scores, clasificación, cobertura, contradicciones

IMPORTANTE — qué significan los números
  · `decision_score` y `confidence_label` miden la COINCIDENCIA y CONSISTENCIA de las
    evidencias. NO son una probabilidad de éxito: un 80 no significa 80% de ganar.
  · Los umbrales de `DecisionConfig` son una elección de diseño razonable y están
    documentados, pero NO están optimizados ni validados para rentabilidad.

─────────────────────────────────────────────────────────────────────
REGLAS (todas deben cumplirse para BUY/SELL; cualquier fallo → WAIT)
─────────────────────────────────────────────────────────────────────
  1 data_sufficiency   clasificación ≠ insufficient_data y cobertura ≥ min_coverage
  2 timeliness         la lectura no supera max_age_minutes respecto a `as_of`
  3 contradictions     sin contradicciones importantes, sin setup contrario activo
                       (candidate/confirmed) y evidencia en contra ≤ max_conflicting_score
  4 direction          la clasificación de confluencia es direccional (no neutral/conflicto)
  5 score              decision_score ≥ min_score
  6 alignment          alineación entre temporalidades en la MISMA dirección y con
                       score ≥ min_alignment_score
  7 setup_confirmation existe un setup activo en esa dirección con estado ≥ min_setup_status
  8 evidence_depth     ≥ min_supporting_evidence evidencias a favor y las categorías
                       requeridas presentes

    decision_score = máx(score alcista, score bajista)
                     − conflict_penalty × evidencia_en_contra            (acotado a 0-100)

Vigencia de un BUY/SELL: mientras el Setup Engine mantenga el setup activo (por defecto hasta 30 velas
tras confirmarse) la decisión puede mantenerse aunque el precio se aleje de la zona. Para evitar
"perseguir" el precio usa `require_price_near_zone=True` (más estricto) o acorta `max_age_minutes`.

INVARIANTES DE SEGURIDAD (no se pueden desactivar con la configuración): nunca BUY/SELL con
entrada inválida, lectura posterior a `as_of`, velas mayores sin cerrar, datos insuficientes,
contexto conflictivo/neutral o contradicciones importantes.

Etiqueta de confianza (solo BUY/SELL; WAIT = LOW): cuenta 6 factores de calidad —margen sobre
el score mínimo, cobertura total, alineación fuerte, evidencia en contra mínima, ≥6 evidencias
a favor, setup confirmado—: HIGH ≥5, MEDIUM ≥3, si no LOW.

─────────────────────────────────────────────────────────────────────
LOOK-AHEAD Y TEMPORALIDAD
─────────────────────────────────────────────────────────────────────
  · Si el resultado de confluencia es POSTERIOR a `as_of`, se descarta (WAIT) y no se
    reproduce nada de su contenido: sería información futura.
  · Si `as_of` es posterior a la lectura por más de `max_age_minutes`, la lectura está
    vencida (WAIT). `valid_until` indica hasta cuándo es vigente una decisión BUY/SELL.
  · Se verifica la integridad temporal del propio resultado: ninguna evidencia puede tener
    `bar_time` posterior al timestamp, y las velas HTF/MTF deben haber cerrado
    (`bar_time + duración ≤ timestamp`).
  · Timestamps con zona horaria se normalizan a UTC sin zona, igual que el resto de módulos.
  · Función pura y determinista: no modifica su entrada, no usa la hora del sistema.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DECISIONES = ("BUY", "SELL", "WAIT")
CLASES_ALCISTAS = ("strong_bullish_confluence", "bullish_confluence")
CLASES_BAJISTAS = ("strong_bearish_confluence", "bearish_confluence")
CLASES_CONFLUENCIA = CLASES_ALCISTAS + CLASES_BAJISTAS + ("neutral", "conflicting_context", "insufficient_data")
CATEGORIAS = ("structure", "liquidity", "fvg", "order_block", "multi_timeframe", "setup")
RANGO_SETUP = {"forming": 0, "candidate": 1, "confirmed": 2}
ALINEACION_ALCISTA = ("strong_bullish_alignment", "bullish_bias")
ALINEACION_BAJISTA = ("strong_bearish_alignment", "bearish_bias")
_ES = {"bullish": "alcista", "bearish": "bajista"}
_OPUESTO = {"bullish": "bearish", "bearish": "bullish"}
_CLAVES_REQUERIDAS = ("classification", "bullish_confluence_score", "bearish_confluence_score", "conflicting_evidence_score",
                      "coverage", "timestamp", "supporting_evidence", "conflicting_evidence", "missing_evidence",
                      "major_contradictions", "alignment", "active_setups", "timeframes", "evidence_count")
_CAMPOS_EVIDENCIA = ("id", "category", "timeframe", "direction", "label", "points", "strength", "source_modules")
AVISO = ("Decisión informativa: el score y la etiqueta de confianza miden coincidencia y consistencia de evidencias, "
         "NO son una probabilidad de éxito. No es una orden ni se ejecuta ninguna operación.")


class DecisionInputError(ValueError):
    """Entrada inválida (solo se lanza si `raise_on_invalid=True`; por defecto se devuelve un WAIT seguro)."""


# ─────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DecisionConfig:
    """Umbrales de decisión. Valores por defecto = elección de diseño, NO optimizados para rentabilidad."""
    min_score: float = 50.0                 # decision_score mínimo (0-100). El motor de confluencia llama 'strong' a ≥55.
    min_coverage: float = 0.75              # cobertura mínima de datos (0-1). 0.75 exige, como mínimo, HTF + LTF disponibles.
    max_conflicting_score: float = 20.0     # evidencia en contra máxima tolerada (0-100).
    conflict_penalty: float = 0.5           # cuánto resta del decision_score cada punto de evidencia en contra.
    min_setup_status: Optional[str] = "confirmed"   # None = no exige setup | "candidate" | "confirmed"
    min_alignment_score: float = 70.0       # alineación HTF/MTF/LTF mínima (0-100). 0 la desactiva (no el chequeo de dirección).
    require_price_near_zone: bool = False   # True: el setup debe tener además price_near_zone (precio cerca de su zona). Por defecto
                                            # NO se exige: en la vela de reacción el cierre queda fuera de la zona por definición, y con
                                            # False un BUY puede mantenerse mientras el setup siga vigente aunque el precio se aleje.
    max_age_minutes: float = 30.0           # vigencia de la lectura (2 velas M15).
    min_supporting_evidence: int = 4        # nº mínimo de evidencias independientes a favor.
    required_categories: Tuple[str, ...] = ("structure",)   # categorías que deben aportar evidencia a favor.
    # etiqueta de confianza (calidad/consistencia, NO probabilidad)
    confidence_score_margin: float = 10.0   # factor 1: score ≥ min_score + margen
    confidence_full_coverage: float = 0.95  # factor 2
    confidence_low_opposition: float = 5.0  # factor 4
    confidence_many_evidence: int = 6       # factor 5
    confidence_high_factors: int = 5        # HIGH si cumple ≥ 5 de 6
    confidence_medium_factors: int = 3      # MEDIUM si cumple ≥ 3 de 6
    raise_on_invalid: bool = False          # True: entrada inválida → DecisionInputError en vez de WAIT seguro

    def validar(self) -> "DecisionConfig":
        def rango(nombre, lo, hi):
            v = getattr(self, nombre)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not (lo <= v <= hi):
                raise ValueError(f"{nombre} debe ser un número entre {lo} y {hi}, no {v!r}")
        rango("min_score", 0, 100); rango("min_coverage", 0, 1); rango("max_conflicting_score", 0, 100)
        rango("conflict_penalty", 0, 10); rango("min_alignment_score", 0, 100); rango("max_age_minutes", 0.0001, 10 ** 7)
        rango("min_supporting_evidence", 0, 100); rango("confidence_score_margin", 0, 100)
        rango("confidence_full_coverage", 0, 1); rango("confidence_low_opposition", 0, 100)
        rango("confidence_many_evidence", 0, 100); rango("confidence_high_factors", 0, 6); rango("confidence_medium_factors", 0, 6)
        if self.confidence_medium_factors > self.confidence_high_factors:
            raise ValueError("confidence_medium_factors no puede superar a confidence_high_factors")
        if self.min_setup_status not in (None, "candidate", "confirmed"):
            raise ValueError("min_setup_status debe ser None, 'candidate' o 'confirmed'")
        cats = tuple(self.required_categories)
        malas = [c for c in cats if c not in CATEGORIAS]
        if malas:
            raise ValueError(f"required_categories contiene categorías desconocidas: {malas}")
        self.required_categories = cats
        if not isinstance(self.require_price_near_zone, bool):
            raise ValueError("require_price_near_zone debe ser bool")
        if not isinstance(self.raise_on_invalid, bool):
            raise ValueError("raise_on_invalid debe ser bool")
        return self


def _resolver_config(config: Any) -> DecisionConfig:
    if config is None:
        return DecisionConfig().validar()
    if isinstance(config, DecisionConfig):
        return replace(config).validar()
    if isinstance(config, dict):
        base = DecisionConfig()
        for k, v in config.items():
            if not hasattr(base, k):
                raise ValueError(f"config desconocida: {k}")
            setattr(base, k, tuple(v) if k == "required_categories" and isinstance(v, list) else v)
        return base.validar()
    raise ValueError("config debe ser None, DecisionConfig o dict")


# ─────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────
def _a_ts(valor: Any, nombre: str) -> Optional[pd.Timestamp]:
    if valor is None:
        return None
    try:
        t = pd.Timestamp(valor)
    except Exception:
        raise ValueError(f"{nombre} no es una fecha válida: {valor!r}")
    if pd.isna(t):
        raise ValueError(f"{nombre} no es una fecha válida: {valor!r}")
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _cap(texto: str) -> str:
    """Mayúscula inicial sin tocar el resto (str.capitalize() pondría en minúsculas los IDs)."""
    return texto[:1].upper() + texto[1:]


def _numero(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _recortar(e: Any) -> dict:
    if not isinstance(e, dict):
        return {"label": str(e)}
    if e.get("category") == "contradiction":
        return {"id": e.get("id"), "category": "contradiction", "label": e.get("label")}
    return {k: e.get(k) for k in _CAMPOS_EVIDENCIA if k in e}


def _inspeccionar(conf: Any) -> Tuple[List[str], List[str]]:
    """Devuelve (problemas de validez, violaciones temporales) del resultado de confluencia."""
    if not isinstance(conf, dict):
        return [f"el resultado de confluencia debe ser un dict, no {type(conf).__name__}"], []
    faltan = [k for k in _CLAVES_REQUERIDAS if k not in conf]
    if faltan:
        return [f"faltan campos del resultado de confluencia: {', '.join(faltan)}"], []
    malos: List[str] = []
    clase = conf["classification"]
    if clase not in CLASES_CONFLUENCIA:
        malos.append(f"clasificación desconocida: {clase!r}")
    for k in ("bullish_confluence_score", "bearish_confluence_score", "conflicting_evidence_score"):
        if not _numero(conf[k]) or not (0 <= conf[k] <= 100):
            malos.append(f"{k} debe ser un número entre 0 y 100, no {conf[k]!r}")
    if not _numero(conf["coverage"]) or not (0 <= conf["coverage"] <= 1):
        malos.append(f"coverage debe ser un número entre 0 y 1, no {conf['coverage']!r}")
    if not isinstance(conf["evidence_count"], int) or isinstance(conf["evidence_count"], bool) or conf["evidence_count"] < 0:
        malos.append(f"evidence_count inválido: {conf['evidence_count']!r}")
    for k in ("supporting_evidence", "conflicting_evidence", "missing_evidence", "major_contradictions", "active_setups"):
        if not isinstance(conf[k], list):
            malos.append(f"{k} debe ser una lista")
    if not isinstance(conf["alignment"], dict) or "alignment" not in conf["alignment"] or "score" not in conf["alignment"]:
        malos.append("alignment debe ser un dict con 'alignment' y 'score'")
    elif not _numero(conf["alignment"]["score"]):
        malos.append("alignment.score no es numérico")
    if not isinstance(conf["timeframes"], dict):
        malos.append("timeframes debe ser un dict")
    if malos:
        return malos, []
    for s in conf["active_setups"]:
        if not isinstance(s, dict) or s.get("direction") not in _ES or s.get("status") not in RANGO_SETUP:
            malos.append(f"setup activo con formato inválido: {s!r}")
            break
    ts = None
    if conf["timestamp"] is None:
        if clase != "insufficient_data":
            malos.append("timestamp ausente en una lectura que no es 'insufficient_data'")
    else:
        try:
            ts = _a_ts(conf["timestamp"], "timestamp")
        except ValueError as e:
            malos.append(str(e))
    bull, bear = conf["bullish_confluence_score"], conf["bearish_confluence_score"]
    if clase in CLASES_ALCISTAS and not bull > bear:
        malos.append("clasificación alcista pero el score alcista no supera al bajista (resultado inconsistente)")
    if clase in CLASES_BAJISTAS and not bear > bull:
        malos.append("clasificación bajista pero el score bajista no supera al alcista (resultado inconsistente)")
    if abs(conf["conflicting_evidence_score"] - min(bull, bear)) > 0.01:
        malos.append("conflicting_evidence_score no coincide con el mínimo entre ambos scores (resultado inconsistente)")
    if malos:
        return malos, []

    # integridad temporal: ninguna evidencia posterior al timestamp; velas HTF/MTF ya cerradas
    violaciones: List[str] = []
    if ts is not None:
        freqs = {}
        for tf, info in conf["timeframes"].items():
            try:
                freqs[tf] = pd.Timedelta(info.get("freq")) if isinstance(info, dict) and info.get("freq") else None
            except Exception:
                freqs[tf] = None
            ult = info.get("last_closed_bar") if isinstance(info, dict) else None
            if ult is not None and tf in ("htf", "mtf") and freqs[tf] is not None:
                try:
                    if _a_ts(ult, "last_closed_bar") + freqs[tf] > ts:
                        violaciones.append(f"la vela {tf.upper()} usada no había cerrado en el timestamp de la lectura")
                except ValueError:
                    violaciones.append(f"last_closed_bar inválido en {tf}")
        pool = list(conf["supporting_evidence"]) + list(conf["conflicting_evidence"]) + list(conf.get("all_evidence") or [])
        for e in pool:
            if not isinstance(e, dict) or e.get("bar_time") is None:
                continue
            try:
                b = _a_ts(e["bar_time"], "bar_time")
            except ValueError:
                violaciones.append(f"evidencia {e.get('id')}: bar_time inválido")
                continue
            if b > ts:
                violaciones.append(f"la evidencia {e.get('id')} es posterior al timestamp de la lectura")
            elif e.get("timeframe") in ("htf", "mtf") and freqs.get(e.get("timeframe")) is not None and b + freqs[e["timeframe"]] > ts:
                violaciones.append(f"la evidencia {e.get('id')} usa una vela {e['timeframe'].upper()} que no había cerrado")
        violaciones = sorted(set(violaciones))
    return [], violaciones


def _resultado_seguro(cfg: DecisionConfig, ref: Optional[pd.Timestamp], motivo: str, estado: str) -> dict:
    """WAIT seguro: sin dirección, sin puntaje y sin reproducir contenido de la entrada."""
    return {
        "decision": "WAIT", "direction": None, "decision_score": 0.0, "confidence_label": "LOW",
        "reasons": [f"WAIT: {motivo}"], "supporting_evidence": [], "conflicting_evidence": [],
        "missing_conditions": [motivo], "invalidation_conditions": [],
        "timestamp": ref, "as_of": ref, "valid_until": None, "input_status": estado,
        "classification": None, "scores": {"bullish": 0.0, "bearish": 0.0, "conflicting": 0.0, "coverage": 0.0},
        "gates": [], "confidence_factors": {}, "evidence_by_module": {}, "disclaimer": AVISO, "config_used": asdict(cfg),
    }


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def analyze_decision(confluence_result: Any, config: Any = None, as_of: Any = None) -> dict:
    """
    Decide BUY / SELL / WAIT a partir del resultado de `analyze_confluence`. Solo informativo.

    Parameters
    ----------
    confluence_result : dict devuelto por `core.confluence_engine.analyze_confluence`.
    config : None, `DecisionConfig` o dict con cualquier subconjunto de sus campos.
    as_of : momento en el que se toma la decisión (p. ej. "ahora"). Si la lectura es posterior a
        `as_of` se descarta; si es más vieja que `max_age_minutes`, está vencida. Si es None no se
        verifica la vigencia (se asume que la decisión se toma en el momento de la lectura).

    Devuelve un dict con: decision, direction, decision_score, confidence_label, reasons,
    supporting_evidence, conflicting_evidence, missing_conditions, invalidation_conditions,
    timestamp, config_used (+ as_of, valid_until, input_status, classification, scores, gates,
    confidence_factors, evidence_by_module, disclaimer).

    Una configuración inválida lanza ValueError. Una entrada inválida devuelve un WAIT seguro
    (o lanza DecisionInputError si `raise_on_invalid=True`).
    """
    cfg = _resolver_config(config)

    def invalido(motivo: str, estado: str, ref: Optional[pd.Timestamp] = None) -> dict:
        if cfg.raise_on_invalid:
            raise DecisionInputError(motivo)
        return _resultado_seguro(cfg, ref, motivo, estado)

    try:
        ref = _a_ts(as_of, "as_of")
    except ValueError as e:
        return invalido(str(e), "invalid")

    problemas, violaciones = _inspeccionar(confluence_result)
    if problemas:
        return invalido("entrada inválida: " + "; ".join(problemas), "invalid", ref)
    if violaciones:
        return invalido("la lectura de confluencia viola la integridad temporal: " + "; ".join(violaciones), "temporal_violation", ref)

    conf = copy.deepcopy(confluence_result)          # la entrada nunca se modifica
    ts = _a_ts(conf["timestamp"], "timestamp")
    if ts is not None and ref is not None and ts > ref:
        return invalido("la lectura de confluencia es posterior a as_of: contiene información futura y se descarta", "future", ref)
    edad = ((ref - ts).total_seconds() / 60.0) if (ts is not None and ref is not None) else None

    clase = conf["classification"]
    bull, bear = float(conf["bullish_confluence_score"]), float(conf["bearish_confluence_score"])
    oposicion = float(conf["conflicting_evidence_score"])
    cobertura = float(conf["coverage"])
    direccion = "bullish" if clase in CLASES_ALCISTAS else "bearish" if clase in CLASES_BAJISTAS else None
    lider = direccion or ("bullish" if bull > bear else "bearish" if bear > bull else None)
    dominante = max(bull, bear)
    decision_score = round(max(0.0, min(100.0, dominante - cfg.conflict_penalty * oposicion)), 4)
    apoyo, contra = conf["supporting_evidence"], conf["conflicting_evidence"]
    mayores = [str(m) for m in conf["major_contradictions"]]
    setups = conf["active_setups"]
    al = conf["alignment"]
    alineacion, al_score = str(al["alignment"]), float(al["score"])

    gates: List[dict] = []

    def gate(gid: str, nombre: str, ok: bool, valor: Any, umbral: Any, detalle: str, falta: str, depende_de_direccion: bool = False) -> None:
        # sin dirección candidata, las condiciones que dependen de ella no son evaluables: no se listan como faltantes
        aplica = not (depende_de_direccion and direccion is None)
        gates.append({"id": gid, "name": nombre, "passed": bool(ok), "applicable": aplica, "value": valor, "threshold": umbral,
                      "detail": detalle if aplica else f"no evaluable: no hay dirección candidata ({detalle})", "missing": falta})

    # 1) suficiencia de datos
    ok1 = clase != "insufficient_data" and cobertura >= cfg.min_coverage
    gate("data_sufficiency", "Suficiencia de datos", ok1, round(cobertura, 4), cfg.min_coverage,
         ("la confluencia reporta datos insuficientes" if clase == "insufficient_data"
          else f"cobertura de datos {cobertura:.0%} " + ("≥" if ok1 else "<") + f" mínima {cfg.min_coverage:.0%}"),
         f"cobertura de datos ≥ {cfg.min_coverage:.0%} (actual {cobertura:.0%})")
    # 2) vigencia
    if edad is None:
        ok2, det2 = True, ("sin as_of: no se verificó la vigencia" if ref is None else "lectura sin timestamp: no aplica")
    else:
        ok2 = edad <= cfg.max_age_minutes
        det2 = f"la lectura tiene {edad:.1f} min " + ("≤" if ok2 else ">") + f" vigencia máxima {cfg.max_age_minutes:g} min"
    gate("timeliness", "Vigencia de la lectura", ok2, None if edad is None else round(edad, 4), cfg.max_age_minutes, det2,
         f"lectura vigente (≤ {cfg.max_age_minutes:g} min; tiene {0 if edad is None else edad:.1f} min)")
    # 3) contradicciones
    contrarios = [s for s in setups if lider and s["direction"] == _OPUESTO[lider] and RANGO_SETUP[s["status"]] >= 1]
    motivos3: List[str] = []
    if clase == "conflicting_context":
        motivos3.append("la confluencia clasifica el contexto como contradictorio")
    motivos3 += mayores
    if oposicion > cfg.max_conflicting_score:
        motivos3.append(f"evidencia en contra {oposicion:.1f} > máximo {cfg.max_conflicting_score:.1f}")
    for s in contrarios:
        motivos3.append(f"setup {_ES[s['direction']]} {s['status']} ({s.get('setup_id')}) en sentido contrario")
    ok3 = not motivos3
    gate("contradictions", "Sin contradicciones", ok3, round(oposicion, 4), cfg.max_conflicting_score,
         ("sin contradicciones importantes; evidencia en contra " + f"{oposicion:.1f} ≤ {cfg.max_conflicting_score:.1f}") if ok3
         else "; ".join(motivos3),
         "sin contradicciones importantes y evidencia en contra ≤ " + f"{cfg.max_conflicting_score:.1f} (hay: {'; '.join(motivos3) or 'ninguna'})")
    # 4) dirección
    gate("direction", "Dirección predominante", direccion is not None, clase, None,
         (f"confluencia {_ES[direccion]} ({clase})" if direccion else f"sin dirección clara (clasificación '{clase}')"),
         "una confluencia direccional (alcista o bajista) sin conflicto")
    # 5) score
    ok5 = direccion is not None and decision_score >= cfg.min_score
    gate("score", "Score de decisión", ok5, decision_score, cfg.min_score,
         f"decision_score {decision_score:.1f} " + ("≥" if decision_score >= cfg.min_score else "<") + f" mínimo {cfg.min_score:.1f}"
         + (f" (lado {_ES[lider]}: {dominante:.1f} − {cfg.conflict_penalty:g}×{oposicion:.1f} de oposición)" if lider else ""),
         f"decision_score ≥ {cfg.min_score:.1f} (actual {decision_score:.1f})", depende_de_direccion=True)
    # 6) alineación entre temporalidades
    if direccion is None:
        ok6, det6 = False, f"alineación {alineacion} (score {al_score:.0f}); sin dirección candidata"
    else:
        alineada = alineacion in (ALINEACION_ALCISTA if direccion == "bullish" else ALINEACION_BAJISTA)
        ok6 = alineada and al_score >= cfg.min_alignment_score
        det6 = (f"alineación {alineacion} con score {al_score:.0f} " + ("≥" if al_score >= cfg.min_alignment_score else "<")
                + f" mínimo {cfg.min_alignment_score:.0f}" + ("" if alineada else f" y no está a favor de la dirección {_ES[direccion]}"))
    gate("alignment", "Alineación HTF/MTF/LTF", ok6, al_score, cfg.min_alignment_score, det6,
         f"alineación a favor de la dirección con score ≥ {cfg.min_alignment_score:.0f} (actual: {alineacion}, {al_score:.0f})", depende_de_direccion=True)
    # 7) confirmación del setup
    if cfg.min_setup_status is None:
        ok7, det7 = True, "no se exige setup"
    else:
        req = RANGO_SETUP[cfg.min_setup_status]
        validos = [s for s in setups if direccion and s["direction"] == direccion and RANGO_SETUP[s["status"]] >= req]
        lejos = []
        if cfg.require_price_near_zone:
            lejos = [s for s in validos if "price_near_zone" in (s.get("conditions_missing") or [])]
            validos = [s for s in validos if s not in lejos]
        ok7 = bool(validos)
        if ok7:
            mejor = sorted(validos, key=lambda s: (-RANGO_SETUP[s["status"]], str(s.get("setup_id"))))[0]
            det7 = f"setup {_ES[direccion]} {mejor['status']} ({mejor.get('setup_id')}) cumple el mínimo '{cfg.min_setup_status}'"
        else:
            propios = [f"{s['status']} ({s.get('setup_id')})" for s in setups if direccion and s["direction"] == direccion]
            det7 = (f"no hay un setup {_ES[direccion]} con estado ≥ '{cfg.min_setup_status}'" if direccion else "sin dirección: no hay setup que confirmar") \
                + (f"; solo {', '.join(propios)}" if propios else "")
            if lejos:
                det7 = f"el setup {lejos[0].get('setup_id')} está {lejos[0]['status']} pero el precio no está cerca de su zona (price_near_zone)"
    gate("setup_confirmation", "Confirmación del Setup Engine", ok7, cfg.min_setup_status, cfg.min_setup_status, det7,
         f"setup {_ES.get(direccion, 'en la dirección')} con estado ≥ '{cfg.min_setup_status}' (Setup Engine)", depende_de_direccion=True)
    # 8) profundidad de evidencia
    cats_apoyo = sorted({e.get("category") for e in apoyo if isinstance(e, dict)})
    faltan_cat = [c for c in cfg.required_categories if c not in cats_apoyo]
    ok8 = direccion is not None and len(apoyo) >= cfg.min_supporting_evidence and not faltan_cat
    gate("evidence_depth", "Profundidad de evidencia", ok8, len(apoyo), cfg.min_supporting_evidence,
         f"{len(apoyo)} evidencias independientes a favor (mínimo {cfg.min_supporting_evidence})"
         + (f"; falta evidencia de: {', '.join(faltan_cat)}" if faltan_cat else ""),
         f"≥ {cfg.min_supporting_evidence} evidencias a favor (hay {len(apoyo)})"
         + (f" y evidencia de: {', '.join(faltan_cat)}" if faltan_cat else ""), depende_de_direccion=True)

    todas = all(g["passed"] for g in gates)   # (si no hay dirección, direction=False ya bloquea)
    decision = ("BUY" if direccion == "bullish" else "SELL") if (todas and direccion) else "WAIT"
    fallidas = [g for g in gates if not g["passed"] and g["applicable"]]

    # ── etiqueta de confianza (solo BUY/SELL) ──
    factores: Dict[str, dict] = {}
    etiqueta = "LOW"
    if decision != "WAIT":
        confirmado = any(s["direction"] == direccion and s["status"] == "confirmed" for s in setups)
        alin_fuerte = alineacion in (("strong_bullish_alignment",) if direccion == "bullish" else ("strong_bearish_alignment",))
        factores = {
            "margen_de_score": {"passed": decision_score >= cfg.min_score + cfg.confidence_score_margin,
                                "detail": f"decision_score {decision_score:.1f} vs mínimo+margen {cfg.min_score + cfg.confidence_score_margin:.1f}"},
            "cobertura_total": {"passed": cobertura >= cfg.confidence_full_coverage, "detail": f"cobertura {cobertura:.0%}"},
            "alineacion_fuerte": {"passed": alin_fuerte, "detail": f"alineación {alineacion}"},
            "poca_oposicion": {"passed": oposicion <= cfg.confidence_low_opposition, "detail": f"evidencia en contra {oposicion:.1f}"},
            "evidencia_abundante": {"passed": len(apoyo) >= cfg.confidence_many_evidence, "detail": f"{len(apoyo)} evidencias a favor"},
            "setup_confirmado": {"passed": confirmado, "detail": "hay un setup confirmado en la dirección" if confirmado else "sin setup confirmado"},
        }
        n = sum(1 for f in factores.values() if f["passed"])
        etiqueta = "HIGH" if n >= cfg.confidence_high_factors else "MEDIUM" if n >= cfg.confidence_medium_factors else "LOW"

    # ── razones ──
    top = sorted((e for e in apoyo if isinstance(e, dict)), key=lambda e: (-(e.get("points") or 0), str(e.get("id"))))[:4]
    valid_until = (ts + pd.Timedelta(minutes=cfg.max_age_minutes)) if (decision != "WAIT" and ts is not None) else None
    reasons: List[str] = []
    if decision != "WAIT":
        reasons.append(f"{decision}: confluencia {_ES[direccion]} ({clase}) con decision_score {decision_score:.1f} (mínimo {cfg.min_score:.1f}), "
                       f"evidencia en contra {oposicion:.1f} (máximo {cfg.max_conflicting_score:.1f}) y cobertura {cobertura:.0%} (mínima {cfg.min_coverage:.0%}).")
        reasons += [_cap(g["detail"]) + "." for g in gates if g["id"] in ("alignment", "setup_confirmation", "evidence_depth")]
        if top:
            reasons.append("Evidencias principales: " + "; ".join(f"{e.get('label')} ({(e.get('points') or 0):.1f} pts)" for e in top) + ".")
        reasons.append(f"Lectura vigente hasta {valid_until} ({cfg.max_age_minutes:g} min).")
    else:
        reasons.append("WAIT: " + fallidas[0]["detail"] + ".")
        reasons += [_cap(g["detail"]) + "." for g in fallidas[1:]]
        pasadas = [g["name"] for g in gates if g["passed"] and g["applicable"]]
        if pasadas:
            reasons.append("Condiciones que sí se cumplen: " + ", ".join(pasadas) + ".")
        reasons.append("Con una o más condiciones sin cumplir no se emite BUY ni SELL: esperar es la decisión segura.")

    # ── condiciones faltantes ──
    faltantes = [g["missing"] for g in fallidas]
    for m in conf["missing_evidence"]:
        m = str(m)
        if m.startswith("setup ") or any(k in m for k in ("sin datos", "insuficientes", "sin resultados", "error en módulo")):
            faltantes.append(m)
    if decision != "WAIT":
        faltantes = [f"(no bloqueante) {m}" for m in faltantes]
    faltantes = list(dict.fromkeys(faltantes))

    # ── condiciones de invalidación (solo BUY/SELL) ──
    invalidacion: List[str] = []
    if decision != "WAIT":
        op = _OPUESTO[direccion]
        for s in setups:
            if s["direction"] == direccion and RANGO_SETUP[s["status"]] >= 1:
                invalidacion.append(f"El setup {s.get('setup_id')} pasa a invalidated o expired en el Setup Engine.")
        zonas = sorted((e for e in conf.get("all_evidence", []) if isinstance(e, dict) and e.get("counted") and e.get("direction") == direccion
                        and isinstance(e.get("zone"), dict) and e.get("timeframe") == "ltf"), key=lambda e: (-(e.get("points") or 0), str(e.get("id"))))
        if zonas:
            z = zonas[0]["zone"]
            nivel, lado = (z.get("bottom"), "por debajo de la base") if direccion == "bullish" else (z.get("top"), "por encima del techo")
            if nivel is not None:
                invalidacion.append(f"Un cierre del LTF {lado} ({nivel:.5g}) de la zona '{zonas[0].get('label')}' (referencia analítica, no es un stop).")
        invalidacion += [
            f"HTF o MTF pasan a tendencia {_ES[op]}.",
            f"La alineación cae por debajo de {cfg.min_alignment_score:.0f} o deja de estar a favor de la dirección {_ES[direccion]}.",
            f"El decision_score baja de {cfg.min_score:.1f} o la evidencia en contra supera {cfg.max_conflicting_score:.1f}.",
            f"Aparece un setup {_ES[op]} candidate/confirmed o una contradicción importante.",
            f"La lectura supera {cfg.max_age_minutes:g} min sin renovarse (vigente hasta {valid_until}).",
        ]

    por_modulo = {}
    for cat, v in (conf.get("breakdown") or {}).items():
        if isinstance(v, dict):
            por_modulo[cat] = {"bullish_points": v.get("bullish_points"), "bearish_points": v.get("bearish_points"),
                               "available_weight": v.get("available_weight")}

    return {
        "decision": decision, "direction": direccion if decision != "WAIT" else None,
        "decision_score": decision_score, "confidence_label": etiqueta,
        "reasons": reasons,
        "supporting_evidence": [_recortar(e) for e in apoyo], "conflicting_evidence": [_recortar(e) for e in contra],
        "missing_conditions": faltantes, "invalidation_conditions": invalidacion,
        "timestamp": ts, "as_of": ref, "valid_until": valid_until, "input_status": "ok",
        "leading_side": lider, "classification": clase,
        "scores": {"bullish": bull, "bearish": bear, "conflicting": oposicion, "coverage": cobertura},
        "gates": gates, "confidence_factors": factores, "evidence_by_module": por_modulo,
        "disclaimer": AVISO, "config_used": asdict(cfg),
    }
