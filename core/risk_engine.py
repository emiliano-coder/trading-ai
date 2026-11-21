"""
core/risk_engine.py
─────────────────────────────────────────────────────────────────────
Risk Engine — evalúa si una operación propuesta por el Decision Engine es ACEPTABLE desde el
punto de vista del riesgo y calcula sus parámetros: Stop Loss, Take Profit, tamaño de posición,
riesgo monetario y relación riesgo-beneficio.

Es un EVALUADOR/CALCULADOR independiente:
  · NO ejecuta ni simula órdenes, NO se conecta a MT5/IBKR, NO toca el BUY/SELL/WAIT anterior,
    `generar_senal`, `calcular_score`, Telegram ni la configuración de señales.
  · `analyze_risk` no importa ningún otro módulo del proyecto: lee diccionarios.

─────────────────────────────────────────────────────────────────────
QUÉ SE ENCONTRÓ AL INSPECCIONAR LOS CAMPOS REALES (y cómo condiciona el diseño)
─────────────────────────────────────────────────────────────────────
  · El resultado de `analyze_decision` NO incluye símbolo ni ningún nivel de precio. Aporta:
    decision, direction, timestamp, as_of, valid_until, input_status y evidencias (el ID del setup
    viaja dentro del ID de evidencia `setup:ltf:<setup_id>`).
  · El resultado de `analyze_confluence` incluye `symbol` y, para FVG/OB, `zone{top,bottom,avail}`
    (avail es un índice de vela, no un timestamp). `active_setups` NO trae el precio de invalidación.
  · El precio de invalidación de un setup (`invalidation_price`) existe solo en el registro de
    `detect_setups`; no viaja por la cadena Confluence → Decision.
  Por eso el Risk Engine NO puede deducir SL/TP únicamente del resultado de decisión: recibe los
  niveles por una interfaz explícita y con marcas de tiempo, `market_data["levels"]`, y NO inventa
  ninguno. Si no hay un nivel válido, rechaza. (`construir_entradas_desde_pipeline` —al final, solo
  para pruebas e interfaz— arma esa interfaz leyendo los campos reales de los demás módulos.)

─────────────────────────────────────────────────────────────────────
ENTRADAS DE `analyze_risk(decision_result, market_data, account_data, instrument_specs, config, as_of)`
─────────────────────────────────────────────────────────────────────
  decision_result  dict de `analyze_decision`
  market_data      {"symbol", "entry_price", "timestamp",
                    "levels": [{"kind": "setup"|"order_block"|"structure"|"liquidity",
                                "role": "stop"|"target", "direction": "bullish"|"bearish",
                                "price", "available_at", "source_id", "label"?}],
                    "confluence": resultado de analyze_confluence (opcional),
                    "proposed_stop_loss"?, "proposed_take_profit"?}
  account_data     {"equity", "currency"?, "daily_pnl", "trades_today", "open_positions": [...],
                    "day_start_equity"?}   (cada posición: symbol, direction, risk_amount, floating_pnl?)
  instrument_specs {"symbol", "tick_size", "tick_value", "contract_size", "volume_min", "volume_max",
                    "volume_step", "quote_currency"?, "volume_unit"?, "min_stop_distance_ticks"?}
                   NINGÚN valor de contrato está escrito en el motor: si falta cualquiera de los
                   indispensables, no se calcula un lote ficticio; se devuelve qué falta.

─────────────────────────────────────────────────────────────────────
FÓRMULAS
─────────────────────────────────────────────────────────────────────
    distancia_SL        = |entrada − SL|
    pérdida_por_lote    = (distancia_SL / tick_size) × tick_value
    presupuesto_riesgo  = mín( equity × riesgo_por_operación%,  equity × riesgo_máximo%,
                               presupuesto diario restante,  presupuesto de exposición restante )
    volumen_bruto       = presupuesto_riesgo / pérdida_por_lote
    volumen             = volume_min + ⌊(volumen_bruto − volume_min) / volume_step⌋ × volume_step
                          (SIEMPRE hacia abajo; nunca por encima del presupuesto; tope volume_max)
    riesgo_real         = volumen × pérdida_por_lote
    beneficio           = volumen × (|TP − entrada| / tick_size) × tick_value
    R:R                 = |TP − entrada| / |entrada − SL|
  SL: nivel de invalidación ∓ margen (sl_buffer_pct) redondeado AL TICK hacia fuera (más lejos de la
  entrada). TP: redondeado al tick hacia la entrada (nunca sobreestima el beneficio).

─────────────────────────────────────────────────────────────────────
SEGURIDAD
─────────────────────────────────────────────────────────────────────
  · WAIT → siempre REJECTED ("NO TRADE"). Solo se evalúan BUY/SELL válidos (`input_status == "ok"`).
  · Nunca se aprueba un SL/TP del lado incorrecto, un R:R < mínimo, un equity inválido, datos
    incompletos o inconsistentes, ni un riesgo superior al máximo configurado.
  · Sin martingala: el tamaño solo depende de equity, % de riesgo, distancia y especificaciones; las
    pérdidas previas SOLO pueden reducirlo o bloquear. Nunca se acepta añadir a una posición perdedora
    de la misma dirección (o cuyo resultado no se pueda verificar).
  · LÍMITES ABSOLUTOS: ninguna configuración puede superarlos (ver LIMITES_ABSOLUTOS) ni desactivar un
    límite (None/0). Son un tope de seguridad de código, elegido de forma conservadora, no una
    recomendación; un valor mayor exige editar este archivo a propósito.
  · Los valores por defecto de RiskConfig son los sugeridos para pruebas; NO están optimizados.

─────────────────────────────────────────────────────────────────────
LOOK-AHEAD
─────────────────────────────────────────────────────────────────────
  · Referencia temporal: `as_of` (o el timestamp del precio si es None). Se rechaza una decisión
    posterior a la referencia o ya vencida (`valid_until`), y un precio posterior o más viejo que
    `max_price_age_minutes`.
  · Un nivel solo se usa si `available_at <= referencia`; los posteriores se IGNORAN.
  · Función pura y determinista; no modifica sus entradas; no usa la hora del sistema.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, getcontext
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

getcontext().prec = 40

LIMITES_ABSOLUTOS = {
    "risk_per_trade_pct": 2.0,        # % del equity por operación
    "max_risk_per_trade_pct": 2.0,
    "max_daily_loss_pct": 5.0,
    "max_daily_trades": 10,
    "max_open_positions": 5,
    "max_total_risk_pct": 5.0,
    "max_stop_distance_pct": 5.0,
}
PISO_RIESGO_BENEFICIO = 1.0           # el R:R mínimo configurable no puede ser inferior a 1:1
KINDS_STOP = ("setup", "order_block", "structure")
TP_MODES = ("rr", "liquidity", "liquidity_or_rr")
_ES = {"bullish": "alcista", "bearish": "bajista"}
_DIR_DE = {"BUY": "bullish", "SELL": "bearish"}
_ESPECS_REQUERIDAS = ("symbol", "tick_size", "tick_value", "contract_size", "volume_min", "volume_max", "volume_step")


# ─────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────
@dataclass
class RiskConfig:
    """Parámetros de riesgo. Los valores por defecto son los sugeridos para pruebas, NO optimizados."""
    risk_per_trade_pct: float = 0.5          # % del equity que se quiere arriesgar por operación (objetivo)
    max_risk_per_trade_pct: float = 1.0      # tope duro por operación (el riesgo real nunca lo supera)
    max_daily_loss_pct: float = 2.0          # pérdida diaria máxima (% del equity al inicio del día)
    max_daily_trades: int = 3                # operaciones máximas por día
    min_risk_reward: float = 2.0             # R:R mínimo aceptado (2 = 2:1)
    target_risk_reward: float = 2.0          # R:R con el que se calcula el TP en modo 'rr' (≥ min_risk_reward)
    min_stop_distance_pct: float = 0.03      # distancia mínima del SL (% del precio de entrada)
    max_stop_distance_pct: float = 2.5       # distancia máxima del SL (% del precio de entrada)
    sl_buffer_pct: float = 0.02              # margen del SL más allá del nivel de invalidación (% del precio)
    sl_source_priority: Tuple[str, ...] = ("setup", "order_block", "structure")
    tp_mode: str = "rr"                      # 'rr' | 'liquidity' | 'liquidity_or_rr'
    max_open_positions: int = 2              # operaciones simultáneas
    max_total_risk_pct: float = 3.0          # exposición: riesgo abierto + nuevo (% del equity)
    max_notional_leverage: Optional[float] = None   # None = no se aplica; si se define, nocional/equity ≤ valor
    allow_min_lot_above_target: bool = False  # True: permite el lote mínimo si su riesgo ≤ max_risk_per_trade_pct
    max_price_age_minutes: float = 15.0      # antigüedad máxima del precio de entrada respecto a la referencia

    def validar(self) -> "RiskConfig":
        def num(nombre, lo, hi, incl_lo=False):
            v = getattr(self, nombre)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > hi or (v <= lo if not incl_lo else v < lo):
                raise ValueError(f"{nombre} debe ser un número {'>=' if incl_lo else '>'} {lo} y <= {hi}, no {v!r}")
        A = LIMITES_ABSOLUTOS
        num("risk_per_trade_pct", 0, A["risk_per_trade_pct"]); num("max_risk_per_trade_pct", 0, A["max_risk_per_trade_pct"])
        num("max_daily_loss_pct", 0, A["max_daily_loss_pct"]); num("max_total_risk_pct", 0, A["max_total_risk_pct"])
        num("max_stop_distance_pct", 0, A["max_stop_distance_pct"]); num("min_stop_distance_pct", 0, A["max_stop_distance_pct"])
        num("sl_buffer_pct", 0, 1.0, incl_lo=True); num("max_price_age_minutes", 0, 10 ** 6)
        num("min_risk_reward", PISO_RIESGO_BENEFICIO, 100, incl_lo=True); num("target_risk_reward", PISO_RIESGO_BENEFICIO, 100, incl_lo=True)
        for n, tope in (("max_daily_trades", A["max_daily_trades"]), ("max_open_positions", A["max_open_positions"])):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int) or not (1 <= v <= tope):
                raise ValueError(f"{n} debe ser un entero entre 1 y {tope}, no {v!r}")
        if self.risk_per_trade_pct > self.max_risk_per_trade_pct:
            raise ValueError("risk_per_trade_pct no puede superar max_risk_per_trade_pct")
        if self.target_risk_reward < self.min_risk_reward:
            raise ValueError("target_risk_reward no puede ser inferior a min_risk_reward")
        if self.min_stop_distance_pct >= self.max_stop_distance_pct:
            raise ValueError("min_stop_distance_pct debe ser menor que max_stop_distance_pct")
        if self.max_notional_leverage is not None:
            num("max_notional_leverage", 0, 10 ** 4)
        prio = tuple(self.sl_source_priority)
        if not prio or any(p not in KINDS_STOP for p in prio) or len(set(prio)) != len(prio):
            raise ValueError(f"sl_source_priority debe ser una selección sin repetir de {KINDS_STOP}")
        self.sl_source_priority = prio
        if self.tp_mode not in TP_MODES:
            raise ValueError(f"tp_mode debe ser uno de {TP_MODES}")
        for n in ("allow_min_lot_above_target",):
            if not isinstance(getattr(self, n), bool):
                raise ValueError(f"{n} debe ser bool")
        return self


def _resolver_config(config: Any) -> RiskConfig:
    if config is None:
        return RiskConfig().validar()
    if isinstance(config, RiskConfig):
        return replace(config).validar()
    if isinstance(config, dict):
        base = RiskConfig()
        for k, v in config.items():
            if not hasattr(base, k):
                raise ValueError(f"config desconocida: {k}")
            setattr(base, k, tuple(v) if k == "sl_source_priority" and isinstance(v, list) else v)
        return base.validar()
    raise ValueError("config debe ser None, RiskConfig o dict")


# ─────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────
def _ok_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _ts(valor: Any) -> Optional[pd.Timestamp]:
    if valor is None:
        return None
    try:
        t = pd.Timestamp(valor)
    except Exception:
        return None
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _d(x: Any) -> Decimal:
    return Decimal(str(x))


def _al_tick(precio: float, tick: float, modo: str) -> float:
    n = _d(precio) / _d(tick)
    n = n.to_integral_value(rounding=ROUND_FLOOR if modo == "floor" else ROUND_CEILING)
    return float(n * _d(tick))


def _simbolo(s: Any) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def _dir_posicion(x: Any) -> Optional[str]:
    t = str(x).strip().lower() if x is not None else ""
    return "bullish" if t in ("bullish", "buy", "long") else "bearish" if t in ("bearish", "sell", "short") else None


class _Rechazo(Exception):
    """Cortocircuito interno: se rechaza con un motivo."""


@dataclass
class _Estado:
    cfg: RiskConfig
    missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    razones: List[str] = field(default_factory=list)
    checks: List[dict] = field(default_factory=list)

    def check(self, id_: str, ok: bool, detalle: str) -> bool:
        self.checks.append({"id": id_, "passed": bool(ok), "detail": detalle})
        if not ok:
            self.razones.append(detalle)
        return ok


def _salida(cfg: RiskConfig, st: _Estado, aprobado: bool, **v) -> dict:
    razon = v.pop("reason", None)
    if aprobado:
        razon = razon or "Operación aceptable: cumple todos los límites de riesgo."
    else:
        razon = razon or (st.razones[0] if st.razones else "Rechazada.")
    base = {
        "approved": bool(aprobado), "status": "APPROVED" if aprobado else "REJECTED", "reason": razon,
        "rejection_reasons": [] if aprobado else list(dict.fromkeys(st.razones or [razon])),
        "symbol": None, "direction": None, "side": None, "entry_price": None, "stop_loss": None, "take_profit": None,
        "risk_amount": None, "risk_percentage": None, "reward_amount": None, "risk_reward_ratio": None,
        "position_size": None, "position_size_unit": None, "maximum_allowed_loss": None,
        "stop_loss_source": None, "take_profit_source": None, "loss_per_lot": None, "raw_volume": None,
        "missing_data": list(dict.fromkeys(st.missing)), "warnings": list(dict.fromkeys(st.warnings)),
        "checks": st.checks, "timestamp": None, "reference_time": None, "calculation_for_reference": None,
        "config_used": asdict(cfg),
    }
    extra = {k: val for k, val in v.items() if k not in base}
    base.update({k: val for k, val in v.items() if k in base})
    if not aprobado:
        # una operación rechazada NUNCA expone SL/TP/tamaño como si fueran utilizables: quedan solo como referencia
        ref = {k: base[k] for k in ("stop_loss", "take_profit", "risk_amount", "risk_percentage", "reward_amount",
                                    "risk_reward_ratio", "position_size", "stop_loss_source", "take_profit_source",
                                    "loss_per_lot", "raw_volume") if base[k] is not None}
        ref.update({k: val for k, val in extra.items() if val is not None})
        base["calculation_for_reference"] = ref or None
        for k in ("stop_loss", "take_profit", "risk_amount", "risk_percentage", "reward_amount", "risk_reward_ratio",
                  "position_size", "position_size_unit", "stop_loss_source", "take_profit_source", "loss_per_lot", "raw_volume"):
            base[k] = None
    return base


# ─────────────────────────────────────────────────────────────────────
#  Motor principal
# ─────────────────────────────────────────────────────────────────────
def analyze_risk(decision_result: Any, market_data: Any, account_data: Any, instrument_specs: Any,
                 config: Any = None, as_of: Any = None) -> dict:
    """
    Evalúa el riesgo de la operación propuesta por el Decision Engine y calcula SL, TP y tamaño.
    Devuelve siempre un dict con `approved`, `status` ('APPROVED'/'REJECTED') y `reason` (ver docstring
    del módulo). Una configuración inválida lanza ValueError; cualquier problema de datos devuelve
    REJECTED sin inventar valores. No ejecuta nada.
    """
    cfg = _resolver_config(config)
    st = _Estado(cfg=cfg)
    ctx: Dict[str, Any] = {"symbol": None, "direction": None, "side": None, "entry_price": None, "timestamp": None, "reference_time": None}
    try:
        return _evaluar(decision_result, market_data, account_data, instrument_specs, cfg, as_of, st, ctx)
    except _Rechazo as e:
        return _salida(cfg, st, False, reason=str(e), **ctx)
    except Exception as e:      # red de seguridad: ante algo inesperado se rechaza, jamás se aprueba
        st.warnings.append(f"error inesperado al evaluar ({type(e).__name__})")
        return _salida(cfg, st, False, reason=f"error inesperado al evaluar la operación ({type(e).__name__}): se rechaza por seguridad", **ctx)


def _evaluar(decision_result, market_data, account_data, specs, cfg: RiskConfig, as_of, st: _Estado, ctx: dict) -> dict:
    def rechazar(id_: str, texto: str):
        st.check(id_, False, texto)
        raise _Rechazo(texto)

    # ── 0) forma de las entradas y decisión ──
    if not isinstance(decision_result, dict):
        rechazar("decision_format", "decision_result debe ser el dict que devuelve analyze_decision")
    decision = decision_result.get("decision")
    if decision == "WAIT":
        rechazar("decision", "NO TRADE: la decisión es WAIT")
    if decision not in ("BUY", "SELL"):
        rechazar("decision", f"decisión inválida o ausente: {decision!r} (solo se evalúan BUY/SELL del Decision Engine)")
    faltan = [f"decision_result.{k}" for k in ("direction", "timestamp", "valid_until", "input_status") if decision_result.get(k) is None]
    if faltan:
        st.missing += faltan
        rechazar("decision_fields", "la decisión está incompleta: faltan " + ", ".join(faltan))
    if decision_result["input_status"] != "ok":
        rechazar("decision_status", f"la decisión no es válida (input_status={decision_result['input_status']!r})")
    if decision_result["direction"] != _DIR_DE[decision]:
        rechazar("decision_direction", f"decisión inconsistente: {decision} con dirección {decision_result['direction']!r}")
    d_ts, d_valid = _ts(decision_result["timestamp"]), _ts(decision_result["valid_until"])
    if d_ts is None or d_valid is None:
        rechazar("decision_time", "timestamp o valid_until de la decisión no son fechas válidas")
    direccion = decision_result["direction"]
    ctx.update(direction=direccion, side=decision)
    for nombre, v in (("market_data", market_data), ("account_data", account_data), ("instrument_specs", specs)):
        if not isinstance(v, dict):
            st.missing.append(nombre)
            rechazar("input_format", f"{nombre} debe ser un dict, no {type(v).__name__}")

    # ── 1) datos de mercado ──
    simbolo = market_data.get("symbol")
    if not isinstance(simbolo, str) or not simbolo.strip():
        st.missing.append("market_data.symbol")
        rechazar("market_symbol", "falta el símbolo (market_data.symbol)")
    ctx["symbol"] = simbolo
    entrada = market_data.get("entry_price")
    if not _ok_num(entrada) or entrada <= 0:
        st.missing.append("market_data.entry_price")
        rechazar("entry_price", f"precio de entrada inválido o ausente: {entrada!r}")
    ctx["entry_price"] = float(entrada)
    m_ts = _ts(market_data.get("timestamp"))
    if m_ts is None:
        st.missing.append("market_data.timestamp")
        rechazar("market_time", "falta la hora del precio de entrada (market_data.timestamp)")
    try:
        ref = _ts(as_of) if as_of is not None else m_ts
    except Exception:
        ref = None
    if ref is None:
        rechazar("as_of", f"as_of no es una fecha válida: {as_of!r}")
    ctx.update(timestamp=ref, reference_time=ref)
    if d_ts > ref:
        rechazar("future_decision", "la decisión es posterior a la referencia temporal (as_of): información futura")
    if ref > d_valid:
        rechazar("decision_expired", f"la decisión ya no está vigente (valid_until {d_valid} < referencia {ref})")
    if m_ts > ref:
        rechazar("future_price", "el precio de entrada es posterior a la referencia temporal (as_of): información futura")
    edad = (ref - m_ts).total_seconds() / 60.0
    if edad > cfg.max_price_age_minutes:
        rechazar("stale_price", f"el precio de entrada tiene {edad:.1f} min > máximo {cfg.max_price_age_minutes:g} min")
    st.check("time", True, f"decisión vigente, precio de {edad:.1f} min de antigüedad")

    # ── 2) especificaciones del instrumento (sin valores por defecto: si faltan, no se calcula nada) ──
    for k in _ESPECS_REQUERIDAS:
        v = specs.get(k)
        if k == "symbol":
            if not isinstance(v, str) or not v.strip():
                st.missing.append("instrument_specs.symbol")
        elif not _ok_num(v) or v <= 0:
            st.missing.append(f"instrument_specs.{k}")
    if any(m.startswith("instrument_specs.") for m in st.missing):
        rechazar("instrument_specs", "faltan o son inválidas las especificaciones del instrumento: "
                 + ", ".join(m for m in st.missing if m.startswith("instrument_specs.")) + " — no se calcula un tamaño de posición ficticio")
    if _simbolo(specs["symbol"]) != _simbolo(simbolo):
        rechazar("symbol_match", f"el símbolo de las especificaciones ({specs['symbol']}) no coincide con el de la operación ({simbolo})")
    tick, tick_val, contrato = specs["tick_size"], specs["tick_value"], specs["contract_size"]
    vmin, vmax, vstep = specs["volume_min"], specs["volume_max"], specs["volume_step"]
    if vmax < vmin:
        rechazar("specs_consistency", "volume_max es menor que volume_min: especificaciones inconsistentes")
    moneda_cuenta = account_data.get("currency")
    quote = specs.get("quote_currency")
    if isinstance(quote, str) and isinstance(moneda_cuenta, str):
        if quote.strip().upper() == moneda_cuenta.strip().upper():
            esperado = _d(tick) * _d(contrato)
            if abs(_d(tick_val) - esperado) > esperado * Decimal("1e-6"):
                rechazar("specs_consistency", f"especificaciones inconsistentes: con moneda de cotización = moneda de la cuenta, "
                         f"tick_value ({tick_val}) debería ser tick_size×contract_size ({float(esperado)})")
        else:
            st.warnings.append("la moneda de cotización difiere de la de la cuenta: tick_value no se pudo contrastar con tick_size×contract_size")
    else:
        st.warnings.append("no se informó quote_currency o currency: tick_value no se pudo contrastar con tick_size×contract_size")
    conf_in = market_data.get("confluence")
    if isinstance(conf_in, dict):
        if conf_in.get("symbol") is not None and _simbolo(conf_in["symbol"]) != _simbolo(simbolo):
            rechazar("symbol_match", f"el símbolo de la confluencia ({conf_in['symbol']}) no coincide con el de la operación ({simbolo})")

    # ── 3) cuenta ──
    equity = account_data.get("equity")
    if not _ok_num(equity) or equity <= 0:
        st.missing.append("account_data.equity")
        rechazar("equity", f"capital (equity) inválido o ausente: {equity!r}")
    for k in ("daily_pnl", "trades_today", "open_positions"):
        if account_data.get(k) is None:
            st.missing.append(f"account_data.{k}")
    if any(m.startswith("account_data.") for m in st.missing):
        rechazar("account_data", "faltan datos de la cuenta indispensables para verificar los límites: "
                 + ", ".join(m for m in st.missing if m.startswith("account_data.")))
    pnl, trades, abiertas = account_data["daily_pnl"], account_data["trades_today"], account_data["open_positions"]
    if not _ok_num(pnl):
        rechazar("daily_pnl", f"daily_pnl inválido: {pnl!r}")
    if isinstance(trades, bool) or not isinstance(trades, int) or trades < 0:
        rechazar("trades_today", f"trades_today inválido: {trades!r}")
    if not isinstance(abiertas, list):
        rechazar("open_positions", "open_positions debe ser una lista")
    equity = float(equity)
    base_dia = account_data.get("day_start_equity", equity)
    if not _ok_num(base_dia) or base_dia <= 0:
        rechazar("day_start_equity", f"day_start_equity inválido: {base_dia!r}")

    # ── 4) límites de actividad y de pérdida ──
    if trades >= cfg.max_daily_trades:
        rechazar("daily_trades", f"límite de operaciones diarias alcanzado ({trades} ≥ {cfg.max_daily_trades})")
    if len(abiertas) >= cfg.max_open_positions:
        rechazar("open_positions_limit", f"límite de operaciones simultáneas alcanzado ({len(abiertas)} ≥ {cfg.max_open_positions})")
    perdida_hoy = max(0.0, -float(pnl))
    tope_dia = cfg.max_daily_loss_pct / 100.0 * float(base_dia)
    if perdida_hoy >= tope_dia:
        rechazar("daily_loss", f"pérdida diaria máxima alcanzada ({perdida_hoy:.2f} ≥ {tope_dia:.2f})")
    restante_dia = tope_dia - perdida_hoy
    riesgo_abierto = 0.0
    for i, p in enumerate(abiertas):
        if not isinstance(p, dict):
            rechazar("open_positions", f"open_positions[{i}] no es un dict")
        r = p.get("risk_amount")
        if not _ok_num(r) or r < 0:
            st.missing.append(f"account_data.open_positions[{i}].risk_amount")
        else:
            riesgo_abierto += float(r)
        dir_p = _dir_posicion(p.get("direction")) if _simbolo(p.get("symbol", "")) == _simbolo(simbolo) else None
        if _simbolo(p.get("symbol", "")) == _simbolo(simbolo) and dir_p is None:
            st.missing.append(f"account_data.open_positions[{i}].direction")
            rechazar("open_positions", f"la posición abierta #{i} del mismo símbolo no tiene una dirección reconocible: no se puede verificar que no se promedie")
        if dir_p == direccion:
            fl = p.get("floating_pnl")
            if not _ok_num(fl):
                st.missing.append(f"account_data.open_positions[{i}].floating_pnl")
                rechazar("no_averaging", "hay una posición abierta en la misma dirección y no se puede verificar si va perdiendo (falta floating_pnl): no se añade")
            if fl < 0:
                rechazar("no_averaging", "no se aumentan posiciones perdedoras: ya hay una posición abierta en la misma dirección con pérdida flotante")
            st.warnings.append("ya hay una posición abierta ganadora en la misma dirección: se estaría aumentando la exposición")
    if any("open_positions[" in m for m in st.missing):
        rechazar("open_positions", "faltan datos de las posiciones abiertas para calcular la exposición: "
                 + ", ".join(m for m in st.missing if "open_positions[" in m))
    tope_total = cfg.max_total_risk_pct / 100.0 * equity
    restante_total = tope_total - riesgo_abierto
    if restante_total <= 0:
        rechazar("total_exposure", f"exposición máxima alcanzada (riesgo abierto {riesgo_abierto:.2f} ≥ {tope_total:.2f})")
    tope_op = cfg.max_risk_per_trade_pct / 100.0 * equity
    objetivo = cfg.risk_per_trade_pct / 100.0 * equity
    presupuesto = min(objetivo, tope_op, restante_dia, restante_total)
    maximo_perdida = min(tope_op, restante_dia, restante_total)
    ctx_extra = {"maximum_allowed_loss": round(maximo_perdida, 8)}
    ctx.update(ctx_extra)
    if presupuesto < objetivo - 1e-12:
        motivo = "pérdida diaria restante" if presupuesto == restante_dia else "exposición total restante"
        st.warnings.append(f"el riesgo se redujo a {presupuesto:.2f} (de {objetivo:.2f}) por el límite de {motivo}")
    st.check("limits", True, f"límites de actividad/pérdida OK; presupuesto de riesgo {presupuesto:.2f}")

    # ── 5) Stop Loss ──
    niveles = _niveles_validos(market_data, conf_in, ref, direccion, st)
    sl, sl_fuente = _calcular_sl(market_data, niveles, ctx["entry_price"], decision, direccion, cfg, tick, specs, st, rechazar,
                                 setup_id=_setup_de(decision_result))
    ctx["stop_loss"] = sl
    ctx["stop_loss_source"] = sl_fuente
    dist = abs(_d(ctx["entry_price"]) - _d(sl))

    # ── 6) Take Profit ──
    tp, tp_fuente = _calcular_tp(market_data, niveles, ctx["entry_price"], sl, decision, cfg, tick, st, rechazar)
    ctx["take_profit"] = tp
    ctx["take_profit_source"] = tp_fuente
    rr = abs(_d(tp) - _d(ctx["entry_price"])) / dist
    ctx["risk_reward_ratio"] = float(round(rr, 8))
    if not st.check("risk_reward", rr >= _d(cfg.min_risk_reward) - Decimal("1e-12"),
                    f"relación riesgo-beneficio {float(rr):.4f} {'≥' if rr >= _d(cfg.min_risk_reward) else '<'} mínima {cfg.min_risk_reward:g}"):
        raise _Rechazo(f"relación riesgo-beneficio insuficiente: {float(rr):.2f} < mínima {cfg.min_risk_reward:g}")

    # ── 7) tamaño de posición (con las especificaciones reales) ──
    perdida_lote = dist / _d(tick) * _d(tick_val)
    ctx["loss_per_lot"] = float(perdida_lote)
    bruto = _d(round(presupuesto, 10)) / perdida_lote
    ctx["raw_volume"] = float(bruto)
    vmin_d, vstep_d, vmax_d = _d(vmin), _d(vstep), _d(vmax)
    if bruto >= vmin_d:
        n = ((bruto - vmin_d) / vstep_d).to_integral_value(rounding=ROUND_FLOOR)
        vol = vmin_d + n * vstep_d
        n_max = ((vmax_d - vmin_d) / vstep_d).to_integral_value(rounding=ROUND_FLOOR)
        vol_max_red = vmin_d + n_max * vstep_d
        if vol > vol_max_red:
            vol = vol_max_red
            st.warnings.append(f"el volumen se limitó al máximo permitido ({float(vol_max_red):g}); el riesgo es menor que el objetivo")
    else:
        riesgo_min_lote = vmin_d * perdida_lote
        if cfg.allow_min_lot_above_target and float(riesgo_min_lote) <= maximo_perdida + 1e-9:
            vol = vmin_d
            st.warnings.append(f"se usa el lote mínimo ({vmin:g}): su riesgo ({float(riesgo_min_lote):.2f}) supera el objetivo ({presupuesto:.2f}) pero respeta el máximo")
        else:
            ctx["calc_min_lot_risk"] = float(riesgo_min_lote)
            rechazar("min_lot", f"el lote mínimo del instrumento ({vmin:g}) arriesgaría {float(riesgo_min_lote):.2f} > presupuesto de riesgo {presupuesto:.2f}: "
                     "no se redondea hacia arriba")
    riesgo_real = vol * perdida_lote
    beneficio = vol * (abs(_d(tp) - _d(ctx["entry_price"])) / _d(tick)) * _d(tick_val)
    if float(riesgo_real) > maximo_perdida + 1e-9 or float(riesgo_real) > tope_op + 1e-9:
        rechazar("risk_cap", f"el riesgo calculado ({float(riesgo_real):.2f}) superaría el máximo permitido ({maximo_perdida:.2f})")
    if cfg.max_notional_leverage is not None:
        _verificar_apalancamiento(cfg, vol, contrato, ctx["entry_price"], specs, account_data, abiertas, equity, st, rechazar)
    st.check("sizing", True, f"volumen {float(vol):g} {specs.get('volume_unit', 'lots')}, riesgo {float(riesgo_real):.2f} ({float(riesgo_real) / equity * 100:.3f}% del equity)")

    return _salida(cfg, st, True, symbol=simbolo, direction=direccion, side=decision, entry_price=ctx["entry_price"],
                   stop_loss=sl, take_profit=tp, risk_amount=float(round(riesgo_real, 8)),
                   risk_percentage=float(round(riesgo_real / _d(equity) * 100, 8)), reward_amount=float(round(beneficio, 8)),
                   risk_reward_ratio=ctx["risk_reward_ratio"], position_size=float(vol),
                   position_size_unit=specs.get("volume_unit") if isinstance(specs.get("volume_unit"), str) else "lots",
                   maximum_allowed_loss=ctx["maximum_allowed_loss"], stop_loss_source=sl_fuente, take_profit_source=tp_fuente,
                   loss_per_lot=float(perdida_lote), raw_volume=float(bruto), timestamp=ref, reference_time=ref)


# ── niveles: solo con available_at <= referencia ──
def _niveles_validos(market_data: dict, conf: Any, ref: pd.Timestamp, direccion: str, st: _Estado) -> List[dict]:
    out: List[dict] = []
    for i, n in enumerate(market_data.get("levels") or []):
        if not isinstance(n, dict) or n.get("kind") not in KINDS_STOP + ("liquidity",) or n.get("role") not in ("stop", "target") \
                or n.get("direction") not in _ES or not _ok_num(n.get("price")) or n["price"] <= 0:
            st.warnings.append(f"nivel #{i} ignorado: formato inválido")
            continue
        disp = _ts(n.get("available_at"))
        if disp is None:
            st.warnings.append(f"nivel #{i} ({n.get('source_id')}) ignorado: no tiene available_at válido")
            continue
        if disp > ref:
            st.warnings.append(f"nivel #{i} ({n.get('source_id')}) ignorado: se conoció después de la referencia temporal")
            continue
        if n["direction"] != direccion:
            continue
        out.append({"kind": n["kind"], "role": n["role"], "price": float(n["price"]), "available_at": disp,
                    "source_id": str(n.get("source_id") or ""), "label": n.get("label")})
    if isinstance(conf, dict):      # zonas de Order Block de la confluencia (campos reales: all_evidence[].zone)
        c_ts = _ts(conf.get("timestamp"))
        if c_ts is None or c_ts > ref:
            st.warnings.append("la confluencia adjunta es posterior a la referencia temporal o no tiene timestamp: se ignora")
        else:
            for e in conf.get("all_evidence") or []:
                if isinstance(e, dict) and e.get("counted") and e.get("category") == "order_block" and e.get("timeframe") == "ltf" \
                        and e.get("direction") == direccion and isinstance(e.get("zone"), dict):
                    z = e["zone"]
                    precio = z.get("bottom") if direccion == "bullish" else z.get("top")
                    if _ok_num(precio) and precio > 0:
                        out.append({"kind": "order_block", "role": "stop", "price": float(precio), "available_at": c_ts,
                                    "source_id": str(e.get("id")), "label": e.get("label")})
    return out


def _setup_de(decision_result: dict) -> Optional[str]:
    for e in decision_result.get("supporting_evidence") or []:
        if isinstance(e, dict) and e.get("category") == "setup" and isinstance(e.get("id"), str):
            return e["id"].split(":", 2)[-1]
    return None


def _calcular_sl(market_data, niveles, entrada, decision, direccion, cfg, tick, specs, st, rechazar, setup_id=None) -> Tuple[float, str]:
    compra = decision == "BUY"
    e = _d(entrada)
    sentido = -1 if compra else 1

    def valida(sl: float, etiqueta: str) -> Optional[str]:
        lado_ok = (sl < entrada) if compra else (sl > entrada)
        if not lado_ok:
            return f"{etiqueta}: el Stop Loss {sl:g} está del lado incorrecto de la entrada {entrada:g} para un {decision}"
        d_pct = float(abs(e - _d(sl)) / e * 100)
        if d_pct < cfg.min_stop_distance_pct:
            return f"{etiqueta}: distancia del SL {d_pct:.3f}% < mínima {cfg.min_stop_distance_pct:g}%"
        if d_pct > cfg.max_stop_distance_pct:
            return f"{etiqueta}: distancia del SL {d_pct:.3f}% > máxima {cfg.max_stop_distance_pct:g}%"
        mt = specs.get("min_stop_distance_ticks")
        if _ok_num(mt) and float(abs(e - _d(sl)) / _d(tick)) < mt:
            return f"{etiqueta}: el SL está a menos de {mt:g} ticks (mínimo del broker)"
        return None

    prop = market_data.get("proposed_stop_loss")
    if prop is not None:
        if not _ok_num(prop) or prop <= 0:
            rechazar("proposed_sl", f"Stop Loss propuesto inválido: {prop!r}")
        sl = _al_tick(float(prop), tick, "floor" if compra else "ceil")
        if sl != float(prop):
            st.warnings.append(f"el Stop Loss propuesto se redondeó al tick hacia fuera: {prop:g} → {sl:g}")
        problema = valida(sl, "Stop Loss propuesto")
        if problema:
            rechazar("stop_loss", problema.split(": ", 1)[1])
        st.check("stop_loss", True, f"Stop Loss propuesto {sl:g}")
        return sl, "propuesto"

    notas: List[str] = []
    for kind in cfg.sl_source_priority:
        cand = [n for n in niveles if n["role"] == "stop" and n["kind"] == kind]
        # preferencia: el nivel del setup que respalda la decisión; luego el más cercano a la entrada del lado correcto
        cand.sort(key=lambda n: (0 if (setup_id and n["source_id"] == setup_id) else 1, abs(n["price"] - entrada), n["source_id"]))
        for n in cand:
            nivel = n["price"]
            if (nivel >= entrada) if compra else (nivel <= entrada):
                notas.append(f"{kind} {n['source_id']}: el nivel de invalidación {nivel:g} está del lado incorrecto de la entrada")
                continue
            margen = e * _d(cfg.sl_buffer_pct) / Decimal(100)
            bruto = float(_d(nivel) + sentido * margen)
            sl = _al_tick(bruto, tick, "floor" if compra else "ceil")
            problema = valida(sl, f"{kind} {n['source_id']}")
            if problema:
                notas.append(problema)
                continue
            st.check("stop_loss", True, f"Stop Loss {sl:g} desde {kind} '{n['source_id']}' (nivel {nivel:g}, margen {cfg.sl_buffer_pct:g}%)")
            return sl, f"{kind}:{n['source_id']}"
    for nota in notas:
        st.warnings.append(nota)
    detalle = ("; ".join(notas) if notas else "no hay ningún nivel de invalidación disponible (setup / order block / estructura)")
    if not notas:
        st.missing.append("market_data.levels (nivel de invalidación para el Stop Loss)")
    rechazar("stop_loss", "no se puede colocar un Stop Loss válido: " + detalle + ". No se inventan niveles.")
    raise AssertionError("inalcanzable")


def _calcular_tp(market_data, niveles, entrada, sl, decision, cfg, tick, st, rechazar) -> Tuple[float, str]:
    compra = decision == "BUY"
    e, s = _d(entrada), _d(sl)
    dist = abs(e - s)
    redondeo = "floor" if compra else "ceil"          # hacia la entrada: nunca sobreestima el beneficio

    def lado_ok(tp):
        return tp > entrada if compra else tp < entrada

    prop = market_data.get("proposed_take_profit")
    if prop is not None:
        if not _ok_num(prop) or prop <= 0:
            rechazar("proposed_tp", f"Take Profit propuesto inválido: {prop!r}")
        if not lado_ok(float(prop)):
            rechazar("take_profit", f"el Take Profit propuesto {prop:g} está del lado incorrecto de la entrada {entrada:g} para un {decision}")
        tp = _al_tick(float(prop), tick, redondeo)
        if tp == entrada:
            rechazar("take_profit", "el Take Profit propuesto coincide con la entrada tras el redondeo al tick")
        st.check("take_profit", True, f"Take Profit propuesto {tp:g}")
        return tp, "propuesto"

    liq = sorted((n for n in niveles if n["role"] == "target" and n["kind"] == "liquidity"), key=lambda n: (abs(n["price"] - entrada), n["source_id"]))
    validos, descartados = [], []
    for n in liq:
        if not lado_ok(n["price"]):
            descartados.append(f"liquidez {n['source_id']} ({n['price']:g}) del lado incorrecto de la entrada")
            continue
        validos.append(n)
    if cfg.tp_mode in ("liquidity", "liquidity_or_rr"):
        for n in validos:
            tp = _al_tick(n["price"], tick, redondeo)
            rr = abs(_d(tp) - e) / dist
            if rr >= _d(cfg.min_risk_reward):
                st.check("take_profit", True, f"Take Profit {tp:g} en liquidez '{n['source_id']}' (R:R {float(rr):.2f})")
                return tp, f"liquidity:{n['source_id']}"
            descartados.append(f"liquidez {n['source_id']} ({n['price']:g}) da R:R {float(rr):.2f} < mínimo {cfg.min_risk_reward:g}")
        for d_ in descartados:
            st.warnings.append(d_)
        if cfg.tp_mode == "liquidity":
            rechazar("take_profit", "no hay un objetivo de liquidez válido con R:R suficiente (modo 'liquidity'). No se inventan niveles. "
                     + ("; ".join(descartados) if descartados else "No hay niveles de liquidez disponibles."))
        st.warnings.append("no hay objetivo de liquidez válido: se usa el Take Profit por relación riesgo-beneficio")
    bruto = e + (1 if compra else -1) * _d(cfg.target_risk_reward) * dist
    tp = _al_tick(float(bruto), tick, redondeo)
    for n in validos:
        if (n["price"] < tp) if compra else (n["price"] > tp):
            st.warnings.append(f"hay liquidez en {n['price']:g} ({n['source_id']}) antes del Take Profit {tp:g}: el precio podría reaccionar allí")
            break
    st.check("take_profit", True, f"Take Profit {tp:g} por R:R objetivo {cfg.target_risk_reward:g}")
    return tp, f"rr:{cfg.target_risk_reward:g}"


def _verificar_apalancamiento(cfg, vol, contrato, entrada, specs, account_data, abiertas, equity, st, rechazar) -> None:
    quote, cur = specs.get("quote_currency"), account_data.get("currency")
    if not (isinstance(quote, str) and isinstance(cur, str) and quote.strip().upper() == cur.strip().upper()):
        st.missing.append("quote_currency == currency (necesario para el apalancamiento nocional)")
        rechazar("leverage", "max_notional_leverage requiere quote_currency igual a la moneda de la cuenta para calcular el nocional")
    nocional = float(vol * _d(contrato) * _d(entrada))
    abierto = 0.0
    for i, p in enumerate(abiertas):
        v = p.get("notional")
        if not _ok_num(v) or v < 0:
            st.missing.append(f"account_data.open_positions[{i}].notional")
            rechazar("leverage", f"max_notional_leverage requiere el nocional de la posición abierta #{i}")
        abierto += float(v)
    apal = (abierto + nocional) / equity
    if apal > cfg.max_notional_leverage:
        rechazar("leverage", f"el apalancamiento nocional {apal:.2f}× superaría el máximo {cfg.max_notional_leverage:g}×")
    st.check("leverage", True, f"apalancamiento nocional {apal:.2f}× ≤ {cfg.max_notional_leverage:g}×")


# ─────────────────────────────────────────────────────────────────────
#  Utilidades para PRUEBAS e INTERFAZ (analyze_risk NO las usa)
#  Leen los campos reales de los demás módulos para armar la interfaz de entrada. Importan de forma
#  diferida, de modo que `analyze_risk` sigue sin depender de ningún otro módulo del proyecto.
# ─────────────────────────────────────────────────────────────────────
ESPECIFICACIONES_DE_EJEMPLO = {
    # EJEMPLOS DE PRUEBA: NO son las especificaciones de tu broker. Cada broker define las suyas; el motor
    # nunca las supone. Los dos primeros muestran que el MISMO instrumento puede tener contratos distintos.
    "XAUUSD · contrato 100 oz (ejemplo A)": {"symbol": "XAUUSD", "tick_size": 0.01, "tick_value": 1.0, "contract_size": 100.0,
                                             "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01, "quote_currency": "USD", "volume_unit": "lots"},
    "XAUUSD · contrato 10 oz (ejemplo B)": {"symbol": "XAUUSD", "tick_size": 0.01, "tick_value": 0.1, "contract_size": 10.0,
                                            "volume_min": 0.01, "volume_max": 500.0, "volume_step": 0.01, "quote_currency": "USD", "volume_unit": "lots"},
    "EURUSD · lote de 100 000 (ejemplo)": {"symbol": "EURUSD", "tick_size": 0.00001, "tick_value": 1.0, "contract_size": 100000.0,
                                           "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01, "quote_currency": "USD", "volume_unit": "lots"},
}


def cuenta_de_ejemplo(equity: float = 20000.0) -> dict:
    return {"equity": float(equity), "currency": "USD", "daily_pnl": 0.0, "trades_today": 0, "open_positions": []}


def escalar_frames(frames: dict, factor: float) -> dict:
    """Multiplica los precios OHLC (los motores de confluencia son invariantes a la escala): sirve para que
    los escenarios sintéticos tengan un orden de magnitud realista (XAUUSD ~2 300, EURUSD ~1.08)."""
    return {k: (v * float(factor) if v is not None else None) for k, v in frames.items()}


def construir_entradas_desde_pipeline(frames: dict, symbol: str, conf_config: Any = None, decision_config: Any = None,
                                      as_of: Any = None) -> dict:
    """Corre confluencia → decisión y arma `market_data` con niveles tomados de los campos REALES:
       · setups:  registro de detect_setups → `invalidation_price`, `created_at`, estado `state_as_of`
       · swings:  columnas swing_high/swing_low de Market Structure (confirmados `swing_order` velas después)
       · liquidez: pools EQH/EQL de Liquidity aún no operados (objetivos)
    Todo con información disponible en la vela evaluada (as_of o la última)."""
    import numpy as np
    from core.confluence_engine import _evaluar, _preparar, _resolver_config as _cfg_conf
    from core.decision_engine import analyze_decision
    from core.market_structure import analyze_market_structure
    from core.multi_timeframe import _asegurar_datetimeindex
    from core.setup_engine import ACTIVOS, state_as_of

    cfg = _cfg_conf(conf_config)
    prep = _preparar(None, symbol, cfg, frames)
    ltf = prep.tf["ltf"]
    if ltf.n == 0:
        raise ValueError("no hay velas LTF utilizables")
    t = ltf.n - 1
    if as_of is not None:
        ref = pd.Timestamp(as_of)
        if ref.tzinfo is not None:
            ref = ref.tz_convert("UTC").tz_localize(None)
        t = int(np.searchsorted(ltf.avail, np.datetime64(ref, "ns"), side="right") - 1)
        if t < 0:
            raise ValueError("as_of es anterior a la primera vela")
    conf = _evaluar(prep, t)
    ts = pd.Timestamp(ltf.avail[t])
    decision = analyze_decision(conf, decision_config, as_of=ts)

    dfl = _asegurar_datetimeindex(frames["ltf"])
    hi, lo = dfl["high"].to_numpy(dtype=float), dfl["low"].to_numpy(dtype=float)
    orden = cfg.swing_order
    niveles: List[dict] = []
    ms_df, _ = analyze_market_structure(dfl, swing_order=orden, break_using="close")
    for pos in np.flatnonzero(ms_df["swing_low"].to_numpy(dtype=bool)):
        if pos + orden <= t:
            niveles.append({"kind": "structure", "role": "stop", "direction": "bullish", "price": float(lo[pos]),
                            "available_at": pd.Timestamp(ltf.avail[pos + orden]), "source_id": f"swing_low:{int(pos)}"})
    for pos in np.flatnonzero(ms_df["swing_high"].to_numpy(dtype=bool)):
        if pos + orden <= t:
            niveles.append({"kind": "structure", "role": "stop", "direction": "bearish", "price": float(hi[pos]),
                            "available_at": pd.Timestamp(ltf.avail[pos + orden]), "source_id": f"swing_high:{int(pos)}"})
    for s_ in prep.setups:
        snap = state_as_of(s_, ts)
        if snap is not None and snap["status"] in ACTIVOS and s_.get("invalidation_price") is not None:
            niveles.append({"kind": "setup", "role": "stop", "direction": s_["direction"], "price": float(s_["invalidation_price"]),
                            "available_at": pd.Timestamp(s_["created_at"]), "source_id": s_["setup_id"]})
    for p_ in ltf.eq_pools:
        if p_["formed"] <= t and (p_["taken"] is None or p_["taken"] > t):
            niveles.append({"kind": "liquidity", "role": "target", "direction": "bullish" if p_["kind"] == "EQH" else "bearish",
                            "price": float(p_["level"]), "available_at": pd.Timestamp(ltf.avail[p_["formed"]]),
                            "source_id": f"eq:{p_['kind']}:{p_['formed']}"})
    niveles.sort(key=lambda n: (n["kind"], n["role"], n["direction"], n["source_id"]))
    market_data = {"symbol": symbol, "entry_price": float(ltf.c[t]), "timestamp": ts, "levels": niveles, "confluence": conf}
    return {"decision": decision, "market_data": market_data, "confluence": conf}


def escenarios_riesgo() -> Dict[str, dict]:
    """Escenarios simulados (para la interfaz y las pruebas): nombre → {"args": kwargs de analyze_risk, "esperado", "descripcion"}.
    Las operaciones BUY/SELL/WAIT salen de la cadena REAL confluencia → decisión sobre datos sintéticos escalados
    (XAUUSD ≈ 2 370, EURUSD ≈ 1.09). Las especificaciones son EJEMPLOS, no las de ningún broker."""
    from core.confluence_engine import _escenarios_confluencia
    esc = _escenarios_confluencia()
    nom = list(esc)
    kb, ks, kc = esc[nom[0]], esc[nom[1]], esc[nom[2]]
    buy = construir_entradas_desde_pipeline(escalar_frames(kb["frames"], 22.0), "XAUUSD", kb["config"])
    sell = construir_entradas_desde_pipeline(escalar_frames(ks["frames"], 0.01), "EURUSD", ks["config"])
    wait = construir_entradas_desde_pipeline(escalar_frames(kc["frames"], 22.0), "XAUUSD", kc["config"])
    sp = ESPECIFICACIONES_DE_EJEMPLO
    spA, spB, spE = (sp[k] for k in ("XAUUSD · contrato 100 oz (ejemplo A)", "XAUUSD · contrato 10 oz (ejemplo B)", "EURUSD · lote de 100 000 (ejemplo)"))
    ts_dec = pd.Timestamp(buy["decision"]["timestamp"])

    def args(ent, specs, equity=20000.0, **cambios):
        a = {"decision_result": copy.deepcopy(ent["decision"]), "market_data": copy.deepcopy(ent["market_data"]),
             "account_data": cuenta_de_ejemplo(equity), "instrument_specs": copy.deepcopy(specs), "config": None, "as_of": None}
        a.update(cambios)
        return a

    def con(ent, specs, md=None, acc=None, **kw):
        a = args(ent, specs, **kw)
        for k, v in (md or {}).items():
            if v is None:
                a["market_data"].pop(k, None)
            else:
                a["market_data"][k] = v
        for k, v in (acc or {}).items():
            a["account_data"][k] = v
        return a
    e_b = buy["market_data"]["entry_price"]
    S: Dict[str, dict] = {}

    def add(nombre, a, esperado, desc):
        S[nombre] = {"args": a, "esperado": esperado, "descripcion": desc}
    add("1 · BUY aprobado — XAUUSD, contrato 100 oz (ejemplo A)", args(buy, spA), "APPROVED", "Cadena real confluencia→decisión BUY; SL en la invalidación del setup, TP a 2:1.")
    add("2 · BUY aprobado — el MISMO XAUUSD con OTRO contrato (10 oz)", args(buy, spB), "APPROVED", "Mismo riesgo en dinero, lote 10 veces mayor: el contrato sale de las especificaciones, no del motor.")
    add("3 · SELL aprobado — EURUSD", args(sell, spE), "APPROVED", "Cadena real confluencia→decisión SELL; SL por encima de la entrada.")
    add("4 · WAIT → NO TRADE", args(wait, spA), "REJECTED", "Una decisión WAIT nunca produce una operación.")
    add("5 · Stop Loss propuesto del lado incorrecto", con(buy, spA, md={"proposed_stop_loss": e_b * 1.01}), "REJECTED", "BUY con Stop Loss por encima de la entrada.")
    add("6 · Take Profit propuesto del lado incorrecto", con(buy, spA, md={"proposed_take_profit": e_b * 0.99}), "REJECTED", "BUY con Take Profit por debajo de la entrada.")
    add("7 · R:R insuficiente (1.5 < 2)", con(buy, spA, md={"proposed_stop_loss": e_b * 0.995, "proposed_take_profit": e_b * 1.0075}), "REJECTED", "Stop 0.5% y objetivo 0.75%: relación 1.5:1.")
    add("8 · Capital muy pequeño: el lote mínimo excede el riesgo permitido", args(buy, spA, equity=300.0), "REJECTED", "No se redondea hacia arriba un lote que rompería el límite.")
    add("9 · Capital inválido (equity ≤ 0)", args(buy, spA, equity=-100.0), "REJECTED", "Nunca se calculan posiciones con equity inválido.")
    add("10 · Datos faltantes (sin precio de entrada)", con(buy, spA, md={"entry_price": None}), "REJECTED", "Falta un dato indispensable.")
    add("11 · Sin niveles de invalidación disponibles", con(buy, spA, md={"levels": [], "confluence": None}), "REJECTED", "No se inventa un Stop Loss.")
    sp_falta = copy.deepcopy(spA); sp_falta.pop("tick_value")
    add("12 · Especificaciones del contrato incompletas (falta tick_value)", args(buy, sp_falta), "REJECTED", "No se calcula un lote ficticio.")
    add("13 · Pérdida diaria máxima alcanzada", con(buy, spA, acc={"daily_pnl": -420.0}), "REJECTED", "Pérdida de 420 ≥ 2% de 20 000.")
    add("14 · Límite de operaciones diarias alcanzado", con(buy, spA, acc={"trades_today": 3}), "REJECTED", "3 de 3 operaciones del día.")
    add("15 · Ya hay una posición BUY perdedora (no se promedia)", con(buy, spA, acc={"open_positions": [
        {"symbol": "XAUUSD", "direction": "BUY", "risk_amount": 100.0, "floating_pnl": -50.0}]}), "REJECTED", "No se aumentan posiciones perdedoras.")
    add("16 · Decisión vencida (45 min después)", args(buy, spA, as_of=ts_dec + pd.Timedelta(minutes=45)), "REJECTED", "La decisión solo es vigente 30 min.")
    add("17 · Información futura (as_of anterior a la decisión)", args(buy, spA, as_of=ts_dec - pd.Timedelta(minutes=15)), "REJECTED", "La decisión es posterior al momento evaluado.")
    add("18 · Aprobada con riesgo REDUCIDO por la pérdida diaria restante", con(buy, spA, acc={"daily_pnl": -350.0}), "APPROVED", "Quedan $50 de pérdida diaria: el riesgo baja del objetivo de $100.")
    return S
