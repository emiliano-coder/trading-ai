# ═══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR — Genera señales y las guarda en signal.json
#  Este módulo se importa dentro de app.py (Streamlit)
#  No corre solo — es parte de la app
# ═══════════════════════════════════════════════════════════════════

import json
import base64
import requests
from datetime import datetime, timezone


SIGNAL_FILE = "signal.json"


def build_signal(pred: int, prob: list, precio: float, sl_long: float,
                 tp_long: float, sl_short: float, tp_short: float,
                 conf: float, smc_bias: str, wyckoff_trend: str) -> dict:
    """
    Construye el dict de señal estandarizado para ser leído por telegram_bot.py
    """
    if pred == 1:
        action = "BUY"
        sl     = sl_long
        tp     = tp_long
    elif pred == -1:
        action = "SELL"
        sl     = sl_short
        tp     = tp_short
    else:
        action = "NO TRADE"
        sl     = sl_long
        tp     = tp_long

    # ID único basado en timestamp
    ts = datetime.now(timezone.utc)
    uid = ts.strftime("%Y%m%d%H%M%S")

    return {
        "id":             uid,
        "symbol":         "XAUUSD",
        "action":         action,
        "entry":          round(precio, 2),
        "sl":             round(sl, 2),
        "tp":             round(tp, 2),
        "confidence":     round(conf, 2),
        "smc_bias":       smc_bias,
        "wyckoff_trend":  wyckoff_trend,
        "timestamp":      ts.isoformat(),
    }


def save_signal_github(signal: dict, gh_token: str, gh_repo: str):
    """
    Guarda signal.json en GitHub para que telegram_bot.py lo lea.
    telegram_bot.py debe apuntar a este archivo en el repo.
    """
    if not gh_token or not gh_repo:
        return False
    try:
        url = f"https://api.github.com/repos/{gh_repo}/contents/{SIGNAL_FILE}"
        r   = requests.get(url, headers={"Authorization": f"token {gh_token}"}, timeout=5)
        sha = r.json().get('sha', '') if r.status_code == 200 else ''
        cnt = base64.b64encode(json.dumps(signal, indent=2).encode()).decode()
        payload = {"message": "MIMI-AI signal update", "content": cnt}
        if sha:
            payload["sha"] = sha
        result = requests.put(url,
                              headers={"Authorization": f"token {gh_token}"},
                              json=payload, timeout=5)
        return result.status_code in [200, 201]
    except Exception as e:
        return False


def load_signal_from_github(gh_token: str, gh_repo: str) -> dict:
    """Carga signal.json desde GitHub (para telegram_bot.py local)."""
    try:
        url = f"https://api.github.com/repos/{gh_repo}/contents/{SIGNAL_FILE}"
        r   = requests.get(url, headers={"Authorization": f"token {gh_token}"}, timeout=5)
        if r.status_code == 200:
            content = base64.b64decode(r.json().get('content', '')).decode()
            return json.loads(content)
    except:
        pass
    return {}
