#!/usr/bin/env python3
"""
solar_poller.py
Coleta periódica de parâmetros de clima espacial, salva em CSV e dispara alertas simples.
Configure ENDPOINTS e THRESHOLDS abaixo.

Execute: python3 solar_poller.py --once
ou: systemd / cron / GitHub Actions para rodar periodicamente.
"""

import os
import time
import requests
import csv
import argparse
from datetime import datetime, timezone
import json

# ---------- CONFIGURAÇÃO ----------
# URL(s) a consultar: coloque aqui a URL JSON/CSV do seu provedor (SWPC, OMNI, ACE, SolarHam, etc)
# Exemplo (não garantido): "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
# Use a URL que você confirmar funcionar. Pode usar múltiplas fontes.
ENDPOINTS = {
    "solar_wind_json": "https://services.swpc.noaa.gov/products/solar-wind/ace-rt.json",
    # "mag_1day": "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json",
}

# Pasta de saída
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# Nome do CSV principal
CSV_FILE = os.path.join(OUT_DIR, "solar_wind_timeseries.csv")

# Thresholds para alertas (ajuste conforme preferir)
THRESHOLDS = {
    "Bz_south_alert_nT": -5.0,   # alerta se Bz <= -5 nT
    "speed_alert_km_s": 700.0,   # velocidade alta
    "density_alert_per_cc": 20.0 # pico de densidade
}

# Telegram alert (opcional) - preencha se quiser alertas via Telegram bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# intervalo padrão entre polls (segundos) — quando for executar localmente
POLL_INTERVAL_S = int(os.getenv("POLL_INTERVAL_S", "300"))  # 5 minutos

# ---------- FUNÇÕES ----------

def fetch_json(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        # alguns endpoints retornam CSV mesmo em .json, trate generico:
        content_type = r.headers.get("Content-Type", "")
        text = r.text.strip()
        # tenta parsear JSON, se não, retorna texto cru
        try:
            return r.json()
        except Exception:
            return text
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] Erro fetch {url}: {e}")
        return None

def parse_ace_rt_json(obj):
    """
    Exemplo de parser para ACE real-time JSON (ajuste conforme formato).
    Retorna um dict padronizado com timestamp UTC e parâmetros chaves.
    """
    if not obj:
        return None
    # Se o endpoint for array de arrays (muitos SWPC endpoints):
    # Estrutura possivel: [[time, ...], [time, ...], ...] - pegar último
    if isinstance(obj, list) and len(obj) > 0:
        # tentar pegar o último registro
        last = obj[-1]
        # dependendo do provider, indices difere; aqui é genérico -> ajuste
        # Exemplo hipotético: [year,month,day,hour,minute,sec,...,Bx,By,Bz,v,density,...]
        # Você precisa mapear para sua fonte real.
        try:
            # tenta identificar se é um timestamp string no primeiro campo
            ts_field = None
            if isinstance(last[0], str):
                ts = last[0]
            else:
                # montar timestamp se tiver campos de data/hora
                ts = datetime.utcnow().isoformat()
            # Busca heurística por valores Bz, speed, density (procura floats)
            # OBS: Substitua por mapeamento correto do seu endpoint.
            # Aqui fazemos um fallback com nomes genéricos.
            # RETURN: {'time':..., 'Bz':..., 'Bt':..., 'speed':..., 'density':...}
            parsed = {
                "time_utc": ts,
                "Bz": try_get_by_key(last, ["bz", "Bz", "B_Z", 6]),
                "Bt": try_get_by_key(last, ["bt", "Bt", "B_T", 7]),
                "speed": try_get_by_key(last, ["speed", "v", "velocity", 10]),
                "density": try_get_by_key(last, ["density", "n", 11])
            }
            return parsed
        except Exception:
            return None
    elif isinstance(obj, dict):
        # se retornar dicionário com campos legíveis:
        # ajuste nomes conforme o provedor
        parsed = {}
        parsed["time_utc"] = obj.get("time_tag") or obj.get("timestamp") or datetime.utcnow().isoformat()
        parsed["Bz"] = obj.get("bz") or obj.get("Bz") or obj.get("B_Z")
        parsed["Bt"] = obj.get("bt") or obj.get("Bt") or obj.get("B_T")
        parsed["speed"] = obj.get("speed") or obj.get("v") or obj.get("velocity")
        parsed["density"] = obj.get("density") or obj.get("n")
        return parsed
    else:
        return None

def try_get_by_key(arr, candidates):
    # tenta encontrar um valor numérico na lista/array por índice(s) ou chave heurística
    for c in candidates:
        try:
            if isinstance(c, int) and c < len(arr):
                val = float(arr[c])
                return val
            # não implementado: lookup por clave string
        except Exception:
            continue
    return None

def append_csv_row(csvfile, rowdict):
    header = ["time_utc", "Bz", "Bt", "speed", "density", "raw_payload"]
    newfile = not os.path.exists(csvfile)
    with open(csvfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if newfile:
            writer.writeheader()
        writer.writerow({
            "time_utc": rowdict.get("time_utc"),
            "Bz": rowdict.get("Bz"),
            "Bt": rowdict.get("Bt"),
            "speed": rowdict.get("speed"),
            "density": rowdict.get("density"),
            "raw_payload": json.dumps(rowdict.get("raw_payload", {}))
        })

def maybe_alert(parsed):
    # checa thresholds e envia alerta se necessário
    alerts = []
    try:
        Bz = parsed.get("Bz")
        if Bz is not None and isinstance(Bz, (int, float)):
            if Bz <= THRESHOLDS["Bz_south_alert_nT"]:
                alerts.append(f"ALERTA: Bz sul detectado {Bz} nT <= {THRESHOLDS['Bz_south_alert_nT']} nT")
        speed = parsed.get("speed")
        if speed is not None and isinstance(speed, (int, float)):
            if speed >= THRESHOLDS["speed_alert_km_s"]:
                alerts.append(f"ALERTA: Velocidade solar alta {speed} km/s >= {THRESHOLDS['speed_alert_km_s']} km/s")
        density = parsed.get("density")
        if density is not None and isinstance(density, (int, float)):
            if density >= THRESHOLDS["density_alert_per_cc"]:
                alerts.append(f"ALERTA: Densidade pico {density} p/cm3 >= {THRESHOLDS['density_alert_per_cc']}")
    except Exception as e:
        print("Erro em checar alerts:", e)
    if alerts:
        msg = "\n".join(alerts) + f"\nTimestamp: {parsed.get('time_utc')}"
        print(msg)
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_alert(msg)
    return alerts

def send_telegram_alert(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print("Erro enviar telegram:", e)
        return False

# ---------- MAIN LOOP ----------
def run_once():
    for name, url in ENDPOINTS.items():
        print(f"[{datetime.utcnow().isoformat()}] Fetching {name} from {url}")
        obj = fetch_json(url)
        parsed = parse_ace_rt_json(obj)
        if parsed is None:
            print("Parsing falhou para endpoint:", name)
            continue
        parsed["raw_payload"] = obj
        # normalizar timestamp
        try:
            # se for string ISO -> usar direto
            parsed["time_utc"] = parsed.get("time_utc") or datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        except:
            parsed["time_utc"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        append_csv_row(CSV_FILE, parsed)
        maybe_alert(parsed)
    print(f"[{datetime.utcnow().isoformat()}] run_once complete.")

def run_daemon(interval_s=POLL_INTERVAL_S):
    print("Starting poller daemon. Ctrl-C to stop.")
    try:
        while True:
            run_once()
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="executa uma vez e sai")
    args = parser.parse_args()
    if args.once:
        run_once()
    else:
        run_daemon()
