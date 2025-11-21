#!/usr/bin/env python3
"""
fetch_omni_auto.py – 2025
Coleta OMNI 1-min dos últimos 12 meses
Testa automaticamente quais variáveis a NASA OMNIWeb aceita no dia atual.
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta

# Lista de variáveis desejadas
CANDIDATE_VARS = {
    8: "speed",
    23: "bz",
    25: "density",
    24: "temp",
    78: "bt",
    80: "bx",
    81: "by",
    82: "bz_gse"
}

BASE_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"


def try_var(var_id, start_s, end_s):
    """Testa 1 variável individualmente para ver se o OMNIWeb aceita."""
    print(f"🔍 Testando variável {var_id}…")

    url = (
        f"{BASE_URL}?activity=ftp&res=1min&spacecraft=omni2"
        f"&start_date={start_s}&end_date={end_s}&vars={var_id}"
    )

    r = requests.get(url)
    txt = r.text

    # NASA responde erro textual
    if "Error" in txt or "wrong variable" in txt:
        print(f"❌ VAR {var_id} REJEITADA pela NASA hoje.")
        return False

    # Tenta descobrir se há link para o .lst
    link = re.search(r'href="(https?://[^"]+\.lst)"', txt)
    if not link:
        print(f"⚠️ VAR {var_id} sem .lst — provavelmente rejeitada.")
        return False

    print(f"✅ VAR {var_id} aceita!")
    return True


def fetch():
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    print(f"\n📡 Coletando OMNI 1-min de {start_s} até {end_s}")
    print("🔬 Testando variáveis disponíveis hoje…\n")

    accepted = []

    # 1) testa todas as variáveis
    for vid in CANDIDATE_VARS:
        if try_var(vid, start_s, end_s):
            accepted.append(vid)

    if not accepted:
        raise RuntimeError("❌ Nenhuma variável disponível hoje! NASA deve estar em manutenção.")

    print("\n📌 Variáveis aceitas hoje:", accepted)

    # 2) Monta URL final só com as variáveis válidas
    params = f"activity=ftp&res=1min&spacecraft=omni2&start_date={start_s}&end_date={end_s}"
    for v in accepted:
        params += f"&vars={v}"

    final_url = f"{BASE_URL}?{params}"

    print("\n⬇ Baixando arquivo completo…")
    r = requests.get(final_url)
    txt = r.text

    # acha link real
    link = re.search(r'href="(https?://[^"]+\.lst)"', txt)
    if not link:
        raise RuntimeError("❌ NASA não devolveu link para .lst")

    lst_url = link.group(1).replace("&amp;", "&")
    print(f"📎 Link real: {lst_url}")

    data = requests.get(lst_url).text
    lines = [l for l in data.splitlines() if l.strip() and not l.startswith("#")]

    rows = [re.split(r"\s+", line.strip()) for line in lines]
    df = pd.DataFrame(rows)

    # 4 colunas de tempo sempre presentes
    df.columns = ["year", "doy", "hour", "minute"] + [
        CANDIDATE_VARS[v] for v in accepted
    ]

    # timestamp
    def make_ts(r):
        try:
            return datetime.strptime(
                f"{r['year']} {r['doy']} {r['hour']} {r['minute']}",
                "%Y %j %H %M"
            )
        except:
            return pd.NaT

    df["timestamp"] = df.apply(make_ts, axis=1)
    df = df.dropna(subset=["timestamp"])

    # converte numérico e remove 99999
    for col in df.columns[4:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace(
        [99999.9, 9999.9, 999.9, -9999.9],
        pd.NA
    ).dropna()

    # salva
    os.makedirs("data_real", exist_ok=True)
    out = "data_real/omni_last12m_auto.csv"
    df.to_csv(out, index=False)

    print(f"\n✅ SUCESSO! {len(df):,} linhas salvas em {out}")

    return out


if __name__ == "__main__":
    fetch()
