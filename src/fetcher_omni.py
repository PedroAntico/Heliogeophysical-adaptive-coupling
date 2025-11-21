#!/usr/bin/env python3
"""
fetch_omni.py — versão fixa 2025
OMNI 1-min com variáveis estáveis (speed, Bz).
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta


def extract_file_link(text):
    """Extrai qualquer link .lst, .txt, .dat, .tmp da página."""
    patterns = [
        r"(https?://[^\s\"']+\.lst)",
        r"(https?://[^\s\"']+\.txt)",
        r"(https?://[^\s\"']+\.dat)",
        r"(https?://[^\s\"']+\.tmp)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return None


def fetch_last_year():
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    sd = start.strftime("%Y%m%d")
    ed = end.strftime("%Y%m%d")

    print(f"📡 OMNI 1-min de {sd} até {ed}")

    base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

    # 🚨 Apenas variáveis 100% estáveis
    VARS = [8, 23]

    params = (
        f"activity=ftp"
        f"&res=1min"
        f"&spacecraft=omni2"
        f"&start_date={sd}"
        f"&end_date={ed}"
        f"&maxdays=400"
    )

    for v in VARS:
        params += f"&vars={v}"

    url = f"{base}?{params}"

    print("🔍 Solicitação à NASA...")
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Erro HTTP {r.status_code}")

    text = r.text

    if "Error" in text or "wrong variable" in text:
        print("\n❌ A NASA rejeitou variáveis solicitadas.")
        print("Conteúdo recebido:\n")
        print(text[:400])
        raise RuntimeError("Variáveis inválidas para este dataset.")

    print("🔍 Encontrando link real...")
    link = extract_file_link(text)
    if not link:
        print("❌ Nenhum link encontrado")
        print(text[:400])
        raise RuntimeError("Não foi possível extrair o arquivo real.")

    print(f"📎 Link real: {link}")
    print("⬇ Baixando arquivo...")
    data = requests.get(link).text

    rows = []
    for line in data.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append(re.split(r"\s+", line.strip()))

    df = pd.DataFrame(rows)

    expected_cols = ["year", "doy", "hhmm", "speed", "bz"]
    df = df.iloc[:, :len(expected_cols)]
    df.columns = expected_cols

    ts = []
    for _, row in df.iterrows():
        try:
            ts.append(datetime.strptime(
                f"{row['year']} {int(row['doy']):03d} {str(row['hhmm']).zfill(4)}",
                "%Y %j %H%M"
            ))
        except:
            ts.append(None)

    df["timestamp"] = ts
    df = df.dropna()

    for c in ["speed", "bz"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    os.makedirs("data_real", exist_ok=True)
    out = "data_real/omni_last12m_basic.csv"
    df.to_csv(out, index=False)

    print(f"✅ Salvo em: {out}")
    print(f"📊 Linhas: {len(df)}")

    return out


if __name__ == "__main__":
    fetch_last_year()
