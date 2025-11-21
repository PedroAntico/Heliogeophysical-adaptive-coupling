#!/usr/bin/env python3
"""
fetch_omni.py — versão blindada
Baixa dados OMNI 1-minute dos últimos 12 meses
Compatível com TODOS os formatos de link da NASA/OMNIWeb.
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta


def extract_any_omni_link(text):
    """
    Extrai qualquer link OMNI válido (.lst, .txt, .dat, .tmp)
    cobrindo TODOS os padrões atuais do OMNIWeb.
    """

    patterns = [
        r"(https?://[\w\./\-]+omni2_[A-Za-z0-9_]+\.lst)",
        r"(https?://[\w\./\-]+omni2_[A-Za-z0-9_]+\.txt)",
        r"(https?://[\w\./\-]+omni2_[A-Za-z0-9_]+\.dat)",
        r"(https?://[\w\./\-]+omni2_[A-Za-z0-9_]+\.tmp)",
        r"(https?://[\w\./\-]+omni2_[A-Za-z0-9_]+)",  # fallback genérico
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)

    return None


def build_request_url(start_date, end_date):
    """Url para OMNI ftp format."""
    base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    vars = [8, 23, 25, 24, 78, 80, 81, 82]

    params = (
        f"activity=ftp"
        f"&res=1min"
        f"&spacecraft=omni2"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&maxdays=366"
    )

    for v in vars:
        params += f"&vars={v}"

    return f"{base}?{params}"


def fetch_last_year():
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    sd = start.strftime("%Y%m%d")
    ed = end.strftime("%Y%m%d")

    print(f"📡 Baixando OMNI 1-min de {sd} até {ed}")

    url = build_request_url(sd, ed)

    print("🔍 Obtendo resposta inicial da NASA...")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Erro OMNIWeb: HTTP {resp.status_code}")

    text = resp.text

    print("🔍 Extraindo link real do arquivo...")
    lst_url = extract_any_omni_link(text)

    if lst_url is None:
        print("❌ Conteúdo recebido (primeiras 400 chars):\n")
        print(text[:400])
        raise RuntimeError("❌ Não consegui encontrar o link do arquivo real (.lst/.txt/.dat/.tmp).")

    print(f"📎 Link detectado: {lst_url}")
    print("⬇ Baixando arquivo real...")

    data = requests.get(lst_url)
    if data.status_code != 200:
        raise RuntimeError("❌ Falha ao baixar arquivo real")

    raw = data.text

    print("📐 Convertendo dados...")

    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = re.split(r"\s+", line)
        rows.append(parts)

    df = pd.DataFrame(rows)

    colnames = ["year", "doy", "hhmm"]
    for v in [8, 23, 25, 24, 78, 80, 81, 82]:
        colnames.append(f"var_{v}")

    df = df.iloc[:, :len(colnames)]
    df.columns = colnames

    print("🕒 Corrigindo timestamps...")
    ts = []
    for _, row in df.iterrows():
        try:
            ts.append(
                datetime.strptime(
                    f"{row['year']} {int(row['doy']):03d} {str(row['hhmm']).zfill(4)}",
                    "%Y %j %H%M"
                )
            )
        except:
            ts.append(pd.NaT)

    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])

    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    os.makedirs("data_real", exist_ok=True)
    out = "data_real/omni_last12m.csv"

    df.to_csv(out, index=False)

    print(f"✅ Dados salvos em: {out}")
    print(f"📊 Linhas: {len(df)}")

    return out


if __name__ == "__main__":
    try:
        p = fetch_last_year()
        print("\n🎉 Finalizado com sucesso!\n")
    except Exception as e:
        print("\n❌ ERRO CRÍTICO:", e)
        print("Tente rodar novamente em 1 minuto — o OMNIWeb às vezes troca de formato.")
