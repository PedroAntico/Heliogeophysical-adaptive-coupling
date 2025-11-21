#!/usr/bin/env python3
"""
fetch_omni.py
Baixa dados OMNI 1-minute dos últimos 12 meses usando a API oficial do OMNIWeb
Formato final: CSV limpo, pronto para treino HAC 5.1

Autor: Pedro Guilherme Antico
Versão: 1.0
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta

OMNI_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Campos necessários (OMNI variable IDs)
# 1-minute OMNI variable indices:
# speed = 8, density = 23, temperature = 25, Bz GSM = 82, Bt = 78,
# Bx GSM = 80, By GSM = 81, proton flux = 130, solar wind pressure = 24
VARS = [
    8,      # speed
    23,     # density
    25,     # temperature
    24,     # pressure
    78,     # Bt
    80,     # Bx
    81,     # By
    82,     # Bz
]

def build_request_url(start_date, end_date):
    """Constrói URL para OMNIWeb data ftp request sem header."""
    params = (
        f"activity=ftp"
        f"&res=1min"
        f"&spacecraft=omni2"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&maxdays=366"
    )
    for v in VARS:
        params += f"&vars={v}"

    return f"{OMNI_URL}?{params}"

def fetch_last_year():
    """Baixa dados OMNI dos últimos 12 meses."""
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    print(f"📡 Baixando OMNI 1-min de {start_s} até {end_s}")

    url = build_request_url(start_s, end_s)

    # 1) Baixa a página contendo o link para o arquivo real .lst
    print("🔍 Obtendo link do arquivo real...")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Erro ao acessar OMNIWeb: {response.status_code}")

    text = response.text

    # Extrai o link real estilo:
    # http://omniweb.gsfc.nasa.gov/staging/omni2_XXXXXX.lst
    match = re.search(r"(https?://[^\s]+omni2_[A-Za-z0-9_]+\.lst)", text)
    if not match:
        raise RuntimeError("❌ Não consegui encontrar o link do arquivo .lst")

    lst_url = match.group(1)
    print(f"📎 Arquivo real encontrado: {lst_url}")

    # 2) Baixa o arquivo real .lst
    print("⬇ Baixando dados...")
    lst_data = requests.get(lst_url)
    if lst_data.status_code != 200:
        raise RuntimeError("❌ Falha ao baixar o arquivo .lst")

    raw_text = lst_data.text

    # 3) Converter .lst em tabela estruturada
    print("📐 Formatando dados...")

    # O formato .lst é fix-width com espaços
    # Vamos separar por qualquer espaçamento
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        rows.append(parts)

    # Criar DataFrame
    df = pd.DataFrame(rows)

    # OMNI 1-minute format: first 3 cols are year, day, minute
    # Depois vêm as variáveis na ordem exata solicitada
    colnames = ["year", "doy", "hhmm"]
    colnames += [f"var_{v}" for v in VARS]

    # Ajustar número de colunas
    df = df.iloc[:, :len(colnames)]
    df.columns = colnames

    # Converter timestamp
    print("🕒 Convertendo timestamps...")
    timestamps = []
    for i, row in df.iterrows():
        try:
            ts = datetime.strptime(
                f"{row['year']} {int(row['doy']):03d} {str(row['hhmm']).zfill(4)}",
                "%Y %j %H%M"
            )
        except:
            ts = pd.NaT
        timestamps.append(ts)

    df["timestamp"] = timestamps

    # Remover linhas inválidas
    df = df.dropna(subset=["timestamp"])

    # Reordenar colunas
    df = df[["timestamp"] + colnames[3:]]

    # Converter tudo para numérico
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    # Criar diretório
    os.makedirs("data_real", exist_ok=True)
    out_path = "data_real/omni_last12m.csv"

    df.to_csv(out_path, index=False)
    print(f"✅ Dados salvos em: {out_path}")
    print(f"📊 Linhas finais: {len(df)}")

    return out_path

if __name__ == "__main__":
    path = fetch_last_year()
    print(f"🎉 Finalizado! Arquivo: {path}")
