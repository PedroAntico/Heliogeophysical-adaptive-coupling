#!/usr/bin/env python3
"""
fetch_omni.py - Versão corrigida 2025
Baixa OMNI 1-minute dos últimos 12 meses (API OMNIWeb + parser robusto)
Salva em data_real/omni_last12m.csv pronto para o HAC 5.1
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta

# Variáveis que você realmente usa no HAC 5.1
VARS = [8, 23, 25, 24, 78, 80, 81, 82]  # speed, density, temp, pressure, Bt, Bx, By, Bz

OMNI_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

def build_url(start_date: str, end_date: str) -> str:
    params = f"activity=ftp&res=1min&spacecraft=omni2&start_date={start_date}&end_date={end_date}&vars={'&vars='.join(map(str, VARS))}"
    return f"{OMNI_URL}?{params}"

def fetch_last_year():
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    print(f"Baixando OMNI 1-min de {start_s} até {end_s}...")

    url = build_url(start_s, end_s)
    r = requests.get(url)
    r.raise_for_status()

    # Novo regex 2025 (o link agora fica dentro de <a href="...">)
    match = re.search(r'href="(https?://omniweb\.gsfc\.nasa\.gov[^"]+\.lst)"', r.text)
    if not match:
        raise RuntimeError("Não encontrou o link do .lst – NASA mudou o HTML de novo?")
    
    lst_url = match.group(1).replace("&amp;", "&")
    print(f"Link encontrado: {lst_url}")

    data = requests.get(lst_url).text

    lines = [l for l in data.splitlines() if not l.startswith('#') and l.strip()]

    # Parser fixo correto do OMNI 1-min (colunas de largura fixa + múltiplos espaços)
    rows = []
    for line in lines:
        # Divide por múltiplos espaços
        parts = re.split(r'\s+', line.strip())
        if len(parts) < 3: 
            continue
        rows.append(parts)

    df = pd.DataFrame(rows)

    # Primeiras colunas sempre: YEAR, DOY, Hour, Minute  (agora são 4 colunas!)
    df.columns = ["year", "doy", "hour", "minute"] + [f"v{i}" for i in range(len(df.columns)-4)]

    # Só pega as colunas que pedimos + timestamp
    df = df.iloc[:, :4 + len(VARS)]

    # Monta timestamp corretamente
    def make_ts(row):
        try:
            return datetime.strptime(f"{row['year']} {row['doy']} {row['hour']} {row['minute']}", "%Y %j %H %M")
        except:
            return pd.NaT

    df["timestamp"] = df.apply(make_ts, axis=1)
    df = df.dropna(subset=["timestamp"])

    # Colunas finais
    colunas_finais = ["timestamp", "speed", "density", "temp", "pressure", "Bt", "Bx", "By", "Bz"]
    df = df[["timestamp"] + [f"v{i}" for i in range(len(VARS))]]
    df.columns = colunas_finais

    # Converte para número (9999.9, 99999. etc são valores faltantes no OMNI)
    for c in colunas_finais[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove valores absurdos/faltantes típicos do OMNI
    df = df.replace([99999.9, 9999.9, 999.9, -9999.9], pd.NA)
    df = df.dropna()

    os.makedirs("data_real", exist_ok=True)
    caminho = "data_real/omni_last12m.csv"
    df.to_csv(caminho, index=False)
    
    print(f"Sucesso! {len(df):,} linhas salvas em {caminho}")
    return caminho

if __name__ == "__main__":
    fetch_last_year()
