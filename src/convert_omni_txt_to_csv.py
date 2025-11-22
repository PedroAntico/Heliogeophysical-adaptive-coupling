#!/usr/bin/env python3
"""
Parser universal para arquivos OMNI baixados via OMNIWeb (.lst.txt)
Detecta automaticamente se o formato é:
A) ANO DOY HORA MIN VARS...
B) ANO MÊS DIA VARS...
C) ANO MÊS DIA HORA VARS...
"""

import os
import re
import pandas as pd
from datetime import datetime

INPUT_FILE = "src/omni2_of3LE00pQF.txt"
OUTPUT_DIR = "data_real"
OUTPUT_FILE = "data_real/omni_converted.csv"


def detect_format(parts):
    """
    Detecta o formato da linha analisando a quantidade e tipo de colunas
    """

    # Ex: 2024 11 22 6.3 -1.2 ...
    if len(parts) >= 5:
        if parts[1].isdigit() and parts[2].isdigit():
            return "YMD"  # Ano mês dia + valores
    
    # Ex: 2024 320 13 00 var var...
    if parts[1].isdigit() and len(parts) >= 6:
        if 1 <= int(parts[1]) <= 366:  # DOY
            return "YDOYHM"
    
    return "UNKNOWN"


def parse_line(parts, detected_format):
    """Converte uma linha para timestamp + valores"""

    if detected_format == "YDOYHM":
        year = int(parts[0])
        doy = int(parts[1])
        hour = int(parts[2])
        minute = int(parts[3])
        ts = datetime.strptime(f"{year} {doy}", "%Y %j").replace(hour=hour, minute=minute)
        values = parts[4:]
        return ts, values

    if detected_format == "YMD":
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])

        # Sem hora → assume 00:00
        ts = datetime(year, month, day)
        values = parts[3:]
        return ts, values

    return None, None


def parse_file(path):
    print(f"📂 Lendo arquivo: {path}")

    rows = []
    detected_format = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = re.split(r"\s+", line.strip())

            if detected_format is None:
                detected_format = detect_format(parts)
                print(f"🔍 Formato detectado: {detected_format}")

                if detected_format == "UNKNOWN":
                    print("❌ Formato desconhecido. Envie 20 linhas aqui no chat.")
                    return None

            ts, values = parse_line(parts, detected_format)

            if ts is None:
                continue

            clean_values = [
                float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else None
                for v in values
            ]

            rows.append([ts] + clean_values)

    if not rows:
        print("❌ Nenhuma linha válida encontrada.")
        return None

    df = pd.DataFrame(rows)
    colnames = ["timestamp"] + [f"var_{i}" for i in range(1, df.shape[1])]
    df.columns = colnames

    print(f"✔️ Linhas processadas: {len(df)}")
    return df


def save(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 CSV salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    df = parse_file(INPUT_FILE)
    if df is not None:
        save(df)
        print("🚀 PRONTO!")
