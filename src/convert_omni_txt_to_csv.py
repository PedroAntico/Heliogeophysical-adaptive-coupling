#!/usr/bin/env python3
"""
Converte arquivo OMNI .txt (baixado do OMNIWeb) para um CSV limpo
Arquivos como: omni2_of3LE00pQF.txt
Detecta automaticamente colunas, timestamp, limpa valores faltantes.
"""

import os
import re
import pandas as pd
from datetime import datetime

INPUT_FILE = "src/omni2_of3LE00pQF.txt"
OUTPUT_DIR = "data_real"
OUTPUT_FILE = "data_real/omni_converted.csv"

def parse_omni_txt(path):
    print(f"📂 Lendo arquivo OMNI: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = re.split(r"\s+", line)

            if len(parts) < 6:
                continue

            try:
                year = int(parts[0])
                doy = int(parts[1])
                hour = int(parts[2])
                minute = int(parts[3])

                ts = datetime.strptime(f"{year} {doy}", "%Y %j")
                ts = ts.replace(hour=hour, minute=minute)

                values = [
                    float(x) if x not in ("9999", "99999", "999.9", "99999.9") else None
                    for x in parts[4:]
                ]

                row = [ts] + values
                rows.append(row)

            except Exception:
                continue

    print(f"✔️ Linhas processadas: {len(rows)}")

    df = pd.DataFrame(rows)

    colnames = ["timestamp"] + [f"var_{i}" for i in range(1, df.shape[1])]
    df.columns = colnames

    print("\n📌 Primeiras linhas:")
    print(df.head())

    return df


def save_csv(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 CSV salvo em: {OUTPUT_FILE}")
    print(f"Total de linhas: {len(df)}")


if __name__ == "__main__":
    df = parse_omni_txt(INPUT_FILE)
    save_csv(df)
    print("\n🎯 Pronto para usar no HAC training!")
