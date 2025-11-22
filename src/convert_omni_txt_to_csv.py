#!/usr/bin/env python3
"""
convert_omni_txt_to_csv.py
Converte arquivos baixados manualmente do OMNIWeb (.lst.txt)
para um CSV estruturado e limpo, pronto para o HAC.

Suporta:
- Formato YEAR MONTH DAY VALUE1 VALUE2 ...
- DAY = 0  (último dia do mês anterior)
"""

import os
import pandas as pd
from datetime import datetime, timedelta

INPUT_FILE = "src/omni2_of3LE00pQF.txt"
OUTPUT_DIR = "data_real"
OUTPUT_FILE = f"{OUTPUT_DIR}/omni_converted.csv"


# ---------------------------------------------------------
# 🔍 Detecta se o arquivo está no formato Y-M-D
# ---------------------------------------------------------
def detect_format(first_data_line):
    parts = first_data_line.split()
    if len(parts) >= 4:
        # Exemplo: "2024 8 0  2.7  -2.3 ..."
        if len(parts[0]) == 4:
            return "YMD"
    return None


# ---------------------------------------------------------
# ⏳ Função para transformar cada linha em timestamp + valores
# ---------------------------------------------------------
def parse_line(parts):
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])  # Pode ser 0!

    # ---------------------------------------------------------
    # 💥 Caso DAY = 0 → último dia do mês anterior
    # ---------------------------------------------------------
    if day == 0:
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        # Último dia do mês (robusto)
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        last_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).day

        day = last_day

    # ---------------------------------------------------------
    # 📅 Criar timestamp
    # ---------------------------------------------------------
    try:
        ts = datetime(year, month, day)
    except ValueError:
        return None, None

    # ---------------------------------------------------------
    # 🔢 Converter valores numéricos
    # ---------------------------------------------------------
    values = []
    for x in parts[3:]:
        try:
            f = float(x)
            # Filtrar valores inválidos típicos da NASA
            if f not in (9999, 999.9, 99999.9, -999.9, -1e5):
                values.append(f)
        except:
            values.append(None)

    return ts, values


# ---------------------------------------------------------
# 📥 Processa o arquivo inteiro
# ---------------------------------------------------------
def parse_file(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Filtrar cabeçalhos (# e textos do HTML)
    data_lines = [l for l in lines if l[0].isdigit()]

    if len(data_lines) == 0:
        print("❌ Nenhuma linha numérica encontrada!")
        return None

    # Detectar formato
    fmt = detect_format(data_lines[0])
    print(f"🔍 Formato detectado: {fmt}")

    rows = []
    for line in data_lines:
        parts = line.split()
        ts, values = parse_line(parts)
        if ts is not None:
            rows.append([ts] + values)

    print(f"✔️ Linhas processadas: {len(rows)}")

    if len(rows) == 0:
        print("❌ Nenhuma linha válida processada.")
        return None

    # Criar DataFrame
    df = pd.DataFrame(rows)
    df.rename(columns={0: "timestamp"}, inplace=True)

    # Nomear colunas numericamente
    for i in range(1, df.shape[1]):
        df.rename(columns={i: f"var_{i}"}, inplace=True)

    return df


# ---------------------------------------------------------
# 💾 Salvar CSV
# ---------------------------------------------------------
def save_csv(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n💾 CSV salvo em: {OUTPUT_FILE}")
    print(f"📈 Total de linhas: {len(df)}")
    print("📌 Primeiras linhas:")
    print(df.head())


# ---------------------------------------------------------
# 🚀 MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"📂 Lendo arquivo: {INPUT_FILE}")

    df = parse_file(INPUT_FILE)

    if df is not None:
        save_csv(df)
        print("\n🎯 Pronto para usar no HAC training!")
    else:
        print("\n❌ Falha na conversão.")
