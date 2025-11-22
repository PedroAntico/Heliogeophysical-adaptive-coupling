#!/usr/bin/env python3
"""
prepare_real_data.py - versão corrigida
Compatível com o arquivo omni_labeled.csv
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

INPUT_FILE = "data_real/omni_labeled.csv"
OUTPUT_DATA = "data_real/omni_scaled.npy"
OUTPUT_SCALERS = "data_real/scalers.pkl"

# Mapeamento correto com base no rename_omni_columns.py
RENAME_MAP = {
    "Speed": "speed",
    "Density": "density",
    "Bz_GSE": "bz",
    "Pressure": "pressure",
    "Bt": "bt"
}

FEATURES = ["speed", "density", "bz", "pressure", "bt"]


def load_and_prepare():
    print("📂 Lendo dados:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    print("🔧 Renomeando colunas para padrão HAC...")
    df = df.rename(columns=RENAME_MAP)

    print("📊 Colunas atuais:", df.columns.tolist())

    # Verificação
    for feat in FEATURES:
        if feat not in df.columns:
            raise ValueError(f"❌ ERRO: coluna '{feat}' não encontrada no CSV!")

    print("✔️ Todas as colunas essenciais encontradas!")

    return df


def scale_data(df):
    print("🔧 Normalizando dados...")
    scalers = {}
    scaled = pd.DataFrame()

    for col in FEATURES:
        sc = MinMaxScaler()
        scaled[col] = sc.fit_transform(df[col].values.reshape(-1, 1)).flatten()
        scalers[col] = sc

    print("✔️ Normalização concluída!")
    return scaled, scalers


def save_outputs(scaled, scalers):
    import pickle

    print("💾 Salvando dados normalizados...")
    np.save(OUTPUT_DATA, scaled.values)

    print("💾 Salvando scalers...")
    with open(OUTPUT_SCALERS, "wb") as f:
        pickle.dump(scalers, f)

    print("🎉 Arquivos salvos:")
    print(" •", OUTPUT_DATA)
    print(" •", OUTPUT_SCALERS)


def main():
    df = load_and_prepare()
    scaled, scalers = scale_data(df)
    save_outputs(scaled, scalers)

    print("\n🎯 TUDO PRONTO!")
    print("Agora você já pode treinar o HAC modelo real-time.")


if __name__ == "__main__":
    main()
