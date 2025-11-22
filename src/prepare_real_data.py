²#!/usr/bin/env python3
"""
prepare_real_data.py
Transforma o arquivo OMNI real em dados de treinamento para o HAC
Gera:
  - X_train, X_val, X_test
  - y_train, y_val, y_test
  - escalonadores
  - horizontes: 1, 3, 6, 12, 24, 48 horas
  - lookback padrão: 168 horas (1 semana)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT = "data_real/omni_converted.csv"
OUTPUT = "data_real/hac_ready.npz"

LOOKBACK = 168  # 7 dias de histórico
HORIZONS = [1, 3, 6, 12, 24, 48]

TARGETS = ["speed", "density", "bz_gse"]  # multioutput HAC

def load_data():
    df = pd.read_csv(INPUT, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # interpolar / limpar
    df = df.interpolate().dropna()

    return df

def scale_data(df):
    scalers = {}
    scaled = {}

    for col in TARGETS:
        sc = MinMaxScaler()
        scaled[col] = sc.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = sc

    return scaled, scalers

def create_sequences(scaled):
    X_list = []
    Y_list = {h: [] for h in HORIZONS}

    n = len(scaled[TARGETS[0]])

    for i in range(LOOKBACK, n - max(HORIZONS)):
        # Entrada: janela de 168h com todas as variáveis
        window = np.stack([scaled[col][i-LOOKBACK:i, 0] for col in TARGETS], axis=1)
        X_list.append(window)

        # Saídas para cada horizonte
        for h in HORIZONS:
            future_idx = i + h
            Y_list[h].append([scaled[col][future_idx, 0] for col in TARGETS])

    X = np.array(X_list)
    Y = {h: np.array(Y_list[h]) for h in HORIZONS}

    return X, Y

def split_data(X, Y):
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    split = {}

    split["X_train"] = X[:train_end]
    split["X_val"]   = X[train_end:val_end]
    split["X_test"]  = X[val_end:]

    for h in HORIZONS:
        split[f"y_train_h{h}"] = Y[h][:train_end]
        split[f"y_val_h{h}"]   = Y[h][train_end:val_end]
        split[f"y_test_h{h}"]  = Y[h][val_end:]

    return split

def main():
    print("📂 Lendo dados reais...")
    df = load_data()

    print(f"✔️ Dados carregados: {len(df)} linhas")
    print("🔧 Normalizando...")
    scaled, scalers = scale_data(df)

    print("📐 Criando sequências...")
    X, Y = create_sequences(scaled)

    print(f"✔️ X shape = {X.shape}")

    print("✂️ Separando treino/validação/teste...")
    split = split_data(X, Y)

    print("💾 Salvando pacote final HAC...")
    os.makedirs("data_real", exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        **split,
        scalers_target_speed = scalers["speed"].min_,  # apenas referência
        scalers_target_density = scalers["density"].min_,
        scalers_target_bz = scalers["bz_gse"].min_,
    )

    print("\n🎉 PRONTO!")
    print(f"Arquivo gerado -> {OUTPUT}")
    print("Agora você pode treinar o HAC usando esse dataset.")

if __name__ == "__main__":
    main()
