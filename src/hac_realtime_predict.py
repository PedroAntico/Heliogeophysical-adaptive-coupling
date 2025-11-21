"""
hac_realtime_predict.py
HAC 3.0 — Previsor em tempo real + alerta de tempestades solares (LSTM + GRU)
"""

import os
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURAÇÃO DE LOGGING E TF
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_RT - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HAC_RT")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass  # se não suportar XLA, segue normal


# ============================================================
# 1. CARREGAR DADOS REAIS
# ============================================================

def load_latest_real_data():
    """
    Carrega o CSV mais recente com dados reais.
    Prioridade: data_real/ > data/raw/ > data/processed/ > data/
    """
    search_folders = ["data_real", "data/raw", "data/processed", "data"]

    for folder in search_folders:
        if not os.path.exists(folder):
            continue

        csv_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".csv")],
            reverse=True,
        )
        if not csv_files:
            continue

        latest = os.path.join(folder, csv_files[0])
        logger.info(f"📥 Carregando dados de: {latest}")

        df = pd.read_csv(latest)

        if df.empty:
            logger.warning(f"Arquivo {latest} está vazio, tentando próximo...")
            continue

        logger.info(f"✅ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        logger.info(f"📋 Colunas: {list(df.columns)}")

        # Normalização do nome de tempo, se existir
        for tcol in ["time_tag", "timestamp", "time", "Epoch"]:
            if tcol in df.columns:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
                df = df.sort_values(tcol).reset_index(drop=True)
                df.rename(columns={tcol: "timestamp"}, inplace=True)
                break

        # Converter colunas numéricas
        for col in df.columns:
            if col != "timestamp":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(how="all")

        if len(df) < 30:
            logger.warning(f"Poucos dados ({len(df)} linhas). Mesmo assim vamos tentar.")
        return df

    raise FileNotFoundError("❌ Nenhum CSV encontrado em data_real/, data/, data/raw/ ou data/processed/.")


# ============================================================
# 2. CRIAÇÃO DE SEQUÊNCIAS PARA LSTM/GRU
# ============================================================

def create_sequences(df, features, target_col="speed", lookback=24, horizon_hours=1, step=1):
    """
    Cria sequências (janelas deslizantes) para treino de LSTM/GRU.
    lookback: nº de pontos no passado
    horizon_hours: quantos passos à frente (cada ponto = 5min ou 1min, depende do dataset)
    Aqui usamos horizon_hours como deslocamento simples.
    """
    logger.info(f"🔄 Criando sequências (lookback={lookback}, horizonte={horizon_hours})")

    available_features = [f for f in features if f in df.columns]
    if not available_features:
        raise ValueError("Nenhuma feature numérica disponível para criar sequências.")

    if target_col not in df.columns:
        # fallback: usa a primeira coluna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Não há colunas numéricas para usar como alvo.")
        target_col = numeric_cols[0]
        logger.warning(f"Target '{target_col}' não encontrado. Usando '{target_col}' como alvo.")

    data = df[available_features].values
    target = df[target_col].values

    X, y = [], []
    for i in range(lookback, len(df) - horizon_hours, step):
        X.append(data[i - lookback:i])
        y.append(target[i + horizon_hours - 1])

    if not X:
        raise ValueError("Não foi possível criar sequências. Dados insuficientes.")

    X = np.array(X)
    y = np.array(y)

    logger.info(f"📊 Sequências criadas: X{X.shape}, y{y.shape}")
    return X, y, target_col, available_features


# ============================================================
# 3. MODELOS LSTM / GRU
# ============================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        LSTM(32, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        GRU(32, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def training_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ============================================================
# 4. TREINAR PARA UM HORIZONTE E PREVER
# ============================================================

def train_and_predict(df, features, horizon_h, model_type="lstm"):
    """
    Treina LSTM/GRU para um horizonte específico e gera previsão
    para o próximo passo desse horizonte (usando a última janela).
    Retorna métricas, modelo e previsão.
    """
    logger.info(f"\n🎯 Treinando {model_type.upper()} para horizonte {horizon_h}h")

    # lookback adaptativo
    lookback = min(36, max(12, len(df) // 4))

    X, y, target_col, used_features = create_sequences(
        df, features, target_col="speed", lookback=lookback, horizon_hours=horizon_h
    )

    if len(X) < 60:
        logger.warning(f"Poucas amostras para treino ({len(X)}). Resultado pode ser instável.")

    # split temporal
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # normalização
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    input_shape = (lookback, len(used_features))

    if model_type == "lstm":
        model = build_lstm(input_shape)
    else:
        model = build_gru(input_shape)

    logger.info(f"🏗️ {model_type.upper()} com {model.count_params()} parâmetros")

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=25,
        batch_size=16,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=training_callbacks(),
        verbose=0,
    )

    # Avaliação
    y_pred_scaled = model.predict(X_val_scaled, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(
            np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        )

    logger.info(
        f"📊 {model_type.upper()} H{horizon_h}h -> "
        f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}% (target: {target_col})"
    )

    # Previsão com a última janela
    last_window = df[used_features].values[-lookback:]
    last_scaled = scaler_X.transform(last_window).reshape(1, lookback, len(used_features))
    next_scaled = model.predict(last_scaled, verbose=0).flatten()[0]
    next_value = float(scaler_y.inverse_transform([[next_scaled]])[0, 0])

    return {
        "model_type": model_type,
        "horizon_h": horizon_h,
        "metrics": {"rmse": rmse, "mae": mae, "mape": mape},
        "target": target_col,
        "features": used_features,
        "prediction": next_value,
    }


# ============================================================
# 5. AVALIAÇÃO DE TEMPESTADES SOLARES
# ============================================================

def compute_storm_risk_row(row):
    """
    Calcula um score de risco geomagnético simples baseado em:
    - speed
    - bz_gse (sul)
    - bt
    - density
    Retorna um dict com 'score' e 'nível'.
    """
    speed = float(row.get("speed", np.nan))
    bz = float(row.get("bz_gse", np.nan))
    bt = float(row.get("bt", np.nan))
    density = float(row.get("density", np.nan))

    score = 0
    reasons = []

    # velocidade
    if not np.isnan(speed):
        if speed > 400:
            score += 1
            reasons.append(f"speed>400 ({speed:.0f} km/s)")
        if speed > 500:
            score += 1
        if speed > 600:
            score += 1
            reasons.append(f"speed>600 ({speed:.0f} km/s)")

    # Bz sul
    if not np.isnan(bz):
        if bz < -2:
            score += 1
        if bz < -5:
            score += 1
            reasons.append(f"bz<-5 ({bz:.1f} nT)")
        if bz < -10:
            score += 1
            reasons.append(f"bz<-10 ({bz:.1f} nT)")

    # Bt forte
    if not np.isnan(bt):
        if bt > 8:
            score += 1
        if bt > 12:
            score += 1
            reasons.append(f"bt>12 ({bt:.1f} nT)")

    # densidade
    if not np.isnan(density):
        if density > 10:
            score += 1
            reasons.append(f"density>10 ({density:.1f} cm^-3)")
        if density > 20:
            score += 1

    # mapeia score para nível
    if score <= 2:
        level = "Baixo"
    elif score <= 4:
        level = "Moderado"
    elif score <= 6:
        level = "Alto"
    else:
        level = "Crítico"

    return {"score": int(score), "level": level, "reasons": reasons}


# ============================================================
# 6. PIPELINE PRINCIPAL
# ============================================================

def run_realtime_hac():
    logger.info("🚀 Iniciando HAC Realtime + Tempestades Solares")

    df = load_latest_real_data()

    # Definir features de interesse
    candidate_features = ["speed", "density", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]
    features = [f for f in candidate_features if f in df.columns]

    if len(features) < 2:
        # fallback: pega primeiras numéricas
        features = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        logger.warning(f"Poucas features padrão, usando: {features}")

    logger.info(f"🎯 Features usadas: {features}")
    logger.info(f"📊 Total de registros: {len(df)}")

    # Ponto atual (última linha)
    current = df.iloc[-1].to_dict()
    storm_now = compute_storm_risk_row(current)

    # Treinar e prever
    horizons = [1, 3, 6]
    model_types = ["lstm", "gru"]
    forecast_results = []

    for model_type in model_types:
        for h in horizons:
            try:
                result = train_and_predict(df, features, h, model_type)
                forecast_results.append(result)
            except Exception as e:
                logger.error(f"Erro ao treinar {model_type} H{h}: {e}")

    # Previsão de risco usando previsão de velocidade (apenas como proxy)
    # Vamos usar a média das previsões LSTM+GRU por horizonte
    risk_forecast = {}
    for h in horizons:
        preds = [r["prediction"] for r in forecast_results if r["horizon_h"] == h]
        if preds:
            avg_speed = float(np.mean(preds))
            # Construímos uma "linha futura" fictícia usando a previsão de speed
            future_row = current.copy()
            future_row["speed"] = avg_speed
            storm_future = compute_storm_risk_row(future_row)
            risk_forecast[h] = {
                "predicted_speed": avg_speed,
                "storm": storm_future,
            }

    # Imprimir resumo bonito
    print("\n" + "=" * 70)
    print("🌞 HAC 3.0 — Previsor em tempo real + Alerta de Tempestades Solares")
    print("=" * 70)

    # Condição atual
    print("\n🕒 Condição atual (última medição):")
    if "timestamp" in df.columns:
        print(f"   Tempo     : {df['timestamp'].iloc[-1]}")
    if "speed" in current:
        print(f"   Velocidade: {current.get('speed', np.nan):6.1f} km/s")
    if "bz_gse" in current:
        print(f"   Bz (GSE)  : {current.get('bz_gse', np.nan):6.2f} nT")
    if "bt" in current:
        print(f"   Bt        : {current.get('bt', np.nan):6.2f} nT")
    if "density" in current:
        print(f"   Densidade : {current.get('density', np.nan):6.2f} cm^-3")

    print(f"\n⚠️  Risco geomagnético AGORA: {storm_now['level']} (score={storm_now['score']})")
    if storm_now["reasons"]:
        print("   Motivos principais: " + "; ".join(storm_now["reasons"]))

    # Previsão
    print("\n🔮 Previsão para tempestades solares (média LSTM+GRU):")
    for h in horizons:
        if h in risk_forecast:
            info = risk_forecast[h]
            storm = info["storm"]
            print(
                f"   ➜ Em {h:2d}h: "
                f"speed ≈ {info['predicted_speed']:6.1f} km/s | "
                f"risco: {storm['level']} (score={storm['score']})"
            )

    # Salvar JSON
    os.makedirs("results/realtime", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/realtime/hac_realtime_{ts}.json"

    output = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "features_used": features,
        "current_conditions": {
            "values": current,
            "storm_risk": storm_now,
        },
        "forecast": risk_forecast,
        "models": [
            {
                "model_type": r["model_type"],
                "horizon_h": r["horizon_h"],
                "metrics": r["metrics"],
                "target": r["target"],
                "prediction": r["prediction"],
            }
            for r in forecast_results
        ],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n💾 Resultado detalhado salvo em:")
    print(f"   {out_path}")
    print("\n✅ Fim da execução do previsor HAC Realtime.\n")


if __name__ == "__main__":
    run_realtime_hac()
