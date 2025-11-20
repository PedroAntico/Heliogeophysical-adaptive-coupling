"""
heliopredictive.py
HAC – Previsor aprimorado (Fase 1)
Melhorias:
    ✓ Features temporais (lags, médias móveis, derivadas)
    ✓ Split temporal correto (treina passado → prevê futuro)
    ✓ Hiperparâmetros ajustados para séries temporais
    ✓ Ensemble mais estável
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 1) Feature Engineering
# ==========================================

def create_features(df, target="speed"):
    """
    Cria features temporais automaticamente:
       - Lags (1,2,3,...)
       - Médias móveis (5, 10 passos)
       - Derivadas (1º e 2º)
    """

    df = df.copy()

    # Lags
    for lag in [1, 2, 3, 6, 12]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    # Médias móveis
    df[f"{target}_ma5"] = df[target].rolling(5).mean()
    df[f"{target}_ma10"] = df[target].rolling(10).mean()

    # Derivadas
    df[f"{target}_diff1"] = df[target].diff()
    df[f"{target}_diff2"] = df[target].diff().diff()

    # Remover valores iniciais inválidos
    df = df.dropna().reset_index(drop=True)

    return df


# ==========================================
# 2) Modelos Base (com hiperparâmetros otimizados)
# ==========================================

def build_models():
    models = {
        "ridge": Ridge(alpha=1.5),
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.07,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist"
        )
    }
    return models


# ==========================================
# 3) Split temporal correto
# ==========================================

def temporal_split(df, horizon_steps):
    """
    Split temporal:
        train = tudo até o final - horizon
        test  = últimos horizon passos
    """

    df = df.reset_index(drop=True)

    train = df.iloc[:-horizon_steps]
    test = df.iloc[-horizon_steps:]

    return train, test


# ==========================================
# 4) Treinamento + Previsão
# ==========================================

def train_and_predict(df, horizon_hours=1, freq_minutes=5):
    """
    Treina o modelo com passado → prevê futuro
    horizon_hours = horizonte em horas (1h, 3h, 6h, 12h...)
    freq_minutes  = frequência dos dados (default = 5 min)
    """

    horizon_steps = int(horizon_hours * 60 / freq_minutes)

    logger.info(f"🎯 Prevendo {horizon_hours}h ({horizon_steps} passos)")

    # Criar features
    df_feat = create_features(df, target="speed")

    # Definir colunas
    feature_cols = [c for c in df_feat.columns if c not in ["time_tag", "speed"]]
    target_col = "speed"

    # Split temporal correto
    train, test = temporal_split(df_feat, horizon_steps)

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    # ===============================
    # Treinar modelos do Ensemble
    # ===============================
    models = build_models()
    predictions = {}

    for name, model in models.items():
        logger.info(f"🔧 Treinando modelo: {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

    # ===============================
    # Ensemble (média)
    # ===============================
    ensemble_pred = np.mean(list(predictions.values()), axis=0)

    # ===============================
    # Métricas
    # ===============================
    rmse = mean_squared_error(y_test, ensemble_pred, squared=False)
    r2 = r2_score(y_test, ensemble_pred)

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "horizon_hours": horizon_hours,
        "steps": horizon_steps
    }

    logger.info(f"📈 Ensemble RMSE={rmse:.3f}  R²={r2:.3f}")

    # Resultado completo
    result = pd.DataFrame({
        "time_tag": test["time_tag"].values,
        "true": y_test.values,
        "pred": ensemble_pred
    })

    return result, metrics


# ==========================================
# 5) Função principal usada no workflow
# ==========================================

def run_hac_forecast(df):
    """
    Executa previsões para múltiplos horizontes:
        1h, 3h, 6h, 12h
    """
    horizons = [1, 3, 6, 12]
    outputs = {}

    for h in horizons:
        result, metrics = train_and_predict(df.copy(), horizon_hours=h)
        outputs[h] = {
            "forecast": result,
            "metrics": metrics
        }

    return outputs


# ==========================================
# Debug local
# ==========================================

if __name__ == "__main__":
    print("⚡ HAC Predictor Fase 1")
    print("→ Importado com sucesso")
