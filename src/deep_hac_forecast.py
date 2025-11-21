"""
deep_hac_forecast.py
HAC 3.0 — Previsão Deep Learning (LSTM + GRU) - CORRIGIDO
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_Deep - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HAC_Deep")

# Configuração do TensorFlow para melhor performance
tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# FUNÇÕES AUXILIARES CORRIGIDAS
# ============================================================

def load_and_validate_real_data():
    """
    Carrega dados priorizando o dataset completo (solar_data_latest.csv).
    Se não existir, tenta os dados reais.
    """
    # 1 — PRIORIDADE TOTAL: dataset completo
    full_dataset_path = "data/solar_data_latest.csv"
    
    if os.path.exists(full_dataset_path):
        logger.info(f"📥 Carregando dataset completo: {full_dataset_path}")
        df = pd.read_csv(full_dataset_path)

        if df.empty:
            logger.error("❌ Dataset completo vazio!")
        else:
            logger.info(f"✅ Dataset completo carregado: {len(df)} registros")
            logger.info(f"📋 Colunas: {list(df.columns)}")
            return df

    # 2 — Caso o dataset completo falhe, tentar dados REAL
    data_folders = ["data_real", "data/raw", "data/processed"]

    for folder in data_folders:
        if not os.path.exists(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")], reverse=True)

        if files:
            latest_file = os.path.join(folder, files[0])
            logger.info(f"📥 Tentando dados reais: {latest_file}")

            try:
                df = pd.read_csv(latest_file)

                if len(df) < 50:
                    logger.warning(f"⚠️ Dados reais insuficientes ({len(df)} registros).")
                    continue

                logger.info(f"✅ Dados reais carregados: {len(df)} registros")
                return df

            except Exception as e:
                logger.error(f"Erro ao carregar {latest_file}: {e}")
                continue

    raise FileNotFoundError("❌ Nenhum dataset válido encontrado (nem completo nem real).")

def create_advanced_sequences(df, features, target_col='speed', 
                            horizon_hours=1, lookback=48, step=1):
    """
    Constrói sequências para treino - CORRIGIDO para dados atuais
    """
    logger.info(f"🔄 Criando sequências: lookback={lookback}, horizon={horizon_hours}h")
    
    # Garantir que todas as features existem
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        logger.warning(f"⚠️ Features faltando: {set(features) - set(available_features)}")
    
    data = df[available_features].values
    
    # Verificar se a coluna target existe
    if target_col not in df.columns:
        # Se não existir, usar a primeira feature numérica disponível
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_col = numeric_cols[0] if len(numeric_cols) > 0 else available_features[0]
        logger.warning(f"⚠️ Target '{target_col}' não encontrado. Usando '{target_col}'")
    
    targets = df[target_col].values
    
    X, y = [], []
    
    for i in range(lookback, len(data) - horizon_hours, step):
        X.append(data[i - lookback:i])
        y.append(targets[i + horizon_hours - 1])  # Previsão single-step corrigida
    
    if len(X) == 0:
        raise ValueError("❌ Não foi possível criar sequências. Dados insuficientes.")
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"📊 Sequências criadas: X{X.shape}, y{y.shape}")
    return X, y, target_col

# ============================================================
# ARQUITETURAS SIMPLIFICADAS (para dados atuais)
# ============================================================

def build_advanced_lstm(input_shape):
    """LSTM simplificado para dados atuais"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        LSTM(32, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)  # Saída single-step
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_advanced_gru(input_shape):
    """GRU simplificado para dados atuais"""
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        GRU(32, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)  # Saída single-step
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# ============================================================
# TREINAMENTO CORRIGIDO
# ============================================================

def create_callbacks():
    """Callbacks para treinamento otimizado"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

def train_model_for_horizon(df, features, horizon_h, model_type='lstm'):
    """
    Treina modelo específico para um horizonte - CORRIGIDO
    """
    logger.info(f"\n🎯 Treinando {model_type.upper()} para horizonte {horizon_h}h")
    
    # Lookback adaptativo baseado nos dados disponíveis
    lookback = min(36, len(df) // 3)  # Reduzido para dados atuais
    if lookback < 12:
        lookback = 12
    
    # Criar sequências (função corrigida)
    X, y, final_target = create_advanced_sequences(
        df, features, horizon_hours=horizon_h, lookback=lookback, step=1
    )
    
    if len(X) < 50:  # Reduzido mínimo para dados atuais
        logger.warning(f"Dados insuficientes para treinamento: {len(X)} amostras")
        return None
    
    # Split temporal (não aleatório para séries temporais)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalização
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape para normalização
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Selecionar modelo
    input_shape = (lookback, len(features))
    
    if model_type == 'lstm':
        model = build_advanced_lstm(input_shape)
    elif model_type == 'gru':
        model = build_advanced_gru(input_shape)
    else:
        raise ValueError(f"Modelo não suportado: {model_type}")
    
    logger.info(f"🏗️ Arquitetura {model_type}: {model.count_params()} parâmetros")
    
    # Treinamento com menos épocas para dados atuais
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=30,  # Reduzido
        batch_size=16,  # Reduzido
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=create_callbacks(),
        verbose=0
    )
    
    # Previsões e métricas
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Métricas robustas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE com proteção contra divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    logger.info(f"📊 {model_type.upper()} H{horizon_h}h -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")
    
    return {
        'model': model,
        'history': history.history,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'horizon': horizon_h,
            'target': final_target
        },
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'lookback': lookback
    }

# ============================================================
# EXECUÇÃO PRINCIPAL CORRIGIDA
# ============================================================

def run_advanced_deep_hac():
    """Executa o pipeline completo de Deep Learning HAC - CORRIGIDO"""
    try:
        logger.info("🚀 Iniciando HAC Deep Learning 3.0 - Versão Corrigida")
        
        # Carregar dados
        df = load_and_validate_real_data()
        
        # FEATURES FLEXÍVEIS - aceita o que estiver disponível
        possible_features = ["speed", "density", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]
        available_features = [f for f in possible_features if f in df.columns]
        
        # ✅ CORREÇÃO 1: Aceitar mínimo de 3 features (ou até menos para teste)
        if len(available_features) < 2:  # Reduzido para 2 para ser mais tolerante
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()[:3]  # Pega primeiras 3 numéricas
            if len(available_features) < 2:
                raise ValueError(f"❌ Features insuficientes. Disponíveis: {list(df.columns)}")
        
        logger.info(f"🎯 Features utilizadas: {available_features}")
        
        # ✅ CORREÇÃO 2: Não depende de timestamp
        # Usar índice como referência temporal se não houver timestamp
        if 'timestamp' not in df.columns:
            logger.warning("⏰ Coluna 'timestamp' não encontrada. Usando índice como referência.")
            # Criar timestamp fictício baseado no índice (assumindo intervalos regulares)
            df = df.reset_index(drop=True)
        
        # Limpeza final - apenas features selecionadas
        df = df[available_features].dropna()
        
        if len(df) < 50:  # Reduzido mínimo
            raise ValueError(f"❌ Dados insuficientes após limpeza: {len(df)} registros")
        
        # Treinar modelos para diferentes horizontes (reduzidos para dados atuais)
        horizons = [1, 3, 6]  # Reduzido: 1h, 3h, 6h
        model_types = ['lstm', 'gru']  # Apenas modelos principais
        
        results = {}
        
        for model_type in model_types:
            results[model_type] = {}
            
            for horizon in horizons:
                try:
                    logger.info(f"\n🔮 Treinando {model_type.upper()} para H{horizon}h")
                    result = train_model_for_horizon(df, available_features, horizon, model_type)
                    results[model_type][horizon] = result
                        
                except Exception as e:
                    logger.error(f"❌ Erro no treino {model_type} H{horizon}: {e}")
                    results[model_type][horizon] = None
        
        # Relatório final
        logger.info("\n" + "="*60)
        logger.info("📈 RELATÓRIO FINAL HAC DEEP LEARNING - CORRIGIDO")
        logger.info("="*60)
        logger.info(f"📊 Total de dados: {len(df)} registros")
        logger.info(f"🎯 Features: {available_features}")
        
        for model_type in model_types:
            logger.info(f"\n🔮 {model_type.upper()}:")
            for horizon in horizons:
                if results[model_type][horizon]:
                    metrics = results[model_type][horizon]['metrics']
                    logger.info(
                        f"  H{horizon:2d}h -> RMSE: {metrics['rmse']:.3f}, "
                        f"MAE: {metrics['mae']:.3f}, MAPE: {metrics['mape']:.2f}%"
                        f" (target: {metrics.get('target', 'speed')})"
                    )
                else:
                    logger.info(f"  H{horizon:2d}h -> ❌ Falha no treinamento")
        
        # Salvar resultados simplificado
        try:
            os.makedirs('results/deep_learning', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Salvar apenas métricas
            metrics_summary = {}
            for model_type in model_types:
                metrics_summary[model_type] = {}
                for horizon in horizons:
                    if results[model_type][horizon]:
                        metrics_summary[model_type][str(horizon)] = results[model_type][horizon]['metrics']
            
            with open(f'results/deep_learning/metrics_{timestamp}.json', 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            
            logger.info(f"💾 Métricas salvas em: results/deep_learning/metrics_{timestamp}.json")
            
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível salvar resultados: {e}")
        
        logger.info("✅ HAC Deep Learning concluído com sucesso!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Falha no HAC Deep Learning: {e}")
        # Tentar fornecer informações úteis para debug
        logger.info("💡 Dica: Verifique se existem arquivos CSV em data_real/, data/raw/ ou data/processed/")
        raise

if __name__ == "__main__":
    run_advanced_deep_hac()
