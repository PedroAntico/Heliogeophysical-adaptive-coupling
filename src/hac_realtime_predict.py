"""
hac_realtime_predict.py
Previsor em tempo real + Alertas de Tempestades Solares
Versão CORRIGIDA — com extração robusta de metadados dos modelos
"""

import os
import json
import numpy as np
import pandas as pd
import re
from datetime import datetime
import logging
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_RT - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HAC_RT")

# ============================================================
# UTILIDADES
# ============================================================

def safe_timestamp(obj):
    """Converte datetime/Timestamp para string JSON-friendly."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return obj

def json_safe(data):
    """Converte recursivamente todos os tipos para JSON-safe."""
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [json_safe(v) for v in data]
    if isinstance(data, (np.float32, np.float64)):
        return float(data)
    if isinstance(data, (np.int32, np.int64)):
        return int(data)
    return safe_timestamp(data)

def extract_model_metadata(filename):
    """
    Extrai metadados do modelo de forma robusta a partir do nome do arquivo.
    Suporta múltiplos formatos:
    - lstm_h1_20251121_023529.keras
    - gru_h3.keras  
    - lstm1.h5
    - model_h6_lstm.tf
    - simple_model.keras (fallback)
    """
    filename = filename.lower()
    
    # Padrões para extração
    patterns = [
        # Padrão: model_h{horizon}_{timestamp}.ext
        (r'(lstm|gru|hybrid)_h(\d+)_\d+\.(keras|h5|tf)', 2),
        # Padrão: model_h{horizon}.ext  
        (r'(lstm|gru|hybrid)_h(\d+)\.(keras|h5|tf)', 2),
        # Padrão: model{horizon}.ext (ex: lstm1.h5)
        (r'(lstm|gru|hybrid)(\d+)\.(keras|h5|tf)', 2),
        # Padrão: {model}_model_h{horizon}.ext
        (r'(\w+)_model_h(\d+)\.(keras|h5|tf)', 2),
        # Padrão: h{horizon}_{model}.ext
        (r'h(\d+)_(\w+)\.(keras|h5|tf)', 1),
    ]
    
    model_type = "unknown"
    horizon = 1  # Default seguro
    
    # Tentar cada padrão
    for pattern, horizon_group in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                # Extrair tipo do modelo
                if horizon_group == 1:
                    model_type = match.group(2)  # h1_lstm -> group2 é lstm
                    horizon = int(match.group(1))  # group1 é 1
                else:
                    model_type = match.group(1)  # lstm_h1 -> group1 é lstm
                    horizon = int(match.group(2))  # group2 é 1
                
                logger.debug(f"✅ Metadados extraídos: {model_type} H{horizon} de {filename}")
                return model_type, horizon
                
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ Padrão encontrado mas extração falhou: {filename} - {e}")
                continue
    
    # Fallback: tentar identificar tipo do modelo
    if 'lstm' in filename:
        model_type = 'lstm'
    elif 'gru' in filename:
        model_type = 'gru' 
    elif 'hybrid' in filename:
        model_type = 'hybrid'
    
    # Tentar extrair horizonte de forma mais agressiva
    horizon_match = re.search(r'h(\d+)', filename)
    if horizon_match:
        try:
            horizon = int(horizon_match.group(1))
        except ValueError:
            pass
    
    logger.info(f"🔍 Metadados (fallback): {model_type} H{horizon} de {filename}")
    return model_type, horizon

# ============================================================
# CARREGAR DADOS - CORRIGIDO
# ============================================================

def load_latest_solar_data():
    """Carrega dados mais recentes com fallback flexível"""
    folders = ["data_real", "data", "data/raw", "data/processed"]
    
    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Pasta {folder} não encontrada")
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")], reverse=True)
        if files:
            latest = os.path.join(folder, files[0])
            logger.info(f"📥 Carregando dados: {latest}")
            try:
                df = pd.read_csv(latest)
                
                if df.empty:
                    logger.warning(f"Arquivo {latest} vazio")
                    continue
                    
                logger.info(f"✅ Dados carregados: {len(df)} registros, colunas: {list(df.columns)}")
                
                # Converter timestamp se existir
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                
                return df
                
            except Exception as e:
                logger.error(f"Erro ao carregar {latest}: {e}")
                continue

    raise FileNotFoundError("❌ Nenhum CSV válido encontrado nas pastas de dados.")

# ============================================================
# CRIAR SEQUÊNCIA PARA PREVISÃO - CORRIGIDO
# ============================================================

def prepare_sequence(df, features, lookback):
    """Prepara sequência para previsão com normalização robusta"""
    # Garantir que temos dados suficientes
    if len(df) < lookback:
        raise ValueError(f"Dados insuficientes: {len(df)} registros, necessário {lookback}")
    
    # Selecionar apenas features disponíveis
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        if len(available_features) < 2:
            raise ValueError(f"Features insuficientes: {available_features}")
    
    logger.info(f"🎯 Usando features: {available_features}")
    
    # Pegar última sequência
    seq = df[available_features].tail(lookback).values
    
    # Normalização robusta
    scaler = StandardScaler()
    seq_scaled = scaler.fit_transform(seq)
    
    logger.info(f"📊 Sequência preparada: {seq_scaled.shape}")
    return seq_scaled.reshape(1, lookback, len(available_features)), scaler, available_features

# ============================================================
# CARREGAMENTO CORRIGIDO DOS MODELOS
# ============================================================

def load_models():
    """Carrega modelos com extração robusta de metadados"""
    model_dirs = ["models/deep_hac", "models", "results/deep_learning"]
    
    models = {}
    loaded_count = 0
    
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        logger.info(f"🔍 Procurando modelos em: {model_dir}")
        
        for fname in os.listdir(model_dir):
            model_path = os.path.join(model_dir, fname)
            
            # Suporte a múltiplos formatos
            if fname.endswith((".keras", ".h5", ".tf")):
                try:
                    logger.info(f"📦 Tentando carregar: {fname}")
                    
                    # Extrair metadados de forma robusta
                    model_type, horizon = extract_model_metadata(fname)
                    
                    # Carregar modelo
                    model = load_model(model_path, compile=False)
                    model.compile(optimizer="adam", loss="mse")
                    
                    # Organizar na estrutura
                    if model_type not in models:
                        models[model_type] = {}
                    
                    # Se já existe modelo para este horizonte, manter o melhor (baseado no nome do arquivo)
                    if horizon in models[model_type]:
                        existing_file = models[model_type][horizon].get('_filename', 'unknown')
                        logger.info(f"⚠️ Horizonte H{horizon} duplicado: {existing_file} vs {fname}")
                        # Preferir modelos mais recentes (com timestamp no nome)
                        if '_' in fname and len(fname) > 20:  # Provavelmente tem timestamp
                            logger.info(f"✅ Usando modelo mais recente: {fname}")
                            models[model_type][horizon] = {
                                'model': model,
                                '_filename': fname,
                                '_path': model_path
                            }
                    else:
                        models[model_type][horizon] = {
                            'model': model, 
                            '_filename': fname,
                            '_path': model_path
                        }
                    
                    loaded_count += 1
                    logger.info(f"✅ Modelo carregado: {model_type} H{horizon} de {fname}")
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao carregar {fname}: {e}")
                    continue
    
    if loaded_count == 0:
        logger.warning("⚠️ Nenhum modelo encontrado. Criando modelo dummy para teste.")
        models = create_dummy_model()
    else:
        logger.info(f"🎉 Total de modelos carregados: {loaded_count}")
        
        # Log de resumo dos modelos
        for model_type in models:
            horizons = list(models[model_type].keys())
            logger.info(f"   {model_type.upper()}: Horizontes {sorted(horizons)}")
    
    return models

def create_dummy_model():
    """Cria modelo simples para demonstração quando não há modelos salvos"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    
    model = Sequential([
        LSTM(32, input_shape=(36, 3)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    return {
        'lstm': {
            1: {'model': model, '_filename': 'dummy_model.keras', '_path': 'generated'},
            3: {'model': model, '_filename': 'dummy_model.keras', '_path': 'generated'}
        }
    }

# ============================================================
# PREVISÃO E ALERTAS - CORRIGIDO
# ============================================================

def classify_risk(speed, bz=None):
    """Classifica risco com valores padrão seguros"""
    score = 0
    
    # Valores padrão se não disponíveis
    if speed is None:
        speed = 400  # Valor médio seguro
    
    if bz is None:
        bz = -2  # Valor neutro seguro
    
    # Pontuação baseada na velocidade
    if speed > 500: score += 1
    if speed > 600: score += 1
    if speed > 700: score += 2
    
    # Pontuação baseada no Bz
    if bz < -5: score += 1
    if bz < -10: score += 2
    if bz < -15: score += 2
    
    return score

def risk_level(score):
    """Define nível de risco baseado na pontuação"""
    if score <= 1: 
        return "Baixo"
    elif score <= 3: 
        return "Moderado"
    elif score <= 5: 
        return "Alto"
    else: 
        return "Crítico"

# ============================================================
# EXECUÇÃO PRINCIPAL - CORRIGIDA
# ============================================================

def run_realtime_hac():
    try:
        logger.info("🚀 HAC 3.0 — Previsor em Tempo Real iniciado")

        # 1. Carregar dados
        df = load_latest_solar_data()
        
        # 2. Definir features possíveis
        possible_features = [
            "speed", "density", "temperature",
            "bx_gse", "by_gse", "bz_gse", "bt"
        ]
        
        # 3. Carregar modelos
        models = load_models()
        
        if not models:
            raise ValueError("❌ Nenhum modelo disponível para previsão")
        
        # 4. Preparar sequência
        lookback = 36
        seq, scaler, features_used = prepare_sequence(df, possible_features, lookback)
        
        # 5. Fazer previsões
        predictions = {}
        model_details = {}
        
        # Calcular estatísticas do target para desnormalização
        target_col = "speed" if "speed" in df.columns else features_used[0]
        target_values = df[target_col].dropna()
        
        if len(target_values) == 0:
            raise ValueError(f"❌ Nenhum dado válido para target: {target_col}")
        
        target_mean = target_values.mean()
        target_std = target_values.std()
        
        logger.info(f"🎯 Target: {target_col}, Média: {target_mean:.2f}, Std: {target_std:.2f}")

        for model_type in models:
            predictions[model_type] = {}
            model_details[model_type] = {}
            
            for horizon_h, model_info in models[model_type].items():
                try:
                    model = model_info['model']
                    
                    # Fazer previsão
                    pred_scaled = model.predict(seq, verbose=0)
                    
                    # Desnormalizar previsão
                    if pred_scaled.ndim > 1:
                        pred_scaled = pred_scaled[0][0]
                    else:
                        pred_scaled = pred_scaled[0]
                    
                    pred_unscaled = float(pred_scaled * target_std + target_mean)
                    predictions[model_type][horizon_h] = pred_unscaled
                    model_details[model_type][horizon_h] = {
                        'filename': model_info.get('_filename', 'unknown'),
                        'source': model_info.get('_path', 'unknown')
                    }
                    
                    logger.info(f"✅ {model_type.upper()} H{horizon_h}: {pred_unscaled:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Erro na previsão {model_type} H{horizon_h}: {e}")
                    predictions[model_type][horizon_h] = None

        # 6. Coletar condições atuais
        current = {
            "timestamp": safe_timestamp(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else None,
            "speed": float(df["speed"].iloc[-1]) if "speed" in df.columns else None,
            "density": float(df["density"].iloc[-1]) if "density" in df.columns else None,
            "temperature": float(df["temperature"].iloc[-1]) if "temperature" in df.columns else None,
            "bz": float(df["bz_gse"].iloc[-1]) if "bz_gse" in df.columns else None,
        }

        # 7. Calcular risco
        score = classify_risk(current["speed"], current.get("bz"))
        risk = risk_level(score)

        # 8. Preparar output
        output = json_safe({
            "generated_at": datetime.utcnow(),
            "data_source": "HAC Real-time Predictor v3.0",
            "current_conditions": current,
            "risk_assessment": {
                "score": score,
                "level": risk,
                "factors": {
                    "high_speed": current["speed"] > 600 if current["speed"] else False,
                    "strong_negative_bz": current.get("bz", 0) < -10 if current.get("bz") else False
                }
            },
            "predictions": predictions,
            "model_info": {
                "features_used": features_used,
                "lookback": lookback,
                "models_loaded": {
                    model_type: list(models[model_type].keys()) 
                    for model_type in models
                },
                "model_details": model_details
            }
        })

        # 9. Salvar resultados
        os.makedirs("results/realtime", exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_file = f"results/realtime/hac_rt_{timestamp}.json"

        with open(out_file, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # 10. Log do relatório
        logger.info("\n" + "="*60)
        logger.info("🌞 RELATÓRIO HAC TEMPO REAL")
        logger.info("="*60)
        
        logger.info(f"📊 Condições Atuais:")
        logger.info(f"   • Velocidade: {current['speed'] or 'N/A'} km/s")
        logger.info(f"   • Densidade: {current['density'] or 'N/A'} p/cm³") 
        logger.info(f"   • Temperatura: {current['temperature'] or 'N/A'} K")
        logger.info(f"   • Bz: {current['bz'] or 'N/A'} nT")
        
        logger.info(f"\n⚠️  Avaliação de Risco:")
        logger.info(f"   • Score: {score}")
        logger.info(f"   • Nível: {risk}")
        
        logger.info(f"\n🔮 Previsões:")
        for model_type in predictions:
            for horizon, value in predictions[model_type].items():
                if value is not None:
                    source = model_details[model_type][horizon]['filename']
                    logger.info(f"   • {model_type.upper()} H{horizon}: {value:.2f} km/s ({source})")

        logger.info(f"\n💾 Resultado salvo em: {out_file}")
        logger.info("="*60)
        
        return output

    except Exception as e:
        logger.error(f"❌ Erro no previsor em tempo real: {e}")
        # Salvar erro para debug
        try:
            os.makedirs("results/realtime", exist_ok=True)
            error_file = f"results/realtime/error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_file, 'w') as f:
                json.dump({"error": str(e), "timestamp": datetime.utcnow().isoformat()}, f, indent=2)
        except:
            pass
        raise

if __name__ == "__main__":
    run_realtime_hac()
