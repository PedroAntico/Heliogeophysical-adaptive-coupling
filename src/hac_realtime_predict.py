"""
hac_realtime_predict.py
HAC 5.0 — Previsor Real-Time Avançado com múltiplos modelos e múltiplos horizontes
Compatível com treino HAC Deep Learning 4.0
"""

import os
import json
import re
import pickle
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("hac_realtime.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HAC_Realtime_Advanced")

# Supressão de warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# CONFIGURAÇÃO
# ============================================================

VALID_MODEL_TYPES = ["lstm", "gru", "bilstm", "cnn_lstm"]
VALID_HORIZONS = [1, 3, 6, 12, 24, 48]
DEFAULT_LOOKBACK = 48
MIN_SEQUENCE_LENGTH = 24

class RealtimeConfig:
    """Configurações para previsão em tempo real"""
    PREDICTION_CONFIDENCE_THRESHOLD = 0.7
    MAX_LOOKBACK = 168
    DATA_QUALITY_THRESHOLD = 0.8  # 80% dos dados devem estar presentes
    ENSEMBLE_WEIGHTS = {
        "lstm": 0.3,
        "gru": 0.3, 
        "bilstm": 0.2,
        "cnn_lstm": 0.2
    }

# ============================================================
# UTILITÁRIOS AVANÇADOS
# ============================================================

def safe_timestamp(obj) -> str:
    """Converte timestamp para string ISO format segura"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif pd.isna(obj):
        return datetime.utcnow().isoformat()
    return str(obj)

def json_safe(data: Any) -> Any:
    """Converte dados para formato JSON seguro"""
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_safe(v) for v in data]
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, (np.ndarray, pd.Series)):
        return data.tolist()
    elif isinstance(data, (datetime, pd.Timestamp)):
        return data.isoformat()
    elif data is None:
        return None
    else:
        return data

def calculate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> float:
    """Calcula qualidade dos dados baseado em valores não nulos"""
    if df.empty:
        return 0.0
    
    quality_scores = []
    for col in required_columns:
        if col in df.columns:
            non_null_ratio = 1 - df[col].isnull().sum() / len(df)
            quality_scores.append(non_null_ratio)
        else:
            quality_scores.append(0.0)
    
    return np.mean(quality_scores) if quality_scores else 0.0

# ============================================================
# 1. CARREGAMENTO DE DADOS AVANÇADO
# ============================================================

def load_latest_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega dados mais recentes com múltiplas estratégias de fallback
    """
    search_paths = ["data_real", "data", "data/raw", "data/processed", "dataset"]
    
    if data_path and os.path.exists(data_path):
        file_path = data_path
        logger.info(f"📥 CARREGANDO DADOS DO CAMINHO ESPECIFICADO: {file_path}")
    else:
        file_path = None
        for folder in search_paths:
            if not os.path.exists(folder):
                continue
            
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
            if not csv_files:
                continue
            
            # Encontra arquivo mais recente
            try:
                latest_file = max(
                    csv_files, 
                    key=lambda x: os.path.getctime(os.path.join(folder, x))
                )
                file_path = os.path.join(folder, latest_file)
                logger.info(f"📥 CARREGANDO DADOS MAIS RECENTES: {file_path}")
                break
            except Exception as e:
                logger.warning(f"⚠️ ERRO AO ACESSAR {folder}: {e}")
                continue
    
    if not file_path:
        raise FileNotFoundError("❌ NENHUM ARQUIVO CSV ENCONTRADO NAS PASTAS DE BUSCA")
    
    # Carregar com tratamento de encoding
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✅ ARQUIVO CARREGADO COM ENCODING: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"⚠️ ERRO COM ENCODING {encoding}: {e}")
            continue
    
    if df is None:
        raise ValueError("❌ NÃO FOI POSSÍVEL DECODIFICAR O ARQUIVO COM NENHUM ENCODING COMUM")
    
    # Processamento de timestamp
    timestamp_cols = ['timestamp', 'time', 'date', 'datetime']
    for col in timestamp_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df = df.sort_values(col).reset_index(drop=True)
                logger.info(f"🕐 COLUNA DE TIMESTAMP IDENTIFICADA: {col}")
                break
            except Exception as e:
                logger.warning(f"⚠️ ERRO AO PROCESSAR COLUNA {col}: {e}")
    
    logger.info(f"📊 DADOS CARREGADOS: {df.shape[0]} LINHAS × {df.shape[1]} COLUNAS")
    
    # Verificar qualidade dos dados
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    quality = calculate_data_quality(df, numeric_cols[:3]) if numeric_cols else 0
    logger.info(f"📈 QUALIDADE DOS DADOS: {quality:.1%}")
    
    if quality < RealtimeConfig.DATA_QUALITY_THRESHOLD:
        logger.warning(f"⚠️ QUALIDADE DOS DADOS ABAIXO DO THRESHOLD: {quality:.1%} < {RealtimeConfig.DATA_QUALITY_THRESHOLD:.1%}")
    
    return df

# ============================================================
# 2. DETECÇÃO E CARREGAMENTO DE MODELOS AVANÇADO
# ============================================================

def parse_model_metadata(filename: str) -> Tuple[str, int]:
    """
    Extrai metadados do modelo do nome do arquivo
    Retorna: (model_type, horizon)
    """
    filename_lower = filename.lower()
    
    # Padrão principal: tipo_horizonte_data.keras
    patterns = [
        r"(lstm|gru|bilstm|cnn_lstm)_h?(\d+)_\d+\.(keras|h5)",
        r"(lstm|gru|bilstm|cnn_lstm)_h?(\d+)\.(keras|h5)",
        r"(\w+)_h?(\d+)_\d+\.(keras|h5)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            model_type = match.group(1)
            horizon = int(match.group(2))
            
            if model_type in VALID_MODEL_TYPES and horizon in VALID_HORIZONS:
                return model_type, horizon
    
    # Fallback: busca por termos conhecidos
    for model_type in VALID_MODEL_TYPES:
        if model_type in filename_lower:
            horizon_match = re.search(r"h?(\d+)", filename_lower)
            horizon = int(horizon_match.group(1)) if horizon_match else 1
            if horizon in VALID_HORIZONS:
                return model_type, horizon
    
    logger.warning(f"🔍 METADADOS NÃO IDENTIFICADOS PARA: {filename}")
    return "unknown", 1

def load_scaler_file(file_path: str) -> Any:
    """Carrega arquivo de scaler com fallback para diferentes métodos"""
    try:
        # Tenta joblib primeiro
        try:
            import joblib
            return joblib.load(file_path)
        except ImportError:
            logger.debug("JOBSLIB NÃO DISPONÍVEL, USANDO PICKLE")
        
        # Fallback para pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
            
    except Exception as e:
        logger.error(f"❌ ERRO AO CARREGAR SCALER {file_path}: {e}")
        return None

def load_all_models() -> Tuple[Dict, Dict, Dict]:
    """
    Carrega todos os modelos, scalers e metadados disponíveis
    Retorna: (models, scalers, metadata)
    """
    model_dir = "models/deep_hac"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ DIRETÓRIO DE MODELOS NÃO ENCONTRADO: {model_dir}")
    
    models = {}
    scalers = {}
    metadata = {}
    
    # Carregar modelos
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        
        if filename.endswith(('.keras', '.h5')):
            model_type, horizon = parse_model_metadata(filename)
            
            if model_type not in models:
                models[model_type] = {}
            
            try:
                model = load_model(file_path, compile=False)
                model.compile(optimizer="adam", loss="mse")
                models[model_type][horizon] = model
                
                # Extrair metadados do modelo
                input_shape = model.input_shape
                lookback = input_shape[1] if input_shape and len(input_shape) > 1 else DEFAULT_LOOKBACK
                
                if model_type not in metadata:
                    metadata[model_type] = {}
                metadata[model_type][horizon] = {
                    'filename': filename,
                    'lookback': lookback,
                    'input_shape': input_shape,
                    'params': model.count_params(),
                    'loaded_at': datetime.utcnow().isoformat()
                }
                
                logger.info(f"📦 MODELO CARREGADO: {model_type.upper()} H{horizon} (LOOKBACK={lookback})")
                
            except Exception as e:
                logger.error(f"❌ ERRO AO CARREGAR MODELO {filename}: {e}")
    
    # Carregar scalers
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(model_dir, filename)
            scaler = load_scaler_file(file_path)
            
            if scaler is None:
                continue
            
            # Identificar modelo e horizonte do scaler
            filename_lower = filename.lower()
            
            for model_type in VALID_MODEL_TYPES:
                if model_type in filename_lower:
                    horizon_match = re.search(r"h?(\d+)", filename_lower)
                    if horizon_match:
                        horizon = int(horizon_match.group(1))
                        
                        if model_type not in scalers:
                            scalers[model_type] = {}
                        if horizon not in scalers[model_type]:
                            scalers[model_type][horizon] = {}
                        
                        if '_x_' in filename_lower or 'scaler_x' in filename_lower:
                            scalers[model_type][horizon]['X'] = scaler
                            logger.debug(f"🔧 SCALER X CARREGADO PARA {model_type.upper()} H{horizon}")
                        elif '_y_' in filename_lower or 'scaler_y' in filename_lower:
                            scalers[model_type][horizon]['Y'] = scaler
                            logger.debug(f"🔧 SCALER Y CARREGADO PARA {model_type.upper()} H{horizon}")
    
    # Log resumo
    total_models = sum(len(m) for m in models.values())
    total_scalers = sum(len(s) for m in scalers.values() for s in m.values())
    logger.info(f"📊 MODELOS CARREGADOS: {total_models}")
    logger.info(f"🔧 SCALERS CARREGADOS: {total_scalers}")
    
    return models, scalers, metadata

# ============================================================
# 3. PREPARAÇÃO DE SEQUÊNCIA AVANÇADA - CORRIGIDA
# ============================================================

def prepare_sequence_for_model(
    df: pd.DataFrame, 
    features: List[str], 
    lookback: int,
    scaler_X: Any = None
) -> np.ndarray:
    """
    Prepara sequência de entrada para um modelo específico
    """
    if len(df) < lookback:
        available = len(df)
        logger.warning(f"⚠️ DADOS INSUFICIENTES: {available} < {lookback}")
        
        # Preencher com zeros se necessário
        if available == 0:
            raise ValueError("❌ NENHUM DADO DISPONÍVEL PARA CRIAR SEQUÊNCIA")
        
        # Repetir os dados disponíveis ou preencher com zeros
        padding_needed = lookback - available
        repeated_data = pd.concat([df] * (lookback // available + 1), ignore_index=True)
        sequence_data = repeated_data[features].iloc[:lookback].values
        logger.warning(f"🔄 SEQUÊNCIA PREENCHIDA: {available} -> {lookback} AMOSTRAS")
    else:
        sequence_data = df[features].tail(lookback).values
    
    # Aplicar scaler se fornecido - COM FALLBACK ROBUSTO
    if scaler_X is not None:
        try:
            # Reshape para 2D, transformar, e reshape de volta para 3D
            original_shape = sequence_data.shape
            sequence_2d = sequence_data.reshape(-1, sequence_data.shape[-1])
            sequence_scaled = scaler_X.transform(sequence_2d)
            sequence_data = sequence_scaled.reshape(original_shape)
            logger.debug(f"🔧 SCALER X APLICADO COM SUCESSO")
        except Exception as e:
            logger.error(f"❌ ERRO AO APLICAR SCALER X: {e}")
            # Fallback: normalização simples com proteção robusta
            try:
                sequence_data = (sequence_data - np.mean(sequence_data, axis=0)) / (np.std(sequence_data, axis=0) + 1e-8)
                logger.debug("🔧 FALLBACK: NORMALIZAÇÃO SIMPLES APLICADA")
            except Exception as fallback_error:
                logger.error(f"❌ ERRO NO FALLBACK DE NORMALIZAÇÃO: {fallback_error}")
                # Último recurso: normalização mínima
                sequence_data = sequence_data / (np.max(np.abs(sequence_data)) + 1e-8)
    else:
        # Normalização básica como fallback
        try:
            sequence_data = (sequence_data - np.mean(sequence_data, axis=0)) / (np.std(sequence_data, axis=0) + 1e-8)
        except Exception as e:
            logger.error(f"❌ ERRO NA NORMALIZAÇÃO BÁSICA: {e}")
            sequence_data = sequence_data / (np.max(np.abs(sequence_data)) + 1e-8)
    
    return sequence_data.reshape(1, lookback, len(features))

# ============================================================
# 4. CLASSIFICAÇÃO DE RISCO AVANÇADA
# ============================================================

def classify_advanced_risk(speed: float, bz: float, density: float = None, bt: float = None) -> Dict[str, Any]:
    """
    Classificação de risco avançada baseada em múltiplos parâmetros
    """
    score = 0
    triggers = []
    
    # Velocidade do vento solar
    if speed is not None:
        if speed > 500:
            score += 1
            triggers.append(f"VELOCIDADE ALTA: {speed:.0f} KM/S")
        if speed > 600:
            score += 1
        if speed > 700:
            score += 2
            triggers.append(f"VELOCIDADE MUITO ALTA: {speed:.0f} KM/S")
        if speed > 800:
            score += 2
    
    # Componente Bz
    if bz is not None:
        if bz < -5:
            score += 1
            triggers.append(f"BZ NEGATIVO: {bz:.1f} NT")
        if bz < -10:
            score += 2
        if bz < -15:
            score += 2
            triggers.append(f"BZ FORTEMENTE NEGATIVO: {bz:.1f} NT")
    
    # Densidade (opcional)
    if density is not None and density > 20:
        score += 1
        triggers.append(f"DENSIDADE ALTA: {density:.1f} P/CM³")
    
    # Campo magnético total (opcional)
    if bt is not None and bt > 10:
        score += 1
        if bt > 20:
            score += 1
            triggers.append(f"CAMPO MAGNÉTICO FORTE: {bt:.1f} NT")
    
    # Determinar nível de risco
    if score <= 2:
        level = "BAIXO"
        color = "green"
    elif score <= 4:
        level = "MODERADO" 
        color = "yellow"
    elif score <= 7:
        level = "ALTO"
        color = "orange"
    else:
        level = "CRÍTICO"
        color = "red"
    
    return {
        "score": score,
        "level": level,
        "color": color,
        "triggers": triggers,
        "max_possible_score": 12
    }

# ============================================================
# 5. PREVISÃO POR ENSEMBLE - CORRIGIDA
# ============================================================

def calculate_ensemble_prediction(predictions: Dict[str, Dict[int, float]]) -> Dict[int, Dict[str, float]]:
    """
    Calcula previsão por ensemble weighted por tipo de modelo
    """
    ensemble = {}
    
    # Agrupar por horizonte
    for horizon in VALID_HORIZONS:
        horizon_predictions = []
        weights = []
        
        for model_type, horizons in predictions.items():
            if horizon in horizons:
                horizon_predictions.append(horizons[horizon])
                weights.append(RealtimeConfig.ENSEMBLE_WEIGHTS.get(model_type, 0.1))
        
        if horizon_predictions:
            # Calcular média ponderada
            weighted_avg = np.average(horizon_predictions, weights=weights)
            
            # Calcular estatísticas COM PROTEÇÃO CONTRA DIVISÃO POR ZERO
            values_array = np.array(horizon_predictions)
            denominator = abs(np.mean(values_array)) + 1e-6  # PROTEÇÃO ADICIONADA
            confidence = max(0, 1 - (np.std(values_array) / denominator))
            
            ensemble[horizon] = {
                "value": float(weighted_avg),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "std": float(np.std(values_array)),
                "models_used": len(horizon_predictions),
                "confidence": confidence
            }
    
    return ensemble

# ============================================================
# 6. EXECUÇÃO PRINCIPAL AVANÇADA
# ============================================================

def run_hac_realtime_advanced(data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Executa previsão em tempo real avançada
    """
    logger.info("🚀 HAC 5.0 — REAL-TIME PREDICTOR AVANÇADO INICIADO")
    start_time = datetime.utcnow()
    
    try:
        # 1. Carregar dados
        df = load_latest_data(data_path)
        
        # 2. Identificar features disponíveis
        preferred_features = [
            "speed", "density", "temperature",
            "bx_gse", "by_gse", "bz_gse", "bt",
            "vx_gse", "vy_gse", "vz_gse", "pressure"
        ]
        
        available_features = [f for f in preferred_features if f in df.columns]
        
        if len(available_features) < 2:
            # Fallback para colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols[:4]  # Primeiras 4 colunas numéricas
            logger.warning(f"⚠️ USANDO FALLBACK FEATURES: {available_features}")
        
        logger.info(f"🎯 FEATURES UTILIZADAS: {available_features}")
        
        # 3. Carregar modelos e scalers
        models, scalers, metadata = load_all_models()
        
        if not any(models.values()):
            raise ValueError("❌ NENHUM MODELO VÁLIDO FOI CARREGADO")
        
        # 4. Coletar condições atuais
        current_conditions = {}
        for feature in ['speed', 'density', 'bz_gse', 'bt'] + available_features[:3]:
            if feature in df.columns and not df[feature].isnull().all():
                current_conditions[feature] = float(df[feature].iloc[-1])
        
        # Adicionar timestamp
        if 'timestamp' in df.columns:
            current_conditions['timestamp'] = safe_timestamp(df['timestamp'].iloc[-1])
        else:
            current_conditions['timestamp'] = datetime.utcnow().isoformat()
        
        # 5. Fazer previsões para cada modelo
        predictions = {}
        prediction_details = {}
        
        for model_type in models:
            predictions[model_type] = {}
            prediction_details[model_type] = {}
            
            for horizon in models[model_type]:
                model = models[model_type][horizon]
                
                # Obter lookback do modelo ou metadata
                lookback = DEFAULT_LOOKBACK
                if (model_type in metadata and 
                    horizon in metadata[model_type] and 
                    'lookback' in metadata[model_type][horizon]):
                    lookback = metadata[model_type][horizon]['lookback']
                else:
                    # Inferir do shape de input
                    if model.input_shape and len(model.input_shape) > 1:
                        lookback = model.input_shape[1]
                
                # Obter scaler apropriado - PULAR MODELOS SEM SCALER_Y PARA MAIOR PRECISÃO
                scaler_X = None
                scaler_Y = None
                
                if (model_type in scalers and 
                    horizon in scalers[model_type]):
                    scaler_X = scalers[model_type][horizon].get('X')
                    scaler_Y = scalers[model_type][horizon].get('Y')
                
                # PULAR MODELOS INCOMPLETOS (SEM SCALER_Y)
                if scaler_Y is None:
                    logger.warning(f"⏭️  PULANDO {model_type.upper()} H{horizon} - SCALER Y NÃO ENCONTRADO")
                    continue
                
                try:
                    # Preparar sequência
                    sequence = prepare_sequence_for_model(
                        df, available_features, lookback, scaler_X
                    )
                    
                    # Fazer previsão
                    pred_scaled = model.predict(sequence, verbose=0).flatten()[0]
                    
                    # Desnormalizar
                    try:
                        pred = scaler_Y.inverse_transform([[pred_scaled]])[0][0]
                    except Exception as e:
                        logger.warning(f"⚠️ ERRO AO DESNORMALIZAR {model_type.upper()} H{horizon}: {e}")
                        pred = float(pred_scaled)
                    
                    predictions[model_type][horizon] = float(pred)
                    
                    # Guardar detalhes
                    prediction_details[model_type][horizon] = {
                        'value': float(pred),
                        'lookback_used': lookback,
                        'features_used': available_features,
                        'scaler_used': scaler_Y is not None,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"📈 {model_type.upper()} H{horizon}: {pred:.2f}")  # LOGGING MELHORADO
                    
                except Exception as e:
                    logger.error(f"❌ ERRO NA PREVISÃO {model_type.upper()} H{horizon}: {e}")
                    continue
        
        # 6. Calcular previsão por ensemble
        ensemble_predictions = calculate_ensemble_prediction(predictions)
        
        # 7. Classificar risco
        current_speed = current_conditions.get('speed')
        current_bz = current_conditions.get('bz_gse')
        current_density = current_conditions.get('density')
        current_bt = current_conditions.get('bt')
        
        risk_assessment = classify_advanced_risk(
            current_speed, current_bz, current_density, current_bt
        )
        
        # 8. Preparar resultado final
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        output = {
            "metadata": {
                "version": "HAC 5.0",
                "generated_at": datetime.utcnow().isoformat(),
                "execution_time_seconds": execution_time,
                "data_points_used": len(df),
                "models_loaded": {mtype: len(models[mtype]) for mtype in models},
                "features_used": available_features
            },
            "current_conditions": current_conditions,
            "risk_assessment": risk_assessment,
            "predictions": {
                "by_model": predictions,
                "ensemble": ensemble_predictions
            },
            "details": prediction_details
        }
        
        # 9. Salvar resultado
        os.makedirs("results/realtime", exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"results/realtime/hac_rt_advanced_{timestamp}.json"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_safe(output), f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 RESULTADO SALVO: {output_filename}")
        logger.info(f"⏱️  TEMPO DE EXECUÇÃO: {execution_time:.2f}s")
        logger.info(f"🎯 PREVISÕES GERADAS: {len(ensemble_predictions)} HORIZONTES")
        logger.info(f"⚠️  NÍVEL DE RISCO: {risk_assessment['level']} (SCORE: {risk_assessment['score']})")
        
        return output
        
    except Exception as e:
        logger.error(f"❌ ERRO CRÍTICO NA EXECUÇÃO: {e}")
        raise

# ============================================================
# FUNÇÃO DE SERVIÇO PARA INTEGRAÇÃO - CORRIGIDA
# ============================================================

def get_latest_prediction() -> Optional[Dict[str, Any]]:
    """
    Retorna a previsão mais recente do diretório de resultados
    Útil para integração com outros sistemas
    """
    results_dir = "results/realtime"
    if not os.path.exists(results_dir):
        return None
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        return None
    
    try:
        # ORDENAR POR TEMPO DE CRIAÇÃO - CORREÇÃO APLICADA
        latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
        latest_path = os.path.join(results_dir, latest_file)
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ ERRO AO CARREGAR PREVISÃO MAIS RECENTE: {e}")
        return None

# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HAC REAL-TIME PREDICTION SYSTEM')
    parser.add_argument('--data_path', type=str, help='CAMINHO PARA ARQUIVO DE DADOS ESPECÍFICO')
    parser.add_argument('--verbose', '-v', action='store_true', help='MODO VERBOSO')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        result = run_hac_realtime_advanced(args.data_path)
        logger.info("✅ PREVISÃO EM TEMPO REAL CONCLUÍDA COM SUCESSO!")
    except Exception as e:
        logger.error(f"❌ FALHA NA PREVISÃO EM TEMPO REAL: {e}")
        exit(1)
