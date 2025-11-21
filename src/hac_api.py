"""
hac_api.py
API Oficial — HAC 5.1 Real-Time Forecasting Server (FastAPI)
Compatível com: HAC Deep Learning 4.0 + HAC Realtime 5.0
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

# ============================================================
# IMPORTS EVITANDO CÍCLO - SÓ O ESSENCIAL NO TOPO
# ============================================================

# Importações básicas que não causam ciclo
from hac_realtime_predict import (
    load_all_models,
    prepare_sequence_for_model,
    VALID_MODEL_TYPES,
    VALID_HORIZONS
)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

class Config:
    """Configurações da aplicação"""
    HOST = os.getenv("HAC_API_HOST", "0.0.0.0")
    PORT = int(os.getenv("HAC_API_PORT", "8501"))
    RELOAD = os.getenv("HAC_API_RELOAD", "false").lower() == "true"
    LOG_LEVEL = os.getenv("HAC_LOG_LEVEL", "INFO")
    MAX_SEQUENCE_LENGTH = 100
    CACHE_DURATION = timedelta(minutes=5)

# ============================================================
# LOGGING AVANÇADO
# ============================================================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hac_api.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("HAC_API")

# ============================================================
# CACHE E ESTADO GLOBAL
# ============================================================

class CacheManager:
    """Gerenciador de cache para previsões"""
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def set(self, key: str, value: Any, duration: timedelta = Config.CACHE_DURATION):
        """Armazena valor no cache"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now() + duration
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache"""
        if key in self._cache and datetime.now() < self._timestamps.get(key, datetime.min):
            return self._cache[key]
        # Remove expirado
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
        return None
    
    def clear(self):
        """Limpa todo o cache"""
        self._cache.clear()
        self._timestamps.clear()

cache = CacheManager()

# ============================================================
# MODELOS PYDANTIC (ATUALIZADO PARA v2)
# ============================================================

class InputFeatures(BaseModel):
    """Modelo para entrada de dados de previsão"""
    speed: Optional[float] = Field(None, ge=0, le=2000, description="Velocidade do vento solar (km/s)")
    density: Optional[float] = Field(None, ge=0, le=100, description="Densidade do plasma (partículas/cm³)")
    bz_gse: Optional[float] = Field(None, ge=-50, le=50, description="Componente Bz do campo magnético (nT)")
    bt: Optional[float] = Field(None, ge=0, le=100, description="Magnitude do campo magnético interplanetário (nT)")
    features: Optional[Dict[str, float]] = Field(None, description="Dicionário completo de features")
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Valida se features contém pelo menos uma variável esperada"""
        if v and not any(key in v for key in ['speed', 'density', 'bz_gse', 'bt']):
            raise ValueError("Features deve conter pelo menos uma das variáveis: speed, density, bz_gse, bt")
        return v

class PredictionResponse(BaseModel):
    """Modelo para resposta de previsão"""
    success: bool
    predictions: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str

class HealthStatus(BaseModel):
    """Modelo para status do sistema"""
    status: str
    version: str
    models_loaded: Dict[str, List[str]]
    total_models: int
    uptime: float
    cache_size: int

# ============================================================
# CARREGAMENTO DE MODELOS (LIFECYCLE)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação"""
    # Startup
    startup_time = datetime.now()
    app.state.startup_time = startup_time
    app.state.requests_processed = 0
    
    logger.info("🚀 Iniciando HAC API 5.1...")
    
    try:
        logger.info("📦 Carregando modelos HAC...")
        models, scalers, metadata = load_all_models()
        
        if not any(len(models[m]) for m in models):
            logger.error("❌ Nenhum modelo carregado!")
            raise RuntimeError("Falha no carregamento dos modelos")
        
        app.state.models = models
        app.state.scalers = scalers
        app.state.metadata = metadata
        
        total_models = sum(len(models[m]) for m in models)
        logger.info(f"✅ Total de modelos carregados: {total_models}")
        
        # Cache de modelos
        cache.set("models_info", {
            "models_loaded": {m: list(models[m].keys()) for m in models},
            "total_models": total_models
        })
        
    except Exception as e:
        logger.error(f"❌ Erro crítico no startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Encerrando HAC API...")

# ============================================================
# FASTAPI INSTANCE
# ============================================================

app = FastAPI(
    title="HAC 5.1 – Real-Time Space Weather Forecasting API",
    description="""
    ## API Oficial do Sistema Heliogeophysical Adaptive Coupling
    """,
    version="5.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================
# MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEPENDÊNCIAS
# ============================================================

async def get_models():
    """Dependency injection para modelos"""
    return app.state.models

async def get_scalers():
    """Dependency injection para scalers"""
    return app.state.scalers

async def get_metadata():
    """Dependency injection para metadata"""
    return app.state.metadata

# ============================================================
# ENDPOINTS CORRIGIDOS
# ============================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Endpoint raíz com informações da API"""
    app.state.requests_processed += 1
    return {
        "message": "🌌 HAC 5.1 – Heliogeophysical Adaptive Coupling API",
        "version": "5.1.0",
        "status": "operational",
        "requests_processed": app.state.requests_processed
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check completo do sistema"""
    uptime = (datetime.now() - app.state.startup_time).total_seconds()
    
    models_info = cache.get("models_info") or {
        "models_loaded": {},
        "total_models": 0
    }
    
    return HealthStatus(
        status="healthy",
        version="5.1.0",
        models_loaded=models_info["models_loaded"],
        total_models=models_info["total_models"],
        uptime=uptime,
        cache_size=len(cache._cache)
    )

@app.get("/models")
async def get_models_endpoint():
    """Lista todos os modelos carregados"""
    models_info = cache.get("models_info")
    if not models_info:
        raise HTTPException(503, "Informações dos modelos não disponíveis")
    
    return {
        "success": True,
        "models": models_info["models_loaded"],
        "total_models": models_info["total_models"]
    }

@app.get("/predict/realtime", response_model=PredictionResponse)
async def realtime_predict(background_tasks: BackgroundTasks):
    """
    Executa previsão em tempo real com os dados mais recentes
    """
    # ✅ CORREÇÃO 1: Import dentro da função para evitar ciclo
    from hac_realtime_predict import run_hac_realtime_advanced
    
    cache_key = "realtime_prediction"
    
    # Verifica cache
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info("📦 Retornando previsão do cache")
        return PredictionResponse(**cached_result)
    
    try:
        logger.info("🔄 Executando previsão em tempo real...")
        result = run_hac_realtime_advanced()
        
        # ✅ CORREÇÃO 3: Garantir estrutura correta do PredictionResponse
        response_data = {
            "success": True,
            "predictions": result,
            "metadata": {
                "type": "realtime", 
                "source": "omni_web",
                "execution_time": datetime.now().isoformat(),
                "models_used": list(result.keys()) if isinstance(result, dict) else []
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Armazena no cache
        cache.set(cache_key, response_data)
        app.state.requests_processed += 1
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Erro na previsão em tempo real: {e}")
        # ✅ CORREÇÃO 3: Response de erro também padronizado
        raise HTTPException(500, detail=f"Erro na previsão: {str(e)}")

@app.get("/predict/latest")
async def get_latest_prediction():
    """
    Retorna a previsão mais recente do sistema
    """
    results_dir = "results/realtime"
    
    if not os.path.exists(results_dir):
        raise HTTPException(404, "Diretório de resultados não encontrado")
    
    try:
        files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        if not files:
            raise HTTPException(404, "Nenhum arquivo de previsão encontrado")
        
        latest_file = max(
            [os.path.join(results_dir, f) for f in files],
            key=os.path.getctime
        )
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        app.state.requests_processed += 1
        
        # ✅ CORREÇÃO 3: Estrutura padronizada mesmo para este endpoint
        return {
            "success": True,
            "predictions": data,
            "metadata": {
                "file": os.path.basename(latest_file),
                "created": datetime.fromtimestamp(os.path.getctime(latest_file)).isoformat(),
                "type": "latest_cached"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar previsão mais recente: {e}")
        raise HTTPException(500, f"Erro ao carregar previsão: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def post_predict(
    request: InputFeatures,
    models: Dict = Depends(get_models),
    scalers: Dict = Depends(get_scalers),
    metadata: Dict = Depends(get_metadata)
):
    """
    Previsão customizada com dados fornecidos pelo usuário
    """
    try:
        # Prepara dados de entrada
        input_data = request.features or {
            "speed": request.speed,
            "density": request.density,
            "bz_gse": request.bz_gse,
            "bt": request.bt
        }
        
        # Remove valores None
        input_data = {k: v for k, v in input_data.items() if v is not None}
        
        if not input_data:
            raise HTTPException(400, "Nenhum dado válido fornecido")
        
        logger.info(f"🎯 Previsão customizada com dados: {input_data}")
        
        # Cria DataFrame
        df = pd.DataFrame([input_data])
        df = df.fillna(method="ffill").fillna(method="bfill")
        
        # Executa previsões
        predictions = {}
        models_used = []
        
        for model_type in models:
            predictions[model_type] = {}
            
            for horizon in models[model_type]:
                model = models[model_type][horizon]
                lookback = metadata[model_type][horizon]["lookback"]
                
                # Obtém scalers
                scaler_X = None
                scaler_Y = None
                
                if model_type in scalers and horizon in scalers[model_type]:
                    scaler_X = scalers[model_type][horizon].get("X")
                    scaler_Y = scalers[model_type][horizon].get("Y")
                
                if scaler_Y is None:
                    continue
                
                # Prepara sequência e prediz
                seq = prepare_sequence_for_model(
                    df, df.columns.tolist(), lookback, scaler_X
                )
                
                if seq is None or len(seq) == 0:
                    continue
                
                pred_scaled = model.predict(seq, verbose=0).flatten()[0]
                
                try:
                    pred = scaler_Y.inverse_transform([[pred_scaled]])[0][0]
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao inverter escala: {e}")
                    pred = float(pred_scaled)
                
                predictions[model_type][horizon] = float(pred)
                models_used.append(f"{model_type}_{horizon}")
        
        app.state.requests_processed += 1
        
        # ✅ CORREÇÃO 3: Estrutura padronizada do PredictionResponse
        return PredictionResponse(
            success=True,
            predictions=predictions,
            metadata={
                "type": "custom",
                "input_features": list(input_data.keys()),
                "models_used": models_used,
                "execution_time": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na previsão customizada: {e}")
        raise HTTPException(500, f"Erro interno na previsão: {str(e)}")

@app.get("/cache/clear")
async def clear_cache():
    """Limpa todo o cache da aplicação"""
    cache_size = len(cache._cache)
    cache.clear()
    
    logger.info("🗑️ Cache limpo")
    
    return {
        "success": True,
        "message": "Cache limpo com sucesso",
        "cleared_entries": cache_size
    }

# ============================================================
# HANDLERS DE ERRO
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler global para HTTPExceptions"""
    logger.warning(f"⚠️ HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global para exceções não tratadas"""
    logger.error(f"💥 Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Erro interno do servidor",
            "type": type(exc).__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":
    logger.info(f"🚀 Iniciando servidor HAC API em {Config.HOST}:{Config.PORT}")
    
    uvicorn.run(
        "hac_api:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD,
        log_level=Config.LOG_LEVEL.lower()
  )
  
