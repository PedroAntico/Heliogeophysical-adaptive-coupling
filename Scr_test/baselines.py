"""
src/baselines.py
Implementações de modelos baseline avançados: ARIMA e LSTM
Para comparação científica com HAC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("baselines")

def baseline_arima(series, horizon, order=(5,1,0)):
    """
    Baseline ARIMA para previsão de séries temporais
    
    Parâmetros:
        series: array-like, série temporal univariada
        horizon: int, número de passos à frente para prever
        order: tuple, parâmetros (p,d,q) do ARIMA
    
    Retorna:
        forecast: array com previsão
        model: modelo treinado (para inspeção)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Garantir que é numpy array
        series = np.array(series, dtype=float)
        
        # Treinar modelo ARIMA
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Fazer previsão
        forecast = fitted_model.forecast(steps=horizon)
        
        logger.info(f"✅ ARIMA({order}) treinado - AIC: {fitted_model.aic:.2f}")
        return forecast, fitted_model
        
    except Exception as e:
        logger.error(f"❌ ARIMA falhou: {e}")
        return np.full(horizon, np.nan), None

def baseline_lstm(series, horizon, lookback=24, epochs=50):
    """
    Baseline LSTM para previsão de séries temporais
    
    Parâmetros:
        series: array-like, série temporal univariada  
        horizon: int, horizonte de previsão
        lookback: int, janela temporal para lookback
        epochs: int, épocas de treinamento
    
    Retorna:
        forecast: array com previsão
        model: modelo treinado (para inspeção)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        
        # Configurar para evitar uso excessivo de GPU
        tf.config.set_visible_devices([], 'GPU')
        
        series = np.array(series, dtype=float).reshape(-1, 1)
        
        # Normalização
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series)
        
        # Preparar dados de treinamento
        X, y = [], []
        for i in range(lookback, len(series_scaled) - horizon):
            X.append(series_scaled[i-lookback:i, 0])
            y.append(series_scaled[i:i+horizon, 0])
        
        if len(X) < lookback:
            logger.warning("Dados insuficientes para LSTM")
            return np.full(horizon, np.nan), None
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # reshape para LSTM
        
        # Modelo LSTM
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Treinamento
        history = model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=32, 
            verbose=0,
            validation_split=0.2
        )
        
        # Previsão
        last_sequence = series_scaled[-lookback:]
        X_pred = last_sequence.reshape(1, lookback, 1)
        y_pred_scaled = model.predict(X_pred, verbose=0)
        
        # Reverter normalização
        forecast = scaler.inverse_transform(y_pred_scaled).flatten()
        
        logger.info(f"✅ LSTM treinado - loss final: {history.history['loss'][-1]:.4f}")
        return forecast, model
        
    except Exception as e:
        logger.error(f"❌ LSTM falhou: {e}")
        return np.full(horizon, np.nan), None

def baseline_prophet(series, horizon, freq='H'):
    """
    Baseline Facebook Prophet para comparação
    
    Parâmetros:
        series: array-like com índice temporal
        horizon: int, passos à frente
        freq: string, frequência dos dados
    
    Retorna:
        forecast: array com previsão
    """
    try:
        from prophet import Prophet
        
        # Preparar dados no formato Prophet
        if hasattr(series, 'index'):
            dates = series.index
        else:
            dates = pd.date_range(start='2020-01-01', periods=len(series), freq=freq)
        
        df = pd.DataFrame({
            'ds': dates,
            'y': series
        })
        
        # Modelo Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(df)
        
        # Criar dataframe futuro
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast_df = model.predict(future)
        
        # Extrair previsão
        forecast = forecast_df['yhat'].values[-horizon:]
        
        logger.info("✅ Prophet treinado com sucesso")
        return forecast, model
        
    except Exception as e:
        logger.error(f"❌ Prophet falhou: {e}")
        return np.full(horizon, np.nan), None

def evaluate_baseline(series, horizon, method='arima'):
    """
    Função unificada para avaliação de baselines
    
    Parâmetros:
        series: série temporal
        horizon: horizonte de previsão
        method: 'arima', 'lstm', ou 'prophet'
    
    Retorna:
        dict: métricas de performance
    """
    # Split treino/teste
    split_idx = int(len(series) * 0.8)
    train = series[:split_idx]
    test = series[split_idx:split_idx + horizon]
    
    if len(test) < horizon:
        logger.warning("Dados insuficientes para avaliação")
        return None
    
    # Selecionar método
    if method == 'arima':
        forecast, _ = baseline_arima(train, horizon)
    elif method == 'lstm':
        forecast, _ = baseline_lstm(train, horizon)
    elif method == 'prophet':
        forecast, _ = baseline_prophet(train, horizon)
    else:
        raise ValueError(f"Método não suportado: {method}")
    
    # Calcular métricas
    if forecast is not None and not np.isnan(forecast).all():
        rmse_val = np.sqrt(mean_squared_error(test, forecast))
        r2_val = r2_score(test, forecast)
        mae_val = np.mean(np.abs(test - forecast))
        
        return {
            'method': method,
            'horizon': horizon,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'forecast': forecast
        }
    else:
        logger.warning(f"❌ {method} retornou previsão inválida")
        return None

if __name__ == "__main__":
    # Teste dos baselines
    logging.basicConfig(level=logging.INFO)
    
    # Dados de exemplo
    t = np.linspace(0, 4*np.pi, 1000)
    series = np.sin(t) + 0.1 * np.random.normal(size=1000)
    
    print("🧪 Testando baselines...")
    
    # Testar ARIMA
    result_arima = evaluate_baseline(series, horizon=24, method='arima')
    if result_arima:
        print(f"📊 ARIMA - RMSE: {result_arima['rmse']:.4f}, R²: {result_arima['r2']:.4f}")
    
    # Testar LSTM
    result_lstm = evaluate_baseline(series, horizon=24, method='lstm') 
    if result_lstm:
        print(f"📊 LSTM - RMSE: {result_lstm['rmse']:.4f}, R²: {result_lstm['r2']:.4f}")
