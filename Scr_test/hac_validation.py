"""
src/hac_validation.py - VERSÃO HAC v2
Validação científica completa com múltiplos baselines e dados OMNI
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

# Importar HACForecaster e novos baselines
try:
    from heliopredictive import HACForecaster
except ImportError:
    HACForecaster = None
    logging.warning("HACForecaster não disponível")

try:
    from baselines import baseline_arima, baseline_lstm, baseline_prophet, evaluate_baseline
except ImportError:
    logging.warning("Baselines avançados não disponíveis")

# Configuração de logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/hac_validation_v2.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("hac_validation_v2")

class HACv2Validator:
    """Validador HAC v2 com baselines avançados e dados multi-fonte"""
    
    def __init__(self):
        self.setup_directories()
        self.data_sources = {}
        
    def setup_directories(self):
        """Cria estrutura de diretórios para HAC v2"""
        for d in ['data', 'results', 'plots', 'logs', 'models']:
            os.makedirs(d, exist_ok=True)
    
    def load_multi_source_data(self):
        """
        Carrega dados de múltiplas fontes prioritariamente:
        1. OMNI históricos (NASA)
        2. NOAA tempo real  
        3. Fallback simulado
        """
        sources_attempted = []
        
        # Tentar OMNI primeiro
        omni_path = "data/omni_historical_data.csv"
        if os.path.exists(omni_path):
            try:
                df_omni = pd.read_csv(omni_path)
                df_omni['time_tag'] = pd.to_datetime(df_omni['time_tag'])
                if len(df_omni) > 1000:  # Dados suficientes
                    self.data_sources['omni'] = {
                        'data': df_omni,
                        'registros': len(df_omni),
                        'periodo': f"{df_omni['time_tag'].min()} a {df_omni['time_tag'].max()}",
                        'fonte': 'NASA_OMNI_CDAWeb'
                    }
                    sources_attempted.append('OMNI')
                    logger.info("✅ Dados OMNI carregados")
                    return df_omni, 'NASA_OMNI'
            except Exception as e:
                logger.warning(f"Falha ao carregar OMNI: {e}")
        
        # Tentar NOAA tempo real
        noaa_path = "data/solar_data_latest.csv"
        if os.path.exists(noaa_path):
            try:
                df_noaa = pd.read_csv(noaa_path)
                df_noaa['time_tag'] = pd.to_datetime(df_noaa['time_tag'])
                if len(df_noaa) > 100:
                    self.data_sources['noaa'] = {
                        'data': df_noaa,
                        'registros': len(df_noaa),
                        'fonte': 'NOAA_REAL_TIME'
                    }
                    sources_attempted.append('NOAA')
                    logger.info("✅ Dados NOAA carregados")
                    return df_noaa, 'NOAA_REAL_TIME'
            except Exception as e:
                logger.warning(f"Falha ao carregar NOAA: {e}")
        
        # Fallback simulado
        logger.warning("⚠️ Usando dados simulados (fallback)")
        from data_fetcher_omni import OMNIDataCollector
        collector = OMNIDataCollector()
        df_sim = collector.create_omni_sample(days=30)
        
        self.data_sources['simulado'] = {
            'data': df_sim,
            'registros': len(df_sim),
            'fonte': 'SIMULADO_FALLBACK'
        }
        sources_attempted.append('SIMULADO')
        
        return df_sim, 'SIMULADO_FALLBACK'
    
    def run_comprehensive_validation(self, df, data_source):
        """
        Executa validação completa com múltiplos baselines
        """
        horizons = [1, 3, 6, 12, 24]
        results = []
        
        # Série principal (speed)
        series = df['speed'].dropna().values
        
        logger.info(f"🔬 Iniciando validação HAC v2 - {len(series)} pontos - Fonte: {data_source}")
        
        for horizon in horizons:
            logger.info(f"🎯 Horizonte {horizon}h")
            
            horizon_result = {
                'horizon_h': horizon,
                'data_source': data_source,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # 1. Baseline Persistência (NOAA)
            try:
                y_true_p, y_pred_p = self.persistence_forecast(series, horizon)
                horizon_result['persistence_rmse'] = float(np.sqrt(mean_squared_error(y_true_p, y_pred_p)))
                horizon_result['persistence_r2'] = float(r2_score(y_true_p, y_pred_p))
            except Exception as e:
                logger.warning(f"Persistência falhou H{horizon}: {e}")
                horizon_result['persistence_rmse'] = None
            
            # 2. Baseline Regressão Linear
            try:
                y_true_lr, y_pred_lr = self.linear_regression_forecast(series, horizon)
                horizon_result['linear_rmse'] = float(np.sqrt(mean_squared_error(y_true_lr, y_pred_lr)))
                horizon_result['linear_r2'] = float(r2_score(y_true_lr, y_pred_lr))
            except Exception as e:
                logger.warning(f"Regressão Linear falhou H{horizon}: {e}")
                horizon_result['linear_rmse'] = None
            
            # 3. Baseline ARIMA
            try:
                result_arima = evaluate_baseline(series, horizon, method='arima')
                if result_arima:
                    horizon_result['arima_rmse'] = result_arima['rmse']
                    horizon_result['arima_r2'] = result_arima['r2']
            except Exception as e:
                logger.warning(f"ARIMA falhou H{horizon}: {e}")
                horizon_result['arima_rmse'] = None
            
            # 4. Baseline LSTM
            try:
                result_lstm = evaluate_baseline(series, horizon, method='lstm')
                if result_lstm:
                    horizon_result['lstm_rmse'] = result_lstm['rmse'] 
                    horizon_result['lstm_r2'] = result_lstm['r2']
            except Exception as e:
                logger.warning(f"LSTM falhou H{horizon}: {e}")
                horizon_result['lstm_rmse'] = None
            
            # 5. HAC Ensemble
            if HACForecaster is not None:
                try:
                    hac_result = self.run_hac_forecast(df, horizon)
                    horizon_result['hac_rmse'] = hac_result.get('rmse')
                    horizon_result['hac_r2'] = hac_result.get('r2')
                    
                    # Calcular melhoria vs persistência
                    if horizon_result['persistence_rmse'] and horizon_result['hac_rmse']:
                        improvement = (horizon_result['persistence_rmse'] - horizon_result['hac_rmse']) / horizon_result['persistence_rmse'] * 100
                        horizon_result['improvement_pct'] = float(improvement)
                    
                except Exception as e:
                    logger.warning(f"HAC falhou H{horizon}: {e}")
                    horizon_result['hac_rmse'] = None
            else:
                horizon_result['hac_rmse'] = None
            
            results.append(horizon_result)
        
        return pd.DataFrame(results)
    
    def run_hac_forecast(self, df, horizon):
        """Executa previsão HAC"""
        try:
            forecaster = HACForecaster()
            result = forecaster.forecast(df, horizon=horizon)
            
            # Extrair scores (compatibilidade com diferentes versões)
            ensemble_scores = (
                result.get("ensemble_scores") or 
                result.get("ensemble") or 
                result.get("scores", {}).get("Ensemble", {})
            )
            
            return {
                'rmse': ensemble_scores.get("RMSE"),
                'r2': ensemble_scores.get("R2"),
                'mae': ensemble_scores.get("MAE")
            }
        except Exception as e:
            logger.error(f"Erro HAC forecast: {e}")
            return {}
    
    def persistence_forecast(self, series, horizon):
        """Baseline de persistência"""
        y_true = series[horizon:]
        y_pred = series[:-horizon]
        # Garantir mesmo tamanho
        min_len = min(len(y_true), len(y_pred))
        return y_true[:min_len], y_pred[:min_len]
    
    def linear_regression_forecast(self, series, horizon, lookback=6):
        """Baseline de regressão linear com lags"""
        X, y = [], []
        
        for i in range(lookback, len(series) - horizon):
            X.append(series[i-lookback:i])
            y.append(series[i+horizon-1])  # Prever horizon steps ahead
        
        if len(X) < 10:
            return None, None
        
        X, y = np.array(X), np.array(y)
        
        # Split treino/teste
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return y_test, y_pred
    
    def create_comprehensive_plot(self, results_df, data_source):
        """Cria visualização completa HAC v2"""
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        horizons = results_df['horizon_h']
        
        # Gráfico 1: Comparação de RMSE
        models = ['persistence', 'linear', 'arima', 'lstm', 'hac']
        colors = ['#e74c3c', '#3498db', '#9b59b6', '#f1c40f', '#2ecc71']
        labels = ['Persistência', 'Reg. Linear', 'ARIMA', 'LSTM', 'HAC Ensemble']
        
        for i, model in enumerate(models):
            rmse_values = results_df[f'{model}_rmse']
            if not rmse_values.isna().all():
                ax1.plot(horizons, rmse_values, 'o-', color=colors[i], label=labels[i], linewidth=2, markersize=8)
        
        ax1.set_xlabel('Horizonte de Previsão (horas)')
        ax1.set_ylabel('RMSE (km/s)')
        ax1.set_title('Comparação de RMSE: HAC vs Baselines Avançados')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Comparação de R²
        for i, model in enumerate(models):
            r2_values = results_df[f'{model}_r2']
            if not r2_values.isna().all():
                ax2.plot(horizons, r2_values, 's-', color=colors[i], label=labels[i], linewidth=2, markersize=8)
        
        ax2.set_xlabel('Horizonte de Previsão (horas)')
        ax2.set_ylabel('R²')
        ax2.set_title('Comparação de R²: HAC vs Baselines Avançados')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Melhoria do HAC vs Persistência
        if 'improvement_pct' in results_df.columns:
            improvements = results_df['improvement_pct'].fillna(0)
            bars = ax3.bar(horizons, improvements, color='#2ecc71', alpha=0.7)
            ax3.set_xlabel('Horizonte de Previsão (horas)')
            ax3.set_ylabel('Melhoria do HAC (%)')
            ax3.set_title('Melhoria do HAC vs Modelo de Persistência')
            ax3.grid(True, alpha=0.3)
            
            for bar, imp in zip(bars, improvements):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 4: Heatmap de performance relativa
        performance_data = []
        baseline_models = ['persistence', 'linear', 'arima', 'lstm']
        
        for model in baseline_models:
            row = []
            for h in horizons:
                hac_rmse = results_df[results_df['horizon_h'] == h]['hac_rmse'].iloc[0]
                model_rmse = results_df[results_df['horizon_h'] == h][f'{model}_rmse'].iloc[0]
                
                if hac_rmse and model_rmse and model_rmse > 0:
                    relative_imp = (model_rmse - hac_rmse) / model_rmse * 100
                else:
                    relative_imp = 0
                row.append(relative_imp)
            performance_data.append(row)
        
        im = ax4.imshow(performance_data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        ax4.set_xticks(range(len(horizons)))
        ax4.set_xticklabels([f'{h}h' for h in horizons])
        ax4.set_yticks(range(len(baseline_models)))
        ax4.set_yticklabels(['Persistência', 'Reg. Linear', 'ARIMA', 'LSTM'])
        ax4.set_title('Melhoria Relativa do HAC vs Outros Modelos (%)')
        
        # Adicionar valores no heatmap
        for i in range(len(baseline_models)):
            for j in range(len(horizons)):
                text = ax4.text(j, i, f'{performance_data[i][j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Melhoria do HAC (%)')
        
        plt.suptitle(f'HAC v2 - Validação Científica Completa\nFonte de Dados: {data_source}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Salvar plot
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        plot_path = f"plots/hac_v2_validation_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Plot HAC v2 salvo: {plot_path}")
        return plot_path
    
    def generate_scientific_report(self, results_df, data_source):
        """Gera relatório científico completo HAC v2"""
        timestamp = datetime.utcnow()
        
        report = {
            'metadata': {
                'timestamp': timestamp.isoformat(),
                'data_source': data_source,
                'data_sources_used': self.data_sources,
                'horizons_tested': list(results_df['horizon_h']),
                'total_models_compared': 5  # persist, linear, arima, lstm, hac
            },
            'summary_metrics': {},
            'detailed_results': results_df.to_dict('records'),
            'conclusions': {}
        }
        
        # Métricas sumarizadas
        valid_hac = results_df['hac_rmse'].notna()
        if valid_hac.any():
            avg_improvement = results_df.loc[valid_hac, 'improvement_pct'].mean()
            avg_hac_r2 = results_df.loc[valid_hac, 'hac_r2'].mean()
            best_horizon = results_df.loc[valid_hac, 'improvement_pct'].idxmax()
        else:
            avg_improvement = 0
            avg_hac_r2 = 0
            best_horizon = 1
        
        report['summary_metrics'] = {
            'average_improvement_vs_persistence': avg_improvement,
            'average_hac_r2': avg_hac_r2,
            'best_performing_horizon': int(results_df.loc[best_horizon, 'horizon_h']),
            'data_quality': 'HIGH' if 'NASA' in data_source else 'MEDIUM'
        }
        
        # Salvar relatório JSON
        report_path = f"results/hac_v2_scientific_report_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Relatório textual executivo
        text_report = f"""
HAC v2 - RELATÓRIO CIENTÍFICO DE VALIDAÇÃO
===========================================

DATA: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}
FONTE DE DADOS: {data_source}
MODELOS COMPARADOS: 5 (Persistência, Regressão Linear, ARIMA, LSTM, HAC Ensemble)

RESULTADOS PRINCIPAIS:
• Melhoria média do HAC: {avg_improvement:.1f}%
• R² médio do HAC: {avg_hac_r2:.3f}
• Melhor horizonte: {report['summary_metrics']['best_performing_horizon']}h
• Qualidade dos dados: {report['summary_metrics']['data_quality']}

DETALHES POR HORIZONTE:
"""
        for _, row in results_df.iterrows():
            if row['hac_rmse']:
                text_report += f"• {row['horizon_h']:2d}h: Persist={row['persistence_rmse']:6.1f} | HAC={row['hac_rmse']:6.1f} | Δ={row.get('improvement_pct', 0):5.1f}%\n"
        
        text_report += f"\nBASELINES AVANÇADOS:"
        for model in ['arima', 'lstm']:
            rmse_vals = results_df[f'{model}_rmse'].dropna()
            if len(rmse_vals) > 0:
                text_report += f"\n• {model.upper()}: RMSE médio = {rmse_vals.mean():.1f}"
        
        text_report += f"\n\nORIGEM DOS DADOS:"
        for source, info in self.data_sources.items():
            text_report += f"\n• {source.upper()}: {info['fonte']} ({info['registros']} registros)"
        
        # Salvar relatório textual
        text_path = f"results/hac_v2_executive_summary_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(text_report)
        logger.info(f"📋 Relatório HAC v2 salvo: {report_path}")
        
        return report
    
    def run_validation_pipeline(self):
        """Pipeline completo de validação HAC v2"""
        logger.info("🚀 INICIANDO HAC v2 - VALIDAÇÃO CIENTÍFICA COMPLETA")
        
        # 1. Carregar dados multi-fonte
        df, data_source = self.load_multi_source_data()
        
        if df.empty or len(df) < 100:
            logger.error("❌ Dados insuficientes para validação")
            return False
        
        # 2. Executar validação com múltiplos baselines
        results_df = self.run_comprehensive_validation(df, data_source)
        
        # 3. Gerar visualizações
        plot_path = self.create_comprehensive_plot(results_df, data_source)
        
        # 4. Salvar resultados
        results_path = f"results/hac_v2_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
        results_df.to_csv(results_path, index=False)
        
        # 5. Gerar relatório científico
        report = self.generate_scientific_report(results_df, data_source)
        
        logger.info("✅ HAC v2 - VALIDAÇÃO CONCLUÍDA COM SUCESSO")
        return True

def main():
    """Função principal HAC v2"""
    validator = HACv2Validator()
    success = validator.run_validation_pipeline()
    
    if success:
        logger.info("🎉 HAC v2 finalizado - Relatórios científicos gerados!")
    else:
        logger.error("💥 HAC v2 encontrou problemas na validação")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
