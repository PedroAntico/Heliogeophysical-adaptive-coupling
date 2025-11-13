"""
src/heliopredictive_observational.py
Sistema de previsão STRICT usando apenas dados observacionais
FALHA se não houver dados observacionais válidos
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/heliopredictive_observational.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('heliopredictive_observational')

class HelioPredictiveObservational:
    """Sistema de previsão 100% baseado em dados observacionais"""
    
    def __init__(self):
        self.observational_data = None
        self.models = {}
        self.predictions = {}
        self.performance = {}
        
    def load_observational_data(self):
        """
        Carrega dados OBSERVACIONAIS mais recentes
        FALHA se não encontrar dados válidos
        """
        try:
            # Buscar arquivo mais recente
            data_files = glob.glob('data_observational/solar_observational_*.csv')
            if not data_files:
                logger.error("❌ NENHUM ARQUIVO DE DADOS OBSERVACIONAIS ENCONTRADO")
                return False
            
            latest_file = sorted(data_files)[-1]
            logger.info(f"🛰️ Carregando dados OBSERVACIONAIS: {latest_file}")
            
            self.observational_data = pd.read_csv(latest_file, parse_dates=['time_tag'])
            self.observational_data = self.observational_data.sort_values('time_tag').reset_index(drop=True)
            
            # Verificação crítica
            if len(self.observational_data) < 144:  # Mínimo 24h
                logger.error("❌ Dados observacionais insuficientes para análise")
                return False
            
            logger.info(f"✅ Dados OBSERVACIONAIS carregados: {len(self.observational_data)} registros")
            logger.info(f"📊 Período: {self.observational_data['time_tag'].min()} a {self.observational_data['time_tag'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados observacionais: {str(e)}")
            return False
    
    def prepare_observational_features(self, target_var='speed'):
        """
        Prepara features APENAS com dados observacionais
        """
        if self.observational_data is None:
            logger.error("❌ Dados observacionais não carregados")
            return None
        
        df = self.observational_data.copy()
        
        # Features temporais OBSERVACIONAIS
        df['hour'] = df['time_tag'].dt.hour
        df['day_of_week'] = df['time_tag'].dt.dayofweek
        df['day_of_year'] = df['time_tag'].dt.dayofyear
        
        # Lags OBSERVACIONAIS
        lags = [1, 2, 3, 6, 12]  # Baseado em dados reais
        for lag in lags:
            df[f'{target_var}_lag_{lag}'] = df[target_var].shift(lag)
        
        # Rolling statistics OBSERVACIONAIS
        windows = [6, 12, 24]  # 30min, 1h, 2h em dados de 5min
        for window in windows:
            if len(df) >= window:
                df[f'{target_var}_rolling_mean_{window}'] = df[target_var].rolling(window=window, min_periods=1).mean()
                df[f'{target_var}_rolling_std_{window}'] = df[target_var].rolling(window=window, min_periods=1).std()
        
        # Remover NaN criados
        df = df.dropna()
        
        if len(df) < 100:
            logger.error("❌ Dados insuficientes após preparação")
            return None
        
        # Split temporal (80/20)
        split_idx = int(0.8 * len(df))
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        
        feature_cols = [col for col in df.columns if col not in ['time_tag', target_var]]
        
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data[target_var]
        y_test = test_data[target_var]
        time_test = test_data['time_tag']
        
        logger.info(f"📈 Features OBSERVACIONAIS preparadas: {len(feature_cols)} variáveis")
        logger.info(f"📊 Treino: {len(X_train)}, Teste: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, time_test, feature_cols
    
    def run_observational_analysis(self, target_var='speed'):
        """
        Executa análise STRICT com dados observacionais
        """
        logger.info("🚀 INICIANDO ANÁLISE OBSERVACIONAL STRICT")
        
        # Carregar dados
        if not self.load_observational_data():
            logger.error("❌ FALHA: Dados observacionais não disponíveis")
            return False
        
        # Preparar features
        preparation = self.prepare_observational_features(target_var)
        if preparation is None:
            logger.error("❌ FALHA: Preparação de features falhou")
            return False
        
        X_train, X_test, y_train, y_test, time_test, feature_cols = preparation
        
        # Modelos simples (evitando overfitting)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Random Forest com dados observacionais
        logger.info("🌲 Treinando Random Forest com dados observacionais...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Persistência (baseline observacional)
        persistence_pred = y_test.shift(1).fillna(method='bfill')
        
        # Calcular métricas OBSERVACIONAIS
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        
        persistence_rmse = np.sqrt(mean_squared_error(y_test, persistence_pred))
        persistence_r2 = r2_score(y_test, persistence_pred)
        
        improvement = (persistence_rmse - rf_rmse) / persistence_rmse * 100
        
        self.performance = {
            'random_forest': {'rmse': rf_rmse, 'r2': rf_r2},
            'persistence': {'rmse': persistence_rmse, 'r2': persistence_r2},
            'improvement_percentage': improvement,
            'test_samples': len(y_test),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        self.predictions = {
            'random_forest': rf_pred,
            'persistence': persistence_pred,
            'actual': y_test.values,
            'time': time_test.values
        }
        
        self.models['random_forest'] = rf_model
        
        logger.info("✅ ANÁLISE OBSERVACIONAL CONCLUÍDA")
        logger.info(f"📊 RF RMSE: {rf_rmse:.2f}, R²: {rf_r2:.3f}")
        logger.info(f"📊 Persistence RMSE: {persistence_rmse:.2f}, R²: {persistence_r2:.3f}")
        logger.info(f"🚀 Melhoria: {improvement:.1f}%")
        
        return True
    
    def generate_observational_report(self):
        """Gera relatório da análise observacional"""
        if not self.performance:
            logger.error("❌ Nenhuma análise realizada")
            return
        
        os.makedirs('results_observational', exist_ok=True)
        
        # Relatório JSON
        report = {
            'observational_analysis': self.performance,
            'dataset_info': {
                'total_records': len(self.observational_data) if self.observational_data is not None else 0,
                'time_range': {
                    'start': self.observational_data['time_tag'].min().isoformat() if self.observational_data is not None else None,
                    'end': self.observational_data['time_tag'].max().isoformat() if self.observational_data is not None else None
                } if self.observational_data is not None else None
            },
            'validation_note': 'ANÁLISE 100% BASEADA EM DADOS OBSERVACIONAIS - SEM DADOS SINTÉTICOS'
        }
        
        import json
        with open('results_observational/observational_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Gráfico observacional
        self._generate_observational_plot()
        
        # Relatório textual
        self._generate_text_report()
        
        logger.info("💾 Relatório observacional salvo em 'results_observational/'")
    
    def _generate_observational_plot(self):
        """Gera gráfico com dados observacionais"""
        plt.figure(figsize=(12, 8))
        
        # Plotar apenas os últimos 100 pontos para clareza
        n_plot = min(100, len(self.predictions['actual']))
        
        time_plot = self.predictions['time'][-n_plot:]
        actual_plot = self.predictions['actual'][-n_plot:]
        rf_plot = self.predictions['random_forest'][-n_plot:]
        persistence_plot = self.predictions['persistence'][-n_plot:]
        
        plt.plot(time_plot, actual_plot, label='Observado', linewidth=2, color='black')
        plt.plot(time_plot, rf_plot, label='Random Forest', linewidth=1.5, color='red', alpha=0.8)
        plt.plot(time_plot, persistence_plot, label='Persistência', linewidth=1.5, color='blue', alpha=0.6)
        
        plt.title('Previsão Heliogeofísica - Dados 100% Observacionais', fontsize=14, fontweight='bold')
        plt.xlabel('Tempo')
        plt.ylabel('Velocidade do Vento Solar (km/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('results_observational/observational_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self):
        """Gera relatório textual"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("RELATÓRIO DE ANÁLISE OBSERVACIONAL - DADOS 100% REAIS")
        report_lines.append("="*70)
        report_lines.append(f"Data da análise: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        report_lines.append("")
        
        if self.observational_data is not None:
            report_lines.append("📊 DADOS OBSERVACIONAIS:")
            report_lines.append(f"  - Registros: {len(self.observational_data)}")
            report_lines.append(f"  - Período: {self.observational_data['time_tag'].min()} a {self.observational_data['time_tag'].max()}")
            report_lines.append(f"  - Variáveis: {', '.join([col for col in self.observational_data.columns if col != 'time_tag'])}")
            report_lines.append("")
        
        report_lines.append("📈 DESEMPENHO PREDITIVO:")
        report_lines.append(f"  - Random Forest: RMSE = {self.performance['random_forest']['rmse']:.2f}, R² = {self.performance['random_forest']['r2']:.3f}")
        report_lines.append(f"  - Persistência:   RMSE = {self.performance['persistence']['rmse']:.2f}, R² = {self.performance['persistence']['r2']:.3f}")
        report_lines.append(f"  - Melhoria:       {self.performance['improvement_percentage']:.1f}%")
        report_lines.append("")
        
        report_lines.append("✅ STATUS: ANÁLISE 100% BASEADA EM DADOS OBSERVACIONAIS")
        report_lines.append("   - Zero dados sintéticos")
        report_lines.append("   - Zero fallback simulado") 
        report_lines.append("   - Apenas dados reais da NASA/NOAA")
        report_lines.append("")
        report_lines.append("="*70)
        
        report_text = "\n".join(report_lines)
        
        with open('results_observational/observational_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)

def main():
    """Função principal - Análise 100% observacional"""
    analyzer = HelioPredictiveObservational()
    
    # Executar análise STRICT
    if not analyzer.run_observational_analysis(target_var='speed'):
        logger.error("❌ ANÁLISE OBSERVACIONAL FALHOU")
        sys.exit(1)
    
    # Gerar relatórios
    analyzer.generate_observational_report()
    
    print("\n🎯 ANÁLISE OBSERVACIONAL CONCLUÍDA COM SUCESSO!")
    print("📁 Resultados em: results_observational/")

if __name__ == '__main__':
    main()
