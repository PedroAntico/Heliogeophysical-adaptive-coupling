import os
import pandas as pd
import numpy as np
import requests
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from heliopredictive import HACForecaster

# ============================================================
# 🚀 VALIDAÇÃO CIENTÍFICA HAC VS NOAA - CORRIGIDO
# ============================================================

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/validation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class SolarDataValidator:
    """Validador científico para comparação HAC vs NOAA"""
    
    def __init__(self):
        self.forecaster = HACForecaster()
    
    def setup_directories(self):
        """Cria estrutura de diretórios para resultados"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
    
    def fetch_noaa_realtime(self, days=5):
        """Coleta dados em tempo real da NOAA com tratamento dinâmico de colunas"""
        logger.info(f"Coletando dados NOAA últimos {days} dias")
        
        plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
        mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
        
        try:
            # Headers para evitar bloqueios
            headers = {'User-Agent': 'SolarValidationBot/1.0'}
            
            # Coletar dados
            plasma_data = requests.get(plasma_url, timeout=15, headers=headers).json()
            mag_data = requests.get(mag_url, timeout=15, headers=headers).json()
            
            logger.info(f"Estrutura Plasma: {len(plasma_data[0])} colunas")
            logger.info(f"Estrutura Mag: {len(mag_data[0])} colunas")
            
            # 🟢 Criação de DataFrames dinâmica e segura
    def to_dataframe_safe(data):
    header = data[0]
    body = [row for row in data[1:] if len(row) == len(header)]
    # Corrige linhas com colunas extras ou a menos
    fixed = []
    for row in data[1:]:
        if len(row) < len(header):
            row = row + [None] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]
        fixed.append(row)
    return pd.DataFrame(fixed, columns=header)

plasma_df = to_dataframe_safe(plasma_data)
mag_df = to_dataframe_safe(mag_data)
            
            # Manter apenas colunas necessárias
            required_plasma = ["time_tag", "density", "speed", "temperature"]
            required_mag = ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt"]
            
            plasma_df = plasma_df[required_plasma]
            mag_df = mag_df[required_mag]
            
            # Converter timestamps
            plasma_df["time_tag"] = pd.to_datetime(plasma_df["time_tag"])
            mag_df["time_tag"] = pd.to_datetime(mag_df["time_tag"])
            
            # Merge dos dados
            df = pd.merge_asof(
                plasma_df.sort_values("time_tag"), 
                mag_df.sort_values("time_tag"),
                on="time_tag", 
                tolerance=pd.Timedelta("5min"), 
                direction="nearest"
            )
            
            # Converter tipos de dados
            numeric_cols = ["density", "speed", "temperature", "bx_gsm", "by_gsm", "bz_gsm", "bt"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filtrar por data
            cutoff = datetime.utcnow() - timedelta(days=days)
            df = df[df["time_tag"] > cutoff].dropna()
            
            logger.info(f"✅ Dados NOAA coletados: {len(df)} registros")
            logger.info(f"📅 Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Falha na coleta NOAA: {e}")
            return self._fallback_data_source()
    
    def _fallback_data_source(self):
        """Fallback para dados locais ou exemplo"""
        try:
            # Tentar arquivo local
            if os.path.exists("data/solar_data_latest.csv"):
                df = pd.read_csv("data/solar_data_latest.csv")
                
                # Verificar colunas necessárias
                required_cols = ["time_tag", "density", "speed", "temperature", "bx_gsm", "by_gsm", "bz_gsm", "bt"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.warning(f"Colunas faltando no backup: {missing_cols}")
                    return self._create_sample_data()
                    
                df["time_tag"] = pd.to_datetime(df["time_tag"])
                logger.info(f"✅ Backup local carregado: {len(df)} registros")
                return df
        except Exception as e:
            logger.warning(f"Falha backup local: {e}")
        
        # Criar dados de exemplo
        logger.info("🔄 Criando dados de exemplo para validação")
        return self._create_sample_data()
    
    def _create_sample_data(self):
        """Cria dados de exemplo para desenvolvimento"""
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=10),
            end=datetime.utcnow(),
            freq='5min'
        )
        
        np.random.seed(42)
        n_samples = len(dates)
        
        df = pd.DataFrame({
            'time_tag': dates,
            'density': np.random.uniform(1, 20, n_samples),
            'speed': np.random.uniform(300, 600, n_samples),
            'temperature': np.random.uniform(50000, 200000, n_samples),
            'bx_gsm': np.random.uniform(-10, 10, n_samples),
            'by_gsm': np.random.uniform(-10, 10, n_samples),
            'bz_gsm': np.random.uniform(-10, 10, n_samples),
            'bt': np.random.uniform(0, 15, n_samples)
        })
        
        # Salvar dados de exemplo
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/solar_data_latest.csv", index=False)
        logger.info("💾 Dados de exemplo salvos em data/solar_data_latest.csv")
        
        return df
    
    def detect_solar_anomalies(self, df):
        """Detecta anomalias solares (CMEs, shocks, etc.)"""
        logger.info("🔍 Analisando anomalias solares...")
        
        anomalies = []
        thresholds = {
            'bz_gsm': 20,    # |Bz| > 20 nT
            'speed': 600,     # Speed > 600 km/s  
            'density': 30,    # Density > 30 p/cc
            'bt': 15          # Bt > 15 nT
        }
        
        bz_anomalies = df[df['bz_gsm'].abs() > thresholds['bz_gsm']]
        speed_anomalies = df[df['speed'] > thresholds['speed']]
        density_anomalies = df[df['density'] > thresholds['density']]
        bt_anomalies = df[df['bt'] > thresholds['bt']]
        
        if len(bz_anomalies) > 0:
            anomalies.append(f"|Bz| > {thresholds['bz_gsm']} nT: {len(bz_anomalies)} eventos")
        if len(speed_anomalies) > 0:
            anomalies.append(f"Speed > {thresholds['speed']} km/s: {len(speed_anomalies)} eventos")
        if len(density_anomalies) > 0:
            anomalies.append(f"Density > {thresholds['density']} p/cc: {len(density_anomalies)} eventos")
        if len(bt_anomalies) > 0:
            anomalies.append(f"Bt > {thresholds['bt']} nT: {len(bt_anomalies)} eventos")
        
        if anomalies:
            logger.info(f"⚠️ Anomalias detectadas: {', '.join(anomalies)}")
        else:
            logger.info("✅ Nenhuma anomalia significativa detectada")
        
        return anomalies
    
    def create_anomaly_plot(self, df):
        """Gera visualização de anomalias solares"""
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Velocidade
        plt.subplot(2, 1, 1)
        plt.plot(df['time_tag'], df['speed'], 'b-', alpha=0.7, label='Velocidade')
        plt.axhline(y=600, color='r', linestyle='--', label='Threshold CME (600 km/s)')
        plt.ylabel('Velocidade (km/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Detecção de Anomalias Solares')
        
        # Plot 2: Componente Bz
        plt.subplot(2, 1, 2)
        plt.plot(df['time_tag'], df['bz_gsm'], 'g-', alpha=0.7, label='Bz GSM')
        plt.axhline(y=-20, color='r', linestyle='--', label='Threshold Reconexão (±20 nT)')
        plt.axhline(y=20, color='r', linestyle='--')
        plt.ylabel('Bz GSM (nT)')
        plt.xlabel('Tempo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/solar_anomalies.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Gráfico de anomalias salvo em plots/solar_anomalies.png")
    
    def run_validation(self, df, horizon):
        """Executa validação para um horizonte específico"""
        try:
            logger.info(f"🎯 Validando horizonte {h}h...")
            res = self.forecaster.forecast(df, horizon=horizon)
            
            # Padronização da extração de scores
            scores = res.get("scores", {})
            persist = res.get("persist_scores", {}) or res.get("persist_score", {})
            ensemble = scores.get("Ensemble", {}) or res.get("ensemble_scores", {}) or res.get("ensemble", {})
            
            # Extrair métricas
            rmse_persist = persist.get("RMSE", np.nan)
            r2_persist = persist.get("R2", np.nan)
            rmse_hac = ensemble.get("RMSE", np.nan)
            r2_hac = ensemble.get("R2", np.nan)
            
            # Calcular melhoria
            if not np.isnan(rmse_persist) and rmse_persist > 0 and not np.isnan(rmse_hac):
                improvement = ((rmse_persist - rmse_hac) / rmse_persist) * 100
            else:
                improvement = np.nan
            
            result = {
                "Horizonte (h)": horizon,
                "RMSE_NOAA": rmse_persist,
                "RMSE_HAC": rmse_hac,
                "R2_NOAA": r2_persist,
                "R2_HAC": r2_hac,
                "Melhoria (%)": improvement,
                "Timestamp": datetime.utcnow()
            }
            
            logger.info(f"📊 Horizonte {horizon}h: NOAA={rmse_persist:.2f}, HAC={rmse_hac:.2f}, Δ={improvement:+.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro validação horizonte {horizon}h: {e}")
            return {
                "Horizonte (h)": horizon,
                "RMSE_NOAA": np.nan,
                "RMSE_HAC": np.nan,
                "R2_NOAA": np.nan,
                "R2_HAC": np.nan,
                "Melhoria (%)": np.nan,
                "Timestamp": datetime.utcnow(),
                "Erro": str(e)
            }
    
    def create_comparison_plot(self, results_df):
        """Gera gráfico de comparação HAC vs NOAA"""
        plt.figure(figsize=(10, 6))
        
        horizons = results_df["Horizonte (h)"]
        x_pos = np.arange(len(horizons))
        bar_width = 0.35
        
        plt.bar(x_pos - bar_width/2, results_df["RMSE_NOAA"], 
                bar_width, alpha=0.7, label="NOAA Persistência", color='red')
        plt.bar(x_pos + bar_width/2, results_df["RMSE_HAC"], 
                bar_width, alpha=0.7, label="HAC Ensemble", color='blue')
        
        plt.xlabel("Horizonte de Previsão (horas)")
        plt.ylabel("RMSE")
        plt.title("Comparação HAC vs NOAA - Erro de Previsão")
        plt.xticks(x_pos, horizons)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (noaa, hac) in enumerate(zip(results_df["RMSE_NOAA"], results_df["RMSE_HAC"])):
            if not np.isnan(noaa):
                plt.text(i - bar_width/2, noaa + 0.1, f'{noaa:.1f}', 
                        ha='center', va='bottom', fontsize=9)
            if not np.isnan(hac):
                plt.text(i + bar_width/2, hac + 0.1, f'{hac:.1f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig("results/hac_validation_plot.png", dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info("📈 Gráfico de comparação salvo em results/hac_validation_plot.png")
    
    def validate_hac_system(self):
        """Executa validação completa do sistema HAC"""
        logger.info("🚀 INICIANDO VALIDAÇÃO HAC vs NOAA")
        
        self.setup_directories()
        results = []
        
        # Coletar dados
        df = self.fetch_noaa_realtime(days=5)
        
        if df.empty:
            logger.error("❌ CRÍTICO: Nenhum dado disponível para validação")
            return False
        
        # Detecção de anomalias
        anomalies = self.detect_solar_anomalies(df)
        self.create_anomaly_plot(df)
        
        # Validação por horizonte
        horizons = [1, 3, 6, 12]
        
        for h in horizons:
            result = self.run_validation(df, h)
            results.append(result)
        
        # Salvar resultados
        results_df = pd.DataFrame(results)
        results_path = "results/hac_validation_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Gerar visualizações
        self.create_comparison_plot(results_df)
        
        # Relatório final
        valid_results = results_df[results_df["Melhoria (%)"].notna()]
        if len(valid_results) > 0:
            avg_improvement = valid_results["Melhoria (%)"].mean()
            success_rate = (len(valid_results) / len(results_df)) * 100
        else:
            avg_improvement = 0
            success_rate = 0
        
        logger.info(f"✅ VALIDAÇÃO CONCLUÍDA")
        logger.info(f"📊 Taxa de sucesso: {success_rate:.1f}%")
        logger.info(f"📈 Melhoria média: {avg_improvement:+.1f}%")
        logger.info(f"⚠️ Anomalias detectadas: {len(anomalies)}")
        logger.info(f"💾 Resultados salvos em: {results_path}")
        
        # Salvar relatório resumido
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'success_rate': success_rate,
            'avg_improvement': avg_improvement,
            'anomalies_detected': len(anomalies),
            'data_points': len(df),
            'validation_period': f"{df['time_tag'].min()} to {df['time_tag'].max()}"
        }
        
        with open("results/validation_summary.json", "w") as f:
            import json
            json.dump(report, f, indent=2)
        
        return True


def main():
    """Função principal"""
    validator = SolarDataValidator()
    success = validator.validate_hac_system()
    
    # Exit code para automação
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
