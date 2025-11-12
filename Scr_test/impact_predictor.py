"""
src/impact_predictor.py
Sistema de alerta precoce para eventos solares de classe X e seus impactos geofísicos
"""

import os
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/impact_predictor.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("impact_predictor")

class SolarImpactPredictor:
    """Sistema de previsão de impactos de eventos solares classe X"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_columns = [
            'speed', 'density', 'temperature', 'bz_gsm', 'bt',
            'speed_std_6h', 'density_std_6h', 'bz_std_6h',
            'speed_trend', 'density_trend', 'bz_trend'
        ]
        
        # Limiares baseados em literatura científica
        self.impact_thresholds = {
            'class_x_warning': {
                'speed_min': 600,
                'bz_min': 15,
                'density_min': 20
            },
            'radiation_storm': {
                'speed_min': 700,
                'density_min': 25
            },
            'geomagnetic_storm': {
                'bz_min': 20,
                'bt_min': 15
            }
        }
    
    def load_solar_data(self):
        """Carrega dados solares mais recentes"""
        try:
            df = pd.read_csv("data/solar_data_latest.csv")
            df["time_tag"] = pd.to_datetime(df["time_tag"])
            df = df.sort_values("time_tag").reset_index(drop=True)
            logger.info(f"📊 Dados solares carregados: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def calculate_features(self, df):
        """Calcula features para previsão de impactos"""
        if len(df) < 12:  # Mínimo de 1 hora de dados
            logger.warning("Dados insuficientes para cálculo de features")
            return df
            
        # Features estatísticas (janela de 6 horas)
        window = min(72, len(df))  # Máximo 6 horas
        
        df['speed_std_6h'] = df['speed'].rolling(window=window, min_periods=1).std()
        df['density_std_6h'] = df['density'].rolling(window=window, min_periods=1).std()
        df['bz_std_6h'] = df['bz_gsm'].abs().rolling(window=window, min_periods=1).std()
        
        # Tendências
        df['speed_trend'] = df['speed'].diff(12)  # Tendência de 1 hora
        df['density_trend'] = df['density'].diff(12)
        df['bz_trend'] = df['bz_gsm'].diff(12)
        
        return df.fillna(method='bfill')
    
    def assess_impact_risk(self, df):
        """Avalia risco de impactos baseado em limiares físicos"""
        latest = df.iloc[-1]
        risks = []
        
        # Verificar condições para alerta classe X
        if (latest['speed'] > self.impact_thresholds['class_x_warning']['speed_min'] and
            abs(latest['bz_gsm']) > self.impact_thresholds['class_x_warning']['bz_min']):
            risks.append({
                'type': 'CLASS_X_WARNING',
                'level': 'HIGH',
                'probability': 0.75,
                'indicators': {
                    'high_speed': latest['speed'],
                    'strong_bz': latest['bz_gsm'],
                    'timestamp': latest['time_tag']
                },
                'expected_impact': 'Potential radio blackouts, satellite disruptions'
            })
        
        # Verificar tempestade de radiação
        if (latest['speed'] > self.impact_thresholds['radiation_storm']['speed_min'] and
            latest['density'] > self.impact_thresholds['radiation_storm']['density_min']):
            risks.append({
                'type': 'RADIATION_STORM_WARNING', 
                'level': 'MEDIUM',
                'probability': 0.6,
                'indicators': {
                    'very_high_speed': latest['speed'],
                    'high_density': latest['density']
                },
                'expected_impact': 'Increased radiation levels, aviation risks'
            })
        
        # Verificar tempestade geomagnética
        if (abs(latest['bz_gsm']) > self.impact_thresholds['geomagnetic_storm']['bz_min'] and
            latest['bt'] > self.impact_thresholds['geomagnetic_storm']['bt_min']):
            risks.append({
                'type': 'GEOMAGNETIC_STORM_WARNING',
                'level': 'HIGH', 
                'probability': 0.8,
                'indicators': {
                    'extreme_bz': latest['bz_gsm'],
                    'strong_bt': latest['bt']
                },
                'expected_impact': 'Power grid fluctuations, aurora at low latitudes'
            })
        
        return risks
    
    def fetch_space_weather_alerts(self):
        """Busca alertas oficiais de clima espacial"""
        try:
            # NOAA Space Weather Alerts
            alert_url = "https://services.swpc.noaa.gov/products/alerts.json"
            response = requests.get(alert_url, timeout=10)
            
            if response.status_code == 200:
                alerts = response.json()
                active_alerts = [
                    alert for alert in alerts 
                    if 'X' in alert.get('message', '') or 'FLARE' in alert.get('message', '')
                ]
                return active_alerts
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível buscar alertas: {e}")
        
        return []
    
    def generate_impact_report(self, df, risks, alerts):
        """Gera relatório completo de impactos"""
        latest = df.iloc[-1]
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_source': 'NOAA_REAL_TIME',
            'current_conditions': {
                'speed_km_s': float(latest['speed']),
                'density_p_cc': float(latest['density']),
                'bz_gsm_nT': float(latest['bz_gsm']),
                'bt_nT': float(latest['bt']),
                'temperature_K': float(latest.get('temperature', 0))
            },
            'risk_assessment': risks,
            'official_alerts': alerts,
            'recommendations': self.generate_recommendations(risks),
            'next_update_utc': (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        return report
    
    def generate_recommendations(self, risks):
        """Gera recomendações baseadas nos riscos detectados"""
        recommendations = []
        
        for risk in risks:
            if risk['type'] == 'CLASS_X_WARNING':
                recommendations.extend([
                    "Monitorar comunicções por satélite",
                    "Verificar sistemas de navegação GPS",
                    "Alerte operadores de rede elétrica",
                    "Prepare-se para possíveis blackouts de rádio"
                ])
            elif risk['type'] == 'RADIATION_STORM_WARNING':
                recommendations.extend([
                    "Tripulações aéreas em rotas polares - monitorar níveis de radiação",
                    "Operadores de satélite - verificar sistemas de proteção",
                    "Possíveis impactos em sistemas de comunicação HF"
                ])
            elif risk['type'] == 'GEOMAGNETIC_STORM_WARNING':
                recommendations.extend([
                    "Operadores de rede elétrica - modo de vigilância aumentada",
                    "Operadores de tubulações - monitorar corrosões induzidas",
                    "Possíveis auroras em latitudes médias"
                ])
        
        if not recommendations:
            recommendations.append("Condições solares normais - nenhuma ação necessária")
            
        return list(set(recommendations))  # Remove duplicatas
    
    def create_impact_visualization(self, df, risks, report):
        """Cria visualização dos impactos previstos"""
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Velocidade e alertas
        plt.subplot(3, 1, 1)
        plt.plot(df['time_tag'], df['speed'], 'b-', linewidth=1, alpha=0.7, label='Velocidade')
        plt.axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Limite Classe X (600 km/s)')
        plt.axhline(y=700, color='darkred', linestyle='--', alpha=0.7, label='Limite Temp. Radiação (700 km/s)')
        
        # Destacar pontos de risco
        risk_times = [pd.to_datetime(risk['indicators']['timestamp']) for risk in risks if 'timestamp' in risk['indicators']]
        risk_speeds = [df[df['time_tag'] == rt]['speed'].values[0] for rt in risk_times if not df[df['time_tag'] == rt].empty]
        
        if risk_times:
            plt.scatter(risk_times, risk_speeds, color='red', s=100, zorder=5, label='Eventos de Risco')
        
        plt.ylabel('Velocidade (km/s)')
        plt.title('Sistema de Alerta Precoce - Impactos Solares Classe X')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Componente Bz
        plt.subplot(3, 1, 2)
        plt.plot(df['time_tag'], df['bz_gsm'], 'g-', linewidth=1, alpha=0.7, label='Bz GSM')
        plt.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Limite Alerta (15 nT)')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Limite Temp. Geomag. (20 nT)')
        plt.axhline(y=-15, color='orange', linestyle='--', alpha=0.7)
        plt.axhline(y=-20, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('Bz GSM (nT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 3: Densidade
        plt.subplot(3, 1, 3)
        plt.plot(df['time_tag'], df['density'], 'purple', linewidth=1, alpha=0.7, label='Densidade')
        plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Limite Alerta (20 p/cc)')
        plt.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Limite Temp. Radiação (25 p/cc)')
        plt.ylabel('Densidade (p/cc)')
        plt.xlabel('Tempo UTC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações de risco
        if risks:
            risk_text = "ALERTAS ATIVOS:\n" + "\n".join([f"• {risk['type']} ({risk['level']})" for risk in risks])
            plt.figtext(0.02, 0.02, risk_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
        
        plt.tight_layout()
        plt.savefig('plots/impact_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📈 Visualização de impactos salva")
    
    def run_impact_analysis(self):
        """Executa análise completa de impactos"""
        logger.info("🚨 INICIANDO ANÁLISE DE IMPACTOS CLASSE X")
        
        # Carregar dados
        df = self.load_solar_data()
        if df is None:
            logger.error("❌ Falha ao carregar dados para análise de impactos")
            return None
        
        # Calcular features
        df = self.calculate_features(df)
        
        # Avaliar riscos
        risks = self.assess_impact_risk(df)
        
        # Buscar alertas oficiais
        alerts = self.fetch_space_weather_alerts()
        
        # Gerar relatório
        report = self.generate_impact_report(df, risks, alerts)
        
        # Criar visualização
        self.create_impact_visualization(df, risks, report)
        
        # Salvar relatório
        os.makedirs("results", exist_ok=True)
        report_path = f"results/impact_forecast_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Log de resultados
        if risks:
            logger.warning(f"⚠️ ALERTAS DETECTADOS: {len(risks)} riscos identificados")
            for risk in risks:
                logger.warning(f"   • {risk['type']} - Nível: {risk['level']}")
        else:
            logger.info("✅ Nenhum risco significativo detectado")
        
        logger.info(f"💾 Relatório de impactos salvo: {report_path}")
        
        return report

def main():
    """Função principal"""
    predictor = SolarImpactPredictor()
    report = predictor.run_impact_analysis()
    
    if report:
        # Imprimir resumo executivo
        print("\n" + "="*60)
        print("🚨 RELATÓRIO DE IMPACTOS SOLARES - CLASSE X")
        print("="*60)
        print(f"📅 Data/Hora: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        
        if report['risk_assessment']:
            print("⚠️  ALERTAS ATIVOS:")
            for risk in report['risk_assessment']:
                print(f"   • {risk['type']} (Nível: {risk['level']})")
                print(f"     Probabilidade: {risk['probability']:.0%}")
                print(f"     Impacto: {risk['expected_impact']}")
        else:
            print("✅ CONDIÇÕES NORMALS - Sem alertas ativos")
        
        print("\n📋 RECOMENDAÇÕES:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
            
        print("="*60)
    else:
        logger.error("Falha na análise de impactos")

if __name__ == "__main__":
    main()
