#!/usr/bin/env python3
"""
Pipeline Principal Aprimorado - Heliogeophysical Adaptive Coupling v3.0
Inclui ML Preditivo, Múltiplas Fontes de Dados e Detecção Avançada
"""
import logging
import yaml
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Importações internas
from src.utils.logger import setup_logging
from src.fetchers.noaa_fetcher import create_noaa_fetcher
from src.fetchers.nasa_cdaweb_fetcher import create_nasa_fetcher
from src.fetchers.ensemble_fetcher import create_ensemble_fetcher
from src.processing.preprocessor import DataPreprocessor
from src.detection.advanced_detector import create_advanced_detector
from src.model.predictive_model import create_predictor

class EnhancedHeliogeophysicalPipeline:
    """Pipeline aprimorado com ML e múltiplas fontes"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = setup_logging(
            self.config['logging']['file'],
            self.config['logging']['level']
        )
        
        # Inicializa componentes avançados
        self.noaa_fetcher = create_noaa_fetcher(self.config['data_sources']['noaa'])
        self.nasa_fetcher = create_nasa_fetcher(self.config['data_sources'].get('nasa_cdaweb', {}))
        self.ensemble_fetcher = create_ensemble_fetcher(self.noaa_fetcher, self.nasa_fetcher)
        self.preprocessor = DataPreprocessor(
            self.config['processing']['resample_frequency']
        )
        self.advanced_detector = create_advanced_detector(self.config['detection'])
        self.predictor = create_predictor()
    
    def _load_config(self) -> dict:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return {}
    
    def run_enhanced_pipeline(self) -> dict:
        """Executa pipeline aprimorado completo"""
        self.logger.info("🚀 Iniciando pipeline HELIOGEOPHYSICAL 3.0")
        
        try:
            # Fase 1: Coleta de Dados com Ensemble
            self.logger.info("📡 Coletando dados de múltiplas fontes...")
            ensemble_data = self._fetch_ensemble_data()
            
            if ensemble_data.empty:
                self.logger.warning("Nenhum dado coletado do ensemble")
                return {"events": [], "status": "no_data"}
            
            # Fase 2: Processamento Avançado
            self.logger.info("⚙️ Processamento avançado de dados...")
            processed_data = self._advanced_processing(ensemble_data)
            
            # Fase 3: Detecção de Eventos (Modo Básico)
            self.logger.info("🔍 Detecção básica de eventos...")
            basic_events = self._basic_event_detection(processed_data)
            
            # Fase 4: Treinamento/Atualização do Modelo Preditivo
            self.logger.info("🧠 Operações de Machine Learning...")
            ml_results = self._ml_operations(processed_data, basic_events)
            
            # Fase 5: Detecção Avançada com ML
            self.logger.info("🎯 Detecção avançada com ML...")
            final_events = self._advanced_event_detection(processed_data, ml_results.get('predictions'))
            
            # Fase 6: Análise e Relatórios
            self.logger.info("📊 Gerando relatórios avançados...")
            analysis_report = self._generate_analysis_report(
                processed_data, final_events, ml_results
            )
            
            # Fase 7: Persistência de Resultados
            self._save_enhanced_results(processed_data, final_events, ml_results, analysis_report)
            
            self.logger.info(f"✅ Pipeline concluído — {len(final_events)} eventos detectados")
            
            return {
                "events": final_events,
                "ml_results": ml_results,
                "analysis": analysis_report,
                "processed_records": len(processed_data),
                "status": "success",
                "pipeline_version": "3.0",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline: {e}")
            return {
                "events": [],
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _fetch_ensemble_data(self) -> pd.DataFrame:
        """Coleta dados usando estratégia de ensemble"""
        return self.ensemble_fetcher.fetch_ensemble_data(days=3)
    
    def _advanced_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processamento avançado com features para ML"""
        processed = self.preprocessor.preprocess_data([data])
        
        if processed is not None:
            # Adiciona features específicas para ML
            processed = self.predictor.prepare_features(processed)
        
        return processed if processed is not None else pd.DataFrame()
    
    def _basic_event_detection(self, data: pd.DataFrame) -> list:
        """Detecção básica de eventos para treinamento do ML"""
        from src.detection.simple_detector import detect_events
        return detect_events(data)
    
    def _ml_operations(self, data: pd.DataFrame, events: list) -> dict:
        """Operações de Machine Learning"""
        ml_results = {
            "training_status": "skipped",
            "prediction_status": "skipped",
            "predictions": pd.DataFrame()
        }
        
        try:
            # Verifica se há dados suficientes para treinamento
            if len(data) >= 100 and len(events) >= 5:
                self.logger.info("🔄 Treinando/Atualizando modelo preditivo...")
                
                # Treina o modelo
                training_result = self.predictor.train(data, events)
                ml_results["training_status"] = training_result.get("status", "unknown")
                ml_results["training_metrics"] = training_result
                
                if training_result.get("status") == "success":
                    self.logger.info(f"✅ Modelo treinado - Acurácia: {training_result.get('accuracy', 0):.3f}")
            
            # Faz previsões com o modelo
            self.logger.info("🔮 Fazendo previsões com ML...")
            predictions, probabilities = self.predictor.predict(data)
            
            if not predictions.empty:
                ml_results["prediction_status"] = "success"
                ml_results["predictions"] = predictions
                ml_results["prediction_confidence"] = probabilities.tolist()
                self.logger.info(f"📈 Previsões ML: {predictions['predicted_event'].sum()} eventos previstos")
            
        except Exception as e:
            self.logger.error(f"❌ Erro nas operações de ML: {e}")
            ml_results["error"] = str(e)
        
        return ml_results
    
    def _advanced_event_detection(self, data: pd.DataFrame, ml_predictions: pd.DataFrame) -> list:
        """Detecção avançada integrando ML"""
        return self.advanced_detector.detect_advanced_events(data, ml_predictions)
    
    def _generate_analysis_report(self, data: pd.DataFrame, events: list, ml_results: dict) -> dict:
        """Gera relatório analítico avançado"""
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_quality": {
                "total_records": len(data),
                "data_completeness": data.notna().mean().mean(),
                "temporal_coverage_hours": self._calculate_temporal_coverage(data),
                "variables_available": list(data.select_dtypes(include=[np.number]).columns)
            },
            "events_summary": {
                "total_events": len(events),
                "by_type": self._count_events_by_type(events),
                "by_severity": self._count_events_by_severity(events),
                "by_detection_method": self._count_events_by_method(events)
            },
            "ml_insights": {
                "model_trained": ml_results.get("training_status") == "success",
                "predictions_made": ml_results.get("prediction_status") == "success",
                "predicted_events": int(ml_results.get("predictions", pd.DataFrame()).get("predicted_event", pd.Series()).sum()),
                "average_confidence": np.mean(ml_results.get("prediction_confidence", [])) if ml_results.get("prediction_confidence") else 0
            },
            "recommendations": self._generate_recommendations(events, ml_results)
        }
        
        return report
    
    def _calculate_temporal_coverage(self, data: pd.DataFrame) -> float:
        """Calcula cobertura temporal em horas"""
        if len(data) < 2:
            return 0
        time_span = data['timestamp'].max() - data['timestamp'].min()
        return time_span.total_seconds() / 3600
    
    def _count_events_by_type(self, events: list) -> dict:
        """Conta eventos por tipo"""
        from collections import Counter
        return dict(Counter(event['type'] for event in events))
    
    def _count_events_by_severity(self, events: list) -> dict:
        """Conta eventos por severidade"""
        from collections import Counter
        return dict(Counter(event['severity'] for event in events))
    
    def _count_events_by_method(self, events: list) -> dict:
        """Conta eventos por método de detecção"""
        from collections import Counter
        return dict(Counter(event.get('detection_method', 'unknown') for event in events))
    
    def _generate_recommendations(self, events: list, ml_results: dict) -> list:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Recomendações baseadas em eventos
        critical_events = [e for e in events if e.get('severity') in ['high', 'critical']]
        if critical_events:
            recommendations.append("⚠️ Eventos críticos detectados - Monitoramento intensivo recomendado")
        
        # Recomendações baseadas em ML
        if ml_results.get("training_status") == "success":
            accuracy = ml_results.get("training_metrics", {}).get("accuracy", 0)
            if accuracy < 0.7:
                recommendations.append("🤖 Modelo ML com acurácia baixa - Considere retreinamento com mais dados")
        
        if not recommendations:
            recommendations.append("✅ Situação normal - Continue monitoramento de rotina")
        
        return recommendations
    
    def _save_enhanced_results(self, data: pd.DataFrame, events: list, ml_results: dict, analysis: dict):
        """Salva resultados aprimorados"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Salva dados processados
        data_path = f"data/processed/helio_enhanced_{timestamp}.csv"
        data.to_csv(data_path, index=False)
        
        # Salva eventos
        if events:
            events_path = f"data/processed/events_enhanced_{timestamp}.json"
            with open(events_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
        
        # Salva resultados ML
        if not ml_results.get("predictions", pd.DataFrame()).empty:
            ml_path = f"data/processed/ml_predictions_{timestamp}.csv"
            ml_results["predictions"].to_csv(ml_path, index=False)
        
        # Salva relatório de análise
        report_path = f"data/processed/analysis_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"💾 Resultados salvos com prefixo: {timestamp}")

def main():
    """Função principal"""
    pipeline = EnhancedHeliogeophysicalPipeline()
    result = pipeline.run_enhanced_pipeline()
    
    # Relatório Executivo Expandido
    print(f"\n{'='*60}")
    print("🌌 RELATÓRIO EXECUTIVO - HELIOGEOPHYSICAL 3.0")
    print(f"{'='*60}")
    print(f"📅 Timestamp: {result['timestamp']}")
    print(f"🔄 Status: {result['status']}")
    print(f"📊 Registros processados: {result.get('processed_records', 0)}")
    print(f"🚨 Eventos detectados: {len(result['events'])}")
    print(f"🤖 Status ML: {result.get('ml_results', {}).get('training_status', 'N/A')}")
    
    # Análise de Eventos
    if result['events']:
        print(f"\n📈 ANÁLISE DE EVENTOS:")
        events_by_type = result.get('analysis', {}).get('events_summary', {}).get('by_type', {})
        for event_type, count in events_by_type.items():
            print(f"   • {event_type}: {count} eventos")
    
    # Recomendações
    recommendations = result.get('analysis', {}).get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMENDAÇÕES:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    print(f"{'='*60}")
    print("🎯 Sistema heliogeofísico avançado operacional!")

if __name__ == "__main__":
    main()
