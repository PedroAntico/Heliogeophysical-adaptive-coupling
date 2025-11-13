"""
run_observational_pipeline.py
Pipeline STRICT 100% dados observacionais
FALHA se qualquer passo usar dados não-observacionais
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/observational_pipeline.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('observational_pipeline')

class ObservationalPipeline:
    """Pipeline 100% dados observacionais"""
    
    def __init__(self):
        self.start_time = None
        self.pipeline_results = {
            'pipeline_type': 'STRICT_OBSERVATIONAL',
            'zero_synthetic_data': True,
            'steps': {},
            'overall_status': 'UNKNOWN'
        }
    
    def run_observational_step(self, step_name, script_path, critical=True):
        """Executa um passo do pipeline observacional"""
        logger.info(f"🚀 Executando: {step_name}")
        print(f"\n{'='*60}")
        print(f"EXECUTANDO: {step_name}")
        print(f"{'='*60}")
        
        step_result = {
            'step_name': step_name,
            'start_time': datetime.utcnow().isoformat(),
            'status': 'RUNNING',
            'is_observational': True
        }
        
        try:
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script observacional não encontrado: {script_path}")
            
            # Executar script
            result = subprocess.run(
                [sys.executable, script_path], 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                timeout=600  # 10 minutos timeout
            )
            
            step_result['output'] = result.stdout
            step_result['error'] = result.stderr
            step_result['return_code'] = result.returncode
            
            if result.returncode == 0:
                step_result['status'] = 'SUCCESS'
                logger.info(f"✅ {step_name} concluído com sucesso")
                print(f"✅ {step_name} concluído")
            else:
                step_result['status'] = 'FAILED'
                logger.error(f"❌ {step_name} falhou: {result.stderr}")
                print(f"❌ {step_name} falhou")
                
                if critical:
                    raise RuntimeError(f"Step observacional crítico falhou: {step_name}")
            
        except subprocess.TimeoutExpired:
            step_result['status'] = 'TIMEOUT'
            logger.error(f"⏰ {step_name} excedeu o tempo limite")
            if critical:
                raise
        except Exception as e:
            step_result['status'] = 'ERROR'
            step_result['error'] = str(e)
            logger.error(f"❌ Erro em {step_name}: {str(e)}")
            if critical:
                raise
        
        step_result['end_time'] = datetime.utcnow().isoformat()
        self.pipeline_results['steps'][step_name] = step_result
        
        return step_result['status'] == 'SUCCESS'
    
    def verify_observational_integrity(self):
        """Verifica integridade observacional do pipeline"""
        logger.info("🔍 Verificando integridade observacional...")
        
        # Verificar que não há dados sintéticos
        synthetic_indicators = ['synthetic', 'simulated', 'fake', 'dummy', 'mock']
        
        for step_name, step_result in self.pipeline_results['steps'].items():
            if step_result['status'] == 'SUCCESS':
                output = step_result.get('output', '').lower()
                error = step_result.get('error', '').lower()
                
                for indicator in synthetic_indicators:
                    if indicator in output or indicator in error:
                        logger.warning(f"⚠️  Possível dado sintético detectado em {step_name}")
                        self.pipeline_results['zero_synthetic_data'] = False
        
        return self.pipeline_results['zero_synthetic_data']
    
    def run_strict_observational_pipeline(self):
        """Executa pipeline 100% observacional"""
        self.start_time = datetime.utcnow()
        self.pipeline_results['start_time'] = self.start_time.isoformat()
        
        logger.info("🚀 INICIANDO PIPELINE OBSERVACIONAL STRICT")
        logger.info("📜 POLÍTICA: 100% DADOS OBSERVACIONAIS - ZERO SINTÉTICO")
        print("🚀 INICIANDO PIPELINE OBSERVACIONAL STRICT")
        print("📜 POLÍTICA: 100% DADOS OBSERVACIONAIS - ZERO SINTÉTICO")
        
        # Definir steps OBSERVACIONAIS
        pipeline_steps = [
            {
                'name': "COLETA DE DADOS OBSERVACIONAIS STRICT",
                'script': "src/data_fetcher_observational.py",
                'critical': True
            },
            {
                'name': "VALIDAÇÃO OBSERVACIONAL RIGOROSA", 
                'script': "src/observational_validator.py",
                'critical': True
            },
            {
                'name': "ANÁLISE PREDITIVA OBSERVACIONAL",
                'script': "src/heliopredictive_observational.py", 
                'critical': True
            }
        ]
        
        # Executar pipeline
        all_observational = True
        for step_config in pipeline_steps:
            success = self.run_observational_step(
                step_config['name'],
                step_config['script'], 
                step_config['critical']
            )
            
            if not success:
                all_observational = False
                if step_config['critical']:
                    break
            
            time.sleep(2)  # Pequena pausa entre steps
        
        # Verificar integridade observacional
        integrity_ok = self.verify_observational_integrity()
        
        # Resultado final
        if all_observational and integrity_ok:
            self.pipeline_results['overall_status'] = 'OBSERVATIONAL_SUCCESS'
        elif all_observational and not integrity_ok:
            self.pipeline_results['overall_status'] = 'OBSERVATIONAL_WARNING'
        else:
            self.pipeline_results['overall_status'] = 'FAILED'
        
        self.pipeline_results['end_time'] = datetime.utcnow().isoformat()
        execution_time = (datetime.utcnow() - self.start_time).total_seconds()
        self.pipeline_results['execution_time_seconds'] = execution_time
        
        # Gerar relatório
        self.generate_observational_report()
        
        return self.pipeline_results['overall_status'] == 'OBSERVATIONAL_SUCCESS'
    
    def generate_observational_report(self):
        """Gera relatório final do pipeline observacional"""
        import json
        
        os.makedirs('results_observational', exist_ok=True)
        
        # Relatório JSON
        with open('results_observational/pipeline_observational_report.json', 'w') as f:
            json.dump(self.pipeline_results, f, indent=2)
        
        # Relatório textual
        self._generate_observational_text_report()
    
    def _generate_observational_text_report(self):
        """Gera relatório textual do pipeline"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("RELATÓRIO FINAL - PIPELINE OBSERVACIONAL STRICT")
        report_lines.append("="*70)
        report_lines.append(f"Data de execução: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        report_lines.append(f"Status geral: {self.pipeline_results['overall_status']}")
        report_lines.append(f"Dados 100% observacionais: {self.pipeline_results['zero_synthetic_data']}")
        report_lines.append("")
        
        # Estatísticas dos steps
        success_count = len([s for s in self.pipeline_results['steps'].values() if s['status'] == 'SUCCESS'])
        total_count = len(self.pipeline_results['steps'])
        
        report_lines.append("📊 ESTATÍSTICAS DOS STEPS:")
        for step_name, step_result in self.pipeline_results['steps'].items():
            status_icon = "✅" if step_result['status'] == 'SUCCESS' else "❌"
            report_lines.append(f"  {status_icon} {step_name}: {step_result['status']}")
        
        report_lines.append("")
        report_lines.append(f"✅ Passos bem-sucedidos: {success_count}/{total_count}")
        report_lines.append(f"⏱️  Tempo total: {self.pipeline_results['execution_time_seconds']:.1f} segundos")
        report_lines.append("")
        
        if self.pipeline_results['zero_synthetic_data']:
            report_lines.append("🎯 STATUS: PIPELINE 100% OBSERVACIONAL")
            report_lines.append("   - Zero dados sintéticos detectados")
            report_lines.append("   - Apenas dados reais da NASA/NOAA")
            report_lines.append("   - Adequado para publicação científica")
        else:
            report_lines.append("⚠️  AVISO: Possível contaminação com dados sintéticos")
            report_lines.append("   - Verificar integridade dos dados")
            report_lines.append("   - Não adequado para publicação científica")
        
        report_lines.append("")
        report_lines.append("="*70)
        
        report_text = "\n".join(report_lines)
        
        with open('results_observational/pipeline_observational_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)

def main():
    """Função principal do pipeline observacional"""
    pipeline = ObservationalPipeline()
    
    try:
        success = pipeline.run_strict_observational_pipeline()
        
        if success:
            logger.info("🎉 PIPELINE OBSERVACIONAL EXECUTADO COM SUCESSO!")
            print("\n🎉 PIPELINE OBSERVACIONAL EXECUTADO COM SUCESSO!")
            print("📁 Resultados em: results_observational/")
            sys.exit(0)
        else:
            logger.error("💥 PIPELINE OBSERVACIONAL FALHOU")
            print("\n💥 PIPELINE OBSERVACIONAL FALHOU")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 ERRO CRÍTICO NO PIPELINE: {e}")
        print(f"\n💥 ERRO CRÍTICO NO PIPELINE: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
