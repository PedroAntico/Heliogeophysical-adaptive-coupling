"""
src/observational_validator.py
Validador STRICT para dados 100% observacionais
Verifica autenticidade dos dados
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('observational_validator')

class ObservationalValidator:
    """Validador rigoroso de dados observacionais"""
    
    def __init__(self):
        self.validation_criteria = {
            'min_data_points': 144,  # 24h de dados
            'max_data_gap_hours': 6,  # Máximo 6h sem dados
            'physical_limits': {
                'speed': (200, 1000),
                'density': (0.1, 100),
                'temperature': (10000, 500000),
                'bz_gse': (-50, 50),
                'bt': (0, 30)
            },
            'variability_thresholds': {
                'speed_std_min': 10,
                'density_std_min': 0.5
            }
        }
    
    def validate_observational_integrity(self, df):
        """Validação completa da integridade observacional"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_validation': 'FAILED',
            'detailed_checks': {},
            'issues': [],
            'warnings': []
        }
        
        # 1. Verificação básica
        basic_check = self._check_basic_requirements(df)
        results['detailed_checks']['basic_requirements'] = basic_check
        if not basic_check['passed']:
            results['issues'].extend(basic_check['issues'])
        
        # 2. Verificação física
        physics_check = self._check_physical_plausibility(df)
        results['detailed_checks']['physical_plausibility'] = physics_check
        if not physics_check['passed']:
            results['issues'].extend(physics_check['issues'])
        
        # 3. Verificação temporal
        temporal_check = self._check_temporal_integrity(df)
        results['detailed_checks']['temporal_integrity'] = temporal_check
        if not temporal_check['passed']:
            results['issues'].extend(temporal_check['issues'])
        
        # 4. Verificação de variabilidade
        variability_check = self._check_observational_variability(df)
        results['detailed_checks']['observational_variability'] = variability_check
        if not variability_check['passed']:
            results['warnings'].extend(variability_check['warnings'])
        
        # Determinar resultado final
        if len(results['issues']) == 0:
            if len(results['warnings']) == 0:
                results['overall_validation'] = 'EXCELLENT'
            else:
                results['overall_validation'] = 'GOOD'
        else:
            results['overall_validation'] = 'FAILED'
        
        return results
    
    def _check_basic_requirements(self, df):
        """Verifica requisitos básicos"""
        check = {'passed': True, 'issues': []}
        
        if df.empty:
            check['passed'] = False
            check['issues'].append('Dataset vazio')
            return check
        
        # Tamanho mínimo
        if len(df) < self.validation_criteria['min_data_points']:
            check['passed'] = False
            check['issues'].append(f"Dados insuficientes: {len(df)} < {self.validation_criteria['min_data_points']}")
        
        # Colunas essenciais
        essential_cols = ['time_tag', 'speed']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            check['passed'] = False
            check['issues'].append(f"Colunas essenciais faltando: {missing_cols}")
        
        return check
    
    def _check_physical_plausibility(self, df):
        """Verifica plausibilidade física dos dados"""
        check = {'passed': True, 'issues': []}
        
        limits = self.validation_criteria['physical_limits']
        
        for var, (min_val, max_val) in limits.items():
            if var in df.columns:
                # Verificar limites
                outliers = df[(df[var] < min_val) | (df[var] > max_val)]
                if len(outliers) > len(df) * 0.05:  # Mais de 5% fora dos limites
                    check['passed'] = False
                    check['issues'].append(f"Muitos valores fora dos limites físicos em {var}: {len(outliers)} registros")
                
                # Verificar se a média está dentro de faixa plausível
                mean_val = df[var].mean()
                if mean_val < min_val * 0.8 or mean_val > max_val * 0.8:
                    check['issues'].append(f"Média de {var} potencialmente implausível: {mean_val:.2f}")
        
        return check
    
    def _check_temporal_integrity(self, df):
        """Verifica integridade temporal"""
        check = {'passed': True, 'issues': []}
        
        if 'time_tag' not in df.columns:
            check['passed'] = False
            check['issues'].append('Coluna time_tag não encontrada')
            return check
        
        df_sorted = df.sort_values('time_tag').reset_index(drop=True)
        
        # Verificar gaps temporais
        time_diffs = df_sorted['time_tag'].diff().dropna()
        if len(time_diffs) > 0:
            max_gap = time_diffs.max()
            max_gap_hours = max_gap.total_seconds() / 3600
            
            if max_gap_hours > self.validation_criteria['max_data_gap_hours']:
                check['passed'] = False
                check['issues'].append(f"Gap temporal muito grande: {max_gap_hours:.1f} horas")
        
        # Verificar se os dados são recentes
        data_age = datetime.utcnow() - df_sorted['time_tag'].max()
        if data_age > timedelta(days=7):
            check['issues'].append(f"Dados podem estar desatualizados: {data_age.days} dias")
        
        return check
    
    def _check_observational_variability(self, df):
        """Verifica variabilidade típica de dados observacionais"""
        check = {'passed': True, 'warnings': []}
        
        thresholds = self.validation_criteria['variability_thresholds']
        
        # Verificar variabilidade da velocidade
        if 'speed' in df.columns:
            speed_std = df['speed'].std()
            if speed_std < thresholds['speed_std_min']:
                check['warnings'].append(f"Baixa variabilidade em speed: std={speed_std:.2f}")
        
        # Verificar variabilidade da densidade
        if 'density' in df.columns:
            density_std = df['density'].std()
            if density_std < thresholds['density_std_min']:
                check['warnings'].append(f"Baixa variabilidade em density: std={density_std:.2f}")
        
        # Verificar presença de eventos extremos (comum em dados reais)
        if 'speed' in df.columns:
            high_speed_events = len(df[df['speed'] > 600])
            if high_speed_events == 0:
                check['warnings'].append("Nenhum evento de alta velocidade detectado")
        
        return check
    
    def generate_validation_report(self, validation_results, output_path=None):
        """Gera relatório detalhado de validação"""
        report = {
            'validation_report': validation_results,
            'summary': self._generate_summary(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_summary(self, results):
        """Gera resumo da validação"""
        status = results['overall_validation']
        issues_count = len(results['issues'])
        warnings_count = len(results['warnings'])
        
        summary = {
            'validation_status': status,
            'total_issues': issues_count,
            'total_warnings': warnings_count,
            'data_points': results['detailed_checks']['basic_requirements'].get('data_points', 0),
            'is_observational': issues_count == 0
        }
        
        return summary
    
    def _generate_recommendations(self, results):
        """Gera recomendações baseadas na validação"""
        recommendations = []
        
        if results['overall_validation'] == 'FAILED':
            recommendations.append("Dados não validados como observacionais. Não use para análise científica.")
            recommendations.append("Verifique as fontes de dados e tente novamente.")
        
        elif results['overall_validation'] == 'GOOD':
            recommendations.append("Dados validados como observacionais com pequenos avisos.")
            recommendations.append("Adequado para análise científica.")
        
        elif results['overall_validation'] == 'EXCELLENT':
            recommendations.append("Dados excelentes - plenamente adequados para análise científica.")
        
        # Recomendações específicas baseadas em issues
        for issue in results['issues']:
            if "insuficientes" in issue.lower():
                recommendations.append("Coletar mais dados para melhor representatividade")
            if "gap temporal" in issue.lower():
                recommendations.append("Considerar preencher gaps temporais com interpolação cuidadosa")
        
        return recommendations

def main():
    """Valida dados observacionais"""
    import glob
    
    validator = ObservationalValidator()
    
    # Encontrar arquivo mais recente
    data_files = glob.glob('data_observational/solar_observational_*.csv')
    if not data_files:
        print("❌ Nenhum arquivo de dados observacional encontrado")
        return
    
    latest_file = sorted(data_files)[-1]
    print(f"🔍 Validando: {latest_file}")
    
    # Carregar dados
    df = pd.read_csv(latest_file, parse_dates=['time_tag'])
    
    # Executar validação
    results = validator.validate_observational_integrity(df)
    
    # Gerar relatório
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
    report_path = f'data_observational/validation_report_{timestamp}.json'
    report = validator.generate_validation_report(results, report_path)
    
    # Output
    print(f"\n📊 RELATÓRIO DE VALIDAÇÃO OBSERVACIONAL")
    print("="*50)
    print(f"Status: {results['overall_validation']}")
    print(f"Registros: {len(df)}")
    print(f"Issues: {len(results['issues'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Relatório: {report_path}")
    
    if results['issues']:
        print("\n❌ ISSUES CRÍTICAS:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    if results['warnings']:
        print("\n⚠️  AVISOS:")
        for warning in results['warnings']:
            print(f"  - {warning}")

if __name__ == '__main__':
    main()
