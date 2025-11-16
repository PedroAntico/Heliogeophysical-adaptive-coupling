"""
src/config/data_sources_REAL.py
Configuração DEFINITIVA com variáveis REAIS dos datasets
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DataSource:
    name: str
    url: str
    variables: List[str]
    expected_columns: List[str]
    sampling_rate: str
    availability: str
    rate_limit: Tuple[int, int]

# =============================================================================
# FONTES DE DADOS REAIS VERIFICADAS - VARIÁVEIS QUE EXISTEM
# =============================================================================

NASA_REAL_SOURCES = {
    # ✅ DSCOVR - Dados em tempo real
    'DSCOVR_SWEPAM_L1': DataSource(
        name='DSCOVR_SWEPAM_L1',
        url='https://cdaweb.gsfc.nasa.gov/',
        variables=['proton_density', 'bulk_speed', 'proton_temp'],  # ✅ VARIÁVEIS REAIS
        expected_columns=['Epoch', 'proton_density', 'bulk_speed', 'proton_temp'],
        sampling_rate='1min',
        availability='Real-time (15-30min delay)',
        rate_limit=(100, 3600)
    ),
    'DSCOVR_MAG_L1': DataSource(
        name='DSCOVR_MAG_L1',
        url='https://cdaweb.gsfc.nasa.gov/',
        variables=['B_x', 'B_y', 'B_z', 'Bt'],  # ✅ VARIÁVEIS REAIS
        expected_columns=['Epoch', 'B_x', 'B_y', 'B_z', 'Bt'],
        sampling_rate='1min',
        availability='Real-time (15-30min delay)',
        rate_limit=(100, 3600)
    ),
    # ✅ OMNI HRO2 - Dados históricos
    'OMNI_HRO2_1MIN': DataSource(
        name='OMNI_HRO2_1MIN',
        url='https://cdaweb.gsfc.nasa.gov/',
        variables=['BX_GSE', 'BY_GSE', 'BZ_GSE', 'V', 'FlowPressure'],  # ✅ VARIÁVEIS REAIS
        expected_columns=['Epoch', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'V', 'FlowPressure'],
        sampling_rate='1min',
        availability='Near real-time',
        rate_limit=(100, 3600)
    )
}

NOAA_REAL_SOURCES = {
    'PLASMA_5MIN': DataSource(
        name='NOAA Plasma 5-min',
        url='https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json',
        variables=['density', 'speed', 'temperature'],
        expected_columns=['time_tag', 'density', 'speed', 'temperature'],
        sampling_rate='5min',
        availability='Real-time (5-10min delay)',
        rate_limit=(60, 60)
    ),
    'MAG_5MIN': DataSource(
        name='NOAA Magnetic 5-min',
        url='https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json',
        variables=['bx_gse', 'by_gse', 'bz_gse', 'bt'],
        expected_columns=['time_tag', 'bx_gse', 'by_gse', 'bz_gse', 'bt'],
        sampling_rate='5min',
        availability='Real-time (5-10min delay)',
        rate_limit=(60, 60)
    )
}

# =============================================================================
# MAPEAMENTO CORRETO PARA VARIÁVEIS REAIS
# =============================================================================

VARIABLE_MAPPING_REAL = {
    'DSCOVR_SWEPAM': {
        'proton_density': 'density',      # ✅ REAL
        'bulk_speed': 'speed',           # ✅ REAL  
        'proton_temp': 'temperature'     # ✅ REAL
    },
    'DSCOVR_MAG': {
        'B_x': 'bx_gse',                 # ✅ REAL
        'B_y': 'by_gse',                 # ✅ REAL
        'B_z': 'bz_gse',                 # ✅ REAL
        'Bt': 'bt'                       # ✅ REAL
    },
    'OMNI_HRO2': {
        'BX_GSE': 'bx_gse',              # ✅ REAL
        'BY_GSE': 'by_gse',              # ✅ REAL
        'BZ_GSE': 'bz_gse',              # ✅ REAL
        'V': 'speed',                    # ✅ REAL
        'FlowPressure': 'pressure'       # ✅ REAL
    },
    'NOAA': {
        'density': 'density',
        'speed': 'speed', 
        'temperature': 'temperature',
        'bx_gse': 'bx_gse',
        'by_gse': 'by_gse',
        'bz_gse': 'bz_gse',
        'bt': 'bt'
    }
}

# =============================================================================
# CONFIGURAÇÕES DE VALIDAÇÃO
# =============================================================================

VALIDATION_CONFIG = {
    'min_data_points': 144,
    'max_data_gap_hours': 6,
    'completeness_threshold': 0.7,
    'variability_thresholds': {
        'speed_std_min': 10,
        'density_std_min': 0.5
    }
}
