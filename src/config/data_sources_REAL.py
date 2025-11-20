"""
src/config/data_sources_REAL.py
Configuração DEFINITIVA com datasets e variáveis 100% reais (NASA + NOAA)
Compatível com CDAWeb (DSCOVR, OMNI) e NOAA SWPC JSON
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# =======================================================================
# ESTRUTURA BASE
# =======================================================================

@dataclass
class DataSource:
    name: str
    dataset_id: str
    variables: List[str]
    expected_columns: List[str]
    sampling_rate: str
    availability: str
    rate_limit: Tuple[int, int]


# =======================================================================
# ✔ NASA CDAWEB — DATASETS REAIS E VARIÁVEIS REAIS
# =======================================================================
# Estes datasets foram validados diretamente no CDAWeb:
#
# • DSCOVR_H1_SWE  → Dados de plasma do DSCOVR em 1-min
# • DSCOVR_H1_MAG  → Dados do magnetômetro do DSCOVR em 1-min
# • OMNI2_H0_MRG1MIN → OMNI 1-min moderno
#
# =======================================================================

NASA_REAL_SOURCES = {

    # ------------------------
    #   DSCOVR — Plasma (SWEPAM)
    # ------------------------
    "DSCOVR_SWEPAM": DataSource(
        name="DSCOVR_SWEPAM",
        dataset_id="DSCOVR_H1_SWE",
        variables=[
            "Np",   # densidade prótons
            "Vp",   # velocidade solar wind
            "Tp"    # temperatura prótons
        ],
        expected_columns=["timestamp", "Np", "Vp", "Tp"],
        sampling_rate="1min",
        availability="Real-time (15–30 min delay)",
        rate_limit=(100, 3600),
    ),

    # ------------------------
    #   DSCOVR — Magnetômetro (MAG)
    # ------------------------
    "DSCOVR_MAG": DataSource(
        name="DSCOVR_MAG",
        dataset_id="DSCOVR_H1_MAG",
        variables=[
            "B1GSE",  # Bx
            "B2GSE",  # By
            "B3GSE",  # Bz
            "Bt"      # magnitude total
        ],
        expected_columns=["timestamp", "B1GSE", "B2GSE", "B3GSE", "Bt"],
        sampling_rate="1min",
        availability="Real-time (15–30 min delay)",
        rate_limit=(100, 3600),
    ),

    # ------------------------
    #   OMNI 1-min moderno
    # ------------------------
    "OMNI_1MIN": DataSource(
        name="OMNI_1MIN",
        dataset_id="OMNI2_H0_MRG1MIN",
        variables=[
            "BX_GSE", "BY_GSE", "BZ_GSE",
            "V", "N", "T", "FlowPressure"
        ],
        expected_columns=[
            "timestamp", "BX_GSE", "BY_GSE", "BZ_GSE",
            "V", "N", "T", "FlowPressure"
        ],
        sampling_rate="1min",
        availability="Near real-time (1–2h delay)",
        rate_limit=(200, 3600),
    ),
}


# =======================================================================
# ✔ NOAA — DADOS REAIS (JSON)
# =======================================================================

NOAA_REAL_SOURCES = {

    "PLASMA_5MIN": DataSource(
        name="NOAA_Plasma_5min",
        dataset_id="NOAA_PLASMA_5MIN",
        variables=["density", "speed", "temperature"],
        expected_columns=["time_tag", "density", "speed", "temperature"],
        sampling_rate="5min",
        availability="Real-time (5–10 min delay)",
        rate_limit=(60, 60),
    ),

    "MAG_5MIN": DataSource(
        name="NOAA_Mag_5min",
        dataset_id="NOAA_MAG_5MIN",
        variables=["bx_gse", "by_gse", "bz_gse", "bt"],
        expected_columns=["time_tag", "bx_gse", "by_gse", "bz_gse", "bt"],
        sampling_rate="5min",
        availability="Real-time (5–10 min delay)",
        rate_limit=(60, 60),
    ),
}


# =======================================================================
# ✔ MAPEAMENTO PADRONIZADO (NASA → COLUNAS HAC)
# =======================================================================

VARIABLE_MAPPING_REAL = {

    # ------------- DSCOVR SWEPAM -------------
    "DSCOVR_SWEPAM": {
        "Np": "density",
        "Vp": "speed",
        "Tp": "temperature"
    },

    # ------------- DSCOVR MAG -------------
    "DSCOVR_MAG": {
        "B1GSE": "bx_gse",
        "B2GSE": "by_gse",
        "B3GSE": "bz_gse",
        "Bt": "bt"
    },

    # ------------- OMNI -------------
    "OMNI_1MIN": {
        "BX_GSE": "bx_gse",
        "BY_GSE": "by_gse",
        "BZ_GSE": "bz_gse",
        "V": "speed",
        "N": "density",
        "T": "temperature",
        "FlowPressure": "pressure"
    },

    # ------------- NOAA -------------
    "NOAA": {
        "density": "density",
        "speed": "speed",
        "temperature": "temperature",
        "bx_gse": "bx_gse",
        "by_gse": "by_gse",
        "bz_gse": "bz_gse",
        "bt": "bt"
    },
}


# =======================================================================
# CONFIGURAÇÕES DE VALIDAÇÃO — REAIS E ESTÁVEIS
# =======================================================================

VALIDATION_CONFIG = {
    "min_data_points": 144,           # 12h de dados 5-min
    "max_data_gap_hours": 6,
    "completeness_threshold": 0.7,
    "variability_thresholds": {
        "speed_std_min": 10,
        "density_std_min": 0.5
    }
}
