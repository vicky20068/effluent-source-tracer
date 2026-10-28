"""
Configuration settings for Industrial Estate Effluent Monitoring & Source Tracing System.
Defines unit profiles, network topology, chemical safety limits, and hydraulic transport parameters.
"""

from typing import Dict, List, Any

# Industrial Estate Drainage Topology & Unit Specs
UNITS_METADATA: Dict[str, Dict[str, Any]] = {
    "U1": {
        "unit_id": "U1",
        "name": "Electroplating & Surface Treatment",
        "sector": "Metallurgy / Plating",
        "distance_m": 120.0,  # Distance to downstream station M0 along drainage channel (meters)
        "operating_shifts": [(8, 16)],  # Operating hours (24h clock)
        "avg_flow_rate_m3h": 25.0,
        "typical_discharge_profile": {
            "pH": 3.8,  # Highly acidic
            "COD_mgL": 140.0,
            "HeavyMetals_mgL": 8.5,  # High Chromium / Nickel
            "TDS_mgL": 1800.0,
        },
        "color": "#E74C3C"  # Red
    },
    "U2": {
        "unit_id": "U2",
        "name": "Textile Dyeing & Wet Processing",
        "sector": "Textiles",
        "distance_m": 280.0,
        "operating_shifts": [(6, 22)],  # Long double shift
        "avg_flow_rate_m3h": 45.0,
        "typical_discharge_profile": {
            "pH": 10.4,  # Highly alkaline
            "COD_mgL": 880.0,  # Very high COD
            "HeavyMetals_mgL": 0.8,
            "TDS_mgL": 3200.0,  # Very high TDS
        },
        "color": "#9B59B6"  # Purple
    },
    "U3": {
        "unit_id": "U3",
        "name": "Specialty Chemical Synthesis",
        "sector": "Chemicals",
        "distance_m": 450.0,
        "operating_shifts": [(0, 8), (16, 24)],  # Night & Evening shifts
        "avg_flow_rate_m3h": 30.0,
        "typical_discharge_profile": {
            "pH": 6.2,
            "COD_mgL": 1250.0,  # Extreme COD
            "HeavyMetals_mgL": 3.2,
            "TDS_mgL": 2400.0,
        },
        "color": "#3498DB"  # Blue
    },
    "U4": {
        "unit_id": "U4",
        "name": "Agro-Food Processing & Canning",
        "sector": "Food Processing",
        "distance_m": 620.0,
        "operating_shifts": [(8, 20)],
        "avg_flow_rate_m3h": 35.0,
        "typical_discharge_profile": {
            "pH": 5.8,
            "COD_mgL": 480.0,  # Organic loading
            "HeavyMetals_mgL": 0.2,
            "TDS_mgL": 1100.0,
        },
        "color": "#2ECC71"  # Green
    },
    "U5": {
        "unit_id": "U5",
        "name": "Pharmaceutical Formulation",
        "sector": "Pharma",
        "distance_m": 750.0,
        "operating_shifts": [(10, 14), (22, 2)],  # Batch runs
        "avg_flow_rate_m3h": 18.0,
        "typical_discharge_profile": {
            "pH": 4.5,
            "COD_mgL": 950.0,
            "HeavyMetals_mgL": 1.1,
            "TDS_mgL": 1950.0,
        },
        "color": "#F1C40F"  # Yellow
    },
    "U6": {
        "unit_id": "U6",
        "name": "Heavy Machine Fabrication",
        "sector": "Engineering",
        "distance_m": 890.0,
        "operating_shifts": [(8, 17)],
        "avg_flow_rate_m3h": 15.0,
        "typical_discharge_profile": {
            "pH": 5.2,
            "COD_mgL": 180.0,
            "HeavyMetals_mgL": 6.8,  # Iron, Zinc
            "TDS_mgL": 1400.0,
        },
        "color": "#E67E22"  # Orange
    }
}

# Hydraulic Transport Constants
HYDRAULICS = {
    "monitoring_station_distance_m": 1000.0,
    "default_stream_velocity_ms": 0.5,  # 0.5 m/s = 30 m/min
    "dispersion_coefficient_m2s": 0.15,  # Advection-dispersion coefficient D
    "background_flow_rate_m3h": 40.0,  # Ambient stream baseflow
}

# Environmental Regulatory Compliance Safety Thresholds
REGULATORY_LIMITS = {
    "pH_min": 6.5,
    "pH_max": 8.5,
    "COD_mgL_max": 250.0,
    "HeavyMetals_mgL_max": 2.0,
    "TDS_mgL_max": 2100.0,
    "FlowRate_m3h_max": 150.0,
}

# Baseline & EWMA Anomaly Detector Parameters
DETECTOR_CONFIG = {
    "ewma_alpha": 0.15,
    "z_score_threshold": 3.0,
    "static_cod_threshold": 250.0,
    "static_heavy_metals_threshold": 2.0,
}

# Fallback Spec for Unknown / Unmapped Upstream Discharge Source
UNKNOWN_UNIT_SPEC = {
    "unit_id": "U_UNK",
    "name": "Unregistered / Illegal Upstream Source",
    "sector": "Unmapped Source",
    "distance_m": 500.0,
    "operating_shifts": [(0, 24)],
    "avg_flow_rate_m3h": 20.0,
    "typical_discharge_profile": {
        "pH": 7.0,
        "COD_mgL": 500.0,
        "HeavyMetals_mgL": 5.0,
        "TDS_mgL": 1500.0,
    },
    "color": "#95A5A6"  # Gray
}

# Uncertainty & Calibration Configuration
CALIBRATION_CONFIG = {
    "high_confidence_margin": 0.35,  # Top-1 vs Top-2 likelihood margin for HIGH confidence
    "medium_confidence_margin": 0.15,
    "entropy_max_bits": 2.80, # log2(7 candidates)
    "unknown_unit_prior_weight": 0.08, # Base likelihood prior weight for unmapped unit
    "calibration_grace_days": 7, # Days after nominal interval before sensor is flagged critical
    "min_sensor_weight": 0.15, # Floor weight for degraded sensor in attribution calculation
}

# Physical Sensor Calibration Specifications (ISO 17025 / NABL Aligned)
SENSOR_CALIBRATION_SPECS: Dict[str, Dict[str, Any]] = {
    "pH": {
        "sensor_id": "SENS-PH-01",
        "sensor_name": "Combination Glass pH Electrode",
        "model": "Thermo Scientific Orion 8102BNUWP",
        "nominal_interval_days": 14,
        "calibration_standard": "NIST 4.01, 7.00, 10.01 Buffer Solution",
        "drift_rate_per_day": 0.008, # Degradation rate per day
        "default_weight": 1.0,
        "primary_target_units": ["U1", "U2", "U4"]
    },
    "COD": {
        "sensor_id": "SENS-COD-01",
        "sensor_name": "UV-Vis Spectrophotometric Immersion Probe",
        "model": "s::can spectro::lyser v3 (200-750nm)",
        "nominal_interval_days": 30,
        "calibration_standard": "Potassium Hydrogen Phthalate (KHP) Standard 500 mg/L",
        "drift_rate_per_day": 0.005,
        "default_weight": 1.0,
        "primary_target_units": ["U2", "U3", "U4", "U5"]
    },
    "HeavyMetals": {
        "sensor_id": "SENS-HM-01",
        "sensor_name": "Anodic Stripping Voltammetry (ASV) Probe",
        "model": "Metrohm 946 Portable VA Analyzer Trace Probe",
        "nominal_interval_days": 7,
        "calibration_standard": "Trace Element Matrix Cr/Ni/Zn/Pb 5.0 mg/L",
        "drift_rate_per_day": 0.014, # Electrodes foul faster under heavy metal flux
        "default_weight": 1.0,
        "primary_target_units": ["U1", "U6"]
    },
    "TDS": {
        "sensor_id": "SENS-TDS-01",
        "sensor_name": "Toroidal Induction Conductivity Sensor",
        "model": "Endress+Hauser Indumax CLS50D",
        "nominal_interval_days": 30,
        "calibration_standard": "Standard KCl 1413 µS/cm Reference",
        "drift_rate_per_day": 0.004,
        "default_weight": 0.8,
        "primary_target_units": ["U1", "U2", "U3", "U5"]
    },
    "FlowRate": {
        "sensor_id": "SENS-FLOW-01",
        "sensor_name": "Open-Channel Ultrasonic Transit Flowmeter",
        "model": "Siemens Sitrans FUS380 Clamped Ultrasonic",
        "nominal_interval_days": 90,
        "calibration_standard": "ISO 1438 Parshall Flume Hydraulic Head Gauge",
        "drift_rate_per_day": 0.002,
        "default_weight": 0.9,
        "primary_target_units": ["ALL"]
    }
}

# Digital Chain-of-Custody Compliance Settings
CHAIN_OF_CUSTODY_CONFIG = {
    "default_hash_algorithm": "sha256",
    "require_gps_validation": True,
    "legal_sample_retention_days": 180,
    "estate_location_lat_long": (11.0168, 76.9558), # Coimbatore Industrial Area Lat/Long
}


