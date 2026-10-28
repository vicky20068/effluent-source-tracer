"""
Synthetic Dataset Generator for Industrial Estate Effluent Monitoring.
Simulates 30 days of 5-minute sampling interval data for downstream station M0
and upstream units, modeling advection-dispersion transport physics and ground-truth anomaly events.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from src.config import UNITS_METADATA, HYDRAULICS, REGULATORY_LIMITS

def is_unit_operating(unit_id: str, timestamp: datetime) -> bool:
    """Check if a given unit is scheduled to operate at a specific timestamp."""
    unit_info = UNITS_METADATA[unit_id]
    hour = timestamp.hour + timestamp.minute / 60.0
    
    for start_h, end_h in unit_info["operating_shifts"]:
        if start_h <= end_h:
            if start_h <= hour < end_h:
                return True
        else:  # Shift crosses midnight (e.g. 22:00 to 02:00)
            if hour >= start_h or hour < end_h:
                return True
    return False

def calculate_advection_dispersion(
    mass_released: float,
    distance_m: float,
    travel_time_sec: float,
    velocity_ms: float,
    dispersion_coeff: float
) -> float:
    """
    Calculate 1D advection-dispersion pollutant concentration C(x,t)
    C(x,t) = (M / sqrt(4 * pi * D * t)) * exp(- (x - v*t)^2 / (4 * D * t))
    """
    if travel_time_sec <= 1.0:
        return 0.0
    
    denom = np.sqrt(4.0 * np.pi * dispersion_coeff * travel_time_sec)
    exponent = - ((distance_m - velocity_ms * travel_time_sec) ** 2) / (4.0 * dispersion_coeff * travel_time_sec)
    
    # Clip exponent to avoid numerical underflow/overflow
    exponent = np.clip(exponent, -50.0, 50.0)
    concentration = (mass_released / denom) * np.exp(exponent)
    return max(0.0, float(concentration))

def generate_synthetic_dataset(
    days: int = 14,
    freq_minutes: int = 5,
    num_anomalies: int = 25,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic sensor data for downstream station M0 and ground-truth anomaly events.
    Returns:
        downstream_df: Timestamps, FlowRate, pH, COD, HeavyMetals, TDS
        ground_truth_df: AnomalyID, Timestamp, TrueSourceUnit, TriggerParameter, PeakMagnitude
    """
    np.random.seed(random_seed)
    
    start_time = datetime(2026, 9, 1, 0, 0, 0)
    total_steps = int((days * 24 * 60) / freq_minutes)
    timestamps = [start_time + timedelta(minutes=i * freq_minutes) for i in range(total_steps)]
    
    # Baseline ambient river stream concentrations
    base_flow = HYDRAULICS["background_flow_rate_m3h"]
    base_pH = 7.2
    base_COD = 45.0
    base_HeavyMetals = 0.3
    base_TDS = 600.0
    
    # Initialize arrays
    flow_arr = np.full(total_steps, base_flow) + np.random.normal(0, 1.5, total_steps)
    ph_arr = np.full(total_steps, base_pH) + np.random.normal(0, 0.05, total_steps)
    cod_arr = np.full(total_steps, base_COD) + np.random.normal(0, 3.0, total_steps)
    hm_arr = np.full(total_steps, base_HeavyMetals) + np.random.normal(0, 0.05, total_steps)
    tds_arr = np.full(total_steps, base_TDS) + np.random.normal(0, 15.0, total_steps)
    
    # Add routine operational background contributions from active units
    v = HYDRAULICS["default_stream_velocity_ms"]
    D = HYDRAULICS["dispersion_coefficient_m2s"]
    
    for i, ts in enumerate(timestamps):
        for uid, uinfo in UNITS_METADATA.items():
            if is_unit_operating(uid, ts):
                # Normal operational discharge
                travel_dist = HYDRAULICS["monitoring_station_distance_m"] - uinfo["distance_m"]
                travel_time = travel_dist / v
                
                # Small operational ripple
                flow_arr[i] += uinfo["avg_flow_rate_m3h"] * 0.1
                prof = uinfo["typical_discharge_profile"]
                cod_arr[i] += (prof["COD_mgL"] * 0.04)
                hm_arr[i] += (prof["HeavyMetals_mgL"] * 0.04)
                tds_arr[i] += (prof["TDS_mgL"] * 0.04)
                
                # pH shift towards unit profile
                ph_arr[i] = ph_arr[i] * 0.95 + prof["pH"] * 0.05
    
    # Inject Ground Truth Anomalies
    anomaly_records = []
    
    # Choose anomaly injection points spread across dataset
    anomaly_indices = np.linspace(50, total_steps - 50, num_anomalies, dtype=int)
    
    for idx_num, idx in enumerate(anomaly_indices):
        ts = timestamps[idx]
        
        # Pick a unit operating at or near this time
        candidate_units = [u for u in UNITS_METADATA.keys() if is_unit_operating(u, ts)]
        if not candidate_units:
            candidate_units = list(UNITS_METADATA.keys())
        
        target_unit = np.random.choice(candidate_units)
        uinfo = UNITS_METADATA[target_unit]
        prof = uinfo["typical_discharge_profile"]
        
        # Determine arrival time at M0
        dist_to_m0 = HYDRAULICS["monitoring_station_distance_m"] - uinfo["distance_m"]
        delay_sec = dist_to_m0 / v
        delay_steps = int(round(delay_sec / (freq_minutes * 60)))
        
        arrival_idx = min(idx + delay_steps, total_steps - 1)
        arrival_ts = timestamps[arrival_idx]
        
        # Spike duration (e.g. 3 to 7 steps)
        duration_steps = np.random.randint(3, 8)
        
        # Primary parameter spike
        if prof["HeavyMetals_mgL"] > 5.0:
            param = "HeavyMetals"
            spike_mag = np.random.uniform(6.0, 14.0)
        elif prof["COD_mgL"] > 800.0:
            param = "COD"
            spike_mag = np.random.uniform(500.0, 1200.0)
        elif prof["pH"] < 4.5 or prof["pH"] > 10.0:
            param = "pH"
            spike_mag = prof["pH"]
        else:
            param = "COD"
            spike_mag = np.random.uniform(400.0, 900.0)
            
        # Apply Gaussian pulse shape to downstream readings around arrival_idx
        for step_offset in range(-duration_steps, duration_steps + 1):
            curr_idx = arrival_idx + step_offset
            if 0 <= curr_idx < total_steps:
                pulse_factor = np.exp(-0.5 * (step_offset / (duration_steps / 2.0)) ** 2)
                
                if param == "COD":
                    cod_arr[curr_idx] += spike_mag * pulse_factor
                    hm_arr[curr_idx] += prof["HeavyMetals_mgL"] * pulse_factor * 0.8
                    tds_arr[curr_idx] += prof["TDS_mgL"] * pulse_factor * 0.5
                elif param == "HeavyMetals":
                    hm_arr[curr_idx] += spike_mag * pulse_factor
                    cod_arr[curr_idx] += prof["COD_mgL"] * pulse_factor * 0.6
                elif param == "pH":
                    target_ph = prof["pH"]
                    ph_arr[curr_idx] = ph_arr[curr_idx] * (1 - 0.7 * pulse_factor) + target_ph * (0.7 * pulse_factor)
                
                flow_arr[curr_idx] += uinfo["avg_flow_rate_m3h"] * 0.8 * pulse_factor

        anomaly_records.append({
            "anomaly_id": f"ANO-{idx_num+1:03d}",
            "discharge_timestamp": ts,
            "arrival_timestamp": arrival_ts,
            "arrival_step": arrival_idx,
            "true_unit_id": target_unit,
            "unit_name": uinfo["name"],
            "primary_parameter": param,
            "spike_magnitude": round(float(spike_mag), 2),
            "travel_delay_mins": round(delay_sec / 60.0, 1)
        })

    # Construct DataFrame
    downstream_df = pd.DataFrame({
        "timestamp": timestamps,
        "FlowRate_m3h": np.round(flow_arr, 2),
        "pH": np.round(ph_arr, 2),
        "COD_mgL": np.round(cod_arr, 2),
        "HeavyMetals_mgL": np.round(hm_arr, 2),
        "TDS_mgL": np.round(tds_arr, 2)
    })
    
    ground_truth_df = pd.DataFrame(anomaly_records)
    
    return downstream_df, ground_truth_df
