"""
Industrial Estate Synthetic Data Generator

Generates realistic time-series effluent telemetry data for a multi-unit industrial estate,
including unit operating schedules, hydraulic travel delays, chemical signatures, and
ground-truth anomaly injections for testing tracing algorithms.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Default Industrial Estate Configuration
DEFAULT_UNITS_CONFIG = {
    "Unit_A": {
        "name": "Unit A (Chemical Processing)",
        "distance_km": 1.5,
        "hydraulic_velocity_mps": 1.0,  # ~25 min lag (5 intervals of 5 min)
        "operating_shifts": [("08:00", "16:00")],
        "base_flow_lps": 12.0,
        "base_ph": 6.8,
        "base_cod": 150.0,
        "base_tds": 400.0,
        "base_heavy_metals": 0.05,
        "spike_signature": {"ph": 3.8, "cod": 850.0, "tds": 600.0, "heavy_metals": 0.1}
    },
    "Unit_B": {
        "name": "Unit B (Electroplating Plant)",
        "distance_km": 3.0,
        "hydraulic_velocity_mps": 1.0,  # ~50 min lag (10 intervals of 5 min)
        "operating_shifts": [("12:00", "20:00")],
        "base_flow_lps": 8.0,
        "base_ph": 7.0,
        "base_cod": 80.0,
        "base_tds": 350.0,
        "base_heavy_metals": 0.02,
        "spike_signature": {"ph": 4.2, "cod": 300.0, "tds": 500.0, "heavy_metals": 4.8}
    },
    "Unit_C": {
        "name": "Unit C (Textile Dyeing Mill)",
        "distance_km": 0.9,
        "hydraulic_velocity_mps": 1.0,  # ~15 min lag (3 intervals of 5 min)
        "operating_shifts": [("00:00", "08:00"), ("16:00", "23:59")],
        "base_flow_lps": 20.0,
        "base_ph": 7.5,
        "base_cod": 200.0,
        "base_tds": 800.0,
        "base_heavy_metals": 0.01,
        "spike_signature": {"ph": 10.5, "cod": 650.0, "tds": 2400.0, "heavy_metals": 0.05}
    },
    "Unit_D": {
        "name": "Unit D (Food Processing)",
        "distance_km": 2.1,
        "hydraulic_velocity_mps": 1.0,  # ~35 min lag (7 intervals of 5 min)
        "operating_shifts": [("06:00", "18:00")],
        "base_flow_lps": 15.0,
        "base_ph": 7.1,
        "base_cod": 250.0,
        "base_tds": 450.0,
        "base_heavy_metals": 0.01,
        "spike_signature": {"ph": 6.5, "cod": 1200.0, "tds": 550.0, "heavy_metals": 0.02}
    }
}


def is_unit_operating(timestamp: pd.Timestamp, shifts: list) -> bool:
    """Checks if a given timestamp falls within any of the unit's operating shifts."""
    time_str = timestamp.strftime("%H:%M")
    for start_str, end_str in shifts:
        if start_str <= time_str <= end_str:
            return True
    return False


def get_hydraulic_lag_steps(distance_km: float, velocity_mps: float, interval_minutes: int = 5) -> int:
    """Calculates hydraulic propagation delay in discrete time intervals."""
    travel_time_seconds = (distance_km * 1000.0) / velocity_mps
    travel_time_minutes = travel_time_seconds / 60.0
    return max(1, int(round(travel_time_minutes / interval_minutes)))


def generate_synthetic_telemetry(
    days: int = 3,
    interval_minutes: int = 5,
    units_config: dict = DEFAULT_UNITS_CONFIG,
    random_seed: int = 42
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Generates time-series sensor telemetry at the downstream monitoring station and
    returns (telemetry_df, ground_truth_anomalies).
    """
    np.random.seed(random_seed)
    start_time = pd.Timestamp("2026-10-01 00:00:00")
    total_steps = int((days * 24 * 60) / interval_minutes)
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_steps)]

    df = pd.DataFrame({"timestamp": timestamps})

    # Unit Operating Status logs (as recorded in schedules / smart meters)
    for unit_id, cfg in units_config.items():
        df[f"{unit_id}_status"] = [
            1 if is_unit_operating(ts, cfg["operating_shifts"]) else 0
            for ts in timestamps
        ]

    # Initialize unit contribution matrices at their discharge outlet
    unit_flows = {u: np.zeros(total_steps) for u in units_config}
    unit_cod = {u: np.zeros(total_steps) for u in units_config}
    unit_ph = {u: np.zeros(total_steps) for u in units_config}
    unit_tds = {u: np.zeros(total_steps) for u in units_config}
    unit_hm = {u: np.zeros(total_steps) for u in units_config}

    # Fill normal baseline operations with slight random Gaussian noise
    for u, cfg in units_config.items():
        operating = df[f"{u}_status"].values
        base_f = cfg["base_flow_lps"]
        # Flow exists during operation; minimal residual flow when idle
        unit_flows[u] = np.where(operating == 1, base_f + np.random.normal(0, 0.5, total_steps), 1.0 + np.random.normal(0, 0.1, total_steps))
        unit_flows[u] = np.clip(unit_flows[u], 0.1, None)

        unit_ph[u] = np.where(operating == 1, cfg["base_ph"] + np.random.normal(0, 0.1, total_steps), 7.0)
        unit_cod[u] = np.where(operating == 1, cfg["base_cod"] + np.random.normal(0, 10.0, total_steps), 20.0)
        unit_tds[u] = np.where(operating == 1, cfg["base_tds"] + np.random.normal(0, 20.0, total_steps), 100.0)
        unit_hm[u] = np.where(operating == 1, cfg["base_heavy_metals"] + np.random.normal(0, 0.005, total_steps), 0.005)

    # Injected Ground Truth Anomalies
    ground_truth_anomalies = []

    # Anomaly 1: Unit_A severe COD & acidic spike (Day 1, 10:00 AM)
    idx_start_1 = int((10 * 60) / interval_minutes)  # 10:00 AM
    duration_1 = 6  # 30 min spike
    unit_cod["Unit_A"][idx_start_1:idx_start_1+duration_1] += 700.0
    unit_ph["Unit_A"][idx_start_1:idx_start_1+duration_1] = 3.8
    unit_flows["Unit_A"][idx_start_1:idx_start_1+duration_1] += 15.0
    ground_truth_anomalies.append({
        "anomaly_id": 1,
        "emission_time": timestamps[idx_start_1].strftime("%Y-%m-%d %H:%M"),
        "source_units": ["Unit_A"],
        "type": "Single Unit Discharge Spike (Acidic/High COD)",
        "notes": "Unit A normal operating window"
    })

    # Anomaly 2: Concurrent discharge Unit_B & Unit_C (Day 1, 17:00 PM)
    idx_start_2 = int((17 * 60) / interval_minutes)
    duration_2 = 8  # 40 min spike
    unit_hm["Unit_B"][idx_start_2:idx_start_2+duration_2] += 5.0
    unit_cod["Unit_B"][idx_start_2:idx_start_2+duration_2] += 250.0

    unit_tds["Unit_C"][idx_start_2:idx_start_2+duration_2] += 1800.0
    unit_ph["Unit_C"][idx_start_2:idx_start_2+duration_2] = 10.2
    ground_truth_anomalies.append({
        "anomaly_id": 2,
        "emission_time": timestamps[idx_start_2].strftime("%Y-%m-%d %H:%M"),
        "source_units": ["Unit_B", "Unit_C"],
        "type": "Concurrent Multi-Unit Overlapping Spike",
        "notes": "Both Unit B and Unit C active simultaneously"
    })

    # Anomaly 3: Unauthorized Off-schedule Dumping by Unit_B (Day 2, 03:00 AM)
    idx_start_3 = int((27 * 60) / interval_minutes)  # Day 2, 3:00 AM
    duration_3 = 6
    if idx_start_3 + duration_3 < total_steps:
        unit_hm["Unit_B"][idx_start_3:idx_start_3+duration_3] += 6.5
        unit_ph["Unit_B"][idx_start_3:idx_start_3+duration_3] = 4.0
        unit_flows["Unit_B"][idx_start_3:idx_start_3+duration_3] += 12.0
        ground_truth_anomalies.append({
            "anomaly_id": 3,
            "emission_time": timestamps[idx_start_3].strftime("%Y-%m-%d %H:%M"),
            "source_units": ["Unit_B"],
            "type": "Unauthorized Illegal Off-Schedule Dumping",
            "notes": "Unit B operating status is logged as 0 (OFF)"
        })

    # Anomaly 4: High COD Spike by Unit_D (Day 2, 14:00 PM)
    idx_start_4 = int((38 * 60) / interval_minutes)
    duration_4 = 6
    if idx_start_4 + duration_4 < total_steps:
        unit_cod["Unit_D"][idx_start_4:idx_start_4+duration_4] += 950.0
        unit_flows["Unit_D"][idx_start_4:idx_start_4+duration_4] += 10.0
        ground_truth_anomalies.append({
            "anomaly_id": 4,
            "emission_time": timestamps[idx_start_4].strftime("%Y-%m-%d %H:%M"),
            "source_units": ["Unit_D"],
            "type": "Food Processing Organic Waste Spike",
            "notes": "High COD without pH extreme"
        })

    # Hydrodynamic Back-propagation / Delay Translation to Downstream Sensor Station Alpha
    downstream_flow = np.zeros(total_steps)
    downstream_cod_mass = np.zeros(total_steps)
    downstream_tds_mass = np.zeros(total_steps)
    downstream_hm_mass = np.zeros(total_steps)
    downstream_h_conc = np.zeros(total_steps)  # [H+] concentration for pH weighted mixing

    for u, cfg in units_config.items():
        lag_steps = get_hydraulic_lag_steps(cfg["distance_km"], cfg["hydraulic_velocity_mps"], interval_minutes)
        # Shift each unit's signal forward in time by lag_steps to simulate travel down the channel
        shifted_flow = np.roll(unit_flows[u], lag_steps)
        shifted_flow[:lag_steps] = cfg["base_flow_lps"]  # warm-up padding

        shifted_cod = np.roll(unit_cod[u], lag_steps)
        shifted_cod[:lag_steps] = cfg["base_cod"]

        shifted_tds = np.roll(unit_tds[u], lag_steps)
        shifted_tds[:lag_steps] = cfg["base_tds"]

        shifted_hm = np.roll(unit_hm[u], lag_steps)
        shifted_hm[:lag_steps] = cfg["base_heavy_metals"]

        shifted_ph = np.roll(unit_ph[u], lag_steps)
        shifted_ph[:lag_steps] = 7.0

        downstream_flow += shifted_flow
        downstream_cod_mass += shifted_cod * shifted_flow
        downstream_tds_mass += shifted_tds * shifted_flow
        downstream_hm_mass += shifted_hm * shifted_flow
        
        # Convert pH to [H+] concentration: 10^(-pH)
        h_plus = 10.0 ** (-shifted_ph)
        downstream_h_conc += h_plus * shifted_flow

    # Flow-weighted concentration at common downstream station
    safe_flow = np.maximum(downstream_flow, 0.1)
    df["downstream_flow_lps"] = downstream_flow + np.random.normal(0, 0.5, total_steps)
    df["downstream_cod_mg_l"] = (downstream_cod_mass / safe_flow) + np.random.normal(0, 5.0, total_steps)
    df["downstream_tds_ppm"] = (downstream_tds_mass / safe_flow) + np.random.normal(0, 10.0, total_steps)
    df["downstream_heavy_metals_ppm"] = (downstream_hm_mass / safe_flow) + np.random.normal(0, 0.002, total_steps)
    
    # Calculate downstream blended pH
    avg_h_conc = downstream_h_conc / safe_flow
    df["downstream_ph"] = -np.log10(np.maximum(avg_h_conc, 1e-14)) + np.random.normal(0, 0.05, total_steps)
    df["downstream_ph"] = np.clip(df["downstream_ph"], 1.0, 14.0)

    # Attach ground truth arrival timestamps downstream
    for anomaly in ground_truth_anomalies:
        emission_ts = pd.Timestamp(anomaly["emission_time"])
        # Downstream arrival is delayed by primary source's lag
        primary_source = anomaly["source_units"][0]
        lag_steps = get_hydraulic_lag_steps(
            units_config[primary_source]["distance_km"],
            units_config[primary_source]["hydraulic_velocity_mps"],
            interval_minutes
        )
        downstream_ts = emission_ts + timedelta(minutes=lag_steps * interval_minutes)
        anomaly["downstream_arrival_time"] = downstream_ts.strftime("%Y-%m-%d %H:%M")

    return df, ground_truth_anomalies


if __name__ == "__main__":
    df, anomalies = generate_synthetic_telemetry()
    print(f"Generated {len(df)} telemetry records with {len(anomalies)} injected anomalies.")
    print("Sample ground truth anomalies:")
    for a in anomalies:
        print(a)
