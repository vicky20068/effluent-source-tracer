"""
Effluent Anomaly Detection Engine

Detects pollution spikes and statistical anomalies in combined downstream sensor signals
using dynamic rolling thresholds (z-score and deviation metrics).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class EffluentAnomalyDetector:
    def __init__(self, cod_z_threshold: float = 3.0, hm_z_threshold: float = 3.0, ph_dev_threshold: float = 1.8, tds_z_threshold: float = 3.0):
        self.cod_z_threshold = cod_z_threshold
        self.hm_z_threshold = hm_z_threshold
        self.ph_dev_threshold = ph_dev_threshold
        self.tds_z_threshold = tds_z_threshold

    def detect_anomalies(self, telemetry_df: pd.DataFrame, rolling_window: int = 48) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Calculates rolling baseline metrics and flags anomalies.
        Returns updated DataFrame with anomaly flags and a list of aggregated Anomaly Events.
        """
        df = telemetry_df.copy()

        # Calculate rolling baseline means and stds (minimum 12 samples)
        for col in ["downstream_cod_mg_l", "downstream_heavy_metals_ppm", "downstream_tds_ppm"]:
            rolling_mean = df[col].rolling(window=rolling_window, min_periods=12).mean()
            rolling_std = df[col].rolling(window=rolling_window, min_periods=12).std().replace(0, 1e-5)
            df[f"{col}_zscore"] = (df[col] - rolling_mean) / rolling_std

        # pH absolute deviation from 7.0
        df["ph_deviation"] = (df["downstream_ph"] - 7.0).abs()

        # Flag composite anomaly condition
        df["is_anomaly"] = (
            (df["downstream_cod_mg_l_zscore"] > self.cod_z_threshold) |
            (df["downstream_heavy_metals_ppm_zscore"] > self.hm_z_threshold) |
            (df["downstream_tds_ppm_zscore"] > self.tds_z_threshold) |
            (df["ph_deviation"] > self.ph_dev_threshold)
        ).astype(int)

        # Aggregate contiguous anomaly timestamps into discrete Anomaly Events
        anomaly_events = []
        in_event = False
        current_event = None

        for idx, row in df.iterrows():
            if row["is_anomaly"] == 1:
                if not in_event:
                    in_event = True
                    current_event = {
                        "event_id": len(anomaly_events) + 1,
                        "start_time": row["timestamp"],
                        "end_time": row["timestamp"],
                        "peak_time": row["timestamp"],
                        "max_cod": row["downstream_cod_mg_l"],
                        "max_heavy_metals": row["downstream_heavy_metals_ppm"],
                        "max_tds": row["downstream_tds_ppm"],
                        "min_ph": row["downstream_ph"],
                        "max_ph": row["downstream_ph"],
                        "peak_flow": row["downstream_flow_lps"],
                        "peak_zscore": max(row["downstream_cod_mg_l_zscore"], row["downstream_heavy_metals_ppm_zscore"]),
                        "raw_rows": [row]
                    }
                else:
                    current_event["end_time"] = row["timestamp"]
                    current_event["raw_rows"].append(row)
                    if row["downstream_cod_mg_l"] > current_event["max_cod"]:
                        current_event["max_cod"] = row["downstream_cod_mg_l"]
                        current_event["peak_time"] = row["timestamp"]
                    if row["downstream_heavy_metals_ppm"] > current_event["max_heavy_metals"]:
                        current_event["max_heavy_metals"] = row["downstream_heavy_metals_ppm"]
                    if row["downstream_tds_ppm"] > current_event["max_tds"]:
                        current_event["max_tds"] = row["downstream_tds_ppm"]
                    if row["downstream_ph"] < current_event["min_ph"]:
                        current_event["min_ph"] = row["downstream_ph"]
                    if row["downstream_ph"] > current_event["max_ph"]:
                        current_event["max_ph"] = row["downstream_ph"]
            else:
                if in_event:
                    in_event = False
                    # Format strings and calculate duration
                    start_ts = pd.Timestamp(current_event["start_time"])
                    end_ts = pd.Timestamp(current_event["end_time"])
                    current_event["duration_minutes"] = int((end_ts - start_ts).total_seconds() / 60) + 5
                    anomaly_events.append(current_event)

        if in_event and current_event:
            start_ts = pd.Timestamp(current_event["start_time"])
            end_ts = pd.Timestamp(current_event["end_time"])
            current_event["duration_minutes"] = int((end_ts - start_ts).total_seconds() / 60) + 5
            anomaly_events.append(current_event)

        return df, anomaly_events


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_telemetry
    df, ground_truth = generate_synthetic_telemetry()
    detector = EffluentAnomalyDetector()
    df_flagged, events = detector.detect_anomalies(df)
    print(f"Detected {len(events)} anomaly events:")
    for ev in events:
        print(f"Event {ev['event_id']}: {ev['start_time']} to {ev['end_time']} (Peak COD: {ev['max_cod']:.1f}, Min pH: {ev['min_ph']:.2f}, Max Metals: {ev['max_heavy_metals']:.3f})")
