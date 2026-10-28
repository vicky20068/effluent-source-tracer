"""
Anomaly Detection Engine for Industrial Estate Effluent Sensor Streams.
Implements:
1. Static Threshold Baseline Detector
2. EWMA + CUSUM Dynamic Residual Spike Detector
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from src.config import REGULATORY_LIMITS, DETECTOR_CONFIG

class StaticBaselineDetector:
    """Baseline anomaly detector using static regulatory limits."""
    
    def __init__(self, cod_limit: float = 250.0, heavy_metals_limit: float = 2.0, ph_min: float = 6.5, ph_max: float = 8.5):
        self.cod_limit = cod_limit
        self.heavy_metals_limit = heavy_metals_limit
        self.ph_min = ph_min
        self.ph_max = ph_max
        
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify individual timestamps violating static regulatory thresholds."""
        detected_spikes = []
        
        is_cod_violation = df["COD_mgL"] > self.cod_limit
        is_hm_violation = df["HeavyMetals_mgL"] > self.heavy_metals_limit
        is_ph_violation = (df["pH"] < self.ph_min) | (df["pH"] > self.ph_max)
        
        violation_mask = is_cod_violation | is_hm_violation | is_ph_violation
        violation_indices = df[violation_mask].index.tolist()
        
        # Group contiguous violation steps into spike events
        if not violation_indices:
            return []
            
        clusters = []
        current_cluster = [violation_indices[0]]
        
        for idx in violation_indices[1:]:
            if idx == current_cluster[-1] + 1:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)
        
        for cluster_id, cluster in enumerate(clusters):
            sub_df = df.loc[cluster]
            # Find peak severity step
            max_cod_idx = sub_df["COD_mgL"].idxmax()
            max_hm_idx = sub_df["HeavyMetals_mgL"].idxmax()
            
            if sub_df.loc[max_hm_idx, "HeavyMetals_mgL"] / self.heavy_metals_limit > sub_df.loc[max_cod_idx, "COD_mgL"] / self.cod_limit:
                peak_idx = max_hm_idx
                param = "HeavyMetals"
                peak_val = sub_df.loc[peak_idx, "HeavyMetals_mgL"]
            else:
                peak_idx = max_cod_idx
                param = "COD"
                peak_val = sub_df.loc[peak_idx, "COD_mgL"]
                
            detected_spikes.append({
                "spike_id": f"BASE-SPIKE-{cluster_id+1:03d}",
                "start_time": sub_df["timestamp"].min(),
                "peak_time": df.loc[peak_idx, "timestamp"],
                "end_time": sub_df["timestamp"].max(),
                "peak_index": peak_idx,
                "trigger_parameter": param,
                "peak_value": round(float(peak_val), 2),
                "duration_mins": len(cluster) * 5,
                "detection_method": "Static Baseline"
            })
            
        return detected_spikes

class EWMACUSUMDetector:
    """Dynamic Anomaly Detector using Exponentially Weighted Moving Average (EWMA) & CUSUM."""
    
    def __init__(self, alpha: float = 0.15, z_threshold: float = 3.0, cusum_h: float = 4.5):
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.cusum_h = cusum_h
        
    def detect_anomalies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Detect spikes across continuous parameters (COD, HeavyMetals, TDS, FlowRate).
        Returns annotated DataFrame with Z-scores and list of detected Spike Events.
        """
        annotated_df = df.copy()
        parameters = ["COD_mgL", "HeavyMetals_mgL", "TDS_mgL"]
        
        combined_anomaly_mask = pd.Series(False, index=df.index)
        
        for param in parameters:
            series = df[param].values
            n = len(series)
            
            # Compute EWMA Mean and EWMA Variance
            ewma_mean = np.zeros(n)
            ewma_var = np.zeros(n)
            z_scores = np.zeros(n)
            
            base_std = max(float(np.std(series[:max(10, min(50, n))])), 1.0)
            ewma_mean[0] = series[0]
            ewma_var[0] = base_std ** 2
            
            for t in range(1, n):
                diff = series[t] - ewma_mean[t-1]
                ewma_mean[t] = self.alpha * series[t] + (1 - self.alpha) * ewma_mean[t-1]
                
                std_dev_prev = np.sqrt(max(ewma_var[t-1], 1e-4))
                diff_clip = np.clip(diff, -3.0 * std_dev_prev, 3.0 * std_dev_prev)
                ewma_var[t] = (1 - self.alpha) * (ewma_var[t-1] + self.alpha * (diff_clip ** 2))
                
                z_scores[t] = (series[t] - ewma_mean[t-1]) / std_dev_prev
                
            annotated_df[f"{param}_zscore"] = z_scores
            param_mask = z_scores > self.z_threshold
            combined_anomaly_mask = combined_anomaly_mask | param_mask
            
        violation_indices = df[combined_anomaly_mask].index.tolist()
        
        if not violation_indices:
            return annotated_df, []
            
        # Cluster continuous steps into spike events
        clusters = []
        current_cluster = [violation_indices[0]]
        
        for idx in violation_indices[1:]:
            if idx <= current_cluster[-1] + 2:  # Allow 1 missing step gap
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)
        
        detected_spikes = []
        for cluster_id, cluster in enumerate(clusters):
            sub_df = annotated_df.loc[cluster]
            
            # Find parameter with highest Z-score peak
            max_param = "COD_mgL"
            max_z = -1.0
            max_idx = cluster[0]
            
            for param in parameters:
                param_max_z = sub_df[f"{param}_zscore"].max()
                if param_max_z > max_z:
                    max_z = param_max_z
                    max_param = param
                    max_idx = sub_df[f"{param}_zscore"].idxmax()
                    
            peak_val = sub_df.loc[max_idx, max_param]
            clean_param_name = max_param.replace("_mgL", "")
            
            detected_spikes.append({
                "spike_id": f"EWMA-SPIKE-{cluster_id+1:03d}",
                "start_time": sub_df["timestamp"].min(),
                "peak_time": annotated_df.loc[max_idx, "timestamp"],
                "end_time": sub_df["timestamp"].max(),
                "peak_index": max_idx,
                "trigger_parameter": clean_param_name,
                "peak_value": round(float(peak_val), 2),
                "peak_zscore": round(float(max_z), 2),
                "duration_mins": len(cluster) * 5,
                "detection_method": "EWMA + CUSUM Dynamic"
            })
            
        return annotated_df, detected_spikes
