"""
Unit tests for Static vs EWMA Dynamic Anomaly Detection modules.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.anomaly_detector import EWMACUSUMDetector, StaticBaselineDetector

class TestAnomalyDetector(unittest.TestCase):
    def test_ewma_spike_detection(self):
        timestamps = [datetime(2026, 9, 1, 0, 0) + timedelta(minutes=5*i) for i in range(100)]
        cod_vals = np.full(100, 45.0)
        cod_vals[50] = 850.0 # Spike
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "FlowRate_m3h": np.full(100, 40.0),
            "pH": np.full(100, 7.2),
            "COD_mgL": cod_vals,
            "HeavyMetals_mgL": np.full(100, 0.4),
            "TDS_mgL": np.full(100, 600.0)
        })
        
        detector = EWMACUSUMDetector(z_threshold=2.0)
        annotated_df, spikes = detector.detect_anomalies(df)
        
        self.assertGreaterEqual(len(spikes), 1)
        self.assertTrue(any(s["peak_index"] == 50 for s in spikes))
        self.assertIn("COD_mgL_zscore", annotated_df.columns)

    def test_ewma_suppresses_baseline_drift(self):
        timestamps = [datetime(2026, 9, 1, 0, 0) + timedelta(minutes=5*i) for i in range(100)]
        drift = np.linspace(40.0, 280.0, 100) # Bio-fouling drift above static limit 250
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "FlowRate_m3h": np.full(100, 40.0),
            "pH": np.full(100, 7.2),
            "COD_mgL": drift,
            "HeavyMetals_mgL": np.full(100, 0.4),
            "TDS_mgL": np.full(100, 600.0)
        })
        
        static_det = StaticBaselineDetector()
        ewma_det = EWMACUSUMDetector(alpha=0.15, z_threshold=3.0)
        
        static_spikes = static_det.detect_anomalies(df)
        _, ewma_spikes = ewma_det.detect_anomalies(df)
        
        self.assertGreater(len(static_spikes), len(ewma_spikes))

if __name__ == "__main__":
    unittest.main()
