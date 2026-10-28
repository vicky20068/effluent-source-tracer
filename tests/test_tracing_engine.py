"""
Unit tests for Baseline and Physics-Informed Bayesian Tracing Engines.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from src.tracing_engine import BaselineTimeProximityTracer, PhysicsInformedBayesianTracer

class TestTracingEngine(unittest.TestCase):
    def test_pibhdm_tracing_electroplating_acid_spike(self):
        start_time = datetime(2026, 9, 10, 10, 0)
        timestamps = [start_time + timedelta(minutes=5*i) for i in range(20)]
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "FlowRate_m3h": [40.0]*20,
            "pH": [7.2]*10 + [3.8]*3 + [7.2]*7, # Acidic pH drop from U1
            "COD_mgL": [50.0]*20,
            "HeavyMetals_mgL": [0.4]*10 + [8.5]*3 + [0.4]*7, # High metals from U1
            "TDS_mgL": [600.0]*20
        })
        
        spike_event = {"peak_index": 11, "peak_time": timestamps[11]}
        tracer = PhysicsInformedBayesianTracer()
        candidates = tracer.trace_spike(spike_event, df)
        
        top1 = candidates[0]
        self.assertEqual(top1["unit_id"], "U1")
        self.assertGreater(top1["likelihood_pct"], 40.0)
        self.assertIn("calibrated_confidence", top1)
        self.assertIn("entropy_bits", top1)

    def test_pibhdm_unknown_source_handling(self):
        start_time = datetime(2026, 9, 10, 2, 0)
        timestamps = [start_time + timedelta(minutes=5*i) for i in range(20)]
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "FlowRate_m3h": [40.0]*20,
            "pH": [7.2]*20,
            "COD_mgL": [50.0]*10 + [950.0]*3 + [50.0]*7,
            "HeavyMetals_mgL": [0.4]*10 + [12.5]*3 + [0.4]*7,
            "TDS_mgL": [100.0]*20
        })
        
        spike_event = {"peak_index": 11, "peak_time": timestamps[11]}
        tracer = PhysicsInformedBayesianTracer()
        candidates = tracer.trace_spike(spike_event, df)
        
        unit_ids = [c["unit_id"] for c in candidates]
        self.assertIn("U_UNK", unit_ids)

if __name__ == "__main__":
    unittest.main()
