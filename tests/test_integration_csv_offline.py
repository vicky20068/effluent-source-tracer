"""
Integration Test Suite for Industrial Effluent Tracing System.
Verifies:
1. End-to-end CSV upload pipeline (parsing, NaN repair, EWMA detection, PI-BHDM tracing).
2. Offline store-and-forward lifecycle (local persistence, SHA-256 hash validation, sync, tamper detection).
3. Sensor calibration pairing and dynamic weighting integration.
"""

import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.field_capture import FieldCaptureManager, generate_sample_hash
from src.anomaly_detector import EWMACUSUMDetector
from src.tracing_engine import PhysicsInformedBayesianTracer
from src.calibration import SensorCalibrationRegistry, generate_calibration_hash

class TestIntegrationCSVAndOfflineSync(unittest.TestCase):
    
    def test_e2e_csv_upload_and_tracing_pipeline(self):
        """Test full workflow from raw CSV file upload to anomaly detection and PI-BHDM attribution."""
        # 1. Generate realistic field CSV with missing values and a textile spike
        base_time = datetime(2026, 9, 15, 8, 0)
        n_steps = 60
        timestamps = [base_time + timedelta(minutes=5*i) for i in range(n_steps)]
        
        flow = [40.0 + 2.0*np.sin(i/5.0) for i in range(n_steps)]
        ph = [7.2]*n_steps
        cod = [50.0 + np.random.normal(0, 3) for _ in range(n_steps)]
        hm = [0.4]*n_steps
        tds = [600.0]*n_steps
        
        # Insert textile unit (U2) spike at step 35 (pH=10.2, COD=850, TDS=3100)
        for offset in range(-2, 3):
            step = 35 + offset
            pulse = np.exp(-0.5 * (offset**2))
            ph[step] = 7.2 + 3.0 * pulse # alkaline
            cod[step] += 800.0 * pulse
            tds[step] += 2500.0 * pulse
            
        # Introduce telemetry dropout / missing values at steps 20-22
        ph[20] = np.nan
        cod[21] = np.nan
        tds[22] = np.nan
        
        raw_df = pd.DataFrame({
            "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "FlowRate_m3h": flow,
            "pH": ph,
            "COD_mgL": cod,
            "HeavyMetals_mgL": hm,
            "TDS_mgL": tds
        })
        
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            raw_df.to_csv(f.name, index=False)
            csv_path = f.name
            
        try:
            # 2. Ingest and repair CSV
            ingested_df = pd.read_csv(csv_path)
            ingested_df["timestamp"] = pd.to_datetime(ingested_df["timestamp"])
            
            # Verify repair of NaN rows
            num_nans_before = ingested_df[["pH", "COD_mgL", "TDS_mgL"]].isna().sum().sum()
            self.assertGreater(num_nans_before, 0)
            
            repaired_df = ingested_df.copy()
            repaired_df[["pH", "COD_mgL", "HeavyMetals_mgL", "TDS_mgL", "FlowRate_m3h"]] = (
                repaired_df[["pH", "COD_mgL", "HeavyMetals_mgL", "TDS_mgL", "FlowRate_m3h"]]
                .interpolate(method="linear")
                .bfill()
                .ffill()
            )
            self.assertEqual(repaired_df.isna().sum().sum(), 0)
            
            # 3. Detect anomaly with EWMA
            detector = EWMACUSUMDetector(z_threshold=2.5)
            annotated_df, spikes = detector.detect_anomalies(repaired_df)
            self.assertGreaterEqual(len(spikes), 1)
            
            # Locate the textile spike near step 35
            target_spike = None
            for s in spikes:
                if abs(s["peak_index"] - 35) <= 3:
                    target_spike = s
                    break
            self.assertIsNotNone(target_spike)
            self.assertIn(target_spike["trigger_parameter"], ["COD", "TDS"])
            
            # 4. Run PI-BHDM tracing

            tracer = PhysicsInformedBayesianTracer()
            candidates = tracer.trace_spike(target_spike, annotated_df)
            
            self.assertGreater(len(candidates), 0)
            top1 = candidates[0]
            # U2 (Textile Dyeing) should be identified as top candidate due to alkaline pH + high COD + high TDS
            self.assertEqual(top1["unit_id"], "U2")
            self.assertGreater(top1["likelihood_pct"], 45.0)
            self.assertIn("parameter_contributions", top1)
            self.assertIn("calibrated_confidence", top1)
            self.assertIn("entropy_bits", top1)
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_offline_store_and_forward_tamper_and_sync_lifecycle(self):
        """Test offline buffer lifecycle: logging, SHA-256 verification, tamper detection, and atomic sync."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            buffer_file = tmp.name
            
        try:
            manager = FieldCaptureManager(buffer_file=buffer_file)
            
            # 1. Add sample offline
            record = manager.add_spot_sample(
                inspector_id="INSP-TAMPER-TEST",
                unit_id="U1",
                unit_name="Electroplating Unit",
                ph=3.95,
                cod=145.0,
                heavy_metals=8.7,
                tds=1850.0,
                notes="Corrosive outfall visual inspection sample"
            )
            
            self.assertEqual(record["sync_status"], "PENDING_SYNC")
            orig_hash = record["sha256_hash"]
            self.assertEqual(len(orig_hash), 64)
            
            # Verify hash recalculation matches
            recalc_hash = generate_sample_hash(
                record["timestamp"],
                record["inspector_id"],
                record["unit_id"],
                record["pH"],
                record["COD_mgL"],
                record["HeavyMetals_mgL"],
                record["TDS_mgL"],
                record["notes"]
            )
            self.assertEqual(orig_hash, recalc_hash)
            
            # 2. Simulate malicious/accidental tampering in local file
            raw_buf = manager.load_buffer()
            raw_buf["records"][-1]["COD_mgL"] = 40.0 # Altered reading
            manager.save_buffer(raw_buf)
            
            tampered_buf = manager.load_buffer()
            tampered_rec = tampered_buf["records"][-1]
            tampered_check_hash = generate_sample_hash(
                tampered_rec["timestamp"],
                tampered_rec["inspector_id"],
                tampered_rec["unit_id"],
                tampered_rec["pH"],
                tampered_rec["COD_mgL"],
                tampered_rec["HeavyMetals_mgL"],
                tampered_rec["TDS_mgL"],
                tampered_rec["notes"]
            )
            # Tampering must cause hash mismatch!
            self.assertNotEqual(tampered_rec["sha256_hash"], tampered_check_hash)
            
            # Restore valid reading and test sync
            raw_buf["records"][-1]["COD_mgL"] = 145.0
            manager.save_buffer(raw_buf)
            
            synced_count = manager.sync_pending_records()
            self.assertGreaterEqual(synced_count, 1)
            
            final_buf = manager.load_buffer()
            synced_rec = [r for r in final_buf["records"] if r["record_id"] == record["record_id"]][0]
            self.assertEqual(synced_rec["sync_status"], "SYNCED")
            
        finally:
            if os.path.exists(buffer_file):
                os.remove(buffer_file)

    def test_calibration_weighting_in_tracing_engine(self):
        """Test that degraded/expired sensors have dynamic weights that influence Bayesian affinity."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            cal_file = tmp.name
            
        try:
            registry = SensorCalibrationRegistry(registry_file=cal_file)
            
            # Baseline: Fresh calibration
            health_fresh = registry.get_sensor_health_and_weight("HeavyMetals")
            self.assertEqual(health_fresh["calibration_status"], "VALID")
            self.assertAlmostEqual(health_fresh["effective_weight"], 1.0, delta=0.1)
            
            # Simulate overdue sensor (45 days elapsed on a 7-day interval)
            old_time = (datetime.now() - timedelta(days=45))
            health_overdue = registry.get_sensor_health_and_weight("HeavyMetals", evaluation_time=datetime.now() + timedelta(days=45))
            self.assertEqual(health_overdue["calibration_status"], "CRITICAL_EXPIRED")
            self.assertLess(health_overdue["effective_weight"], 0.4)
            
            # Verify tracer accepts custom sensor weights and reflects them in parameter contributions
            degraded_weights = {"pH": 1.0, "COD": 1.0, "HeavyMetals": 0.2, "TDS": 0.8}
            tracer = PhysicsInformedBayesianTracer(sensor_weights=degraded_weights)
            
            spike = {
                "peak_index": 0,
                "peak_time": datetime.now(),
                "trigger_parameter": "HeavyMetals_mgL"
            }
            df_dummy = pd.DataFrame([{
                "timestamp": datetime.now(),
                "pH": 4.0,
                "COD_mgL": 140.0,
                "HeavyMetals_mgL": 8.5,
                "TDS_mgL": 1800.0
            }])
            
            candidates = tracer.trace_spike(spike, df_dummy)
            weights_used = candidates[0]["parameter_contributions"]["weights_applied"]
            self.assertEqual(weights_used["HeavyMetals"], 0.2)
            
        finally:
            if os.path.exists(cal_file):
                os.remove(cal_file)

if __name__ == "__main__":
    unittest.main()
