"""
Integration tests for Field Capture Manager, Offline Store-and-Forward Buffer, Batch Ingest, and SHA-256 Chain-of-Custody.
"""

import os
import unittest
import tempfile
import pandas as pd
from src.field_capture import FieldCaptureManager, generate_sample_hash

class TestFieldCapture(unittest.TestCase):
    def test_offline_store_and_forward_sync(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            mgr = FieldCaptureManager(buffer_file=tmp_path)
            buf1 = mgr.load_buffer()
            initial_count = len(buf1["records"])
            
            # Add new spot sample offline
            rec = mgr.add_spot_sample(
                inspector_id="INSP-TEST",
                unit_id="U1",
                unit_name="Electroplating Unit",
                ph=3.9,
                cod=150.0,
                heavy_metals=8.0,
                tds=1800.0,
                notes="Offline spot check test"
            )
            
            self.assertEqual(rec["sync_status"], "PENDING_SYNC")
            self.assertIn("sha256_hash", rec)
            self.assertEqual(len(rec["sha256_hash"]), 64)
            
            # Verify buffer pending status
            buf2 = mgr.load_buffer()
            self.assertEqual(len(buf2["records"]), initial_count + 1)
            
            # Execute sync
            synced_count = mgr.sync_pending_records()
            self.assertGreaterEqual(synced_count, 1)
            
            buf3 = mgr.load_buffer()
            synced_recs = [r for r in buf3["records"] if r["record_id"] == rec["record_id"]]
            self.assertEqual(synced_recs[0]["sync_status"], "SYNCED")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_batch_csv_upload_ingest(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            mgr = FieldCaptureManager(buffer_file=tmp_path)
            df_batch = pd.DataFrame([
                {"timestamp": "2026-09-10 12:00:00", "unit_id": "U1", "unit_name": "Plating", "pH": 4.0, "COD_mgL": 140, "HeavyMetals_mgL": 8.1, "TDS_mgL": 1800, "notes": "Batch 1"},
                {"timestamp": "2026-09-10 13:00:00", "unit_id": "U2", "unit_name": "Textile", "pH": 10.2, "COD_mgL": 850, "HeavyMetals_mgL": 0.5, "TDS_mgL": 3100, "notes": "Batch 2"}
            ])
            
            added = mgr.ingest_batch_df(df_batch, inspector_id="BATCH-TEST")
            self.assertEqual(added, 2)
            
            buf = mgr.load_buffer()
            self.assertGreaterEqual(len(buf["records"]), 4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
