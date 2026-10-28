"""
Field Inspector Data Capture & Offline Buffer Manager.
Enables offline-first data collection, manual spot sample entry,
batch CSV log ingest, local storage buffer synchronization,
and Digital Chain-of-Custody SHA-256 cryptographic sample verification.
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from src.config import CHAIN_OF_CUSTODY_CONFIG

BUFFER_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "offline_buffer.json")

def generate_sample_hash(
    timestamp: str,
    inspector_id: str,
    unit_id: str,
    ph: float,
    cod: float,
    heavy_metals: float,
    tds: float,
    notes: str
) -> str:
    """Generate cryptographic SHA-256 hash for sample chain-of-custody verification."""
    payload = f"{timestamp}|{inspector_id}|{unit_id}|{ph:.2f}|{cod:.1f}|{heavy_metals:.2f}|{tds:.1f}|{notes}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

class FieldCaptureManager:
    """Manages offline local buffer and Chain-of-Custody log for field inspector data collection."""
    
    def __init__(self, buffer_file: str = BUFFER_FILE_PATH):
        self.buffer_file = os.path.abspath(buffer_file)
        self._ensure_buffer_exists()
        
    def _ensure_buffer_exists(self):
        if not os.path.exists(self.buffer_file) or os.path.getsize(self.buffer_file) == 0:
            t1 = "2026-09-08 11:30:00"
            h1 = generate_sample_hash(t1, "INSP-402", "U1", 4.1, 160.0, 9.2, 1900.0, "Corrosive odor detected near outfall pipe. Physical sample taken.")
            
            t2 = "2026-09-09 09:15:00"
            h2 = generate_sample_hash(t2, "INSP-108", "U2", 10.1, 820.0, 0.6, 3100.0, "Deep purple coloration in discharge channel.")
            
            initial_data = {
                "last_sync_timestamp": None,
                "records": [
                    {
                        "record_id": "FIELD-LOG-001",
                        "timestamp": t1,
                        "inspector_id": "INSP-402",
                        "unit_id": "U1",
                        "unit_name": "Electroplating & Surface Treatment",
                        "sample_type": "Manual Spot Sample",
                        "pH": 4.1,
                        "COD_mgL": 160.0,
                        "HeavyMetals_mgL": 9.2,
                        "TDS_mgL": 1900.0,
                        "gps_lat_long": CHAIN_OF_CUSTODY_CONFIG["estate_location_lat_long"],
                        "notes": "Corrosive odor detected near outfall pipe. Physical sample taken.",
                        "sha256_hash": h1,
                        "verification_status": "VERIFIED_LEGAL",
                        "sync_status": "SYNCED"
                    },
                    {
                        "record_id": "FIELD-LOG-002",
                        "timestamp": t2,
                        "inspector_id": "INSP-108",
                        "unit_id": "U2",
                        "unit_name": "Textile Dyeing & Wet Processing",
                        "sample_type": "Manual Spot Sample",
                        "pH": 10.1,
                        "COD_mgL": 820.0,
                        "HeavyMetals_mgL": 0.6,
                        "TDS_mgL": 3100.0,
                        "gps_lat_long": CHAIN_OF_CUSTODY_CONFIG["estate_location_lat_long"],
                        "notes": "Deep purple coloration in discharge channel.",
                        "sha256_hash": h2,
                        "verification_status": "PENDING_LAB_VERIFICATION",
                        "sync_status": "PENDING_SYNC"
                    }
                ]
            }
            with open(self.buffer_file, "w") as f:
                json.dump(initial_data, f, indent=2)
                
    def load_buffer(self) -> Dict[str, Any]:
        with open(self.buffer_file, "r") as f:
            return json.load(f)
            
    def save_buffer(self, data: Dict[str, Any]):
        with open(self.buffer_file, "w") as f:
            json.dump(data, f, indent=2)
            
    def add_spot_sample(
        self,
        inspector_id: str,
        unit_id: str,
        unit_name: str,
        ph: float,
        cod: float,
        heavy_metals: float,
        tds: float,
        notes: str,
        gps_lat_long: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        buffer = self.load_buffer()
        new_id = f"FIELD-LOG-{len(buffer['records'])+1:03d}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sha_hash = generate_sample_hash(
            now_str, inspector_id, unit_id, ph, cod, heavy_metals, tds, notes
        )
        
        gps_coords = gps_lat_long if gps_lat_long is not None else CHAIN_OF_CUSTODY_CONFIG["estate_location_lat_long"]
        
        record = {
            "record_id": new_id,
            "timestamp": now_str,
            "inspector_id": inspector_id,
            "unit_id": unit_id,
            "unit_name": unit_name,
            "sample_type": "Manual Spot Sample",
            "pH": float(ph),
            "COD_mgL": float(cod),
            "HeavyMetals_mgL": float(heavy_metals),
            "TDS_mgL": float(tds),
            "gps_lat_long": list(gps_coords),
            "notes": notes,
            "sha256_hash": sha_hash,
            "verification_status": "PENDING_LAB_VERIFICATION",
            "sync_status": "PENDING_SYNC"
        }
        
        buffer["records"].append(record)
        self.save_buffer(buffer)
        return record

    def ingest_batch_df(self, df: pd.DataFrame, inspector_id: str = "BATCH-UPLOAD") -> int:
        buffer = self.load_buffer()
        added_count = 0
        
        for _, row in df.iterrows():
            record_id = f"FIELD-LOG-{len(buffer['records'])+1:03d}"
            now_str = str(row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            unit_id = str(row.get("unit_id", "U1"))
            unit_name = str(row.get("unit_name", "Ingested Unit"))
            ph = float(row.get("pH", 7.0))
            cod = float(row.get("COD_mgL", 50.0))
            heavy_metals = float(row.get("HeavyMetals_mgL", 0.5))
            tds = float(row.get("TDS_mgL", 600.0))
            notes = str(row.get("notes", "Batch CSV ingest offline log"))
            
            sha_hash = generate_sample_hash(
                now_str, inspector_id, unit_id, ph, cod, heavy_metals, tds, notes
            )
            
            record = {
                "record_id": record_id,
                "timestamp": now_str,
                "inspector_id": inspector_id,
                "unit_id": unit_id,
                "unit_name": unit_name,
                "sample_type": "Batch CSV Import",
                "pH": ph,
                "COD_mgL": cod,
                "HeavyMetals_mgL": heavy_metals,
                "TDS_mgL": tds,
                "gps_lat_long": list(CHAIN_OF_CUSTODY_CONFIG["estate_location_lat_long"]),
                "notes": notes,
                "sha256_hash": sha_hash,
                "verification_status": "PENDING_LAB_VERIFICATION",
                "sync_status": "PENDING_SYNC"
            }
            buffer["records"].append(record)
            added_count += 1
            
        self.save_buffer(buffer)
        return added_count

    def sync_pending_records(self) -> int:
        buffer = self.load_buffer()
        synced_count = 0
        
        for rec in buffer["records"]:
            if rec["sync_status"] == "PENDING_SYNC":
                rec["sync_status"] = "SYNCED"
                synced_count += 1
                
        buffer["last_sync_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_buffer(buffer)
        return synced_count
