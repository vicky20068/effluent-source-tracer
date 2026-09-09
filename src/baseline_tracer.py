"""
Baseline Source Tracing Algorithm

Simple heuristic method required by project baseline specification:
Flags whichever unit(s) were logged as OPERATING nearest to the wall-clock time of the downstream spike,
without hydraulic travel delay calculation or chemical signature alignment.
"""

import pandas as pd
from typing import Dict, List, Any


class BaselineTracer:
    def __init__(self, units_config: Dict[str, Any]):
        self.units_config = units_config

    def trace_spike(self, event: Dict[str, Any], telemetry_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Traces a downstream anomaly event using the simple temporal proximity baseline heuristic.
        
        Returns a list of unit candidate dictionaries ordered by likelihood rank.
        """
        spike_time = pd.Timestamp(event["start_time"])

        # Find closest row in telemetry DataFrame
        idx = (telemetry_df["timestamp"] - spike_time).abs().idxmin()
        row = telemetry_df.loc[idx]

        candidates = []
        for unit_id, cfg in self.units_config.items():
            status_col = f"{unit_id}_status"
            is_active = row.get(status_col, 0) == 1
            
            # Simple baseline likelihood: 1.0 if operating at spike time, 0.1 if idle
            score = 1.0 if is_active else 0.1
            
            candidates.append({
                "unit_id": unit_id,
                "unit_name": cfg["name"],
                "likelihood_score": round(score, 2),
                "is_operating_at_spike": is_active,
                "hydraulic_delay_considered": False,
                "chemical_match_considered": False,
                "explanation": f"Operating status = {'ACTIVE' if is_active else 'IDLE'} at wall-clock spike time {row['timestamp']}"
            })

        # Rank candidates descending by score
        candidates.sort(key=lambda x: x["likelihood_score"], reverse=True)
        return candidates
