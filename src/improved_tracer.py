"""
Improved Hydraulic & Probabilistic Effluent Source Tracer

Traces downstream pollution spikes back to plausible source units by integrating:
1. Hydraulic travel-time back-propagation (distance / flow velocity delay)
2. Back-projected operating schedule alignment
3. Multi-parameter chemical signature matching (pH, COD, TDS, Heavy Metals)
4. Multi-source concurrent attribution & confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.data_generator import get_hydraulic_lag_steps


class ImprovedTracer:
    def __init__(self, units_config: Dict[str, Any]):
        self.units_config = units_config

    def _calculate_chemical_match(self, event: Dict[str, Any], unit_cfg: Dict[str, Any]) -> float:
        """
        Computes compatibility score (0.0 to 1.0) between downstream spike chemical profile
        and the unit's known effluent spike signature.
        """
        sig = unit_cfg["spike_signature"]
        score = 0.0
        weights_sum = 0.0

        # pH matching
        min_ph = event["min_ph"]
        max_ph = event["max_ph"]
        expected_ph = sig["ph"]
        
        # If expected pH is acidic (< 5.5) and observed pH is acidic
        if expected_ph < 5.5:
            ph_score = max(0.0, 1.0 - abs(min_ph - expected_ph) / 3.0)
            score += 2.5 * ph_score
            weights_sum += 2.5
        elif expected_ph > 8.5:
            ph_score = max(0.0, 1.0 - abs(max_ph - expected_ph) / 3.0)
            score += 2.5 * ph_score
            weights_sum += 2.5

        # Heavy Metals matching
        max_hm = event["max_heavy_metals"]
        expected_hm = sig["heavy_metals"]
        if expected_hm > 0.5:
            hm_score = min(1.0, max_hm / (expected_hm * 0.5)) if max_hm > 0.1 else 0.05
            score += 3.0 * hm_score
            weights_sum += 3.0

        # TDS matching
        max_tds = event["max_tds"]
        expected_tds = sig["tds"]
        if expected_tds > 1000:
            tds_score = min(1.0, max_tds / expected_tds) if max_tds > 800 else 0.1
            score += 2.0 * tds_score
            weights_sum += 2.0

        # COD matching
        max_cod = event["max_cod"]
        expected_cod = sig["cod"]
        cod_score = min(1.0, max_cod / expected_cod) if max_cod > 250 else 0.2
        score += 1.5 * cod_score
        weights_sum += 1.5

        return float(np.clip(score / weights_sum if weights_sum > 0 else 0.5, 0.0, 1.0))

    def trace_spike(self, event: Dict[str, Any], telemetry_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Traces a downstream anomaly event using hydraulic back-propagation, schedule verification,
        and chemical fingerprinting.
        """
        spike_start = pd.Timestamp(event["start_time"])
        spike_end = pd.Timestamp(event["end_time"])

        candidate_results = []
        raw_scores = {}

        for unit_id, cfg in self.units_config.items():
            # 1. Hydraulic Delay Back-Propagation
            lag_steps = get_hydraulic_lag_steps(cfg["distance_km"], cfg["hydraulic_velocity_mps"], interval_minutes=5)
            lag_minutes = lag_steps * 5
            
            # Back-projected emission window at unit outlet
            emission_start = spike_start - pd.Timedelta(minutes=lag_minutes)
            emission_end = spike_end - pd.Timedelta(minutes=lag_minutes)

            # 2. Operating Schedule Check during back-projected window
            window_df = telemetry_df[
                (telemetry_df["timestamp"] >= emission_start - pd.Timedelta(minutes=5)) &
                (telemetry_df["timestamp"] <= emission_end + pd.Timedelta(minutes=5))
            ]

            status_col = f"{unit_id}_status"
            if not window_df.empty and status_col in window_df.columns:
                was_active = (window_df[status_col].mean() > 0.2)
            else:
                was_active = False

            # Schedule score (1.0 if operating, 0.4 if off to allow for unauthorized dumping detection)
            schedule_score = 1.0 if was_active else 0.4

            # 3. Chemical Signature Fingerprint Matching
            chem_score = self._calculate_chemical_match(event, cfg)

            # 4. Combine into Composite Likelihood Score
            # Heavy weight on chemical fingerprint + hydraulic back-propagated schedule alignment
            composite_score = (0.55 * chem_score) + (0.45 * schedule_score)
            raw_scores[unit_id] = composite_score

            candidate_results.append({
                "unit_id": unit_id,
                "unit_name": cfg["name"],
                "raw_score": composite_score,
                "chemical_match_score": round(chem_score, 2),
                "schedule_alignment_score": round(schedule_score, 2),
                "was_operating_in_emission_window": was_active,
                "hydraulic_delay_minutes": lag_minutes,
                "estimated_emission_window": f"{emission_start.strftime('%H:%M')} - {emission_end.strftime('%H:%M')}",
                "explanation": f"Lag: -{lag_minutes}m. Schedule: {'ACTIVE' if was_active else 'OFF (Possible Illegal Dump)'}. Chem Match: {int(chem_score*100)}%"
            })

        # Normalize raw scores into likelihood probabilities
        total_raw = sum(raw_scores.values()) or 1.0
        for cand in candidate_results:
            prob = cand["raw_score"] / total_raw
            cand["likelihood_probability"] = round(prob, 3)

        # Rank descending
        candidate_results.sort(key=lambda x: x["likelihood_probability"], reverse=True)

        # Multi-source analysis
        top_candidates = [c for c in candidate_results if c["likelihood_probability"] >= 0.28]
        is_multi_source = len(top_candidates) > 1

        return {
            "event_id": event.get("event_id"),
            "spike_start": spike_start.strftime("%Y-%m-%d %H:%M"),
            "spike_end": spike_end.strftime("%Y-%m-%d %H:%M"),
            "is_multi_source": is_multi_source,
            "top_source_unit": candidate_results[0]["unit_id"],
            "plausible_sources": [c["unit_id"] for c in top_candidates],
            "ranked_candidates": candidate_results
        }


if __name__ == "__main__":
    from src.data_generator import generate_synthetic_telemetry, DEFAULT_UNITS_CONFIG
    from src.anomaly_detector import EffluentAnomalyDetector

    df, ground_truth = generate_synthetic_telemetry()
    detector = EffluentAnomalyDetector()
    df_flagged, events = detector.detect_anomalies(df)

    tracer = ImprovedTracer(DEFAULT_UNITS_CONFIG)
    print("\n--- Improved Tracer Analysis ---")
    for ev in events[:5]:  # view first 5 events
        res = tracer.trace_spike(ev, df_flagged)
        print(f"\nDownstream Spike at {res['spike_start']} (Peak COD: {ev['max_cod']:.1f}, Min pH: {ev['min_ph']:.2f}, Max HM: {ev['max_heavy_metals']:.3f})")
        print(f"Plausible Sources: {res['plausible_sources']} (Multi-Source: {res['is_multi_source']})")
        for c in res["ranked_candidates"]:
            print(f"  -> {c['unit_id']}: Prob={c['likelihood_probability']:.3f} | {c['explanation']}")
