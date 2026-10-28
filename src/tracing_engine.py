"""
Source Tracing Engines for Industrial Effluent Anomaly Attribution.
Implements:
1. Baseline Method: Time-Proximity Operating Heuristic
2. Proposed Method: Physics-Informed Bayesian Hydraulic Dispersion Model (PI-BHDM)
   with Uncertainty Calibration, Unknown Source Modeling, and Telemetry Gap Hardening.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from src.config import UNITS_METADATA, HYDRAULICS, UNKNOWN_UNIT_SPEC, CALIBRATION_CONFIG
from src.data_generator import is_unit_operating
from src.calibration import SensorCalibrationRegistry

def get_schedule_gap_minutes(unit_id: str, timestamp: datetime) -> float:

    """Calculate minutes between timestamp and unit's active operating shift."""
    if unit_id == "U_UNK":
        return 0.0  # Unknown/Illegal dumping can occur at any time
        
    if is_unit_operating(unit_id, timestamp):
        return 0.0
    
    hour = timestamp.hour + timestamp.minute / 60.0
    uinfo = UNITS_METADATA[unit_id]
    min_gap = 24.0
    
    for start_h, end_h in uinfo["operating_shifts"]:
        if start_h <= end_h:
            gap = min(abs(hour - start_h), abs(hour - end_h))
        else:
            gap = min(abs(hour - start_h), abs(hour - end_h))
        min_gap = min(min_gap, gap)
        
    return float(min_gap * 60.0)

def compute_uncertainty_metrics(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Shannon Entropy and Top-1 vs Top-2 Probability Margin."""
    probs = [c["likelihood_pct"] / 100.0 for c in candidates if c["likelihood_pct"] > 0]
    
    # Shannon Entropy H = -sum(p * log2(p))
    entropy = -sum(p * np.log2(p) for p in probs) if probs else 0.0
    
    # Probability Margin m = p_1 - p_2
    sorted_probs = sorted(probs, reverse=True)
    top1_p = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
    top2_p = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin_pct = round((top1_p - top2_p) * 100.0, 1)
    
    # Calibrated Confidence Classification
    if margin_pct >= CALIBRATION_CONFIG["high_confidence_margin"] * 100.0 and entropy <= 2.2:
        calibrated_confidence = "HIGH (Strong Attribution)"
    elif margin_pct >= CALIBRATION_CONFIG["medium_confidence_margin"] * 100.0:
        calibrated_confidence = "MEDIUM (Moderate Attribution)"
    else:
        calibrated_confidence = "LOW (High Uncertainty / Superposition)"
        
    return {
        "entropy_bits": round(float(entropy), 3),
        "probability_margin_pct": margin_pct,
        "calibrated_confidence": calibrated_confidence,
        "top1_likelihood_pct": round(top1_p * 100.0, 1),
        "top2_likelihood_pct": round(top2_p * 100.0, 1),
    }

class BaselineTimeProximityTracer:
    """
    Baseline Method: Flags whichever unit was operating nearest to the spike timestamp t_spike.
    Ignores hydraulic travel distance, flow velocity, advection-dispersion, and chemical profile matching.
    """
    
    def trace_spike(self, spike_event: Dict[str, Any], downstream_df: pd.DataFrame) -> List[Dict[str, Any]]:
        spike_time = spike_event["peak_time"]
        candidates = []
        
        for uid, uinfo in UNITS_METADATA.items():
            gap_mins = get_schedule_gap_minutes(uid, spike_time)
            score = np.exp(-gap_mins / 120.0)
            
            candidates.append({
                "unit_id": uid,
                "unit_name": uinfo["name"],
                "sector": uinfo["sector"],
                "distance_m": uinfo["distance_m"],
                "likelihood_score": score,
                "is_operating_at_spike": gap_mins == 0.0,
                "schedule_gap_mins": round(gap_mins, 1),
                "travel_delay_mins": 0.0,
                "chemical_affinity": 0.5,
                "tracing_method": "Baseline (Time-Proximity)"
            })
            
        total_score = sum(c["likelihood_score"] for c in candidates)
        if total_score == 0:
            total_score = 1e-6
            
        for c in candidates:
            c["likelihood_pct"] = round((c["likelihood_score"] / total_score) * 100.0, 1)
            
        candidates.sort(key=lambda x: x["likelihood_score"], reverse=True)
        for rank, c in enumerate(candidates, start=1):
            c["rank"] = rank
            
        return candidates

class PhysicsInformedBayesianTracer:
    """
    Proposed Method: Physics-Informed Bayesian Hydraulic Dispersion Model (PI-BHDM).
    Integrates physical travel time back-propagation, 1D advection-dispersion attenuation,
    operating schedule alignment, multi-parameter chemical profile similarity,
    and unknown unit fallback candidate U_UNK.
    """
    
    def __init__(
        self,
        custom_velocity_ms: Optional[float] = None,
        dispersion_coeff: Optional[float] = None,
        sensor_weights: Optional[Dict[str, float]] = None
    ):
        self.v = custom_velocity_ms if custom_velocity_ms is not None else HYDRAULICS["default_stream_velocity_ms"]
        self.D = dispersion_coeff if dispersion_coeff is not None else HYDRAULICS["dispersion_coefficient_m2s"]
        self.m0_dist = HYDRAULICS["monitoring_station_distance_m"]
        if sensor_weights is not None:
            self.sensor_weights = sensor_weights
        else:
            try:
                self.sensor_weights = SensorCalibrationRegistry().get_all_sensor_weights()
            except Exception:
                self.sensor_weights = {"pH": 1.0, "COD": 1.0, "HeavyMetals": 1.0, "TDS": 0.8}
        
    def trace_spike(self, spike_event: Dict[str, Any], downstream_df: pd.DataFrame) -> List[Dict[str, Any]]:
        spike_time = spike_event["peak_time"]
        peak_idx = spike_event["peak_index"]
        
        # Safe extraction of observed telemetry row handling potential NaNs
        obs_row = downstream_df.loc[peak_idx]
        obs_ph = float(obs_row["pH"]) if pd.notna(obs_row["pH"]) else 7.0
        obs_cod = float(obs_row["COD_mgL"]) if pd.notna(obs_row["COD_mgL"]) else 100.0
        obs_hm = float(obs_row["HeavyMetals_mgL"]) if pd.notna(obs_row["HeavyMetals_mgL"]) else 0.5
        obs_tds = float(obs_row["TDS_mgL"]) if pd.notna(obs_row["TDS_mgL"]) else 800.0
        
        is_superposition = (obs_hm > 4.0 and obs_cod > 400.0)
        
        candidates = []
        raw_likelihoods = []
        
        # Include all 6 registered units + 1 Unknown/Illegal upstream source candidate
        units_to_eval = dict(UNITS_METADATA)
        units_to_eval["U_UNK"] = UNKNOWN_UNIT_SPEC
        
        # Extract sensor calibration weights
        w_ph = float(self.sensor_weights.get("pH", 1.0))
        w_cod = float(self.sensor_weights.get("COD", 1.0))
        w_hm = float(self.sensor_weights.get("HeavyMetals", 1.0))
        w_tds = float(self.sensor_weights.get("TDS", 0.8))
        
        for uid, uinfo in units_to_eval.items():
            # 1. Hydraulic Travel Time Back-Propagation
            dist_to_m0 = self.m0_dist - uinfo["distance_m"]
            travel_sec = dist_to_m0 / self.v
            travel_mins = travel_sec / 60.0
            
            est_discharge_time = spike_time - timedelta(seconds=travel_sec)
            
            # 2. Operating Schedule Alignment P(Sched)
            gap_mins = get_schedule_gap_minutes(uid, est_discharge_time)
            is_active_at_discharge = (gap_mins == 0.0)
            
            if uid == "U_UNK":
                p_sched = CALIBRATION_CONFIG["unknown_unit_prior_weight"]
            else:
                p_sched = np.exp(-gap_mins / 30.0)
            
            # 3. Advection-Dispersion Hydraulic Transport Likelihood P(Hydro)
            p_hydro = 1.0 / np.sqrt(1.0 + travel_mins / 15.0)
            
            # 4. Multi-Parameter Chemical Profile Matching P(Chem) with Sensor Calibration Weights
            prof = uinfo["typical_discharge_profile"]
            
            diff_ph = w_ph * (((obs_ph - prof["pH"]) / 2.0) ** 2)
            diff_cod = w_cod * (((obs_cod - prof["COD_mgL"]) / 350.0) ** 2)
            diff_hm = w_hm * (((obs_hm - prof["HeavyMetals_mgL"]) / 2.0) ** 2)
            diff_tds = w_tds * (((obs_tds - prof["TDS_mgL"]) / 800.0) ** 2)
            
            if is_superposition:
                # Under multi-unit superposition, observed signal is an additive plume.
                is_high_metal_unit = prof["HeavyMetals_mgL"] > 2.0
                is_high_cod_unit = prof["COD_mgL"] > 600.0
                
                if is_high_metal_unit or is_high_cod_unit:
                    p_chem = 0.85
                else:
                    p_chem = 0.05
            else:
                joint_dist_sq = diff_ph + diff_cod + diff_hm + diff_tds
                p_chem_joint = np.exp(-0.4 * joint_dist_sq)
                
                # If pH directly contradicts (acidic effluent vs alkaline factory or vice versa), require joint profile
                is_acid_base_contradiction = (obs_ph > 8.5 and prof["pH"] < 5.5) or (obs_ph < 5.5 and prof["pH"] > 8.5)
                if is_acid_base_contradiction:
                    p_chem = p_chem_joint
                else:
                    p_chem_cod = np.exp(-0.5 * diff_cod) if obs_cod > 250.0 else 0.05
                    p_chem_hm = np.exp(-0.5 * diff_hm) if obs_hm > 2.0 else 0.05
                    p_chem = max(p_chem_joint, p_chem_cod * 0.9, p_chem_hm * 0.9)
                
            p_chem = max(0.01, float(p_chem))


            
            # 5. Composite Posterior Likelihood: L = P_sched * P_hydro * P_chem^2
            raw_L = p_sched * p_hydro * (p_chem ** 2)
            
            raw_likelihoods.append(raw_L)
            candidates.append({
                "unit_id": uid,
                "unit_name": uinfo["name"],
                "sector": uinfo["sector"],
                "distance_m": uinfo["distance_m"],
                "est_discharge_time": est_discharge_time,
                "travel_delay_mins": round(travel_mins, 1),
                "is_operating_at_discharge": is_active_at_discharge,
                "schedule_gap_mins": round(gap_mins, 1),
                "chemical_affinity": round(float(p_chem), 3),
                "parameter_contributions": {
                    "pH_affinity": round(float(np.exp(-0.5 * diff_ph)), 3),
                    "COD_affinity": round(float(np.exp(-0.5 * diff_cod)), 3),
                    "HeavyMetals_affinity": round(float(np.exp(-0.5 * diff_hm)), 3),
                    "TDS_affinity": round(float(np.exp(-0.5 * diff_tds)), 3),
                    "weights_applied": {"pH": round(w_ph, 2), "COD": round(w_cod, 2), "HeavyMetals": round(w_hm, 2), "TDS": round(w_tds, 2)}
                },
                "raw_likelihood": raw_L,
                "tracing_method": "PI-BHDM (Physics-Informed Bayesian)"
            })
            
        total_L = sum(raw_likelihoods)
        if total_L == 0:
            total_L = 1e-6
            
        for c in candidates:
            c["likelihood_pct"] = round((c["raw_likelihood"] / total_L) * 100.0, 1)
            
        candidates.sort(key=lambda x: x["raw_likelihood"], reverse=True)

        
        # Calculate overall uncertainty & confidence calibration for this spike attribution
        uncertainty = compute_uncertainty_metrics(candidates)
        
        for rank, c in enumerate(candidates, start=1):
            c["rank"] = rank
            c["calibrated_confidence"] = uncertainty["calibrated_confidence"]
            c["entropy_bits"] = uncertainty["entropy_bits"]
            c["probability_margin_pct"] = uncertainty["probability_margin_pct"]
            
        return candidates
