"""
Benchmarking and Experiment Suite for Industrial Effluent Tracing.
Evaluates Baseline Method vs Physics-Informed Bayesian Model across synthetic & empirical trials.
Computes:
1. Disambiguated Detection Metrics: Precision, Recall, F1-Score, Detection Delay
2. Disambiguated Attribution Metrics: Top-1 Accuracy, Top-3 Accuracy, Mean Reciprocal Rank (MRR), Confusion Matrix
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from src.data_generator import generate_synthetic_dataset
from src.anomaly_detector import EWMACUSUMDetector, StaticBaselineDetector
from src.tracing_engine import BaselineTimeProximityTracer, PhysicsInformedBayesianTracer

class BenchmarkEvaluator:
    """Quantitative evaluation suite comparing Baseline vs PI-BHDM performance."""
    
    def __init__(self):
        self.baseline_tracer = BaselineTimeProximityTracer()
        self.physics_tracer = PhysicsInformedBayesianTracer()
        self.detector = EWMACUSUMDetector()
        
    def run_benchmark_experiment(self, num_days: int = 14, num_anomalies: int = 30) -> Dict[str, Any]:
        """Run systematic experiment across ground-truth anomalies."""
        print(f"[BENCHMARK] Generating synthetic dataset with {num_anomalies} anomalies across {num_days} days...")
        downstream_df, ground_truth_df = generate_synthetic_dataset(
            days=num_days,
            freq_minutes=5,
            num_anomalies=num_anomalies,
            random_seed=42
        )
        
        # 1. Evaluate Anomaly Detection Engine (Detection Metrics)
        annotated_df, detected_spikes = self.detector.detect_anomalies(downstream_df)
        
        # Match detected spikes to ground truth events to calculate TP, FP, FN
        matched_gt_ids = set()
        matched_spike_indices = set()
        detection_delays_min = []
        
        for gt_idx, gt_row in ground_truth_df.iterrows():
            gt_arrival_ts = gt_row["arrival_timestamp"]
            for s_i, spike in enumerate(detected_spikes):
                dt_sec = (spike["peak_time"] - gt_arrival_ts).total_seconds()
                if abs(dt_sec) <= 1800: # Detected within +- 30 minutes of ground truth arrival
                    matched_gt_ids.add(gt_row["anomaly_id"])
                    matched_spike_indices.add(s_i)
                    detection_delays_min.append(dt_sec / 60.0)
                    break
                    
        tp = len(matched_gt_ids)
        fp = len(detected_spikes) - len(matched_spike_indices)
        fn = len(ground_truth_df) - tp
        
        det_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        det_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        det_f1 = (2 * det_precision * det_recall) / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0.0
        avg_det_delay = float(np.mean(detection_delays_min)) if detection_delays_min else 0.0
        
        # 2. Evaluate Source Attribution Engine (Attribution Metrics)
        baseline_ranks = []
        physics_ranks = []
        baseline_latencies = []
        physics_latencies = []
        trial_records = []
        
        unit_ids = ["U1", "U2", "U3", "U4", "U5", "U6", "U_UNK"]
        confusion_matrix = {t: {p: 0 for p in unit_ids} for t in unit_ids}
        
        for gt_idx, gt_row in ground_truth_df.iterrows():
            gt_arrival_ts = gt_row["arrival_timestamp"]
            gt_unit = gt_row["true_unit_id"]
            
            # Find closest detected spike event
            matched_spike = None
            min_dt = float("inf")
            
            for spike in detected_spikes:
                dt = abs((spike["peak_time"] - gt_arrival_ts).total_seconds())
                if dt <= 1200 and dt < min_dt:
                    min_dt = dt
                    matched_spike = spike
                    
            if matched_spike is None:
                matched_spike = {
                    "peak_index": gt_row["arrival_step"],
                    "peak_time": gt_arrival_ts,
                    "trigger_parameter": gt_row["primary_parameter"],
                    "peak_value": gt_row["spike_magnitude"]
                }
                
            # Run Baseline Method
            t0 = time.perf_counter()
            b_candidates = self.baseline_tracer.trace_spike(matched_spike, annotated_df)
            t1 = time.perf_counter()
            baseline_latencies.append((t1 - t0) * 1000.0)
            
            # Run Proposed PI-BHDM Method
            t2 = time.perf_counter()
            p_candidates = self.physics_tracer.trace_spike(matched_spike, annotated_df)
            t3 = time.perf_counter()
            physics_latencies.append((t3 - t2) * 1000.0)
            
            # Find ranks
            b_rank = next((c["rank"] for c in b_candidates if c["unit_id"] == gt_unit), len(b_candidates))
            p_rank = next((c["rank"] for c in p_candidates if c["unit_id"] == gt_unit), len(p_candidates))
            
            baseline_ranks.append(b_rank)
            physics_ranks.append(p_rank)
            
            p_top1_unit = p_candidates[0]["unit_id"]
            if gt_unit in confusion_matrix and p_top1_unit in confusion_matrix[gt_unit]:
                confusion_matrix[gt_unit][p_top1_unit] += 1
                
            trial_records.append({
                "anomaly_id": gt_row["anomaly_id"],
                "true_unit": gt_unit,
                "arrival_time": gt_arrival_ts,
                "baseline_top1": b_candidates[0]["unit_id"],
                "baseline_rank": b_rank,
                "physics_top1": p_top1_unit,
                "physics_rank": p_rank,
                "physics_top1_pct": p_candidates[0]["likelihood_pct"],
                "calibrated_confidence": p_candidates[0]["calibrated_confidence"],
                "entropy_bits": p_candidates[0]["entropy_bits"],
                "margin_pct": p_candidates[0]["probability_margin_pct"],
                "trigger_param": gt_row["primary_parameter"]
            })
            
        trials_df = pd.DataFrame(trial_records)
        confusion_df = pd.DataFrame(confusion_matrix).T
        n_trials = len(trials_df)
        
        b_top1 = np.mean([1 if r == 1 else 0 for r in baseline_ranks]) * 100.0
        p_top1 = np.mean([1 if r == 1 else 0 for r in physics_ranks]) * 100.0
        
        b_top3 = np.mean([1 if r <= 3 else 0 for r in baseline_ranks]) * 100.0
        p_top3 = np.mean([1 if r <= 3 else 0 for r in physics_ranks]) * 100.0
        
        b_mrr = np.mean([1.0 / r for r in baseline_ranks])
        p_mrr = np.mean([1.0 / r for r in physics_ranks])
        
        results_summary = {
            "num_trials": n_trials,
            "detection_metrics": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision_pct": round(det_precision * 100.0, 1),
                "recall_pct": round(det_recall * 100.0, 1),
                "f1_score": round(float(det_f1), 3),
                "avg_detection_delay_mins": round(avg_det_delay, 1)
            },
            "attribution_metrics": {
                "baseline": {
                    "top1_accuracy_pct": round(b_top1, 1),
                    "top3_accuracy_pct": round(b_top3, 1),
                    "mrr": round(float(b_mrr), 3),
                    "avg_latency_ms": round(float(np.mean(baseline_latencies)), 2)
                },
                "proposed_pibhdm": {
                    "top1_accuracy_pct": round(p_top1, 1),
                    "top3_accuracy_pct": round(p_top3, 1),
                    "mrr": round(float(p_mrr), 3),
                    "avg_latency_ms": round(float(np.mean(physics_latencies)), 2)
                },
                "confusion_matrix_df": confusion_df
            },
            "improvements": {
                "top1_gain_pct": round(p_top1 - b_top1, 1),
                "mrr_gain": round(p_mrr - b_mrr, 3)
            },
            "trials_df": trials_df,
            "ground_truth_df": ground_truth_df,
            "downstream_df": downstream_df
        }
        
        return results_summary
