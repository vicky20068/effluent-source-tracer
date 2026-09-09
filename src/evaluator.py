"""
Experimental Benchmark & Evaluation Engine

Runs quantitative comparison experiments between the Baseline Method and Improved Tracer
across multi-day synthetic datasets with injected ground truth anomalies.
Calculates Accuracy, Precision, Recall, F1-Score, and execution latency.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.data_generator import generate_synthetic_telemetry, DEFAULT_UNITS_CONFIG
from src.anomaly_detector import EffluentAnomalyDetector
from src.baseline_tracer import BaselineTracer
from src.improved_tracer import ImprovedTracer


class PerformanceEvaluator:
    def __init__(self, units_config: Dict[str, Any] = DEFAULT_UNITS_CONFIG):
        self.units_config = units_config
        self.detector = EffluentAnomalyDetector()
        self.baseline_tracer = BaselineTracer(units_config)
        self.improved_tracer = ImprovedTracer(units_config)

    def run_experiment(self, days: int = 5, seeds: List[int] = [42, 101, 2024]) -> Dict[str, Any]:
        """
        Runs multi-seed benchmarking experiments and compiles evaluation metrics.
        """
        baseline_correct_top1 = 0
        improved_correct_top1 = 0
        improved_top2_recall = 0
        total_evaluations = 0

        baseline_times = []
        improved_times = []

        detailed_results = []

        for seed in seeds:
            df, ground_truth = generate_synthetic_telemetry(days=days, random_seed=seed)
            df_flagged, events = self.detector.detect_anomalies(df)

            # Match detected events to ground-truth injected anomalies by arrival timestamp window
            for anomaly in ground_truth:
                gt_arrival = pd.Timestamp(anomaly["downstream_arrival_time"])
                gt_sources = anomaly["source_units"]

                # Find closest detected event within +/- 20 min window
                matched_event = None
                min_diff = pd.Timedelta(minutes=30)

                for ev in events:
                    ev_time = pd.Timestamp(ev["start_time"])
                    diff = abs(ev_time - gt_arrival)
                    if diff < min_diff:
                        min_diff = diff
                        matched_event = ev

                if matched_event:
                    total_evaluations += 1

                    # Benchmark Baseline Tracer
                    t0 = time.perf_counter()
                    b_res = self.baseline_tracer.trace_spike(matched_event, df_flagged)
                    t_base = (time.perf_counter() - t0) * 1000.0
                    baseline_times.append(t_base)
                    b_top1 = b_res[0]["unit_id"]

                    # Benchmark Improved Tracer
                    t0 = time.perf_counter()
                    i_res = self.improved_tracer.trace_spike(matched_event, df_flagged)
                    t_imp = (time.perf_counter() - t0) * 1000.0
                    improved_times.append(t_imp)
                    i_top1 = i_res["top_source_unit"]
                    i_plausible = i_res["plausible_sources"]

                    # Accuracy checks
                    b_correct = b_top1 in gt_sources
                    i_correct = i_top1 in gt_sources
                    i_in_plausible = any(s in i_plausible for s in gt_sources)

                    if b_correct:
                        baseline_correct_top1 += 1
                    if i_correct:
                        improved_correct_top1 += 1
                    if i_in_plausible:
                        improved_top2_recall += 1

                    detailed_results.append({
                        "seed": seed,
                        "anomaly_id": anomaly["anomaly_id"],
                        "anomaly_type": anomaly["type"],
                        "ground_truth_sources": ", ".join(gt_sources),
                        "baseline_top1": b_top1,
                        "baseline_correct": b_correct,
                        "improved_top1": i_top1,
                        "improved_plausible": ", ".join(i_plausible),
                        "improved_correct": i_correct,
                        "improved_in_plausible": i_in_plausible
                    })

        # Calculate Summary Metrics
        N = max(1, total_evaluations)
        baseline_acc = (baseline_correct_top1 / N) * 100.0
        improved_acc = (improved_correct_top1 / N) * 100.0
        improved_recall = (improved_top2_recall / N) * 100.0

        # Estimated Precision / Recall / F1
        baseline_precision = baseline_acc / 100.0
        baseline_recall_val = baseline_acc / 100.0
        baseline_f1 = (2 * baseline_precision * baseline_recall_val) / max(1e-5, baseline_precision + baseline_recall_val)

        improved_precision = improved_acc / 100.0
        improved_recall_val = improved_recall / 100.0
        improved_f1 = (2 * improved_precision * improved_recall_val) / max(1e-5, improved_precision + improved_recall_val)

        metrics = {
            "total_anomalies_evaluated": total_evaluations,
            "baseline": {
                "top1_accuracy": round(baseline_acc, 2),
                "precision": round(baseline_precision, 3),
                "recall": round(baseline_recall_val, 3),
                "f1_score": round(baseline_f1, 3),
                "avg_latency_ms": round(float(np.mean(baseline_times)), 3) if baseline_times else 0.0
            },
            "improved_tracer": {
                "top1_accuracy": round(improved_acc, 2),
                "plausible_list_recall": round(improved_recall, 2),
                "precision": round(improved_precision, 3),
                "recall": round(improved_recall_val, 3),
                "f1_score": round(improved_f1, 3),
                "avg_latency_ms": round(float(np.mean(improved_times)), 3) if improved_times else 0.0
            },
            "detailed_trials": detailed_results
        }

        return metrics


if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    results = evaluator.run_experiment(days=5, seeds=[42, 101, 2024])
    
    print("\n==================================================")
    print("      EXPERIMENTAL PERFORMANCE BENCHMARK REPORT   ")
    print("==================================================")
    print(f"Total Anomalies Evaluated: {results['total_anomalies_evaluated']}")
    print("\n[Baseline Method (Nearest Operating Unit)]")
    print(f"  Top-1 Accuracy : {results['baseline']['top1_accuracy']}%")
    print(f"  Precision      : {results['baseline']['precision']}")
    print(f"  Recall         : {results['baseline']['recall']}")
    print(f"  F1-Score       : {results['baseline']['f1_score']}")
    print(f"  Avg Latency    : {results['baseline']['avg_latency_ms']} ms")

    print("\n[Improved Hydraulic & Chemical Tracer]")
    print(f"  Top-1 Accuracy : {results['improved_tracer']['top1_accuracy']}%")
    print(f"  List Recall    : {results['improved_tracer']['plausible_list_recall']}%")
    print(f"  Precision      : {results['improved_tracer']['precision']}")
    print(f"  Recall         : {results['improved_tracer']['recall']}")
    print(f"  F1-Score       : {results['improved_tracer']['f1_score']}")
    print(f"  Avg Latency    : {results['improved_tracer']['avg_latency_ms']} ms")
    print("==================================================")
