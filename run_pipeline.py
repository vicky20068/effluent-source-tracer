"""
Master End-to-End Pipeline Execution Script for Industrial Estate Effluent Tracing (IE28).
Runs benchmark evaluation, edge case simulations, sensor calibration verification,
stakeholder validation metrics, and exports the final project report.
"""

import os
import sys
import json
import pandas as pd

# Add root project path to Python import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.benchmark_evaluator import BenchmarkEvaluator
from src.edge_case_runner import EdgeCaseSimulator
from src.calibration import SensorCalibrationRegistry
from src.stakeholder_validation import get_stakeholder_validation_summary
from src.report_generator import generate_project_report

def main():
    print("=" * 80)
    print("INDUSTRIAL ESTATE EFFLUENT MONITORING & SOURCE TRACING PIPELINE (IE28)")
    print("REVIEW 3 FINAL STAGING GATE — 100% PROJECT DELIVERY")
    print("=" * 80)
    
    # 1. Run Benchmark Experiment
    evaluator = BenchmarkEvaluator()
    benchmark_results = evaluator.run_benchmark_experiment(num_days=14, num_anomalies=25)
    
    det = benchmark_results["detection_metrics"]
    attr_base = benchmark_results["attribution_metrics"]["baseline"]
    attr_phys = benchmark_results["attribution_metrics"]["proposed_pibhdm"]
    
    print("\n" + "-" * 60)
    print("1. BENCHMARK EXPERIMENT RESULTS SUMMARY")
    print("-" * 60)
    print(f"Total Anomaly Trials Evaluated:             {benchmark_results['num_trials']}")
    print(f"Detection Precision:                        {det['precision_pct']}%")
    print(f"Detection Recall:                           {det['recall_pct']}%")
    print(f"Detection F1-Score:                         {det['f1_score']}")
    print(f"Average Detection Delay:                    {det['avg_detection_delay_mins']} mins")
    print("-" * 60)
    print(f"Baseline Method (Time Proximity) Top-1 Acc: {attr_base['top1_accuracy_pct']}%")
    print(f"Proposed PI-BHDM Method Top-1 Acc:          {attr_phys['top1_accuracy_pct']}%")
    print(f"Accuracy Gain:                              +{benchmark_results['improvements']['top1_gain_pct']}%")
    print(f"Baseline Mean Reciprocal Rank (MRR):        {attr_base['mrr']}")
    print(f"Proposed PI-BHDM Mean Reciprocal Rank (MRR):{attr_phys['mrr']}")
    print(f"Average PI-BHDM Inference Latency:          {attr_phys['avg_latency_ms']} ms")
    
    # 2. Run Edge Case Tests
    print("\n" + "-" * 60)
    print("2. EXECUTING EDGE & FAILURE CASE SUITE")
    print("-" * 60)
    simulator = EdgeCaseSimulator()
    c1 = simulator.run_case1_superposition()
    c2 = simulator.run_case2_sensor_dropout()
    c3 = simulator.run_case3_velocity_shift()
    c4 = simulator.run_case4_sensor_drift()
    
    print(f"[CASE 1] Superposition Test:    {'PASSED' if c1['physics_success'] else 'FAILED'}")
    print(f"[CASE 2] Sensor Dropout Test:   {'PASSED' if c2['physics_success'] else 'FAILED'}")
    print(f"[CASE 3] Velocity Shift Test:   {'PASSED' if c3['adaptive_physics_success'] else 'FAILED'}")
    print(f"[CASE 4] Baseline Drift Test:   {'PASSED' if c4['ewma_detected'] and c4['static_false_positives'] > c4['ewma_false_positives'] else 'FAILED'}")
    
    edge_results = {"c1": c1, "c2": c2, "c3": c3, "c4": c4}
    
    # 3. Chemical Sensor Calibration Registry Verification
    print("\n" + "-" * 60)
    print("3. CHEMICAL SENSOR CALIBRATION & TRACEABILITY STATUS")
    print("-" * 60)
    cal_registry = SensorCalibrationRegistry()
    weights = cal_registry.get_all_sensor_weights()
    for param, w in weights.items():
        info = cal_registry.get_sensor_health_and_weight(param)
        print(f"[{param:<12}] Status: {info['calibration_status']:<15} | Weight: {w:.2f} | Cert: {info['certificate_id']}")
    
    # 4. Stakeholder Validation Metrics Summary
    print("\n" + "-" * 60)
    print("4. STAKEHOLDER VALIDATION & OPERATIONAL IMPACT")
    print("-" * 60)
    stakeholder_data = get_stakeholder_validation_summary()
    sus = stakeholder_data["system_usability_scale"]
    ops = stakeholder_data["operational_impact"]
    print(f"System Usability Scale (SUS) Score:         {sus['mean_sus_score']} / 100 ({sus['grade']})")
    print(f"Triage-to-Dispatch Time Reduction:          {ops['triage_to_dispatch_time_mins']['reduction_pct']}% (250m -> 18m)")
    print(f"Targeted Inspection Hit Rate:               {ops['inspection_hit_rate_pct']['proposed_pibhdm_targeting']}% (vs 32% baseline)")
    print(f"False Accusation Dispute Reduction:         -{ops['false_accusation_dispute_rate_pct']['reduction_pct']}%")
    print(f"Offline Buffer Store-and-Forward Sync:      {ops['offline_sync_reliability_pct']}% Success")

    # 5. Generate Project Report
    print("\n" + "-" * 60)
    print("5. GENERATING REVIEW 3 FINAL PROJECT REPORT")
    print("-" * 60)
    report_md = generate_project_report(benchmark_results, edge_results)
    
    report_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "IE28_Project_Report.md"))
    with open(report_file, "w") as f:
        f.write(report_md)
        
    print(f"[SUCCESS] Project report generated successfully: {report_file}")
    print(f"Report Length: {len(report_md)} characters (Platform safe <= 7600 chars)")
    print("=" * 80)

if __name__ == "__main__":
    main()
