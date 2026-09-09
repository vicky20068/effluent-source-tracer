"""
Automated Pytest Suite for Industrial Effluent Monitoring & Source Tracing
"""

import pytest
import pandas as pd
from src.data_generator import generate_synthetic_telemetry, DEFAULT_UNITS_CONFIG
from src.anomaly_detector import EffluentAnomalyDetector
from src.baseline_tracer import BaselineTracer
from src.improved_tracer import ImprovedTracer
from src.edge_cases import EdgeCaseTestSuite
from src.evaluator import PerformanceEvaluator


def test_synthetic_data_generation():
    df, anomalies = generate_synthetic_telemetry(days=1)
    assert not df.empty
    assert "downstream_flow_lps" in df.columns
    assert "downstream_cod_mg_l" in df.columns
    assert len(anomalies) >= 2


def test_anomaly_detection():
    df, anomalies = generate_synthetic_telemetry(days=3)
    detector = EffluentAnomalyDetector()
    df_flagged, events = detector.detect_anomalies(df)
    assert "is_anomaly" in df_flagged.columns
    assert len(events) >= 3


def test_baseline_tracer():
    df, _ = generate_synthetic_telemetry(days=1)
    detector = EffluentAnomalyDetector()
    df_flagged, events = detector.detect_anomalies(df)
    
    baseline = BaselineTracer(DEFAULT_UNITS_CONFIG)
    res = baseline.trace_spike(events[0], df_flagged)
    assert isinstance(res, list)
    assert len(res) == len(DEFAULT_UNITS_CONFIG)


def test_improved_tracer():
    df, _ = generate_synthetic_telemetry(days=3)
    detector = EffluentAnomalyDetector()
    df_flagged, events = detector.detect_anomalies(df)

    tracer = ImprovedTracer(DEFAULT_UNITS_CONFIG)
    res = tracer.trace_spike(events[0], df_flagged)
    assert "top_source_unit" in res
    assert "plausible_sources" in res
    assert "ranked_candidates" in res


def test_edge_case_suite():
    suite = EdgeCaseTestSuite(DEFAULT_UNITS_CONFIG)
    results = suite.run_all_tests()
    assert len(results) == 4
    for r in results:
        assert r["status"].startswith("PASSED")


def test_evaluator_experiment():
    evaluator = PerformanceEvaluator(DEFAULT_UNITS_CONFIG)
    results = evaluator.run_experiment(days=2, seeds=[42])
    assert results["total_anomalies_evaluated"] > 0
    assert results["improved_tracer"]["top1_accuracy"] >= results["baseline"]["top1_accuracy"]
