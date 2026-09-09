# IE28 Industrial Estate Effluent Anomaly Detection & Source Tracing Prototype

A field-ready prototype designed for multi-unit industrial estate effluent monitoring, anomaly detection, and hydraulic back-propagation source tracing.

## Features
- **Synthetic Data Generator**: Realistic 5-minute interval sensor telemetry with unit operating schedules, chemical signatures (pH, COD, Heavy Metals, TDS), and hydraulic travel times.
- **Dynamic Anomaly Detector**: Rolling z-score and pH deviation thresholding for identifying pollution spikes.
- **Baseline Tracer Method**: Temporal proximity heuristic ("nearest operating unit") for benchmark comparison.
- **Improved Source Tracer**: Hydraulic travel-time back-propagation, operating schedule alignment, and multi-parameter chemical fingerprint matching.
- **Edge-Case Diagnostics**: 4 robust edge-case tests (concurrent discharge, sensor dropout, flow velocity shifts, unauthorized off-schedule night dumping).
- **Experimental Evaluator**: Automated benchmark suite reporting Top-1 Accuracy, List Recall, Precision, F1-Score, and Latency.
- **Field-Ready Dashboard (Streamlit)**: Offline-capable UI supporting batch CSV uploads, interactive spike inspection, edge-case tester, and field inspector audit logging.

## Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/
```

### 3. Run Benchmark Experiment
```bash
python -m src.evaluator
```

### 4. Launch Streamlit Field Dashboard
```bash
streamlit run app.py
```
View in browser at `http://localhost:8501`.
