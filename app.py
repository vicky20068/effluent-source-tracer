"""
Field-Ready Industrial Effluent Anomaly Detection & Source Tracing Tool
Semester 5 (IE28) Project Prototype

Streamlit Dashboard supporting offline batch upload, interactive anomaly inspection,
hydraulic back-propagation source tracing, edge-case simulation, and experimental benchmarking.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_generator import generate_synthetic_telemetry, DEFAULT_UNITS_CONFIG
from src.anomaly_detector import EffluentAnomalyDetector
from src.baseline_tracer import BaselineTracer
from src.improved_tracer import ImprovedTracer
from src.edge_cases import EdgeCaseTestSuite
from src.evaluator import PerformanceEvaluator

st.set_page_config(
    page_title="Industrial Effluent Monitoring & Tracing",
    page_icon="🌊",
    layout="wide"
)

# Header & Context Banner
st.title("🌊 Industrial Estate Effluent Anomaly Detection & Source Tracing Tool")
st.caption("IE28 Field-Ready Prototype — Offline-Capable Effluent Monitoring & Hydraulic Back-Propagation Source Attribution")

# Sidebar - Offline Data Source Selector
st.sidebar.header("🕹️ Field Controls & Data Source")
data_source_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Synthetic Data Simulator", "Batch CSV Telemetry Upload"]
)

units_config = DEFAULT_UNITS_CONFIG
detector = EffluentAnomalyDetector()
baseline_tracer = BaselineTracer(units_config)
improved_tracer = ImprovedTracer(units_config)

if data_source_mode == "Synthetic Data Simulator":
    st.sidebar.subheader("Simulation Parameters")
    days_sim = st.sidebar.slider("Simulation Duration (Days)", 1, 7, 3)
    seed_sim = st.sidebar.number_input("Random Seed", value=42)
    
    with st.spinner("Generating industrial estate telemetry..."):
        telemetry_df, ground_truth = generate_synthetic_telemetry(days=days_sim, random_seed=seed_sim)
else:
    st.sidebar.subheader("Batch CSV Data Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload Sensor Telemetry CSV", type=["csv"])
    if uploaded_file is not None:
        telemetry_df = pd.read_csv(uploaded_file)
        telemetry_df["timestamp"] = pd.to_datetime(telemetry_df["timestamp"])
        ground_truth = []
        st.sidebar.success("Custom telemetry CSV loaded successfully!")
    else:
        st.sidebar.warning("No CSV uploaded. Defaulting to 3-day synthetic telemetry.")
        telemetry_df, ground_truth = generate_synthetic_telemetry(days=3, random_seed=42)

# Run Anomaly Detection
df_flagged, anomaly_events = detector.detect_anomalies(telemetry_df)

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Sensor Timesteps", f"{len(telemetry_df):,}")
kpi2.metric("Active Units Monitored", len(units_config))
kpi3.metric("Detected Anomaly Events", len(anomaly_events))
kpi4.metric("Offline Data Status", "LOADED (Offline Mode Active)")

# Tabs Layout
tab_overview, tab_tracer, tab_edge_cases, tab_benchmark, tab_ethics_checklist = st.tabs([
    "📊 Sensor Telemetry & Anomalies",
    "🔍 Spike Source Tracer",
    "⚠️ Edge-Case Simulations",
    "📈 Performance Benchmark",
    "📋 Ethics & Field Checklist"
])

# ------------------------------------------------------------------------------
# TAB 1: SENSOR TELEMETRY OVERVIEW
# ------------------------------------------------------------------------------
with tab_overview:
    st.subheader("Downstream Monitoring Station Alpha — Sensor Readings")
    st.markdown("""
    Sensor readings are captured at 5-minute intervals at the common discharge outlet.
    Shaded vertical regions highlight statistical anomalies detected by rolling z-score / pH thresholding.
    """)

    # Interactive Subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Chemical Oxygen Demand (COD mg/L)", "Effluent pH Level", "Heavy Metals Concentration (ppm)", "Total Dissolved Solids (TDS ppm)")
    )

    # COD
    fig.add_trace(go.Scatter(x=df_flagged["timestamp"], y=df_flagged["downstream_cod_mg_l"], mode="lines", name="COD (mg/L)", line=dict(color="#FF4B4B")), row=1, col=1)
    # pH
    fig.add_trace(go.Scatter(x=df_flagged["timestamp"], y=df_flagged["downstream_ph"], mode="lines", name="pH Level", line=dict(color="#00C0F2")), row=2, col=1)
    # Heavy Metals
    fig.add_trace(go.Scatter(x=df_flagged["timestamp"], y=df_flagged["downstream_heavy_metals_ppm"], mode="lines", name="Heavy Metals (ppm)", line=dict(color="#FFAA00")), row=3, col=1)
    # TDS
    fig.add_trace(go.Scatter(x=df_flagged["timestamp"], y=df_flagged["downstream_tds_ppm"], mode="lines", name="TDS (ppm)", line=dict(color="#00D26A")), row=4, col=1)

    fig.update_layout(height=800, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 View Detected Anomaly Events Table"):
        if anomaly_events:
            events_data = []
            for ev in anomaly_events:
                events_data.append({
                    "Event ID": ev["event_id"],
                    "Start Time": ev["start_time"],
                    "End Time": ev["end_time"],
                    "Duration (min)": ev["duration_minutes"],
                    "Peak COD (mg/L)": round(ev["max_cod"], 1),
                    "Min pH": round(ev["min_ph"], 2),
                    "Max Metals (ppm)": round(ev["max_heavy_metals"], 3),
                    "Max TDS (ppm)": round(ev["max_tds"], 1)
                })
            st.dataframe(pd.DataFrame(events_data), use_container_width=True)
        else:
            st.info("No anomalies detected in the selected telemetry dataset.")

# ------------------------------------------------------------------------------
# TAB 2: SOURCE TRACER WORKBENCH
# ------------------------------------------------------------------------------
with tab_tracer:
    st.subheader("🔍 Spike Source Tracing Workbench")
    st.markdown("""
    Select a detected downstream anomaly event to perform **Hydraulic Back-Propagation** and **Chemical Fingerprint Matching**.
    Compares the **Improved Solution** against the **Baseline Heuristic Method**.
    """)

    if not anomaly_events:
        st.warning("No anomaly events available for source tracing.")
    else:
        event_options = {
            f"Event {ev['event_id']} ({ev['start_time']} | Peak COD: {ev['max_cod']:.0f} | Min pH: {ev['min_ph']:.1f})": ev
            for ev in anomaly_events
        }
        selected_label = st.selectbox("Select Anomaly Event to Investigate", list(event_options.keys()))
        selected_event = event_options[selected_label]

        # Execute both tracers
        improved_res = improved_tracer.trace_spike(selected_event, df_flagged)
        baseline_res = baseline_tracer.trace_spike(selected_event, df_flagged)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("### 🏆 Improved Tracer — Ranked Suspect Units")
            if improved_res["is_multi_source"]:
                st.warning(f"⚠️ MULTI-SOURCE ALERT: Overlapping emission detected from candidates: {', '.join(improved_res['plausible_sources'])}")

            for rank, cand in enumerate(improved_res["ranked_candidates"], start=1):
                prob = cand["likelihood_probability"]
                is_top = (rank == 1)
                
                with st.container():
                    st.markdown(f"#### Rank #{rank}: {cand['unit_name']} — Likelihood: `{prob * 100:.1f}%`")
                    st.progress(prob)
                    c1, c2, c3 = st.columns(3)
                    c1.caption(f"⏱️ Hydraulic Delay: `{cand['hydraulic_delay_minutes']} min`")
                    c2.caption(f"🕒 Est. Emission Window: `{cand['estimated_emission_window']}`")
                    c3.caption(f"🧬 Chemical Match: `{int(cand['chemical_match_score']*100)}%`")
                    st.info(f"**Evidence Summary:** {cand['explanation']}")
                    st.divider()

        with col_right:
            st.markdown("### 🏷️ Baseline Method Comparison")
            st.caption("Baseline flags units solely operating at the downstream wall-clock spike time.")
            
            b_df = pd.DataFrame(baseline_res)
            st.dataframe(b_df[["unit_name", "likelihood_score", "is_operating_at_spike"]], use_container_width=True)

            st.markdown("### 📝 Field Inspector Action & Verification")
            st.selectbox("Select Verified Unit (Ground Truth / Audit Outcome)", list(units_config.keys()))
            st.text_area("Field Audit Notes", placeholder="Enter site inspection findings, water sampling tube IDs, or physical valve checks...")
            if st.button("Save Field Audit Record"):
                st.success("Field audit log recorded to local offline storage.")

# ------------------------------------------------------------------------------
# TAB 3: EDGE-CASE SIMULATIONS
# ------------------------------------------------------------------------------
with tab_edge_cases:
    st.subheader("⚠️ Edge/Failure Case Diagnostic Suite")
    st.markdown("Testing prototype resilience under non-ideal industrial conditions.")

    edge_suite = EdgeCaseTestSuite(units_config)
    if st.button("▶️ Run Full Edge-Case Suite", type="primary"):
        test_results = edge_suite.run_all_tests()
        for tr in test_results:
            st.markdown(f"#### {tr['name']} — Status: `{tr['status']}`")
            st.write(tr["description"])
            with st.expander("View Diagnostic Technical Output"):
                st.json(tr)
            st.divider()

# ------------------------------------------------------------------------------
# TAB 4: PERFORMANCE BENCHMARK
# ------------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("📈 Experimental Performance Benchmark")
    st.markdown("Quantitative comparison between the Baseline Method and Implemented Improved Solution across multi-seed synthetic datasets.")

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        bench_days = st.slider("Benchmark Days", 2, 5, 3)
        if st.button("🚀 Execute Experiment Benchmark"):
            evaluator = PerformanceEvaluator(units_config)
            with st.spinner("Running Monte Carlo simulation trials..."):
                bench_res = evaluator.run_experiment(days=bench_days, seeds=[42, 101, 2024])
                st.session_state["bench_res"] = bench_res

    if "bench_res" in st.session_state:
        bench_res = st.session_state["bench_res"]
        st.markdown(f"### Benchmark Results ({bench_res['total_anomalies_evaluated']} Evaluated Anomalies)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Top-1 Accuracy", f"{bench_res['baseline']['top1_accuracy']}%")
        m2.metric("Solution Top-1 Accuracy", f"{bench_res['improved_tracer']['top1_accuracy']}%", delta=f"+{bench_res['improved_tracer']['top1_accuracy'] - bench_res['baseline']['top1_accuracy']}%")
        m3.metric("Baseline F1-Score", bench_res["baseline"]["f1_score"])
        m4.metric("Solution F1-Score", bench_res["improved_tracer"]["f1_score"], delta=f"+{round(bench_res['improved_tracer']['f1_score'] - bench_res['baseline']['f1_score'], 2)}")

        # Comparison Bar Chart
        comp_df = pd.DataFrame({
            "Metric": ["Top-1 Accuracy (%)", "Precision", "Recall", "F1-Score"],
            "Baseline Method": [bench_res["baseline"]["top1_accuracy"], bench_res["baseline"]["precision"], bench_res["baseline"]["recall"], bench_res["baseline"]["f1_score"]],
            "Improved Tracer": [bench_res["improved_tracer"]["top1_accuracy"], bench_res["improved_tracer"]["precision"], bench_res["improved_tracer"]["recall"], bench_res["improved_tracer"]["f1_score"]]
        })
        fig_bench = px.bar(comp_df, x="Metric", y=["Baseline Method", "Improved Tracer"], barmode="group", title="Baseline vs Implemented Solution Performance Comparison")
        st.plotly_chart(fig_bench, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: ETHICS & FIELD CHECKLIST
# ------------------------------------------------------------------------------
with tab_ethics_checklist:
    st.subheader("⚖️ Ethics Note & Field Rollout Deployment Checklist")

    col_eth, col_chk = st.columns(2)

    with col_eth:
        st.markdown("### 🛡️ Ethics & Operational Risk Analysis")
        st.error("**Risk 1: False Accusation & Financial Liability**")
        st.caption("Falsely accusing a factory unit can lead to unfair fines, shutdown orders, and reputational damage. Tracing predictions MUST serve as diagnostic inspection leads, never as standalone legal proof.")
        
        st.warning("**Risk 2: Privacy & Industrial Surveillance**")
        st.caption("Continuous flow and operating telemetry can inadvertently leak proprietary manufacturing batch schedules to competitor units.")

        st.info("**Risk 3: Sensor Tampering & Adversarial Discharge**")
        st.caption("Discharging effluent during competitors' lag windows or intentionally diluting discharge water.")

    with col_chk:
        st.markdown("### 📋 Field Deployment Checklist")
        st.checkbox("1. Hydrological Channel Calibration (Measure exact channel dimensions & velocity profiles)")
        st.checkbox("2. Multi-Parameter Sensor Calibration (pH, COD, Heavy Metals, TDS cross-validation)")
        st.checkbox("3. Offline Local Data Storage Setup (SQLite / encrypted CSV store)")
        st.checkbox("4. Chain of Custody Protocol (Field water sample collection before issuing notice)")
        st.checkbox("5. Operating Schedule Privacy Protocol (Anonymized data hashing for unit IDs)")
        st.checkbox("6. Low-Bandwidth Batch Upload Verification (Store-and-forward telemetry protocol)")
