from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_data import load_schedule_summary


st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="PM", layout="wide")

summary = load_schedule_summary()

st.title("Predictive Maintenance Dashboard")
st.caption("Week 3 multipage scaffold for forecasting, scheduling, alerts, and reporting.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Scheduled Tasks", int(summary.get("task_count", 0)))
col2.metric("Total Cost", f"${float(summary.get('total_cost', 0.0)):,.0f}")
col3.metric("Risk Cost", f"${float(summary.get('total_risk_cost', 0.0)):,.0f}")
col4.metric("On-Time Rate", f"{float(summary.get('on_or_before_preferred_rate', 0.0)):.1%}")

st.markdown(
    """
    Use the pages in the sidebar to explore:

    - Overview: current schedule health and sensitivity analysis
    - Equipment Detail: per-unit forecast and maintenance recommendations
    - Alerts Configuration: thresholds and escalation rules
    - Reports: exported artifacts and reviewable summaries
    """
)
