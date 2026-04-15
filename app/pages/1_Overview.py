from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_data import load_schedule_summary, load_sensitivity_table


st.title("Overview")
st.caption("Operational summary of the latest maintenance schedule and sensitivity analysis.")

summary = load_schedule_summary()
sensitivity = load_sensitivity_table()

col1, col2, col3 = st.columns(3)
col1.metric("Solver Status", str(summary.get("solver_status", "n/a")))
col2.metric("Tasks After Preferred Day", int(summary.get("tasks_after_preferred_day", 0)))
col3.metric("Technician Hours", float(summary.get("total_technician_hours", 0.0)))

st.subheader("Cost vs Risk Trade-off")
st.scatter_chart(sensitivity, x="total_risk_cost", y="total_cost", color="scenario")

st.subheader("Sensitivity Scenarios")
st.dataframe(sensitivity, use_container_width=True)
