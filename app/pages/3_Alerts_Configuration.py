from __future__ import annotations

import streamlit as st


st.title("Alerts Configuration")
st.caption("Configure thresholds that turn forecasts and schedules into actionable alerts.")

with st.form("alerts_form"):
    predicted_rul_threshold = st.slider("Predicted RUL alert threshold (days)", min_value=1, max_value=30, value=7)
    risk_cost_threshold = st.number_input("Risk cost escalation threshold", min_value=0, value=25_000, step=1_000)
    daily_downtime_threshold = st.number_input("Daily downtime SLA threshold (hours)", min_value=1.0, value=16.0, step=1.0)
    notify_email = st.text_input("Notification email", value="maintenance@example.com")
    submitted = st.form_submit_button("Save Alert Rules")

if submitted:
    st.success("Alert rule draft saved. Wire this form to persistence and notifications in the next iteration.")

st.info(
    "Recommended next step: connect these settings to scheduler outputs and trigger alerts when a unit is scheduled after its preferred day or predicted RUL falls below the configured threshold."
)
