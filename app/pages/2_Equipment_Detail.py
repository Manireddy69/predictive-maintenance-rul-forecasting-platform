from __future__ import annotations

import streamlit as st

from app.dashboard_data import load_prediction_table, load_schedule_table


st.title("Equipment Detail")
st.caption("Inspect the latest forecast and scheduled action for each asset.")

predictions = load_prediction_table()
schedule = load_schedule_table()

unit_options = sorted(predictions["unit"].astype(int).unique().tolist()) if "unit" in predictions.columns else [1]
selected_unit = st.selectbox("Equipment Unit", options=unit_options)

unit_predictions = predictions[predictions["unit"].astype(int) == int(selected_unit)].copy() if "unit" in predictions.columns else predictions.copy()
unit_schedule = schedule[schedule["equipment_id"] == f"unit_{selected_unit}"].copy() if "equipment_id" in schedule.columns else schedule.copy()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Forecast History")
    st.line_chart(unit_predictions.set_index("target_cycle")[["prediction"]] if {"target_cycle", "prediction"}.issubset(unit_predictions.columns) else unit_predictions)
with col2:
    st.subheader("Maintenance Recommendation")
    st.dataframe(unit_schedule, use_container_width=True)

st.subheader("Prediction Records")
st.dataframe(unit_predictions, use_container_width=True)
