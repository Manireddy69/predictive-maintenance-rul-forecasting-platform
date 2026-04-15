from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_data import latest_week2_schedule_json, latest_week2_sensitivity_csv


st.title("Reports")
st.caption("Review exportable artifacts from the latest checkpoint run.")

schedule_json = latest_week2_schedule_json()
sensitivity_csv = latest_week2_sensitivity_csv()

if schedule_json is not None:
    st.subheader("Latest Schedule JSON")
    payload = json.loads(schedule_json.read_text(encoding="utf-8"))
    st.json(payload.get("summary", payload))
    st.download_button(
        label="Download Schedule JSON",
        data=schedule_json.read_bytes(),
        file_name=schedule_json.name,
        mime="application/json",
    )
else:
    st.warning("No schedule JSON found yet. Run the week 2 checkpoint pipeline first.")

if sensitivity_csv is not None:
    st.subheader("Latest Sensitivity CSV")
    st.download_button(
        label="Download Sensitivity CSV",
        data=sensitivity_csv.read_bytes(),
        file_name=sensitivity_csv.name,
        mime="text/csv",
    )
else:
    st.warning("No sensitivity CSV found yet.")
