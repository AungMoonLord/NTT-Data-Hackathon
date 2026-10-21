import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import simulate_signal  # noqa: E402

st.set_page_config(page_title="MechAgent", layout="wide")
st.title("MechAgent: Agentic AI Predictive Maintenance")
api = st.sidebar.text_input("API URL", os.environ.get("MECHAGENT_API", "")).rstrip("/")
machine = st.sidebar.selectbox("Machine", ["PUMP-01", "MOTOR-07", "FAN-03"])
bearing = "6208" if machine == "FAN-03" else "6205"
fault = st.sidebar.selectbox("Simulated fault", ["normal", "inner_race", "outer_race", "ball", "cage"], index=1)
severity = st.sidebar.slider("Severity (0 = new, 1 = failure)", 0.0, 1.0, 0.6, 0.05)
rpm = st.sidebar.slider("Shaft speed (RPM)", 900, 3600, 1800, 50)

if st.sidebar.button("Simulate & Analyze", use_container_width=True) and api:
    x = simulate_signal(fault, severity, float(rpm), np.random.default_rng(), bearing=bearing)
    st.session_state["signal"] = x
    r = requests.post(f"{api}/analyze", timeout=30, json={"machine_id": machine, "rpm": rpm, "fs": 16000,
                                                          "bearing": bearing, "signal": np.round(x, 4).tolist()})
    job_id = r.json()["job_id"]
    with st.status(f"Job {job_id}: running", expanded=True) as status:
        job = {}
        for _ in range(60):
            job = requests.get(f"{api}/reports/{job_id}", timeout=15).json().get("job", {})
            status.update(label=f"Job {job_id}: {job.get('status')}")
            if job.get("done"):
                break
            time.sleep(4)
    st.session_state["job"] = job

if "signal" in st.session_state:
    st.caption("Raw vibration: first 0.1 s")
    st.line_chart(pd.DataFrame({"acc (g)": st.session_state["signal"][:1600]}))

job = st.session_state.get("job")
if job:
    pred, s = job.get("prediction", {}), job.get("report", {}).get("summary", {})
    c = st.columns(5)
    c[0].metric("Fault location", s.get("fault_th", "-"))
    c[1].metric("Degradation", f"{s.get('degradation_pct', 0):.1f}%")
    c[2].metric("Replace within", s.get("replace_within", {}).get("text", "-"))
    c[3].metric("Urgency", s.get("urgency", "-"))
    c[4].metric("Status", job.get("status", "-"))
    t1, t2, t3, t4 = st.tabs(["Report", "Envelope spectrum", "Agent reasoning", "Raw JSON"])
    with t1:
        st.markdown(job.get("report", {}).get("markdown", "-"))
        if job.get("plan"):
            st.subheader("Agent plan (pending approval)")
            st.json(job["plan"])
    with t2:
        spec = pred.get("physics", {}).get("envelope_order_spectrum")
        if spec:
            st.line_chart(pd.DataFrame({"envelope amplitude": spec["amp"]}, index=pd.Index(spec["order"], name="order")))
            st.dataframe(pd.DataFrame(pred["physics"]["scores"]).T)
    with t3:
        ag = job.get("agent") or {}
        st.markdown(ag.get("final_text") or "Gatekeeper: no LLM call (healthy)")
        if ag.get("trace"):
            st.dataframe(pd.DataFrame(ag["trace"]))
            st.json(ag.get("usage", {}))
    with t4:
        st.json(job)
    