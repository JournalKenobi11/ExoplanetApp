import re
import pandas as pd
import streamlit as st

from inference import run_inference

st.set_page_config(
    page_title="TransitAI",
    layout="wide"
)

st.title("TransitAI - Exoplanet Candidate Finder")

text = st.text_area(
    "Enter TIC IDs",
    height=200,
    placeholder="123456789\n987654321\nor comma separated"
)

if st.button("Analyze"):

    lines = re.split(r'[,\s]+', text)

    tic_ids = []

    for line in lines:

        line = line.strip()

        if line:

            try:
                tic_ids.append(int(line))

            except:
                pass

    if len(tic_ids) == 0:

        st.error("No valid TIC IDs found")

    else:

        results = []

        progress = st.progress(0)

        status = st.empty()

        for idx, tic in enumerate(tic_ids):

            status.text(
                f"Processing TIC {tic}"
            )

            try:

                result = run_inference(tic)

                results.append(result)

            except Exception as exc:

                results.append({
                    "tic_id": tic,
                    "probability": -1.0,
                    "candidate": False,
                    "snr": 0.0,
                    "period": 0.0,
                    "bls_power": 0.0,
                    "error": str(exc),
                })

            progress.progress(
                (idx + 1) / len(tic_ids)
            )

        df = pd.DataFrame(results)

        df = df.sort_values(
            by="probability",
            ascending=False
        )

        st.success("Inference completed")

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(index=False)

        st.download_button(
            label="Export CSV",
            data=csv,
            file_name="transitai_results.csv",
            mime="text/csv"
        )