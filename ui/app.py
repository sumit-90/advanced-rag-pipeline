import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Advanced RAG Pipeline", layout="wide")
st.title("Advanced RAG Pipeline")
st.caption("A production-style RAG pipeline POC")

ingest_tab, query_tab, evaluate_tab = st.tabs(["Ingest", "Query", "Evaluate"])


# ─────────────────────────────────────────────
# Tab 1: Ingest
# ─────────────────────────────────────────────
with ingest_tab:
    st.header("Ingest Documents")
    st.info("Load documents from a subdirectory under `data/` into the vector store.")

    directory = st.text_input("Directory name", placeholder="e.g. my_docs")

    if st.button("Ingest", key="ingest_btn"):
        if not directory.strip():
            st.error("Please enter a directory name.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    response = requests.post(f"{API_URL}/ingest", json={"directory": directory})
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Ingestion successful!")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Documents Loaded", result["documents_loaded"])
                        col2.metric("Chunks Created", result["chunks_created"])
                        col3.metric("Collection", result["collection"])
                    else:
                        st.error(f"Ingestion failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")


# ─────────────────────────────────────────────
# Tab 2: Query
# ─────────────────────────────────────────────
with query_tab:
    st.header("Query the Pipeline")
    st.info("Ask a question and get an answer from your ingested documents.")

    query = st.text_area("Your question", placeholder="e.g. What is self-attention in transformers?", height=100)

    if st.button("Ask", key="query_btn"):
        if not query.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    response = requests.post(f"{API_URL}/query", json={"query": query})
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Answer generated!")

                        st.markdown("### Answer")
                        st.write(result["answer"])

                        st.markdown("### Sources")
                        for source in result["sources"]:
                            st.markdown(f"- `{source}`")

                        st.caption(f"Model: {result['model']}")
                    else:
                        st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")


# ─────────────────────────────────────────────
# Tab 3: Evaluate
# ─────────────────────────────────────────────
with evaluate_tab:
    st.header("Evaluate the Pipeline")
    st.info("Provide question and ground truth pairs to evaluate the pipeline using RAGAS metrics.")

    st.markdown("Add your evaluation samples in the table below. Click **+** to add a new row.")

    eval_df = st.data_editor(
        pd.DataFrame({"question": [""], "ground_truth": [""]}),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "ground_truth": st.column_config.TextColumn("Ground Truth", width="large"),
        }
    )

    if st.button("Evaluate", key="evaluate_btn"):
        valid_samples = eval_df.dropna().query("question != '' and ground_truth != ''").to_dict(orient="records")

        if not valid_samples:
            st.error("Please enter at least one complete question and ground truth pair.")
        else:
            with st.spinner(f"Running RAGAS evaluation on {len(valid_samples)} sample(s)..."):
                try:
                    response = requests.post(f"{API_URL}/evaluate", json={"eval_dataset": valid_samples})
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Evaluation complete!")
                        st.markdown("### RAGAS Scores")
                        st.json(result["results"])
                    else:
                        st.error(f"Evaluation failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
