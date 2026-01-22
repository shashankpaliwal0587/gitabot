#!/bin/bash
# Run the ingestion script
python3 ingest.py

# Start the Streamlit app
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
