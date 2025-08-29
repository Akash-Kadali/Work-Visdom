# backend/inference_jobs/swinv2_critical.py
import os
from typing import Dict, Any
from backend.modal_request import classify_critical_swinv2_to_csv

# Defaults (override via env if you want a different file name or folders)
OUTPUT_CRITICAL_FOLDER_ID = os.getenv("OUTPUT_CRITICAL_FOLDER_ID", "").strip()
OUTPUT_CLASSIFY_FOLDER_ID = os.getenv("OUTPUT_CLASSIFY_FOLDER_ID", "").strip()
CSV_NAME = os.getenv("CRITICAL_SWINV2_CSV_NAME", "critical_swinv2_results.csv").strip()

def run_inference() -> Dict[str, Any]:
    return classify_critical_swinv2_to_csv(
        input_folder_id=OUTPUT_CRITICAL_FOLDER_ID,
        output_csv_folder_id=OUTPUT_CLASSIFY_FOLDER_ID,
        csv_name=CSV_NAME,
    )

if __name__ == "__main__":
    res = run_inference()
    print(res)
