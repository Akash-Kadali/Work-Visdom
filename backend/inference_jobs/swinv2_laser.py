# backend/inference_jobs/swinv2_laser.py
import os
from typing import Dict, Any
from backend.modal_request import classify_laser_swinv2_to_csv

OUTPUT_LASER2_FOLDER_ID = os.getenv("OUTPUT_LASER2_FOLDER_ID", "").strip()
OUTPUT_CLASSIFY_FOLDER_ID = os.getenv("OUTPUT_CLASSIFY_FOLDER_ID", "").strip()
CSV_NAME = os.getenv("LASER_SWINV2_CSV_NAME", "laser_swinv2_results.csv").strip()

def run_inference() -> Dict[str, Any]:
    return classify_laser_swinv2_to_csv(
        input_folder_id=OUTPUT_LASER2_FOLDER_ID,
        output_csv_folder_id=OUTPUT_CLASSIFY_FOLDER_ID,
        csv_name=CSV_NAME,
    )

if __name__ == "__main__":
    res = run_inference()
    print(res)
