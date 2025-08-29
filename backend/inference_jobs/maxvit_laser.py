# backend/inference_jobs/maxvit_laser.py
import os
from typing import Dict, Any
from backend.modal_request import classify_laser_maxvit_to_csv

OUTPUT_LASER2_FOLDER_ID = os.getenv("OUTPUT_LASER2_FOLDER_ID", "").strip()
OUTPUT_CLASSIFY_FOLDER_ID = os.getenv("OUTPUT_CLASSIFY_FOLDER_ID", "").strip()
CSV_NAME = os.getenv("LASER_MAXVIT_CSV_NAME", "laser_maxvit_results.csv").strip()

def run_inference() -> Dict[str, Any]:
    return classify_laser_maxvit_to_csv(
        input_folder_id=OUTPUT_LASER2_FOLDER_ID,
        output_csv_folder_id=OUTPUT_CLASSIFY_FOLDER_ID,
        csv_name=CSV_NAME,
    )

if __name__ == "__main__":
    res = run_inference()
    print(res)
