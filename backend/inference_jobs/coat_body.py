# backend/inference_jobs/coat_body.py
import os
from typing import Dict, Any
from backend.modal_request import classify_body_coat_to_csv

OUTPUT_BODY_FOLDER_ID = os.getenv("OUTPUT_BODY_FOLDER_ID", "").strip()
OUTPUT_CLASSIFY_FOLDER_ID = os.getenv("OUTPUT_CLASSIFY_FOLDER_ID", "").strip()
CSV_NAME = os.getenv("BODY_COAT_CSV_NAME", "body_coat_results.csv").strip()

def run_inference() -> Dict[str, Any]:
    return classify_body_coat_to_csv(
        input_folder_id=OUTPUT_BODY_FOLDER_ID,
        output_csv_folder_id=OUTPUT_CLASSIFY_FOLDER_ID,
        csv_name=CSV_NAME,
    )

if __name__ == "__main__":
    res = run_inference()
    print(res)
