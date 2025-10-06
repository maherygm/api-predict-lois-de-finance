# backend/services/merge_forecast.py
import sys
import subprocess
import json

def merge_forecasts(file_path):
    # Appel Random Forest
    rf_proc = subprocess.run(
        ["python3", "services/forecast_loi_finance.py", file_path],
        capture_output=True, text=True
    )
    rf_result = json.loads(rf_proc.stdout)

    # Appel Prophet
    prophet_proc = subprocess.run(
        ["python3", "services/prophet_budget_national.py", file_path],
        capture_output=True, text=True
    )
    prophet_result = json.loads(prophet_proc.stdout)

    # Fusionner dans un dictionnaire final
    output = {
        "forecast_regional": rf_result,
        "forecast_national": prophet_result
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_forecast.py <excel_file>")
        sys.exit(1)
    merge_forecasts(sys.argv[1])
