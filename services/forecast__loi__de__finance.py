# backend/services/forecast_loi_finance.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
import json
import traceback

def forecast_regional(file_path):
    try:
        data = pd.read_excel(file_path)
        data["Année"] = data["Année"].astype(int)
        data = data.sort_values(["Région", "Année"])

        regions = data["Région"].unique()
        predictions = []

        for region in regions:
            df = data[data["Région"] == region].copy()
            required_cols = ["Année", "Budget_Santé", "Population", "Croissance", "Dépenses_Santé"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Colonnes manquantes pour la région {region}: {required_cols}")

            X = df[["Année", "Budget_Santé", "Population", "Croissance"]]
            y = df["Dépenses_Santé"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Prévision 3 prochaines années
            last_year = df["Année"].max()
            next_years = pd.DataFrame({
                "Année": [last_year + i for i in range(1, 4)],
                "Budget_Santé": [df["Budget_Santé"].iloc[-1] * (1 + 0.05*i) for i in range(1, 4)],
                "Population": [df["Population"].iloc[-1] * (1 + 0.015*i) for i in range(1, 4)],
                "Croissance": [df["Croissance"].iloc[-1]]*3
            })
            next_pred = model.predict(next_years)
            next_years["Dépenses_Prédites"] = next_pred
            next_years["Région"] = region
            predictions.append(next_years)

        forecast_df = pd.concat(predictions)
        print(json.dumps(forecast_df.to_dict(orient="records")))

    except Exception as e:
        # Toujours renvoyer un JSON même en cas d'erreur
        error_info = {
            "error": str(e),
            "trace": traceback.format_exc()
        }
        print(json.dumps(error_info))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python forecast_loi_finance.py <excel_file>"}))
        sys.exit(1)
    forecast_regional(sys.argv[1])
