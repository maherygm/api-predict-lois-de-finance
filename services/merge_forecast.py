#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_forecast_all_in_one.py
-----------------------------------
Fusion des scripts :
- forecast_loi_finance.py
- prophet_budget_national.py
- merge_forecast.py

Utilisation :
$ python3 merge_forecast_all_in_one.py data.xlsx
"""

import sys
import json
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet


# ============================================================
# 🔹 1. Forecast Régional (RandomForest)
# ============================================================
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
        return forecast_df.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# ============================================================
# 🔹 2. Forecast National (Prophet)
# ============================================================
def forecast_national(file_path):
    try:
        data = pd.read_excel(file_path)
        df = data.groupby("Année")[["Dépenses_Santé"]].sum().reset_index()
        df.rename(columns={"Année": "ds", "Dépenses_Santé": "y"}, inplace=True)

        model = Prophet(yearly_seasonality=False)
        df["ds"] = pd.to_datetime(df["ds"], format="%Y")
        model.fit(df)

        future = model.make_future_dataframe(periods=3, freq="Y")
        forecast = model.predict(future)

        result = forecast[["ds", "yhat"]].tail(3)
        result["ds"] = result["ds"].dt.year
        return result.rename(columns={"ds": "Année", "yhat": "Dépenses_Prédites"}).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# ============================================================
# 🔹 3. Fusion des deux prévisions
# ============================================================
def merge_forecasts(file_path):
    try:
        rf_result = forecast_regional(file_path)
        prophet_result = forecast_national(file_path)

        output = {
            "forecast_regional": rf_result,
            "forecast_national": prophet_result
        }

        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "trace": traceback.format_exc()
        }))


# ============================================================
# 🔹 4. Main
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python merge_forecast_all_in_one.py <excel_file>"}))
        sys.exit(1)

    merge_forecasts(sys.argv[1])
