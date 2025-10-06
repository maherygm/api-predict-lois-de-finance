# backend/services/prophet_budget_national.py
import pandas as pd
from prophet import Prophet
import sys
import json

def forecast_national(file_path):
    data = pd.read_excel(file_path)
    df = data.groupby("Année")[["Dépenses_Santé"]].sum().reset_index()
    df.rename(columns={"Année": "ds", "Dépenses_Santé": "y"}, inplace=True)

    model = Prophet(yearly_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=3, freq="Y")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat"]].tail(3)
    result["ds"] = result["ds"].dt.year
    print(json.dumps(result.rename(columns={"ds": "Année", "yhat": "Dépenses_Prédites"}).to_dict(orient="records")))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prophet_budget_national.py <excel_file>")
        sys.exit(1)
    forecast_national(sys.argv[1])
