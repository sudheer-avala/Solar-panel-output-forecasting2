from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Solar Output Forecasting", page_icon="☀️", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "solar_weather_data.csv"
FEATURES = ["temperature_c", "humidity_pct", "wind_speed_ms", "irradiance_wm2", "cloud_cover_pct", "panel_temperature_c"]
TARGET = "daily_energy_kwh"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # Median imputation handles occasional missing sensor readings in the demo data.
    data[FEATURES] = data[FEATURES].apply(lambda col: col.fillna(col.median()))
    data[TARGET] = data[TARGET].fillna(data[TARGET].median())
    return data

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=2, loss="squared_error", random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {"MAE": mean_absolute_error(y_test, predictions), "RMSE": mean_squared_error(y_test, predictions) ** 0.5, "R²": r2_score(y_test, predictions)}
    test = pd.DataFrame({"Actual (kWh)": y_test, "Predicted (kWh)": predictions}).sort_index()
    return model, metrics, test

st.title("☀️ Solar Panel Output Forecasting")
st.caption("Gradient Boosting Regression • Renewable-energy planning dashboard")
df = load_data()
model, metrics, test_results = train_model(df)

with st.sidebar:
    st.header("Forecast inputs")
    st.caption("Adjust expected weather and panel conditions.")
    temperature = st.slider("Ambient temperature (°C)", -5.0, 48.0, 29.0, 0.5)
    humidity = st.slider("Relative humidity (%)", 5, 100, 48)
    wind = st.slider("Wind speed (m/s)", 0.0, 20.0, 3.0, 0.1)
    irradiance = st.slider("Solar irradiance (W/m²)", 0, 1100, 650, 10)
    cloud = st.slider("Cloud cover (%)", 0, 100, 25)
    panel_temp = st.slider("Panel temperature (°C)", -5.0, 85.0, 42.0, 0.5)
    system_kw = st.number_input("Installed capacity (kW)", min_value=0.1, max_value=10000.0, value=5.0, step=0.5)

input_row = pd.DataFrame([[temperature, humidity, wind, irradiance, cloud, panel_temp]], columns=FEATURES)
prediction = max(0.0, float(model.predict(input_row)[0]))
capacity_adjusted = prediction * (system_kw / 5.0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast daily generation", f"{capacity_adjusted:,.1f} kWh", help="Model output scaled from the reference 5 kW sample system to the capacity you entered.")
c2.metric("Test MAE", f"{metrics['MAE']:.2f} kWh")
c3.metric("Test RMSE", f"{metrics['RMSE']:.2f} kWh")
c4.metric("Test R²", f"{metrics['R²']:.3f}")
st.info("Demo notice: the included dataset is synthetic example data, not measured field data. Replace it with validated local weather-station and solar-meter readings before operational use. Capacity scaling is a simple prototype assumption.")

left, right = st.columns(2)
with left:
    st.subheader("Historical daily generation")
    chart_df = df.sort_values("date").tail(120)
    st.plotly_chart(px.line(chart_df, x="date", y=TARGET, labels={TARGET: "Energy (kWh)", "date": "Date"}, title="Last 120 sample days"), use_container_width=True)
with right:
    st.subheader("Weather and generation")
    st.plotly_chart(px.scatter(df, x="irradiance_wm2", y=TARGET, color="cloud_cover_pct", hover_data=["date"], labels={"irradiance_wm2":"Irradiance (W/m²)", TARGET:"Energy (kWh)", "cloud_cover_pct":"Cloud cover (%)"}, title="Generation vs solar irradiance"), use_container_width=True)

st.subheader("Model evaluation: held-out test data")
actual_pred = test_results.reset_index(drop=True).head(100)
st.plotly_chart(px.scatter(actual_pred, x="Actual (kWh)", y="Predicted (kWh)", title="Actual vs predicted generation", trendline="ols"), use_container_width=True)
with st.expander("View sample data"):
    st.dataframe(df.tail(30), use_container_width=True)
    st.download_button("Download dataset CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="solar_weather_data.csv", mime="text/csv")
with st.expander("Feature importance"):
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", title="Gradient Boosting feature importance"), use_container_width=True)

st.markdown("---")
st.caption("Prototype for academic demonstration. Predictions are estimates and should not be used for grid operations or financial commitments without real data validation.")
