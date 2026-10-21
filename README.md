# Solar Panel Output Forecasting — Gradient Boosting Regression

A working academic prototype that forecasts daily solar energy generation from weather and panel sensor variables, evaluates a Gradient Boosting Regressor on held-out data, and displays results in an interactive dashboard.

## Features
- Gradient Boosting Regression (`scikit-learn`)
- 80/20 train/test split and MAE, RMSE, R² metrics
- Interactive weather/panel input sliders and daily-generation forecast
- Historical generation, irradiance relationship, actual-vs-predicted chart, feature importance
- Dataset preview and CSV download

## Run locally
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

## Dataset
`data/solar_weather_data.csv` contains 730 days of **synthetic demonstration data**. It is included so the prototype runs immediately; it is not real weather station or solar-panel sensor data. For a valid project evaluation, replace it with observed readings and keep the same columns:

- `date`
- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `irradiance_wm2`
- `cloud_cover_pct`
- `panel_temperature_c`
- `daily_energy_kwh` (target)

Ensure dates, units, missing values, outliers, and meter readings are checked before training. Avoid using future information as features. The sample capacity scaling in the dashboard is a simple prototype assumption, not a calibrated physical model.

## Methodology
1. Load and inspect the daily weather/panel dataset.
2. Use the six weather and panel variables as model features; predict `daily_energy_kwh`.
3. Split records into train/test partitions (80/20, fixed random seed for reproducibility).
4. Train `GradientBoostingRegressor` on training data.
5. Report MAE, RMSE and R² on held-out data.
6. Present a forecast and evaluation charts in the dashboard.

## Suggested project report chapters
1. Introduction and problem statement
2. Literature review
3. Requirements and system design
4. Dataset and preprocessing
5. Gradient Boosting methodology
6. Implementation and dashboard
7. Results and evaluation
8. Limitations and future enhancements
9. Conclusion
