"""
forecasting.py
================
Module 2 of QuantumGrid.
Responsibility: predict near-term aggregate system demand from historical
consumption, calendar features, and temperature, then disaggregate that
single forecast into per-bus loads consumed by Module 3 (renewable
adjustment) and Module 5 (power balance constraint).
Design decision (justified in the accompanying explanation, not asserted):
gradient-boosted regression trees, not an LSTM, given the dataset size and
hackathon timeline. XGBoost is used if installed; otherwise this falls back
to scikit-learn's GradientBoostingRegressor, which implements the same
underlying algorithm family (Friedman, 2001) and requires no extra install.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGBOOST = False
# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------
def build_features(demand_pu: pd.Series, temperature: pd.Series = None) -> pd.DataFrame:
    """
    Build the feature matrix x(t) described in the mathematical formulation:
    lag-1h, lag-24h, lag-168h demand, calendar features, and (optionally)
    temperature. Rows with insufficient history for the longest lag (168h)
    are dropped rather than filled with guessed values.
    Parameters
    ----------
    demand_pu : per-unit demand series, hourly-indexed (output of Module 1)
    temperature : optional per-hour temperature series, same index.
                  If None, the temperature feature is omitted and this is
                  logged explicitly rather than silently zero-filled.
    """
    df = pd.DataFrame({"P_demand_pu": demand_pu})
    df["lag_1h"] = df["P_demand_pu"].shift(1)
    df["lag_24h"] = df["P_demand_pu"].shift(24)
    df["lag_168h"] = df["P_demand_pu"].shift(168)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    # US federal holiday calendar is a defensible default for the PJM
    # (Eastern US) footprint; swap out if using a different region's data.
    us_holidays = USFederalHolidayCalendar().holidays(
        start=df.index.min(), end=df.index.max()
    )
    df["is_holiday"] = df.index.normalize().isin(us_holidays).astype(int)
    if temperature is not None:
        df["temperature"] = temperature.reindex(df.index)
    else:
        print("WARNING: no temperature series supplied — training without "
              "the strongest known demand driver. Forecast quality will "
              "likely be degraded; this is stated explicitly rather than "
              "silently substituting a placeholder value.")
    df = df.dropna()
    return df
FEATURE_COLUMNS = ["lag_1h", "lag_24h", "lag_168h", "hour",
                   "day_of_week", "month", "is_holiday"]
# ---------------------------------------------------------------------------
# 2. Model training and evaluation
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    model: object
    feature_columns: list
    mae: float
    rmse: float
    mape: float
def train_demand_forecaster(features_df: pd.DataFrame,
                             target_col: str = "P_demand_pu",
                             test_fraction: float = 0.2) -> ForecastResult:
    """
    Train a gradient-boosted regressor to predict next-hour demand.
    Uses a chronological (not random/shuffled) train/test split, since
    shuffling time series data leaks future information into training and
    would produce an artificially optimistic error estimate.
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
    if "temperature" in features_df.columns:
        feature_cols.append("temperature")
    X = features_df[feature_cols].values
    y = features_df[target_col].values
    n = len(X)
    split_idx = int(n * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    if _HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42,
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    # avoid division-by-zero on any near-zero true demand values
    nonzero = y_test != 0
    mape = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100)
    return ForecastResult(model=model, feature_columns=feature_cols,
                           mae=mae, rmse=rmse, mape=mape)
# ---------------------------------------------------------------------------
# 3. Per-bus disaggregation
# ---------------------------------------------------------------------------
def compute_allocation_factors(graph) -> dict:
    """
    a_i = P_load_i^pu / sum_k P_load_k^pu
    graph : NetworkGraph instance from data_loader.py (Module 1).
    Returns {bus_id: a_i}, with sum(a_i) == 1 (validated below).
    """
    total_load = sum(bus.P_load_pu for bus in graph.buses.values())
    if total_load <= 0:
        raise ValueError("Total base-case load is zero or negative; "
                          "cannot compute allocation factors.")
    factors = {bus_id: bus.P_load_pu / total_load
               for bus_id, bus in graph.buses.items()}
    assert abs(sum(factors.values()) - 1.0) < 1e-9, "Allocation factors must sum to 1."
    return factors
def disaggregate_forecast(system_demand_forecast_pu: float,
                           allocation_factors: dict) -> dict:
    """
    L_i(t) = a_i * P_demand_hat(t)   -- SINGLE-TIMESTEP version.
    Returns {bus_id: L_i_pu} for one forecasted timestep. Useful for a
    live/real-time dashboard update, but NOT what multi-timestep pipeline
    code (e.g. renewable.py's compute_net_load) expects -- see
    disaggregate_forecast_series below for that case.
    """
    return {bus_id: a_i * system_demand_forecast_pu
            for bus_id, a_i in allocation_factors.items()}
def disaggregate_forecast_series(system_demand_forecast_pu: pd.Series,
                                  allocation_factors: dict) -> pd.DataFrame:
    """
    Vectorized, FULL-TIME-SERIES version of the same equation:
        L_i(t) = a_i * P_demand_hat(t)  for every t at once.
    Returns a DataFrame (index=time, columns=bus_id) -- this is the form
    renewable.py's compute_net_load() actually expects. Added after
    integration testing surfaced that the single-timestep dict version
    above doesn't match what a full pipeline run needs; rather than loop
    the dict version per-timestep in main.py, this computes it directly
    via an outer product, which is both correct and considerably faster.
    """
    return pd.DataFrame(
        {bus_id: a_i * system_demand_forecast_pu
         for bus_id, a_i in allocation_factors.items()}
    )
if __name__ == "__main__":
    # Example only — replace with real bundle from data_loader.load_all().
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    rng = np.random.default_rng(42)
    synthetic_demand = pd.Series(
        0.5 + 0.1 * np.sin(2 * np.pi * idx.hour / 24) + 0.01 * rng.standard_normal(len(idx)),
        index=idx,
    )
    feats = build_features(synthetic_demand)
    result = train_demand_forecaster(feats)
    print(f"Using {'XGBoost' if _HAS_XGBOOST else 'sklearn GradientBoostingRegressor'}")
    print(f"MAE={result.mae:.5f} pu, RMSE={result.rmse:.5f} pu, MAPE={result.mape:.2f}%")