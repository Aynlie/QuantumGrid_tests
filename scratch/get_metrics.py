import sys
import pandas as pd
sys.path.append("c:/Users/Jaymee/Documents/SEA HACKATHON/code/QuantumGrid_tests_fresh")
import data_loader as dl
import forecasting as fc

bundle = dl.load_all(
    network_csv="c:/Users/Jaymee/Documents/SEA HACKATHON/code/QuantumGrid_tests_fresh/network_topology.csv",
    demand_csv="c:/Users/Jaymee/Documents/SEA HACKATHON/code/QuantumGrid_tests_fresh/PJME_hourly.csv",
    solar_csv="c:/Users/Jaymee/Documents/SEA HACKATHON/code/QuantumGrid_tests_fresh/solar_generation.csv",
    S_base_mva=10.0, V_base_kv=12.66
)

network = bundle["graph"]
nominal_total_load_pu = sum(b.P_load_pu for b in network.buses.values())
demand_shape = bundle["demand_pu"] / bundle["demand_pu"].mean()
demand_pu = demand_shape * nominal_total_load_pu

features = fc.build_features(demand_pu)
forecast_res = fc.train_demand_forecaster(features)
print(f"MAE: {forecast_res.mae:.4f}")
print(f"RMSE: {forecast_res.rmse:.4f}")
print(f"MAPE: {forecast_res.mape:.2f}%")
