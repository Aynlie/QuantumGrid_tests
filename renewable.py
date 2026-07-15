"""
renewable.py
================
Module 3 of QuantumGrid.
Responsibility: compute per-bus available and dispatched solar generation,
apply a hosting-capacity/curtailment cap, and produce the net residual load
per bus that Module 5 (power balance) consumes. Supports two input paths:
  (A) measured system-level solar generation data (preferred), or
  (B) irradiance + cell temperature via the PVWatts-style conversion model
      (fallback, used only when (A) is unavailable).
This module does NOT invent a hosting-capacity value from nothing: if the
caller does not supply per-bus PV capacity or a hosting-capacity limit, it
raises an error rather than assuming a default silently.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# 1. PV capacity allocation (Case A: measured system-level generation)
# ---------------------------------------------------------------------------
def compute_pv_allocation_factors(pv_capacity_pu: dict) -> dict:
    """
    b_i = C_i^PV / sum_k C_k^PV
    pv_capacity_pu : {bus_id: installed PV capacity, per-unit}. Buses with no
                     PV should be present with capacity 0.0, not omitted, so
                     that downstream code has an explicit entry for every bus.
    """
    total_capacity = sum(pv_capacity_pu.values())
    if total_capacity <= 0:
        raise ValueError("Total installed PV capacity is zero; cannot "
                          "allocate a system-level solar profile across "
                          "buses. Provide per-bus PV capacity explicitly.")
    return {bus_id: c / total_capacity for bus_id, c in pv_capacity_pu.items()}
def allocate_measured_solar(system_solar_pu: pd.Series,
                             pv_allocation_factors: dict) -> pd.DataFrame:
    """
    P_r,i^avail(t) = b_i * P_solar^system(t)
    Returns a DataFrame indexed by time, one column per bus_id.
    """
    return pd.DataFrame(
        {bus_id: b_i * system_solar_pu for bus_id, b_i in pv_allocation_factors.items()}
    )
# ---------------------------------------------------------------------------
# 2. PVWatts-style physical conversion model (Case B: irradiance fallback)
# ---------------------------------------------------------------------------
def pv_output_from_irradiance(pv_capacity_pu: dict,
                               irradiance_w_m2: pd.Series,
                               cell_temp_c: pd.Series,
                               gamma_per_c: float = -0.004,
                               g_stc: float = 1000.0,
                               t_stc: float = 25.0) -> pd.DataFrame:
    """
    P_r,i^avail(t) = C_i^PV * (G(t)/G_stc) * [1 + gamma * (T_cell(t) - T_stc)]
    gamma_per_c: manufacturer temperature coefficient of power (1/°C).
                 -0.004/°C is a commonly cited default for crystalline
                 silicon modules; replace with your actual panel datasheet
                 value if known (see NREL PVWatts documentation reference).
    """
    if irradiance_w_m2.index is not cell_temp_c.index:
        cell_temp_c = cell_temp_c.reindex(irradiance_w_m2.index)
    shape = (irradiance_w_m2 / g_stc) * (1 + gamma_per_c * (cell_temp_c - t_stc))
    # Physical output cannot be negative even if the linear temperature
    # term would otherwise push it below zero at very low irradiance.
    shape = shape.clip(lower=0.0)
    return pd.DataFrame({bus_id: c_i * shape for bus_id, c_i in pv_capacity_pu.items()})
# ---------------------------------------------------------------------------
# 3. Curtailment / hosting-capacity cap
# ---------------------------------------------------------------------------
@dataclass
class RenewableDispatchResult:
    dispatched_pu: pd.DataFrame   # P_r,i(t): what's actually used
    curtailed_pu: pd.DataFrame    # C_i(t) = P_r,i^avail(t) - P_r,i(t), >= 0
def apply_hosting_capacity(available_pu: pd.DataFrame,
                            hosting_capacity_pu: dict) -> RenewableDispatchResult:
    """
    P_r,i(t) = min(P_r,i^avail(t), P_r,i^max)
    C_i(t)   = P_r,i^avail(t) - P_r,i(t)  (>= 0 by construction)
    hosting_capacity_pu : {bus_id: P_r,i^max}. This is a PRELIMINARY bound —
    a rigorous value requires checking against the connected line's S_ij^max
    from Module 4/5, which isn't available yet at this stage of the pipeline.
    Flagging this explicitly rather than presenting the cap as final.
    """
    missing = set(available_pu.columns) - set(hosting_capacity_pu.keys())
    if missing:
        raise ValueError(f"No hosting-capacity limit provided for buses: {missing}")
    cap_series = pd.Series(hosting_capacity_pu)
    dispatched = available_pu.clip(upper=cap_series, axis=1)
    curtailed = available_pu - dispatched
    return RenewableDispatchResult(dispatched_pu=dispatched, curtailed_pu=curtailed)
# ---------------------------------------------------------------------------
# 4. Net residual load
# ---------------------------------------------------------------------------
def compute_net_load(demand_by_bus_pu: pd.DataFrame,
                      dispatched_solar_pu: pd.DataFrame) -> pd.DataFrame:
    """
    P_net,i(t) = L_i(t) - P_r,i(t)
    Negative values are valid (bus i is a net exporter at that hour) and
    are NOT clipped to zero -- doing so would silently discard real
    reverse-power-flow behavior.
    """
    common_cols = demand_by_bus_pu.columns.intersection(dispatched_solar_pu.columns)
    missing_cols = demand_by_bus_pu.columns.difference(dispatched_solar_pu.columns)
    net = demand_by_bus_pu[common_cols] - dispatched_solar_pu[common_cols]
    # Buses with no PV asset at all simply have zero dispatched solar.
    for col in missing_cols:
        net[col] = demand_by_bus_pu[col]
    return net[demand_by_bus_pu.columns]  # preserve original bus ordering
if __name__ == "__main__":
    # Example only, using synthetic data for two PV-equipped buses out of three.
    idx = pd.date_range("2024-06-01", periods=48, freq="h")
    rng = np.random.default_rng(0)
    demand_by_bus = pd.DataFrame({
        1: 0.0,  # slack bus, no local load
        2: 0.3 + 0.05 * np.sin(2 * np.pi * idx.hour / 24) + 0.005 * rng.standard_normal(48),
        3: 0.7 + 0.05 * np.cos(2 * np.pi * idx.hour / 24) + 0.005 * rng.standard_normal(48),
    }, index=idx)
    pv_capacity = {1: 0.0, 2: 0.4, 3: 0.1}  # bus 2 has more installed PV than bus 3
    b_factors = compute_pv_allocation_factors(pv_capacity)
    print("PV allocation factors:", b_factors)
    # Synthetic daytime solar bell curve
    daylight = np.clip(np.sin(np.pi * (idx.hour - 6) / 12), 0, None)
    system_solar = pd.Series(0.5 * daylight, index=idx)
    available = allocate_measured_solar(system_solar, b_factors)
    hosting_caps = {1: 0.0, 2: 0.15, 3: 0.05}  # deliberately tight to trigger curtailment
    result = apply_hosting_capacity(available, hosting_caps)
    print("\nMax available vs. dispatched vs. curtailed at bus 2:")
    print(f"  available max = {available[2].max():.4f} pu")
    print(f"  dispatched max = {result.dispatched_pu[2].max():.4f} pu (capped at {hosting_caps[2]})")
    print(f"  curtailed max  = {result.curtailed_pu[2].max():.4f} pu")
    net_load = compute_net_load(demand_by_bus, result.dispatched_pu)
    print("\nNet load sample at midday (bus 2, bus 3):")
    noon = idx[12]
    print(net_load.loc[noon])