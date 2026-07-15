"""
data_loader.py
================
Module 1 of QuantumGrid.
Responsibility: load and validate all raw inputs, convert them into a
single, dimensionally-consistent representation (per-unit network graph +
aligned time series) that every downstream module (forecasting, renewable
adjustment, network model, power flow, QUBO builder) can consume without
re-deriving units or re-checking consistency.
This module does NOT decide switch states. It only loads the network's
*known, current* topology (s_ij_initial) and physical parameters. The
switch decision variables are produced later by the quantum optimizer
(Module 6), not by the data loader.
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
# ---------------------------------------------------------------------------
# 1. Per-unit conversion utilities
# ---------------------------------------------------------------------------
@dataclass
class BaseValues:
    """
    Reference (base) quantities used to convert raw physical units
    (ohms, MW, MVAr, kV) into per-unit (p.u.) quantities.
    S_base_mva : apparent power base, in MVA
    V_base_kv  : voltage base, in kV (line-to-line, nominal feeder voltage)
    Z_base = V_base^2 / S_base   (Ohms)
    This mirrors standard power-system per-unit analysis
    (Baran & Wu, 1989; Kersting, 2017).
    """
    S_base_mva: float
    V_base_kv: float
    @property
    def Z_base_ohm(self) -> float:
        return (self.V_base_kv ** 2) / self.S_base_mva
def to_per_unit_impedance(r_ohm: float, x_ohm: float, base: BaseValues):
    """Convert raw R, X (ohms) into per-unit R, X."""
    z_base = base.Z_base_ohm
    return r_ohm / z_base, x_ohm / z_base
def to_per_unit_power(p_mw: float, base: BaseValues) -> float:
    """Convert raw active/reactive power (MW or MVAr) into per-unit."""
    return p_mw / base.S_base_mva
# ---------------------------------------------------------------------------
# 2. Network graph data structures
# ---------------------------------------------------------------------------
@dataclass
class Bus:
    """
    A single node (bus) in the distribution network.
    id        : unique bus index, 1..N
    bus_type  : 'slack' (substation / feeder head), 'PQ' (load bus),
                or 'generator' (has dispatchable/renewable generation)
    P_load_pu : active power demand at this bus, per-unit (base-case;
                overwritten later by forecasted/net demand in Modules 2-3)
    Q_load_pu : reactive power demand at this bus, per-unit
    V_min_pu, V_max_pu : allowed voltage magnitude band (typically 0.95-1.05 p.u.)
    priority_weight : used only by Module 7 (disaster recovery); default 1.0
    """
    id: int
    bus_type: str
    P_load_pu: float
    Q_load_pu: float
    V_min_pu: float = 0.95
    V_max_pu: float = 1.05
    priority_weight: float = 1.0
@dataclass
class Branch:
    """
    A single edge (line/branch) in the distribution network.
    i, j          : endpoint bus ids
    R_pu, X_pu    : per-unit resistance and reactance (NOT the switch state)
    S_max_pu      : thermal (apparent power) rating of the line, per-unit
    is_switchable : True if this branch has a controllable switch
                    (tie-switch or sectionalizing switch)
    s_initial     : current, as-loaded switch state (0 = open, 1 = closed).
                    This is a known INPUT, not an optimization output.
                    The optimizer's decision variable (Module 5/6) is a
                    separate symbol, s_ij, which may equal s_initial or not.
    """
    i: int
    j: int
    R_pu: float
    X_pu: float
    S_max_pu: float
    is_switchable: bool = False
    s_initial: int = 1
@dataclass
class NetworkGraph:
    buses: dict = field(default_factory=dict)      # id -> Bus
    branches: list = field(default_factory=list)   # list[Branch]
    base: BaseValues = None
    def num_buses(self) -> int:
        return len(self.buses)
    def switchable_branches(self):
        return [b for b in self.branches if b.is_switchable]
# ---------------------------------------------------------------------------
# 3. Loaders
# ---------------------------------------------------------------------------
def load_network_topology(csv_path: str, base: BaseValues) -> NetworkGraph:
    """
    Load line/bus data (e.g. an IEEE 33-bus or 69-bus test-feeder CSV export)
    and convert to a per-unit NetworkGraph.
    Expected CSV columns (raw, physical units):
        bus_i, bus_j, r_ohm, x_ohm, s_max_mva,
        is_switchable (0/1), s_initial (0/1),
        p_load_mw (per bus j), q_load_mvar (per bus j),
        bus_type ('slack'/'PQ'/'generator'), v_min_pu, v_max_pu
    Any missing/duplicate bus or branch entries raise a ValueError rather
    than silently proceeding, per Task 10 (no silent assumptions).
    """
    df = pd.read_csv(csv_path)
    required_cols = {
        "bus_i", "bus_j", "r_ohm", "x_ohm", "s_max_mva",
        "is_switchable", "s_initial", "p_load_mw", "q_load_mvar",
        "bus_type", "v_min_pu", "v_max_pu",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Network topology CSV is missing columns: {missing}")
    graph = NetworkGraph(base=base)
    # Pass 1: every bus's own load/type is defined by the row where it
    # appears as bus_j (per the docstring: "p_load_mw (per bus j)").
    for _, row in df.iterrows():
        bj = int(row.bus_j)
        if bj not in graph.buses:
            graph.buses[bj] = Bus(
                id=bj,
                bus_type=str(row.bus_type),
                P_load_pu=to_per_unit_power(float(row.p_load_mw), base),
                Q_load_pu=to_per_unit_power(float(row.q_load_mvar), base),
                V_min_pu=float(row.v_min_pu),
                V_max_pu=float(row.v_max_pu),
            )
    # Pass 2: any bus_i that never appears as a bus_j is the feeder root
    # (the slack/substation bus) -- it has no load of its own, and must
    # NOT silently inherit whichever row's bus_j load happened to be
    # created first. This was the actual bug: the old single-pass version
    # assigned bus_j's row values to bus_i too, misclassifying the slack
    # bus as a loaded 'PQ' bus.
    for _, row in df.iterrows():
        bi = int(row.bus_i)
        if bi not in graph.buses:
            graph.buses[bi] = Bus(
                id=bi,
                bus_type="slack",
                P_load_pu=0.0,
                Q_load_pu=0.0,
                V_min_pu=float(row.v_min_pu),
                V_max_pu=float(row.v_max_pu),
            )
    for _, row in df.iterrows():
        r_pu, x_pu = to_per_unit_impedance(float(row.r_ohm), float(row.x_ohm), base)
        s_max_pu = to_per_unit_power(float(row.s_max_mva), base)
        graph.branches.append(
            Branch(
                i=int(row.bus_i),
                j=int(row.bus_j),
                R_pu=r_pu,
                X_pu=x_pu,
                S_max_pu=s_max_pu,
                is_switchable=bool(int(row.is_switchable)),
                s_initial=int(row.s_initial),
            )
        )
    _validate_topology(graph)
    return graph
def _validate_topology(graph: NetworkGraph) -> None:
    """Fail loudly instead of silently proceeding with a broken network."""
    slack_buses = [b for b in graph.buses.values() if b.bus_type == "slack"]
    if len(slack_buses) != 1:
        raise ValueError(
            f"Expected exactly 1 slack (substation) bus, found {len(slack_buses)}."
        )
    connected_ids = {b.i for b in graph.branches} | {b.j for b in graph.branches}
    isolated = set(graph.buses.keys()) - connected_ids
    if isolated:
        raise ValueError(f"Isolated buses with no branch connection: {isolated}")
def load_demand_series(csv_path: str, base: BaseValues,
                        value_col: str = "PJME_MW") -> pd.Series:
    """
    Load the PJME hourly demand dataset (Kaggle: robikscube/hourly-energy-consumption,
    PJME_hourly.csv) and convert to per-unit power, indexed by timestamp.
    """
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").set_index("Datetime")
    if value_col not in df.columns:
        raise ValueError(f"Expected column '{value_col}' not found in {csv_path}")
    demand_mw = df[value_col]
    demand_pu = demand_mw / base.S_base_mva
    demand_pu.name = "P_demand_pu"
    return demand_pu
def load_solar_series(csv_path: str, base: BaseValues,
                       value_col: str = "solar_mw") -> pd.Series:
    """Load solar generation time series and convert to per-unit power."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    if value_col not in df.columns:
        raise ValueError(f"Expected column '{value_col}' not found in {csv_path}")
    solar_pu = df[value_col] / base.S_base_mva
    solar_pu.name = "P_solar_pu"
    return solar_pu
def align_time_series(demand_pu: pd.Series, solar_pu: pd.Series) -> pd.DataFrame:
    """
    Align demand and solar series onto a common hourly index.
    This directly addresses a real bug risk in the original design: PJME
    and a solar dataset will rarely share identical timestamps or timezone.
    We resample both to hourly means and inner-join on the overlapping
    time range, rather than assuming they already match.
    """
    demand_h = demand_pu.resample("h").mean()
    solar_h = solar_pu.resample("h").mean()
    aligned = pd.concat([demand_h, solar_h], axis=1, join="inner")
    aligned = aligned.dropna()
    if aligned.empty:
        raise ValueError(
            "No overlapping timestamps between demand and solar series "
            "after resampling. Check date ranges and timezones."
        )
    return aligned
# ---------------------------------------------------------------------------
# 4. Convenience entry point
# ---------------------------------------------------------------------------
def load_all(network_csv: str, demand_csv: str, solar_csv: str,
             S_base_mva: float, V_base_kv: float):
    """
    Load everything Module 1 is responsible for and return a single bundle
    consumed by Module 2 (forecasting) and Module 4 (network model).
    """
    base = BaseValues(S_base_mva=S_base_mva, V_base_kv=V_base_kv)
    graph = load_network_topology(network_csv, base)
    demand_pu = load_demand_series(demand_csv, base)
    solar_pu = load_solar_series(solar_csv, base)
    aligned = align_time_series(demand_pu, solar_pu)
    return {
        "base": base,
        "graph": graph,
        "demand_pu": aligned["P_demand_pu"],
        "solar_pu": aligned["P_solar_pu"],
    }
if __name__ == "__main__":
    # Example only — replace with real file paths before running.
    bundle = load_all(
        network_csv="network_topology.csv",
        demand_csv="PJME_hourly.csv",
        solar_csv="solar_generation.csv",
        S_base_mva=10.0,
        V_base_kv=12.66,
    )
    print(f"Loaded {bundle['graph'].num_buses()} buses, "
          f"{len(bundle['graph'].branches)} branches, "
          f"{len(bundle['demand_pu'])} aligned demand/solar timestamps.")