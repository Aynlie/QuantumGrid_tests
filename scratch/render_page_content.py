import sys
from pathlib import Path

# Set UTF-8 encoding for stdout to prevent Unicode errors on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock

# Mock streamlit before importing dashboard
st_mock = MagicMock()

# Set up decorators to be transparent
st_mock.cache_data = lambda f: f
st_mock.cache_resource = lambda f: f

# Setup session state mock
class SessionState(dict):
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

session_state = SessionState({
    "page": "Why This Recommendation",
    "active_fault": (5, 6),
    "simulation_result": None,
    "net_injection": None,
    "peak_hour": None,
    "solver_results": None,
    "selected_fault_edge": (5, 6),
})
st_mock.session_state = session_state

# We also mock st.expander to be a context manager
class DummyExpander:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        print(f"\n--- [EXPANDER START: {self.label}] ---")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"--- [EXPANDER END: {self.label}] ---\n")

st_mock.expander = DummyExpander

# Capture markdown calls
markdown_calls = []
def mock_markdown(content, unsafe_allow_html=False):
    markdown_calls.append(content)
    try:
        print(content)
    except Exception:
        print(content.encode('utf-8', errors='replace').decode('utf-8'))

st_mock.markdown = mock_markdown

sys.modules['streamlit'] = st_mock

# Now we can import the dashboard and load components
import data_loader as dl
import disaster_recovery as dr
import qubo_builder as qb
import quantum_optimizer as qo

# Load cached data
bundle = dl.load_all(
    network_csv="network_topology.csv",
    demand_csv="PJME_hourly.csv",
    solar_csv="solar_generation.csv",
    S_base_mva=10.0, V_base_kv=12.66,
)

network = bundle["graph"]
nominal_total_load_pu = sum(b.P_load_pu for b in network.buses.values())
demand_shape = bundle["demand_pu"] / bundle["demand_pu"].mean()
demand_pu = demand_shape * nominal_total_load_pu

PV_CAPACITY_PU = 0.35
solar_shape = bundle["solar_pu"] / bundle["solar_pu"].max()
solar_pu = solar_shape * PV_CAPACITY_PU

import forecasting as fc
features = fc.build_features(demand_pu)
_forecast = fc.train_demand_forecaster(features)
allocation_factors = fc.compute_allocation_factors(network)
demand_by_bus_pu = fc.disaggregate_forecast_series(demand_pu, allocation_factors)

import renewable as rw
PV_BUS = 18
PV_HOSTING_CAP_PU = 0.30
pv_capacity = {bus_id: (PV_CAPACITY_PU if bus_id == PV_BUS else 0.0)
               for bus_id in network.buses}
pv_factors = rw.compute_pv_allocation_factors(
    {k: (v if v > 0 else 1e-9) for k, v in pv_capacity.items()}
)
available_solar = rw.allocate_measured_solar(solar_pu, pv_factors)
hosting_caps = {bus_id: (PV_HOSTING_CAP_PU if bus_id == PV_BUS else 0.0)
                for bus_id in network.buses}
dispatch = rw.apply_hosting_capacity(available_solar, hosting_caps)
net_load_by_bus = rw.compute_net_load(demand_by_bus_pu, dispatch.dispatched_pu)

# Simulate fault
peak_hour = demand_pu.idxmax()
net_injection = {b: net_load_by_bus.loc[peak_hour, b] for b in bundle["graph"].buses}
result = dr.simulate_fault(bundle["graph"], (5, 6), net_injection, root=1)

session_state.simulation_result = result
session_state.net_injection = net_injection
session_state.peak_hour = peak_hour

loops = qb.find_switchable_loops(result.new_dist_graph)
costs = qb.compute_loop_open_costs(result.new_dist_graph, loops, net_injection, root=1)
Q, var_order = qb.build_qubo(loops, costs)
sa_assignment, sa_energy = qo.solve_with_classical_sa(Q, var_order)
bf_assignment, bf_energy = qb.brute_force_solve(Q, var_order)
qaoa_available = False # Mocked as False for simplicity

session_state.solver_results = {
    "classical_sa": {"assignment": sa_assignment, "energy": sa_energy, "time_ms": 12.5},
    "brute_force": {"assignment": bf_assignment, "energy": bf_energy, "time_ms": 1.2},
    "qaoa": {
        "available": qaoa_available,
        "assignment": {},
        "energy": 0.0,
        "time_ms": 150.0,
        "precomputed_date": None
    }
}

# Run the page content generator of render_dashboard()
import dashboard
dashboard.render_dashboard()
