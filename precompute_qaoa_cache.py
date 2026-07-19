"""
precompute_qaoa_cache.py
========================
Standalone script — run manually BEFORE deploying the dashboard.

For every faulted-edge scenario listed in SCENARIOS below, this script:
  1. Builds the IEEE 33-bus network and simulates the fault.
  2. Constructs the QUBO via qubo_builder.
  3. Submits the QAOA job to Quapp Cloud (requires QUAPP_API_TOKEN in .env).
  4. Decodes the returned counts, picks the lowest-energy bitstring.
  5. Writes the result into qaoa_cache.json keyed by scenario.

The dashboard reads qaoa_cache.json at runtime and NEVER contacts Quapp.

Usage:
    python precompute_qaoa_cache.py            # run all scenarios
    python precompute_qaoa_cache.py "(5, 6)"   # run a single scenario
"""
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import disaster_recovery as dr
import qubo_builder as qb
import quapp_client
from data_loader import load_all

CACHE_PATH = Path(__file__).resolve().parent / "qaoa_cache.json"

# All backbone + switchable edges that are meaningful fault scenarios.
# Extend this list as needed.
SCENARIOS = [
    (5, 6),
    (2, 3),
    (3, 4),
    (6, 7),
    (9, 10),
    (10, 11),
]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def precompute_scenario(faulted_edge: tuple, graph, net_injection: dict, root: int = 1):
    """Run a single scenario through Quapp and return the cache entry, or None."""
    result = dr.simulate_fault(graph, faulted_edge, net_injection, root=root)

    if not result.restorable:
        print(f"  Scenario {faulted_edge}: not restorable (no QUBO to solve). Skipping.")
        return None

    loops = qb.find_switchable_loops(result.new_dist_graph)
    costs = qb.compute_loop_open_costs(result.new_dist_graph, loops, net_injection, root=root)
    Q, var_order = qb.build_qubo(loops, costs)

    # Build the Quapp-compatible scalar QUBO (same logic the old dashboard used)
    scalar_var_order = [str(i) for i in range(len(var_order))]
    edge_of_var = dict(zip(scalar_var_order, var_order))
    scalar_Q = {}
    for i_idx, u in enumerate(var_order):
        scalar_Q[f"{i_idx},{i_idx}"] = Q.get((u, u), 0.0)
    for (u, v), coeff in Q.items():
        if u != v:
            i_idx = var_order.index(u)
            j_idx = var_order.index(v)
            k_str = f"{i_idx},{j_idx}" if i_idx < j_idx else f"{j_idx},{i_idx}"
            scalar_Q[k_str] = coeff

    reps = 1
    job_input = {
        "var_order": scalar_var_order,
        "Q": scalar_Q,
        "reps": reps,
        "betas": [1.0] * reps,
        "gammas": [0.5] * reps,
        "shots": 1024,
    }

    print(f"  Submitting to Quapp for scenario {faulted_edge}...")
    job_result = quapp_client.submit_job(job_input)
    counts = job_result["counts"]

    # Decode and score
    scored = []
    for bitstring, freq in counts.items():
        bits = bitstring[::-1]
        assignment = {
            edge_of_var[str(i)]: int(bits[i])
            for i in range(len(scalar_var_order))
            if i < len(bits)
        }
        energy = qb.evaluate_qubo(Q, assignment)
        scored.append({
            "assignment": assignment,
            "energy": energy,
            "frequency": freq,
        })
    best = sorted(scored, key=lambda r: (r["energy"], -r["frequency"]))[0]

    # Serialize assignment keys (tuples) to strings for JSON
    serialized_assignment = {str(k): v for k, v in best["assignment"].items()}

    return {
        "assignment": serialized_assignment,
        "energy": best["energy"],
        "precomputed_date": datetime.now(timezone.utc).isoformat(),
    }


def main():
    # Load network data
    bundle = load_all(
        network_csv="network_topology.csv",
        demand_csv="PJME_hourly.csv",
        solar_csv="solar_generation.csv",
        S_base_mva=10.0,
        V_base_kv=12.66,
    )
    graph = bundle["graph"]
    demand = bundle["demand_pu"]
    solar = bundle["solar_pu"]
    network = graph


    # Use peak-hour net injection as the representative load profile
    from forecasting import compute_allocation_factors, disaggregate_forecast_series
    from renewable import (compute_pv_allocation_factors, allocate_measured_solar,
                           apply_hosting_capacity, compute_net_load)

    nominal_total_load_pu = sum(b.P_load_pu for b in network.buses.values())
    demand_shape = demand / demand.mean()
    demand_pu = demand_shape * nominal_total_load_pu

    PV_CAPACITY_PU = 0.35
    solar_shape = solar / solar.max()
    solar_pu = solar_shape * PV_CAPACITY_PU

    allocation_factors = compute_allocation_factors(network)
    demand_by_bus_pu = disaggregate_forecast_series(demand_pu, allocation_factors)

    PV_BUS = 18
    PV_HOSTING_CAP_PU = 0.30
    pv_capacity = {bus_id: (PV_CAPACITY_PU if bus_id == PV_BUS else 0.0)
                   for bus_id in network.buses}
    pv_factors = compute_pv_allocation_factors(
        {k: (v if v > 0 else 1e-9) for k, v in pv_capacity.items()}
    )
    available_solar = allocate_measured_solar(solar_pu, pv_factors)
    hosting_caps = {bus_id: (PV_HOSTING_CAP_PU if bus_id == PV_BUS else 0.0)
                    for bus_id in network.buses}
    dispatch = apply_hosting_capacity(available_solar, hosting_caps)
    net_load_by_bus = compute_net_load(demand_by_bus_pu, dispatch.dispatched_pu)

    peak_hour = demand_pu.idxmax()
    net_injection = {b: net_load_by_bus.loc[peak_hour, b] for b in graph.buses}

    # Determine which scenarios to run
    if len(sys.argv) > 1:
        # Parse a specific scenario from command line, e.g. "(5, 6)"
        arg = sys.argv[1].strip()
        try:
            val = ast.literal_eval(arg)
            if not isinstance(val, tuple) or len(val) != 2 or not all(isinstance(x, int) for x in val):
                raise ValueError("Scenario must be a 2-tuple of integers")
            scenario_list = [val]
        except Exception:
            print(f"Error: could not parse scenario '{arg}'. Expected format: \"(5, 6)\"")
            sys.exit(1)
    else:
        scenario_list = SCENARIOS

    cache = _load_cache()
    success_count = 0

    for faulted_edge in scenario_list:
        key = str(faulted_edge)
        print(f"\n--- Scenario: {key} ---")
        try:
            entry = precompute_scenario(faulted_edge, graph, net_injection)
            if entry is not None:
                cache[key] = entry
                _save_cache(cache)
                print(f"  [Cached] energy={entry['energy']:.4f}, date={entry['precomputed_date']}")
                success_count += 1
            else:
                print("  [Skipped] (not restorable)")
        except Exception as e:
            print(f"  [FAILED] {e}")

    print(f"\nDone. {success_count}/{len(scenario_list)} scenarios cached to {CACHE_PATH}")


if __name__ == "__main__":
    main()
