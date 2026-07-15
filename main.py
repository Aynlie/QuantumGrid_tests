"""
main.py
================
QuantumGrid end-to-end pipeline, tying together Modules 1-8.
This is the integration point: every function called below is real code
from the corresponding module file, not a re-implementation. Running this
file is the actual test of whether the eight modules, each individually
corrected and verified, also work correctly TOGETHER.

This version runs on REAL data:
  - network_topology.csv : real IEEE 33-bus feeder (32 fixed branches +
    5 standard tie switches), built from the uploaded distribution
    network dataset.
  - PJME_hourly.csv       : real hourly demand (Kaggle PJM Interconnection).
  - solar_generation.csv  : real solar plant output (Kaggle Plant_1),
    aggregated across inverters and re-based onto PJME's date range.

IMPORTANT SCALE NOTE: PJME's demand is a whole utility system (tens of
thousands of MW) and Plant_1 is a utility-scale solar farm (~29 MW peak)
-- both are far larger than this 10 MVA feeder. Using their raw MW values
directly would be physically meaningless (e.g. 350+ pu of demand). So
only their SHAPE (normalized intraday/seasonal pattern) is used here,
rescaled onto this feeder's own real baseline load (from
network_topology.csv) and an assumed installed PV capacity. This is
stated explicitly rather than silently feeding in unphysical numbers.
"""
import os
import pandas as pd
import numpy as np
import data_loader as dl
import forecasting as fc
import renewable as rw
import network_model as nm
import power_flow as pf
import qubo_builder as qb
import quantum_optimizer as qo
import disaster_recovery as dr
import dashboard as db
import quapp_client

# Enable this only when you have a valid Quapp token and a verified
# function endpoint. It defaults to False so the pipeline can still run
# end-to-end without Quapp access.
RUN_QAOA_ON_QUAPP = os.getenv("RUN_QAOA_ON_QUAPP", "false").lower() in {"1", "true", "yes", "on"}


def build_synthetic_network(base: dl.BaseValues) -> dl.NetworkGraph:
    """
    Stand-in for data_loader.load_network_topology() when no real feeder
    CSV is available -- a small 5-bus network with one loop, a hospital
    (high priority), and a PV installation, exercising every module.
    Kept as a fallback/quick-test fixture; NOT called by run_pipeline()
    below, which uses the real data loaded from CSV instead.
    """
    net = dl.NetworkGraph(base=base)
    net.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    net.buses[2] = dl.Bus(id=2, bus_type="PQ", P_load_pu=0.30, Q_load_pu=0.10,
                           priority_weight=5_000)
    net.buses[3] = dl.Bus(id=3, bus_type="PQ", P_load_pu=0.20, Q_load_pu=0.05,
                           priority_weight=1_000_000)  # hospital
    net.buses[4] = dl.Bus(id=4, bus_type="generator", P_load_pu=0.25, Q_load_pu=0.08,
                           priority_weight=100_000)     # school, has local PV
    net.branches = [
        dl.Branch(i=1, j=2, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=True, s_initial=1),
        dl.Branch(i=2, j=3, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=3, j=4, R_pu=0.05, X_pu=0.08, S_max_pu=1.0, is_switchable=True, s_initial=1),
        dl.Branch(i=4, j=1, R_pu=0.01, X_pu=0.02, S_max_pu=1.0, is_switchable=True, s_initial=0),
    ]
    return net


# Assumption: the real datasets don't specify WHICH bus has the solar
# installation, so bus 18 (a real feeder-end lateral bus in the IEEE
# 33-bus system) is chosen as a representative PV location. State this
# assumption plainly rather than burying it -- swap this if you know the
# real PV bus.
PV_BUS = 18
PV_CAPACITY_PU = 0.35
PV_HOSTING_CAP_PU = 0.30  # slightly below capacity -> some real curtailment shows up

# The fixed (non-switchable) backbone edge to fault-test in Stage 7.
FAULT_EDGE = (5, 6)


def run_pipeline():
    print("=" * 70)
    print("STAGE 1 -- Data Loader")
    print("=" * 70)
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    bundle = dl.load_all(
        network_csv="network_topology.csv",
        demand_csv="PJME_hourly.csv",
        solar_csv="solar_generation.csv",
        S_base_mva=base.S_base_mva,
        V_base_kv=base.V_base_kv,
    )
    network = bundle["graph"]

    # Rescale PJME's real shape onto this feeder's own real nominal load
    # (rather than feeding in PJME's raw system-wide MW figures directly).
    nominal_total_load_pu = sum(b.P_load_pu for b in network.buses.values())
    demand_shape = bundle["demand_pu"] / bundle["demand_pu"].mean()
    demand_pu = demand_shape * nominal_total_load_pu

    # Rescale Plant_1's real shape onto an assumed installed PV capacity.
    solar_shape = bundle["solar_pu"] / bundle["solar_pu"].max()
    solar_pu = solar_shape * PV_CAPACITY_PU

    print(f"Loaded real network: {network.num_buses()} buses, "
          f"{len(network.branches)} branches; {len(demand_pu)} hourly "
          f"demand/solar points (PJME shape + Plant_1 shape, rescaled to "
          f"this feeder's real nominal load of {nominal_total_load_pu:.4f} pu).")

    print("\n" + "=" * 70)
    print("STAGE 2 -- AI Demand Forecasting")
    print("=" * 70)
    features = fc.build_features(demand_pu)
    forecast_result = fc.train_demand_forecaster(features)
    print(f"Forecast MAE={forecast_result.mae:.4f} pu, RMSE={forecast_result.rmse:.4f} pu, "
          f"MAPE={forecast_result.mape:.2f}%")
    allocation_factors = fc.compute_allocation_factors(network)
    demand_by_bus_pu = fc.disaggregate_forecast_series(demand_pu, allocation_factors)
    print(f"Per-bus demand disaggregated across {len(allocation_factors)} buses.")

    print("\n" + "=" * 70)
    print("STAGE 3 -- Renewable Integration")
    print("=" * 70)
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
    print(f"Peak curtailment at bus {PV_BUS}: {dispatch.curtailed_pu[PV_BUS].max():.4f} pu")

    print("\n" + "=" * 70)
    print("STAGE 4 -- Distribution Network Modeling")
    print("=" * 70)
    dist_graph = nm.build_distribution_graph(network)

    print("\n" + "=" * 70)
    print("STAGE 5 -- Power Flow & QUBO Builder (snapshot: peak demand hour)")
    print("=" * 70)
    peak_hour = demand_pu.idxmax()
    net_injection = {bus: net_load_by_bus.loc[peak_hour, bus] for bus in network.buses}
    print(f"Optimizing for peak hour: {peak_hour}")
    loops = qb.find_switchable_loops(dist_graph)
    costs = qb.compute_loop_open_costs(dist_graph, loops, net_injection, root=1)
    Q, var_order = qb.build_qubo(loops, costs)
    print(f"QUBO built: {len(var_order)} switchable decision variables.")

    print("\n" + "=" * 70)
    print("STAGE 6 -- Quantum / Classical Optimization")
    print("=" * 70)
    sa_assignment, sa_energy = qo.solve_with_classical_sa(Q, var_order)
    bf_assignment, bf_energy = qb.brute_force_solve(Q, var_order)
    assert sa_assignment == bf_assignment, "Solver disagreement detected!"
    print(f"[Classical SA]      {sa_assignment} (energy={sa_energy:.4f})")
    print(f"[Brute force]       {bf_assignment} (energy={bf_energy:.4f})")
    print("Classical SA matches brute-force ground truth.")

    if RUN_QAOA_ON_QUAPP:
        try:
            # NOTE: handler.py (deployed on Quapp) parses Q keys as
            # "a,b".split(",") and uses var_order entries directly as dict
            # keys/labels (no int conversion) -- so var labels must be
            # STRINGS, and Q keys must be COMMA-separated, not pipe-separated.
            # It also expects "reps"/"betas"/"gammas", not "p_layers"/"beta"/"gamma".
            scalar_var_order = [str(i) for i in range(len(var_order))]
            edge_of_var = dict(zip(scalar_var_order, var_order))
            scalar_Q = {
                f"{i},{i}": Q.get((var_order[int(i)], var_order[int(i)]), 0.0)
                for i in scalar_var_order
            }
            reps = 1
            job_input = {
                "var_order": scalar_var_order,
                "Q": scalar_Q,
                "reps": reps,
                "betas": [1.0] * reps,
                "gammas": [0.5] * reps,
                "shots": 1024,
            }
            job_result = quapp_client.submit_job(job_input)
            counts = job_result["counts"]

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
                    "bitstring": bitstring,
                    "assignment": assignment,
                    "energy": energy,
                    "frequency": freq,
                })
            best = sorted(scored, key=lambda r: (r["energy"], -r["frequency"]))[0]
            qaoa_result = {
                "switch_config": {
                    edge: ("closed" if state == 1 else "open")
                    for edge, state in best["assignment"].items()
                },
                "energy": best["energy"],
                "frequency": best["frequency"],
            }
            print(f"[QAOA on Quapp]     {qaoa_result['switch_config']} "
                  f"(energy={qaoa_result['energy']:.4f}, "
                  f"frequency={qaoa_result['frequency']})")
            if qaoa_result["energy"] == sa_energy:
                print("QAOA matches the classical optimum exactly.")
            else:
                gap = qaoa_result["energy"] - sa_energy
                print(f"QAOA is within {gap:.4f} energy of the classical optimum "
                      f"(expected at this qubit count with a shallow p_layers=1 circuit).")
        except Exception as exc:
            print(f"[QAOA on Quapp] SKIPPED -- {exc}")
    else:
        print("[QAOA on Quapp] SKIPPED (RUN_QAOA_ON_QUAPP=False)")

    closed_edges = set(dist_graph.fixed_edges)
    closed_edges.update(e for e, closed in sa_assignment.items() if closed)
    flows = pf.compute_tree_flows(dist_graph, closed_edges, net_injection, root=1)
    total_loss = pf.total_ohmic_loss(dist_graph, flows)
    q_flows = {k: 0.0 for k in flows}  # reactive flows not modeled at this stage
    voltage_check = pf.check_voltage_feasibility(dist_graph, flows, q_flows, root=1)
    print(f"Total ohmic loss at peak hour: {total_loss:.5f} pu")
    print(f"Voltage violations: {voltage_check['violations']}")

    print("\n" + "=" * 70)
    print("STAGE 7 -- Disaster Recovery (simulate a fault on the fixed backbone)")
    print("=" * 70)
    fault_result = dr.simulate_fault(network, FAULT_EDGE, net_injection, root=1)
    if fault_result.restorable:
        print(f"Restored via: {fault_result.new_switch_assignment}")
    else:
        print(f"UNAVOIDABLE OUTAGE at bus(es): {fault_result.stranded_buses}")

    print("\n" + "=" * 70)
    print("STAGE 8 -- Dashboard metrics")
    print("=" * 70)
    total_load = sum(max(v, 0.0) for v in net_injection.values())
    efficiency = db.compute_grid_efficiency(total_load, total_loss)
    total_renewable = dispatch.dispatched_pu.loc[peak_hour].sum()
    renewable_frac = db.compute_renewable_fraction(total_renewable, total_load)
    print(f"Grid efficiency at peak hour: {efficiency:.2f}%")
    print(f"Renewable contribution at peak hour: {renewable_frac:.2f}%")
    print("\nPipeline completed successfully end-to-end.")
    return {
        "network": network, "dist_graph": dist_graph,
        "switch_assignment": sa_assignment, "total_loss": total_loss,
        "voltage_check": voltage_check, "efficiency": efficiency,
    }


if __name__ == "__main__":
    run_pipeline()