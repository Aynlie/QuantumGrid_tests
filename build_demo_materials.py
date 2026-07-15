"""
build_demo_materials.py
================
Produces the disaster-recovery demo assets for the hackathon presentation:
  - before_fault.png   : topology with the standard radial configuration
  - after_fault.png    : topology after the fault + tie-switch restoration
  - solver_comparison.csv : classical SA / brute-force / QAOA side-by-side
                            (QAOA row filled in only if RUN_QAOA_ON_QUAPP=True)

Run this once your Quapp token is fresh if you want the QAOA row included;
otherwise it still produces both figures and the classical-only table.
"""
import os
from pathlib import Path
import data_loader as dl
import forecasting as fc
import renewable as rw
import network_model as nm
import power_flow as pf
import qubo_builder as qb
import quantum_optimizer as qo
import disaster_recovery as dr
import dashboard as db

RUN_QAOA_ON_QUAPP = os.getenv("RUN_QAOA_ON_QUAPP", "false").lower() in {"1", "true", "yes", "on"}
PV_BUS = 18
PV_CAPACITY_PU = 0.35
PV_HOSTING_CAP_PU = 0.30
FAULT_EDGE = (5, 6)
OUTPUT_DIR = Path(__file__).resolve().parent / "submission_assets"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    bundle = dl.load_all(
        network_csv="network_topology.csv",
        demand_csv="PJME_hourly.csv",
        solar_csv="solar_generation.csv",
        S_base_mva=base.S_base_mva,
        V_base_kv=base.V_base_kv,
    )
    network = bundle["graph"]
    nominal_total_load_pu = sum(b.P_load_pu for b in network.buses.values())
    demand_shape = bundle["demand_pu"] / bundle["demand_pu"].mean()
    demand_pu = demand_shape * nominal_total_load_pu
    solar_shape = bundle["solar_pu"] / bundle["solar_pu"].max()
    solar_pu = solar_shape * PV_CAPACITY_PU

    features = fc.build_features(demand_pu)
    allocation_factors = fc.compute_allocation_factors(network)
    demand_by_bus_pu = fc.disaggregate_forecast_series(demand_pu, allocation_factors)
    pv_capacity = {b: (PV_CAPACITY_PU if b == PV_BUS else 0.0) for b in network.buses}
    pv_factors = rw.compute_pv_allocation_factors(
        {k: (v if v > 0 else 1e-9) for k, v in pv_capacity.items()}
    )
    available_solar = rw.allocate_measured_solar(solar_pu, pv_factors)
    hosting_caps = {b: (PV_HOSTING_CAP_PU if b == PV_BUS else 0.0) for b in network.buses}
    dispatch = rw.apply_hosting_capacity(available_solar, hosting_caps)
    net_load_by_bus = rw.compute_net_load(demand_by_bus_pu, dispatch.dispatched_pu)
    dist_graph = nm.build_distribution_graph(network)

    peak_hour = demand_pu.idxmax()
    net_injection = {bus: net_load_by_bus.loc[peak_hour, bus] for bus in network.buses}

    # Compute ONE stable layout from the full pre-fault topology and reuse
    # it for both figures below. Without this, the after-fault graph (one
    # branch fewer) gets its own independent spring_layout solve and nodes
    # end up in visually unrelated positions between the two images, even
    # though it's the same physical feeder with only one switch changed.
    shared_pos = db.compute_stable_layout(dist_graph)

    # --- BEFORE: standard radial configuration (all ties open) ---
    loops = qb.find_switchable_loops(dist_graph)
    costs = qb.compute_loop_open_costs(dist_graph, loops, net_injection, root=1)
    Q, var_order = qb.build_qubo(loops, costs)
    sa_assignment, sa_energy = qo.solve_with_classical_sa(Q, var_order)
    bf_assignment, bf_energy = qb.brute_force_solve(Q, var_order)

    fig_before = db.render_topology_figure(
        dist_graph, sa_assignment,
        title=f"Before fault -- standard radial config ({peak_hour})",
        pos=shared_pos,
    )
    fig_before.savefig(OUTPUT_DIR / "before_fault.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'before_fault.png'}")

    # --- AFTER: simulate the fault, get the restoration switch assignment ---
    fault_result = dr.simulate_fault(network, FAULT_EDGE, net_injection, root=1)
    if fault_result.restorable:
        fig_after = db.render_topology_figure(
            fault_result.new_dist_graph, fault_result.new_switch_assignment,
            title=f"After fault on {FAULT_EDGE} -- restored via tie switch",
            pos=shared_pos,
        )
        fig_after.savefig(OUTPUT_DIR / "after_fault.png", dpi=150, bbox_inches="tight")
        print(f"Saved {OUTPUT_DIR / 'after_fault.png'}")
        print(f"Restoration switch assignment: {fault_result.new_switch_assignment}")
    else:
        print(f"UNAVOIDABLE OUTAGE at bus(es): {fault_result.stranded_buses} "
              f"-- no after_fault.png generated (this itself may be worth "
              f"showing: not every fault is restorable).")

    # --- Solver comparison table ---
    results = {
        "Classical SA": {"assignment": sa_assignment, "energy": sa_energy},
        "Brute force": {"assignment": bf_assignment, "energy": bf_energy},
    }

    if RUN_QAOA_ON_QUAPP:
        try:
            import quapp_client
            scalar_var_order = [str(i) for i in range(len(var_order))]
            edge_of_var = dict(zip(scalar_var_order, var_order))
            scalar_Q = {
                f"{i},{i}": Q.get((var_order[int(i)], var_order[int(i)]), 0.0)
                for i in scalar_var_order
            }
            job_input = {
                "var_order": scalar_var_order, "Q": scalar_Q,
                "reps": 1, "betas": [1.0], "gammas": [0.5], "shots": 1024,
            }
            job_result = quapp_client.submit_job(job_input)
            counts = job_result["counts"]
            scored = []
            for bitstring, freq in counts.items():
                bits = bitstring[::-1]
                assignment = {edge_of_var[str(i)]: int(bits[i])
                              for i in range(len(scalar_var_order)) if i < len(bits)}
                energy = qb.evaluate_qubo(Q, assignment)
                scored.append({"assignment": assignment, "energy": energy, "frequency": freq})
            best = sorted(scored, key=lambda r: (r["energy"], -r["frequency"]))[0]
            results["QAOA (Quapp)"] = {"assignment": best["assignment"], "energy": best["energy"]}
            print(f"QAOA row added: energy={best['energy']:.4f}, "
                  f"assignment={best['assignment']}")
        except Exception as exc:
            # Surfaced loudly on purpose -- a QAOA row silently missing from
            # the CSV with no visible error is exactly what happened before.
            print(f"*** QAOA row skipped -- {type(exc).__name__}: {exc} ***")
    else:
        print("QAOA row skipped (RUN_QAOA_ON_QUAPP is not set to true).")

    # index=False is safe now: build_solver_comparison_table() puts the
    # solver name into an explicit "solver" column via reset_index(),
    # instead of leaving it only in the (dropped-on-export) DataFrame index.
    table = db.build_solver_comparison_table(results)
    table.to_csv(OUTPUT_DIR / "solver_comparison.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'solver_comparison.csv'}")
    print(table)


if __name__ == "__main__":
    main()