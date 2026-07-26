import os
import data_loader as dl
import forecasting as fc
import renewable as rw
import network_model as nm
import power_flow as pf
import qubo_builder as qb
import quantum_optimizer as qo
import benders_loop as bl
import disaster_recovery as dr

PV_BUS = 18
PV_CAPACITY_PU = 0.35
PV_HOSTING_CAP_PU = 0.30
FAULT_EDGE = (5, 6)


def run_algorithm():
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

    pv_capacity = {bus_id: (PV_CAPACITY_PU if bus_id == PV_BUS else 0.0)
                   for bus_id in network.buses}
    pv_factors = rw.compute_pv_allocation_factors(
        {k: (v if v > 0 else 1e-9) for k, v in pv_capacity.items()}
    )
    available_solar = rw.allocate_measured_solar(bundle["solar_pu"] * PV_CAPACITY_PU / bundle["solar_pu"].max(), pv_factors)
    hosting_caps = {bus_id: (PV_HOSTING_CAP_PU if bus_id == PV_BUS else 0.0)
                    for bus_id in network.buses}
    dispatch = rw.apply_hosting_capacity(available_solar, hosting_caps)
    net_load_by_bus = rw.compute_net_load(demand_pu, dispatch.dispatched_pu)

    dist_graph = nm.build_distribution_graph(network)
    peak_hour = demand_pu.idxmax()
    net_injection = {bus: net_load_by_bus.loc[peak_hour, bus] for bus in network.buses}
    loops = qb.find_switchable_loops(dist_graph)
    costs = qb.compute_loop_open_costs(dist_graph, loops, net_injection, root=1)

    benders_result = bl.run_benders_qaoa_loop(
        dist_graph, loops, costs, net_injection, root=1,
        q_injection={}, solver="sa", max_iters=10,
    )

    sa_assignment = benders_result.switch_assignment
    sa_energy = benders_result.energy
    print(f"Benders + SA assignment: {sa_assignment}")
    print(f"Benders + SA energy: {sa_energy:.4f}")

    final_Q, final_var_order = qb.build_master_qubo(loops, costs, cuts=benders_result.cuts)
    bf_assignment, bf_energy = qb.brute_force_solve(final_Q, final_var_order)
    print(f"Brute force assignment: {bf_assignment}")
    print(f"Brute force energy: {bf_energy:.4f}")

    if sa_assignment != bf_assignment:
        raise AssertionError("SA and brute-force assignments disagree")

    print("Classical solver agreement confirmed.")

    return {
        "sa_assignment": sa_assignment,
        "sa_energy": sa_energy,
        "bf_assignment": bf_assignment,
        "bf_energy": bf_energy,
    }


if __name__ == "__main__":
    results = run_algorithm()
    print(results)
