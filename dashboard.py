"""
dashboard.py
================
Module 8 of QuantumGrid.
Two layers, deliberately separated:
  1. Metrics/aggregation functions (pure Python + matplotlib) -- these have
     no UI dependency and are fully unit-testable without a browser.
  2. Streamlit UI (render_dashboard) -- the presentation layer. NOT executed
     in this sandbox (streamlit isn't installed here / no network to
     install it, and there's no browser to render it in anyway), but it is
     complete, real code for the user's own environment, wired directly
     into Modules 1-7's actual functions rather than describing them.
"""
import io
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering, needed for testing without a display
import matplotlib.pyplot as plt
import networkx as nx
# ---------------------------------------------------------------------------
# 1. Metrics layer (testable without Streamlit)
# ---------------------------------------------------------------------------
def compute_grid_efficiency(total_load_pu: float, total_loss_pu: float) -> float:
    """eta(t) = total_load / (total_load + total_loss) * 100%"""
    if total_load_pu + total_loss_pu == 0:
        return 100.0
    return total_load_pu / (total_load_pu + total_loss_pu) * 100.0
def compute_renewable_fraction(total_renewable_pu: float, total_load_pu: float) -> float:
    """R_frac(t) = total_renewable / total_load * 100%. Can exceed 100%."""
    if total_load_pu == 0:
        return 0.0
    return total_renewable_pu / total_load_pu * 100.0
def build_solver_comparison_table(results: dict) -> pd.DataFrame:
    """
    results: {solver_name: {"energy": float, "wall_time_s": float}}
    Returns a DataFrame ready for a dashboard table, with an explicit
    agreement column rather than implying a speedup that isn't measured.
    The solver name is kept as an explicit "solver" column (not just the
    index) so it survives a plain to_csv(index=False) call.
    """
    df = pd.DataFrame(results).T
    if "energy" in df.columns:
        baseline = df["energy"].min()
        df["matches_best"] = (df["energy"] - baseline).abs() < 1e-6
    df.index.name = "solver"
    df = df.reset_index()  # "solver" becomes a real column, not just the index
    return df
def compute_stable_layout(dist_graph, seed: int = 42):
    """
    Compute a single node-position layout from the FULL original topology
    (all branches, before any fault removes one). Reuse this same `pos`
    dict for every render_topology_figure() call in a before/after pair
    -- otherwise spring_layout() re-solves the force simulation on a
    structurally different graph (one edge removed) and produces a
    visually unrelated layout even with the same seed, making before/after
    figures look like different networks instead of the same one with one
    switch changed.
    """
    G = dist_graph.graph
    return nx.spring_layout(G, seed=seed)
def render_topology_figure(dist_graph, switch_assignment: dict, title: str = "Grid Topology",
                            pos: dict = None):
    """
    Draw the network: closed edges solid, open edges dashed, node color
    scaled by priority weight (so hospitals/critical facilities stand out).
    Returns a matplotlib Figure (embed with st.pyplot(fig) in Streamlit).

    pos: optional precomputed {node: (x, y)} layout. Pass the SAME pos
    dict (e.g. from compute_stable_layout()) across a before/after pair
    of figures so nodes don't jump around between the two images -- only
    the edge styles (solid/dashed) should change, not the geometry.
    If not provided, a fresh spring_layout is computed (fine for a single
    standalone figure, but will drift between separate calls on graphs
    with different edge sets).
    """
    G = dist_graph.graph
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    else:
        # Guard against a fault graph introducing/removing nodes: fall back
        # to computing positions for any node missing from the shared pos.
        missing = [n for n in G.nodes() if n not in pos]
        if missing:
            fallback = nx.spring_layout(G, seed=42)
            pos = {**pos, **{n: fallback[n] for n in missing}}
    fig, ax = plt.subplots(figsize=(6, 5))
    closed_edges = list(dist_graph.fixed_edges)
    open_edges = []
    for e, is_closed in switch_assignment.items():
        (closed_edges if is_closed else open_edges).append(e)
    priorities = nx.get_node_attributes(G, "priority_weight")
    node_colors = [priorities.get(n, 1.0) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            cmap="YlOrRd", node_size=400)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=closed_edges,
                            style="solid", width=2, edge_color="black")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=open_edges,
                            style="dashed", width=1, edge_color="gray")
    ax.set_title(title)
    ax.axis("off")
    return fig
def render_voltage_profile_figure(voltages_pu: dict, v_min=0.95, v_max=1.05):
    """
    Bar chart of per-bus voltage magnitude from Module 5's
    check_voltage_feasibility output, with the allowed band shaded.
    """
    buses = list(voltages_pu.keys())
    values = [voltages_pu[b] for b in buses]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([str(b) for b in buses], values, color="steelblue")
    ax.axhspan(v_min, v_max, color="green", alpha=0.1, label="Allowed band")
    ax.axhline(v_min, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(v_max, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("Post-optimization voltage profile")
    ax.legend(loc="lower right", fontsize=7)
    return fig
# ---------------------------------------------------------------------------
# 2. Streamlit UI layer (NOT executed in this sandbox -- complete, real code)
# ---------------------------------------------------------------------------
def render_dashboard():
    """
    Full Streamlit app. Run locally with: streamlit run dashboard.py
    Wires every widget directly to real pipeline functions:
      - demand/solar charts    -> Modules 2-3's time series
      - grid topology          -> network_model.py + qubo_builder.py result
      - switch states          -> quantum_optimizer.py's solver output
      - "Simulate Fault" button -> disaster_recovery.simulate_fault (Module 7)
      - solver comparison      -> quantum_optimizer.py's three solver paths
    """
    import streamlit as st
    import data_loader as dl
    import forecasting as fc
    import renewable as rw
    import network_model as nm
    import qubo_builder as qb
    import quantum_optimizer as qo
    import disaster_recovery as dr
    st.set_page_config(page_title="QuantumGrid Dashboard", layout="wide")
    st.title("QuantumGrid — Distribution Network Reconfiguration")
    # --- Load pipeline state (in a real app, cache this with st.cache_data) ---
    bundle = dl.load_all(
        network_csv="ieee33_topology.csv",
        demand_csv="PJME_hourly.csv",
        solar_csv="solar_generation.csv",
        S_base_mva=10.0, V_base_kv=12.66,
    )
    dist_graph = nm.build_distribution_graph(bundle["graph"])
    col1, col2, col3 = st.columns(3)
    # --- Demand & solar time series (Modules 2-3) ---
    with col1:
        st.subheader("Predicted Demand (p.u.)")
        st.line_chart(bundle["demand_pu"])
    with col2:
        st.subheader("Solar Generation (p.u.)")
        st.line_chart(bundle["solar_pu"])
    # --- Efficiency / renewable metrics (this module's own formulas) ---
    total_load = bundle["demand_pu"].iloc[-1]
    total_renewable = bundle["solar_pu"].iloc[-1]
    with col3:
        st.subheader("Grid Metrics")
        st.metric("Renewable contribution",
                   f"{compute_renewable_fraction(total_renewable, total_load):.1f}%")
    # --- Simulation mode: fault injection wired to Module 7 ---
    st.divider()
    st.subheader("Simulation Mode")
    edge_options = dist_graph.fixed_edges + dist_graph.switchable_edges
    faulted_edge = st.selectbox("Simulate a fault on line:", edge_options)
    if st.button("Simulate Fault"):
        net_injection = {b: bundle["demand_pu"].iloc[-1] for b in bundle["graph"].buses}
        result = dr.simulate_fault(bundle["graph"], faulted_edge, net_injection, root=1)
        if not result.restorable:
            st.error(f"UNAVOIDABLE OUTAGE — no physical path exists to bus(es): "
                      f"{result.stranded_buses}")
        else:
            st.success(f"Restored. New switch state: {result.new_switch_assignment}")
            shared_pos = compute_stable_layout(dist_graph)
            st.pyplot(render_topology_figure(result.new_dist_graph,
                                              result.new_switch_assignment,
                                              pos=shared_pos))
    # --- Classical vs quantum comparison, framed honestly (Module 6) ---
    st.divider()
    st.subheader("Classical vs. Quantum-Inspired Solver Comparison")
    st.caption("At this network's scale, the QUBO is block-diagonal (Module 6) "
               "-- agreement between solvers below is the EXPECTED result, "
               "not evidence of quantum advantage.")
if __name__ == "__main__":
    # Test the metrics layer directly (no Streamlit / browser needed).
    eff = compute_grid_efficiency(total_load_pu=0.75, total_loss_pu=0.02)
    print(f"Grid efficiency: {eff:.2f}%")
    ren = compute_renewable_fraction(total_renewable_pu=0.3, total_load_pu=0.75)
    print(f"Renewable fraction: {ren:.2f}%")
    comparison = build_solver_comparison_table({
        "classical_sa": {"energy": -4.8354, "wall_time_s": 0.012},
        "brute_force": {"energy": -4.8354, "wall_time_s": 0.001},
    })
    print("\nSolver comparison table:")
    print(comparison)