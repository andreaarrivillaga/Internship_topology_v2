import os
import math
import sqlite3
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


# =========================
# CONFIG
# =========================
DEFAULT_DB = "youtube_topology_v3.sqlite"


# =========================
# DB helpers
# =========================
@st.cache_data(show_spinner=False)
def load_channels(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT channel_id, channel_title, seed_source
        FROM channels
        """,
        conn,
    )
    conn.close()
    return df


@st.cache_data(show_spinner=False)
def load_edges(db_path: str, edge_type: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            source_channel_id AS source,
            target_channel_id AS target,
            edge_weight AS weight,
            edge_type,
            query,
            discovered_at
        FROM channel_edges
        WHERE edge_type = ?
        """,
        conn,
        params=(edge_type,),
    )
    conn.close()
    return df


def attach_titles(G: nx.DiGraph, channels: pd.DataFrame) -> None:
    title_map = dict(zip(channels["channel_id"], channels["channel_title"]))
    seed_map = dict(zip(channels["channel_id"], channels["seed_source"]))
    for n in G.nodes():
        G.nodes[n]["title"] = title_map.get(n) or n
        G.nodes[n]["seed_source"] = seed_map.get(n) or ""


def build_graph(edges: pd.DataFrame, min_weight: int) -> nx.DiGraph:
    edges = edges[edges["weight"] >= min_weight].copy()

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        u, v = r["source"], r["target"]
        w = int(r["weight"])
        if not u or not v or u == v:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def weighted_degree_table(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    in_w = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    in_d = dict(G.in_degree())
    out_d = dict(G.out_degree())

    for n in G.nodes():
        rows.append(
            {
                "channel_id": n,
                "channel_title": G.nodes[n].get("title", n),
                "in_degree": in_d.get(n, 0),
                "out_degree": out_d.get(n, 0),
                "in_weight": int(in_w.get(n, 0)),
                "out_weight": int(out_w.get(n, 0)),
                "total_degree": in_d.get(n, 0) + out_d.get(n, 0),
                "total_weight": int(in_w.get(n, 0) + out_w.get(n, 0)),
                "seed_source": G.nodes[n].get("seed_source", ""),
            }
        )

    df = pd.DataFrame(rows).sort_values(["total_weight", "total_degree"], ascending=False)
    return df


def compute_metrics(G: nx.DiGraph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    wcc = list(nx.weakly_connected_components(G))
    wcc_sizes = sorted([len(c) for c in wcc], reverse=True)
    largest_wcc = wcc_sizes[0] if wcc_sizes else 0

    scc = list(nx.strongly_connected_components(G))
    scc_sizes = sorted([len(c) for c in scc], reverse=True)
    largest_scc = scc_sizes[0] if scc_sizes else 0

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    isolates = [n for n in G.nodes() if in_deg.get(n, 0) == 0 and out_deg.get(n, 0) == 0]
    sources_only = [n for n in G.nodes() if in_deg.get(n, 0) == 0 and out_deg.get(n, 0) > 0]
    sinks_only = [n for n in G.nodes() if in_deg.get(n, 0) > 0 and out_deg.get(n, 0) == 0]

    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "weak_components": len(wcc),
        "largest_weak_component_size": largest_wcc,
        "largest_weak_component_pct": (largest_wcc / n) if n else 0.0,
        "strong_components": len(scc),
        "largest_strong_component_size": largest_scc,
        "isolates": len(isolates),
        "sources_only": len(sources_only),
        "sinks_only": len(sinks_only),
    }


def to_pyvis_html(
    G: nx.DiGraph,
    show_labels: bool,
    physics: bool,
    height_px: int = 800,
) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    if physics:
        net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=130, spring_strength=0.04, damping=0.09)
    else:
        net.toggle_physics(False)

    # Improve UX: enable a settings widget in the HTML (lets your prof tweak layout)
    net.show_buttons(filter_=["physics"])

    weighted_total = {n: G.in_degree(n, weight="weight") + G.out_degree(n, weight="weight") for n in G.nodes()}
    max_w = max(weighted_total.values()) if weighted_total else 1

    for n in G.nodes():
        title = G.nodes[n].get("title", n)
        w = weighted_total.get(n, 0)
        size = 8 + 35 * (w / max_w) if max_w else 10

        tooltip = (
            f"{title}<br>"
            f"channel_id: {n}<br>"
            f"in_degree: {G.in_degree(n)} | out_degree: {G.out_degree(n)}<br>"
            f"in_weight: {int(G.in_degree(n, weight='weight'))} | out_weight: {int(G.out_degree(n, weight='weight'))}"
        )

        label = title if show_labels else ""
        net.add_node(n, label=label, title=tooltip, size=size)

    for u, v, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        width = 1 + math.log(1 + w)
        net.add_edge(u, v, value=w, width=width, title=f"weight={w}")

    return net.generate_html()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="YouTube Topology Dashboard", layout="wide")
st.title("YouTube Creator Topology Dashboard")

with st.sidebar:
    st.header("Data Source")
    db_path = st.text_input("SQLite DB path", value=DEFAULT_DB)
    if not os.path.exists(db_path):
        st.error("DB path not found. Put the DB in this folder or update the path.")

    st.header("Filters")
    edge_type = "collab_link"
    min_weight = st.slider("Min edge weight", min_value=1, max_value=10, value=1)

    st.header("Subgraph")
    top_k = st.slider("Top-K nodes (by total_weight)", min_value=10, max_value=1500, value=50, step=5)
    ego_mode = st.checkbox("Ego network (focus on one node)", value=False)
    ego_depth = st.slider("Ego depth (hops)", 1, 3, 1) if ego_mode else 1

    st.header("Visualization")
    show_labels = st.checkbox("Show node labels", value=True)
    physics = st.checkbox("Enable physics layout", value=True)
    height_px = st.slider("Graph height (px)", 500, 1100, 850, 50)

if os.path.exists(db_path):
    channels_df = load_channels(db_path)
    edges_df = load_edges(db_path, edge_type=edge_type)

    if edges_df.empty:
        st.warning(f"No edges found for edge_type='{edge_type}'. Try the other edge type.")
        st.stop()

    G_full = build_graph(edges_df, min_weight=min_weight)
    attach_titles(G_full, channels_df)

    # Rank nodes and keep Top-K
    ranking = weighted_degree_table(G_full)
    top_nodes = set(ranking.head(top_k)["channel_id"].tolist())

    G = G_full.subgraph(top_nodes).copy()

    # Optional ego subgraph
    selected_node = None
    if ego_mode:
        options = ranking["channel_title"].fillna(ranking["channel_id"]).tolist()
        title_to_id = dict(zip(ranking["channel_title"].fillna(ranking["channel_id"]), ranking["channel_id"]))
        selected_title = st.sidebar.selectbox("Select node", options)
        selected_node = title_to_id.get(selected_title)

        if selected_node and selected_node in G_full:
            nodes_ego = set([selected_node])
            frontier = {selected_node}
            for _ in range(ego_depth):
                nxt = set()
                for x in frontier:
                    nxt |= set(G_full.predecessors(x))
                    nxt |= set(G_full.successors(x))
                nodes_ego |= nxt
                frontier = nxt
            G = G_full.subgraph(nodes_ego).copy()
            attach_titles(G, channels_df)

    # Metrics + tables
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Coverage Metrics")
        metrics = compute_metrics(G)
        st.json(metrics)

    with colB:
        st.subheader("Top Nodes (current view)")
        rank_view = weighted_degree_table(G).head(25)
        st.dataframe(rank_view, use_container_width=True, height=380)

    # Downloads
    st.subheader("Exports")
    nodes_export = weighted_degree_table(G)
    edges_export = pd.DataFrame(
        [
            {
                "source_channel_id": u,
                "target_channel_id": v,
                "edge_weight": int(d.get("weight", 1)),
                "source_title": G.nodes[u].get("title", u),
                "target_title": G.nodes[v].get("title", v),
            }
            for u, v, d in G.edges(data=True)
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download nodes CSV",
            data=nodes_export.to_csv(index=False).encode("utf-8"),
            file_name="nodes_export_streamlit.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download edges CSV",
            data=edges_export.to_csv(index=False).encode("utf-8"),
            file_name="edges_export_streamlit.csv",
            mime="text/csv",
        )

    # Render graph
    st.subheader("Interactive Network")
    html = to_pyvis_html(G, show_labels=show_labels, physics=physics, height_px=height_px)
    components.html(html, height=height_px + 50, scrolling=True)

    st.caption(
        "Tip: Use the Physics button panel (top-right of the PyVis widget) to tune spacing if nodes overlap."
    )
