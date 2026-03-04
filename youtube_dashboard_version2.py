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

st.set_page_config(
    page_title="YT Creator Topology",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# GLOBAL STYLES
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1f1f1f;
}
[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
[data-testid="stSidebar"] .stSlider > div > div { background: #1a1a1a; }

/* ---------- main background ---------- */
.stApp { background: #0a0a0a; color: #e0e0e0; }

/* ---------- metric cards ---------- */
.kpi-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 130px;
    background: #111;
    border: 1px solid #222;
    border-top: 2px solid #ff4500;
    border-radius: 4px;
    padding: 14px 18px;
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-label { font-size: 10px; letter-spacing: 2px; color: #666; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 600; color: #f0f0f0; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #555; margin-top: 4px; }

/* ---------- section headers ---------- */
.section-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #ff4500;
    border: 1px solid #ff4500;
    padding: 2px 10px;
    margin-bottom: 14px;
    border-radius: 2px;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: #111;
    border-bottom: 1px solid #222;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    color: #555 !important;
    border-radius: 0;
    padding: 10px 22px;
    border-bottom: 2px solid transparent;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #ff4500 !important;
    border-bottom: 2px solid #ff4500 !important;
}

/* ---------- dataframes ---------- */
.stDataFrame { border: 1px solid #1f1f1f; border-radius: 4px; }

/* ---------- download buttons ---------- */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #333 !important;
    color: #aaa !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    border-radius: 3px !important;
    transition: border-color .2s, color .2s;
}
.stDownloadButton > button:hover {
    border-color: #ff4500 !important;
    color: #ff4500 !important;
}

/* ---------- title area ---------- */
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: 1px;
    margin-bottom: 2px;
}
.dash-sub {
    font-size: 12px;
    color: #444;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 24px;
}

/* ---------- info boxes ---------- */
.insight-box {
    background: #111;
    border-left: 3px solid #ff4500;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
    color: #bbb;
    font-family: 'IBM Plex Mono', monospace;
}

/* misc */
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# =========================
# DATA LOADERS
# =========================
@st.cache_data(show_spinner=False)
def load_channels(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT channel_id, channel_title, seed_source FROM channels", conn)
    conn.close()
    return df


@st.cache_data(show_spinner=False)
def load_edges(db_path: str, edge_type: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            source_channel_id AS source,
            target_channel_id AS target,
            edge_weight        AS weight,
            edge_type,
            query,
            discovered_at
        FROM channel_edges
        WHERE edge_type = ?
    """, conn, params=(edge_type,))
    conn.close()
    return df


@st.cache_data(show_spinner=False)
def load_crawl_status(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM crawl_status ORDER BY outlinks_found DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


# =========================
# GRAPH HELPERS
# =========================
def build_graph(edges: pd.DataFrame, min_weight: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in edges[edges["weight"] >= min_weight].iterrows():
        u, v, w = r["source"], r["target"], int(r["weight"])
        if not u or not v or u == v:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def attach_titles(G: nx.DiGraph, channels: pd.DataFrame) -> None:
    title_map = dict(zip(channels["channel_id"], channels["channel_title"]))
    seed_map  = dict(zip(channels["channel_id"], channels["seed_source"]))
    for n in G.nodes():
        G.nodes[n]["title"]       = title_map.get(n) or n
        G.nodes[n]["seed_source"] = seed_map.get(n)  or ""


def weighted_degree_table(G: nx.DiGraph) -> pd.DataFrame:
    in_w  = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    in_d  = dict(G.in_degree())
    out_d = dict(G.out_degree())
    rows  = []
    for n in G.nodes():
        rows.append({
            "channel_id":    n,
            "channel_title": G.nodes[n].get("title", n),
            "in_degree":     in_d.get(n, 0),
            "out_degree":    out_d.get(n, 0),
            "in_weight":     int(in_w.get(n, 0)),
            "out_weight":    int(out_w.get(n, 0)),
            "total_degree":  in_d.get(n, 0) + out_d.get(n, 0),
            "total_weight":  int(in_w.get(n, 0) + out_w.get(n, 0)),
            "seed_source":   G.nodes[n].get("seed_source", ""),
        })
    return pd.DataFrame(rows).sort_values(["total_weight", "total_degree"], ascending=False)


def compute_metrics(G: nx.DiGraph) -> dict:
    n = G.number_of_nodes()
    wcc        = list(nx.weakly_connected_components(G))
    scc        = list(nx.strongly_connected_components(G))
    wcc_sizes  = sorted([len(c) for c in wcc], reverse=True)
    scc_sizes  = sorted([len(c) for c in scc], reverse=True)
    in_deg     = dict(G.in_degree())
    out_deg    = dict(G.out_degree())
    isolates   = sum(1 for x in G.nodes() if in_deg[x] == 0 and out_deg[x] == 0)
    sources    = sum(1 for x in G.nodes() if in_deg[x] == 0 and out_deg[x] > 0)
    sinks      = sum(1 for x in G.nodes() if in_deg[x] > 0  and out_deg[x] == 0)
    reciprocal = sum(1 for u, v in G.edges() if G.has_edge(v, u)) // 2
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 6) if n > 1 else 0.0,
        "weak_components": len(wcc),
        "largest_wcc_size": wcc_sizes[0] if wcc_sizes else 0,
        "largest_wcc_%": round((wcc_sizes[0] / n * 100) if n else 0, 1),
        "strong_components": len(scc),
        "largest_scc_size": scc_sizes[0] if scc_sizes else 0,
        "reciprocal_pairs": reciprocal,
        "isolates": isolates,
        "sources_only": sources,
        "sinks_only": sinks,
    }


def get_ego_subgraph(G_full: nx.DiGraph, node: str, depth: int) -> nx.DiGraph:
    nodes = {node}
    frontier = {node}
    for _ in range(depth):
        nxt = set()
        for x in frontier:
            nxt |= set(G_full.predecessors(x)) | set(G_full.successors(x))
        nodes |= nxt
        frontier = nxt
    return G_full.subgraph(nodes).copy()


def to_pyvis_html(G: nx.DiGraph, show_labels: bool, physics: bool, height_px: int, seed_node: str = None) -> str:
    net = Network(
        height=f"{height_px}px", width="100%", directed=True,
        bgcolor="#ffffff", font_color="#111111"
    )
    if physics:
        net.barnes_hut(gravity=-25000, central_gravity=0.25, spring_length=150, spring_strength=0.035, damping=0.09)
    else:
        net.toggle_physics(False)
    net.show_buttons(filter_=["physics"])

    wt = {n: G.in_degree(n, weight="weight") + G.out_degree(n, weight="weight") for n in G.nodes()}
    max_w = max(wt.values()) if wt else 1

    # color scheme: seed node = bright orange, others by out_degree intensity
    for n in G.nodes():
        title_label = G.nodes[n].get("title", n)
        w     = wt.get(n, 0)
        size  = 8 + 38 * (w / max_w)
        is_seed = (n == seed_node)

        ind = G.in_degree(n)
        outd = G.out_degree(n)
        inw  = int(G.in_degree(n, weight="weight"))
        outw = int(G.out_degree(n, weight="weight"))

        tooltip = (
            f"<b>{title_label}</b><br>"
            f"<span style='color:#888'>id:</span> {n}<br>"
            f"in: {ind} nodes / {inw} links &nbsp;|&nbsp; out: {outd} nodes / {outw} links<br>"
            f"seed: {G.nodes[n].get('seed_source','')}"
        )

        # Color gradient: low → #1a1a2e, mid → #3a86ff, high → #ff4500
        ratio = w / max_w
        if is_seed:
            color = "#ff4500"
            border = "#ffffff"
            size  = max(size, 30)
        elif ratio > 0.6:
            color = "#ff6b35"
        elif ratio > 0.3:
            color = "#3a86ff"
        elif ratio > 0.1:
            color = "#1a6fb5"
        else:
            color = "#ccccdd"
            border = "#aaaacc"

        border = "#ffffff" if is_seed else ("#444" if ratio < 0.1 else color)

        net.add_node(
            n,
            label=title_label if show_labels else "",
            title=tooltip,
            size=size,
            color={"background": color, "border": border, "highlight": {"background": "#ff4500", "border": "#fff"}},
            font={"size": max(9, int(8 + 6 * ratio)), "color": "#e0e0e0"},
        )

    for u, v, data in G.edges(data=True):
        w     = int(data.get("weight", 1))
        width = 0.5 + math.log(1 + w)
        net.add_edge(u, v, value=w, width=width, title=f"weight={w}",
                     color={"color": "#bbbbbb", "highlight": "#ff4500", "hover": "#ff6b35"},
                     arrows={"to": {"enabled": True, "scaleFactor": 0.5}})

    return net.generate_html()


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### 🕸️ YT TOPOLOGY")
    st.markdown("---")

    db_path = st.text_input("SQLite DB path", value=DEFAULT_DB)
    db_ok   = os.path.exists(db_path)
    if not db_ok:
        st.error("DB not found.")

    st.markdown("**Filters**")
    min_weight = st.slider("Min edge weight", 1, 10, 1)

    st.markdown("**Subgraph**")
    top_k = st.slider("Top-K nodes (by total weight)", 10, 1000, 60, step=5)

    ego_mode  = st.checkbox("Ego network mode", value=False)
    ego_depth = st.slider("Ego depth (hops)", 1, 3, 1) if ego_mode else 1

    st.markdown("**Visual**")
    show_labels = st.checkbox("Show labels", value=True)
    physics     = st.checkbox("Physics layout", value=True)
    height_px   = st.slider("Graph height (px)", 500, 1100, 850, 50)

    st.markdown("---")
    st.markdown(
        "<span style='font-size:10px;color:#444;font-family:monospace'>"
        "collab_link edges only</span>",
        unsafe_allow_html=True
    )

# =========================
# MAIN
# =========================
st.markdown('<div class="dash-title">CREATOR TOPOLOGY</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">youtube collaboration network · collab_link edges</div>', unsafe_allow_html=True)

if not db_ok:
    st.stop()

# Load data
with st.spinner("Loading graph data…"):
    channels_df   = load_channels(db_path)
    edges_df      = load_edges(db_path, "collab_link")
    crawl_df      = load_crawl_status(db_path)

if edges_df.empty:
    st.warning("No `collab_link` edges found in this database.")
    st.stop()

G_full = build_graph(edges_df, min_weight=min_weight)
attach_titles(G_full, channels_df)

ranking   = weighted_degree_table(G_full)
top_nodes = set(ranking.head(top_k)["channel_id"].tolist())
G         = G_full.subgraph(top_nodes).copy()

# Ego network override
ego_node = None
if ego_mode and not ranking.empty:
    options       = ranking["channel_title"].fillna(ranking["channel_id"]).tolist()
    title_to_id   = dict(zip(ranking["channel_title"].fillna(ranking["channel_id"]), ranking["channel_id"]))
    selected_title= st.sidebar.selectbox("Center node", options)
    ego_node      = title_to_id.get(selected_title)
    if ego_node and ego_node in G_full:
        G = get_ego_subgraph(G_full, ego_node, ego_depth)
        attach_titles(G, channels_df)

metrics = compute_metrics(G)
rank_view = weighted_degree_table(G)

# =========================
# KPI ROW
# =========================
wcc_pct = metrics["largest_wcc_%"]
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">nodes (view)</div>
    <div class="kpi-value">{metrics['nodes']:,}</div>
    <div class="kpi-sub">of {G_full.number_of_nodes():,} total</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">edges (view)</div>
    <div class="kpi-value">{metrics['edges']:,}</div>
    <div class="kpi-sub">min weight ≥ {min_weight}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">density</div>
    <div class="kpi-value">{metrics['density']:.5f}</div>
    <div class="kpi-sub">directed</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">weak components</div>
    <div class="kpi-value">{metrics['weak_components']:,}</div>
    <div class="kpi-sub">largest = {metrics['largest_wcc_size']} nodes ({wcc_pct}%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">reciprocal pairs</div>
    <div class="kpi-value">{metrics['reciprocal_pairs']:,}</div>
    <div class="kpi-sub">mutual collabs</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">sinks only</div>
    <div class="kpi-value">{metrics['sinks_only']:,}</div>
    <div class="kpi-sub">never linked out</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🕸️  Network Graph",
    "📊  Node Rankings",
    "🔍  Crawl Status",
    "📥  Export",
])

# ── TAB 1: GRAPH ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-tag">interactive network</div>', unsafe_allow_html=True)

    n_shown  = G.number_of_nodes()
    e_shown  = G.number_of_edges()
    st.markdown(
        f'<div class="insight-box">Showing <b>{n_shown}</b> nodes · <b>{e_shown}</b> edges'
        f'{"  ·  ego mode: <b>" + (G.nodes[ego_node].get("title", ego_node) if ego_node else "") + "</b>" if ego_mode and ego_node else ""}'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Rendering graph…"):
        html = to_pyvis_html(G, show_labels=show_labels, physics=physics,
                              height_px=height_px, seed_node=ego_node)
    components.html(html, height=height_px + 50, scrolling=True)

    st.caption("🟠 High-degree nodes · 🔵 Mid-degree · ⬛ Low-degree · Node size = total weighted degree")

# ── TAB 2: NODE RANKINGS ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-tag">node rankings</div>', unsafe_allow_html=True)

    col_sort, col_dir = st.columns([2, 1])
    with col_sort:
        sort_col = st.selectbox("Sort by", ["total_weight", "in_weight", "out_weight", "total_degree", "in_degree", "out_degree"])
    with col_dir:
        sort_asc = st.checkbox("Ascending", value=False)

    display_df = rank_view.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
    display_df.index += 1  # 1-based rank

    st.dataframe(
        display_df[["channel_title", "in_degree", "out_degree", "in_weight", "out_weight", "total_weight", "seed_source"]].rename(columns={
            "channel_title": "Channel",
            "in_degree":     "In-Deg",
            "out_degree":    "Out-Deg",
            "in_weight":     "In-W",
            "out_weight":    "Out-W",
            "total_weight":  "Total-W",
            "seed_source":   "Seed",
        }),
        use_container_width=True,
        height=500,
    )

    # Top hub vs top authority split
    st.markdown('<div class="section-tag">hubs vs authorities</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top outlinkers (hubs)**")
        hubs = rank_view.sort_values("out_weight", ascending=False).head(10)[["channel_title", "out_degree", "out_weight"]]
        hubs.columns = ["Channel", "Out-Deg", "Out-W"]
        st.dataframe(hubs, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Top cited (authorities)**")
        auths = rank_view.sort_values("in_weight", ascending=False).head(10)[["channel_title", "in_degree", "in_weight"]]
        auths.columns = ["Channel", "In-Deg", "In-W"]
        st.dataframe(auths, use_container_width=True, hide_index=True)

# ── TAB 3: CRAWL STATUS ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-tag">crawl status</div>', unsafe_allow_html=True)

    if crawl_df.empty:
        st.info("No crawl_status table found in this database.")
    else:
        # Summary stats
        total_scanned  = len(crawl_df)
        total_videos   = int(crawl_df["videos_checked"].sum()) if "videos_checked" in crawl_df.columns else 0
        total_outlinks = int(crawl_df["outlinks_found"].sum()) if "outlinks_found" in crawl_df.columns else 0
        channels_w_links = int((crawl_df["outlinks_found"] > 0).sum()) if "outlinks_found" in crawl_df.columns else 0

        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card"><div class="kpi-label">channels scanned</div><div class="kpi-value">{total_scanned:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">videos checked</div><div class="kpi-value">{total_videos:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">outlinks found</div><div class="kpi-value">{total_outlinks:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">channels w/ links</div><div class="kpi-value">{channels_w_links:,}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Merge channel titles into crawl status
        crawl_display = crawl_df.merge(
            channels_df[["channel_id", "channel_title"]], on="channel_id", how="left"
        )
        crawl_display["channel_title"] = crawl_display["channel_title"].fillna(crawl_display["channel_id"])

        display_cols = [c for c in ["channel_title", "scanned_at", "videos_checked", "desc_with_links", "outlinks_found", "stop_reason"] if c in crawl_display.columns]
        st.dataframe(
            crawl_display[display_cols].rename(columns={
                "channel_title":    "Channel",
                "scanned_at":       "Scanned At",
                "videos_checked":   "Videos",
                "desc_with_links":  "Desc w/ Links",
                "outlinks_found":   "Outlinks",
                "stop_reason":      "Stop Reason",
            }),
            use_container_width=True,
            height=500,
        )

        # Stop reason breakdown
        if "stop_reason" in crawl_df.columns:
            st.markdown('<div class="section-tag">stop reasons</div>', unsafe_allow_html=True)
            reason_counts = crawl_df["stop_reason"].fillna("success").value_counts().reset_index()
            reason_counts.columns = ["Reason", "Count"]
            st.dataframe(reason_counts, use_container_width=True, hide_index=True)

# ── TAB 4: EXPORT ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-tag">data export</div>', unsafe_allow_html=True)
    st.markdown("All exports reflect the **current view** (top-K / ego filter applied).")

    nodes_export = weighted_degree_table(G)
    edges_export = pd.DataFrame([
        {
            "source_channel_id": u,
            "target_channel_id": v,
            "edge_weight":       int(d.get("weight", 1)),
            "source_title":      G.nodes[u].get("title", u),
            "target_title":      G.nodes[v].get("title", v),
        }
        for u, v, d in G.edges(data=True)
    ])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇  nodes.csv",
            data=nodes_export.to_csv(index=False).encode("utf-8"),
            file_name="topology_nodes.csv", mime="text/csv",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "⬇  edges.csv",
            data=edges_export.to_csv(index=False).encode("utf-8"),
            file_name="topology_edges.csv", mime="text/csv",
            use_container_width=True
        )
    with c3:
        if not crawl_df.empty:
            st.download_button(
                "⬇  crawl_status.csv",
                data=crawl_df.to_csv(index=False).encode("utf-8"),
                file_name="crawl_status.csv", mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("**Preview — nodes**")
    st.dataframe(nodes_export.head(20), use_container_width=True, hide_index=True)
    st.markdown("**Preview — edges**")
    st.dataframe(edges_export.head(20), use_container_width=True, hide_index=True)