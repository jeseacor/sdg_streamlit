import streamlit as st
import pandas as pd
import numpy as np
import hashlib, json, re, os
from pandas.util import hash_pandas_object
from base64 import b64encode
from pathlib import Path
from streamlit_option_menu import option_menu
import plotly.io as pio

# ---- New API (plotly.io.defaults.*) ----
try:
    pio.defaults.width  = 900
    pio.defaults.height = 600
    pio.defaults.scale  = 1
    pio.defaults.format = "png"
except Exception:
    pass

# ---- Legacy API (kaleido.scope.*) for older combos ----
try:
    sc = pio.kaleido.scope
    if hasattr(sc, "default_width"):  sc.default_width  = 900
    if hasattr(sc, "default_height"): sc.default_height = 600
    if hasattr(sc, "default_scale"):  sc.default_scale  = 1
    if hasattr(sc, "default_format"): sc.default_format = "png"
    if hasattr(sc, "mathjax"):        sc.mathjax = None
except Exception:
    pass


key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if key:
    os.environ["OPENAI_API_KEY"] = key
else:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets or your local .env")
    st.stop()

from sdg_module import ProjectKit

st.set_page_config(page_title="GoalScope SDG Analytics Hub", layout="wide")

def _insights_store() -> dict:
    """Per-session dict for AI insights (plots & tables)."""
    if "insights_store" not in st.session_state:
        st.session_state.insights_store = {}
    return st.session_state.insights_store

def _fig_hash(fig) -> str:
    try:
        payload = json.dumps(fig.to_plotly_json(), sort_keys=True, default=str)
    except Exception:
        payload = repr(fig)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _df_hash(df) -> str:
    try:
        m = hashlib.sha1()
        m.update(str(df.shape).encode())
        m.update("|".join(map(str, df.columns)).encode())
        m.update(hash_pandas_object(df, index=True).values.tobytes())
        return m.hexdigest()
    except Exception:
        payload = df.to_json(orient="split", index=True, date_format="iso", default_handler=str)
        return hashlib.sha1(payload.encode()).hexdigest()

def _insight_key(page_key: str, *, fig=None, df=None, tag: str | None = None) -> str:
    if tag:
        return f"{page_key}:tag:{tag}"
    if df is not None:
        return f"{page_key}:df:{_df_hash(df)}"
    if fig is not None:
        return f"{page_key}:{_fig_hash(fig)}"
    return page_key

def load_insight(page_key: str, fig=None, *, df=None, tag: str | None = None) -> str | None:
    store = _insights_store()
    return store.get(_insight_key(page_key, fig=fig, df=df, tag=tag))

def save_insight(page_key: str, fig, text: str, *, df=None, tag: str | None = None) -> None:
    # keeps your existing (page_key, fig, text) signature working
    store = _insights_store()
    store[_insight_key(page_key, fig=fig, df=df, tag=tag)] = text

def clear_insight(page_key: str, fig=None, *, df=None, tag: str | None = None) -> None:
    store = _insights_store()
    if any(x is not None for x in (fig, df, tag)):
        store.pop(_insight_key(page_key, fig=fig, df=df, tag=tag), None)
    else:
        for k in [k for k in list(store) if k.startswith(page_key + ":") or k == page_key]:
            store.pop(k, None)


def linked_image_local(img_path: str, url: str, *, width: int|None=None, radius: int | str = 12, sidebar=False, shadow: bool = False):
    data = Path(img_path).read_bytes()
    b64  = b64encode(data).decode("utf-8")
    ext = (Path(img_path).suffix or ".png").lstrip(".")
    wcss = f"width:{width}px;" if width else "width:100%;"
    rcss = f"{int(radius)}px" if isinstance(radius, (int, float)) else str(radius)
    scss = "box-shadow:0 2px 12px rgba(0,0,0,.25);" if shadow else ""
    html = f"""
    <a href="{url}" target="_blank" rel="noopener noreferrer" style="display:block; text-decoration:none;">
      <div style="{wcss} border-radius:{rcss}; overflow:hidden; {scss} line-height:0;">
        <img src="data:image/{ext};base64,{b64}" style="width:100%; height:auto; display:block; border:0;" />
      </div>
    </a>
    """
    (st.sidebar if sidebar else st).markdown(html, unsafe_allow_html=True)


def show_plot(df = None, fig = None, page_key = ""):
    PAGE_KEY = page_key
    if df is not None:
        st.dataframe(df, hide_index=True, use_container_width=True)        
        if st.session_state.get("chat_open"):
            kit.update_plot_context(df=df, title=page_key) 
        else:
            kit.update_plot_context(title=page_key)
        saved = load_insight(PAGE_KEY, df=df)
        gen = st.button("🤖 Generate Insight", key=page_key + "btn")
        if gen:
            with st.spinner("Generating insight…"):
                md = kit.generate_nlp_insight(df=df, title=page_key) or "_No insight returned._"
            save_insight(PAGE_KEY, None, md, df=df) 
            saved = md
            st.toast("Insight saved for this table.", icon="💾")

    elif fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        if st.session_state.get("chat_open"):
            kit.update_plot_context(fig=fig, title=page_key)
        else:
            kit.update_plot_context(title=fig.layout.title.text or page_key)
        saved = load_insight(PAGE_KEY, fig=fig)
        gen = st.button("🤖 Generate Insight", key=page_key + "btn")
        if gen:
            with st.spinner("Generating insight…"):
                md = kit.generate_nlp_insight(fig=fig, title=page_key) or "_No insight returned._"
            save_insight(PAGE_KEY, fig, md) 
            saved = md
            st.toast("Insight saved for this table.", icon="💾")

    if saved:
        st.markdown(saved)
    else:
        st.info("No saved insight yet. Click **🤖 Generate Insight**.")    


# --- Chat state ---
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {'role','content'}

@st.dialog("🤖 Chat with GoalScope Assistant")
def chat_dialog():
    st.caption("Ask about goals, indicators, methods, or how to interpret a chart.")

    # history
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # input
    user_msg = st.chat_input("Type your question…")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = kit.chat(
                    user_msg,
                    history=st.session_state.chat_history,
                    model="gpt-5-mini",
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # footer actions
    st.divider()
    a, b = st.columns(2)
    with a:
        if st.button("🧹 Clear", key="chat_clear_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with b:
        if st.button("✖ Close", key="chat_close_btn", use_container_width=True):
            st.session_state.chat_open = False
            st.rerun()


def youtube_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url  # allow passing the raw 11-char ID

def youtube_autoplay(url_or_id: str, *, start=0, loop=False, height=360):
    vid = youtube_id(url_or_id)
    # loop requires playlist=VIDEO_ID
    loop_params = f"&loop=1&playlist={vid}" if loop else ""
    src = (
        f"https://www.youtube.com/embed/{vid}"
        f"?autoplay=1&mute=1&controls=1&modestbranding=1&rel=0"
        f"&playsinline=1&start={start}{loop_params}"
    )
    html = f"""
    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;">
      <iframe
        src="{src}"
        frameborder="0"
        allow="autoplay; encrypted-media; picture-in-picture"
        allowfullscreen
        style="position:absolute;top:0;left:0;width:100%;height:100%;">
      </iframe>
    </div>
    """
    st.components.v1.html(html, height=height, scrolling=False)



# -------------------- Data Loading --------------------
@st.cache_data(show_spinner=True)
def load_data():
    kit = ProjectKit()
    df_sdg, df_lookup = kit.get_clean_data()
    df_lookup = df_lookup.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return df_sdg, df_lookup

@st.cache_resource(show_spinner=False)
def get_kit():
    return ProjectKit()

try:
    df_sdg, df_lookup = load_data()
    kit = get_kit()
    kit.set_data_for_chat(df_sdg, df_lookup)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()



# -------------------- Lists --------------------
YEARS = sorted([int(y) for y in df_sdg["Year"].dropna().unique().tolist()])
COUNTRIES = sorted(df_sdg["Country"].dropna().unique().tolist())
REGIONS = sorted(df_sdg["Region"].dropna().unique().tolist())
GOAL_COLS = [c for c in df_sdg.columns if str(c).startswith("Goal_")]
SDG_COLS  = [c for c in df_sdg.columns if str(c).lower().startswith("sdg") and len(c) > 3]
GROUPS = ["economic", "social", "environmental", "partnership"]


# -------------------- Sidebar Navigation --------------------
PAGES = [
    "Home",
    "Ranking",
    "Trends & Timelines",
    "Network & Structure",
    "Correlations",
    "About"
]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

with st.sidebar:

    #here = Path(__file__).parent
    #logo_path = here / "sdg_logo_b.png" 
    #if logo_path.exists():
    #    st.image(str(logo_path), use_container_width=True)
    #    #st.caption("Te Kunenga ki Pūrehuroa")
    #else:
    #    st.info("image not found")

    linked_image_local("assets/mu_logo.png", "https://www.massey.ac.nz", width=220, radius=5, sidebar=True)
    st.divider()

    page = option_menu(
        None,  # no title inside the menu
        PAGES,
        icons=["house", "trophy", "graph-up",
               "diagram-3", "link-45deg", "info-circle"],
        menu_icon="cast",
        default_index=PAGES.index(st.session_state.nav_page),
        key="nav_menu",
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            #"icon": {"color": "#9aa0a6", "font-size": "18px"},
            "icon": {"color": "#e4a024", "font-size": "18px"},
            "nav-link": {
                "font-size": "12px", "text-align": "left",
                "margin": "4px 0", "padding": "8px 12px",
                "border-radius": "10px",
            },
            "nav-link-selected": {"background-color": "#4789C8", "color": "white"},
        },
    )
    # keep current page in session so your app remembers it on rerun
    st.session_state.nav_page = page

    prev_page = st.session_state.get("last_nav_page")
    if prev_page != page:
        st.session_state.chat_open = False
    st.session_state.last_nav_page = page


    linked_image_local("assets/sdg_logo_b.png", "https://sdgs.un.org/goals", width=220, sidebar=True)
    
    st.divider()
    if st.button("🤖 Chatbot", key="chat_sidebar_btn", use_container_width=True):
        st.session_state.chat_open = True
        chat_dialog()    


# --- Floating Chat Button (bottom-right) ---
st.markdown("""
<style>
#chat-fab { position: fixed; right: 20px; bottom: 20px; z-index: 9999; }
#chat-fab button { border-radius: 999px; padding: .65rem 1rem; font-weight: 700; }
</style>
<div id="chat-fab"></div>
""", unsafe_allow_html=True)



# -------------------- Pages --------------------

st.markdown("""
<style>
/* Main hero section */
.hero {
    font-famliy: Arial;
    text-align: center;
    margin-bottom: 2em;
}
.hero h1 {
    font-famliy: Arial;
    font-size: 2.2em;
    font-weight: 900;
    letter-spacing: -1.5px;
    line-height: 1.1;
    color: #d9d9d9;
    text-shadow: 1px 1px 8px rgba(0,0,0,1);
}
.hero b {
    color: white;
}
.hero .brand {
    font-famliy: Arial;
    font-size: 2em;
    color: #d4a024;
}
.hero .dot {
    font-size: 2em;
    color: #4789C8;
}
.hero .sub {
    font-size: 0.8em;
    color: #d9d9d9;
}
.hero p {
    font-size: 1.2em;
    color: #aaa;
    margin-top: 0.5em;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}
.section-title {
    font-size: 3em;
    font-weight: 700;
    margin-bottom: 0.4em;
    text-shadow: 1px 4px 12px rgba(0,0,0,0.4);
    color: #4789C8;
}
.feature-card {
    font-size: 1.3em;
    font-weight: 700;
    margin-bottom: 0.4em;
    color: #4789C8;      
    padding: 1.2em;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,1);
    margin-bottom: 1.5em;
}
.feature-card h4 {
    margin-top: 0.2em;
    margin-bottom: 0.4em;
    font-weight: 800;
    font-size: 1.8em;
    color: white;
}
.feature-card ul {
    margin: 0;
}
.feature-card li {
    font-size: 0.95em;
    color: #ccc;
}
.meta { font-size:.9rem; opacity:.85; margin-top:.25rem; }
</style>
""", unsafe_allow_html=True)

#logo_path = here / "sdg_grid.png" 
# -------------------- Home --------------------
if page == "Home":

    #st.image("assets/goalscope_logo.jpg", use_container_width=True)    
    #<h1><span class="dot">GOAL</span><span class="brand">Scope </span><span class="sub">{ SDG Analytics Hub }</span></h1>
    st.markdown("""
    <div class="hero">
        <h1><span class="dot">GOAL</span><span class="brand">Scope </span><span class="sub">{ SDG Analytics Hub }</span></h1>
        <p>
            Welcome to Massey University GOALScope. A complete application for analyzing the latest, dynamic UN Sustainable Development Goals (SDG)
            data — augmented with fast AI insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #st.divider()
    c1a, c2a = st.columns([1, 1], vertical_alignment="top")
    with c1a:
        with st.container(border=True):
            st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card"> 
                <ul>
                    <li><b>Rankings:</b> Compare Overall Score or any Goal (1–17) / indicator across countries or regions.</li>
                    <li><b>Timelines:</b> Track changes through time for selected places and measures.</li>
                    <li><b>Correlations:</b> Scan associations between measures to spot patterns and trade-offs.</li>
                    <li><b>Summaries:</b> One-click AI insights saved per chart/table for easy revisits.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-title">Benefits</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card"> 
                <ul>
                    <li>A single, reliable space to <b>explore, compare, and explain</b> SDG performance.</li>
                    <li>Clear visuals backed by on-demand AI narratives — great for coursework, reports, and presentations.</li>
                    <li>Flexible filters for <b>Year</b>, <b>Scope</b> (Country/Region), and <b>Goal/Indicator</b> or <b>Overall</b>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)        
                        
    with c2a:
        with st.container(border=True):    
            st.markdown('<div class="section-title">The 17 SDGs</div>', unsafe_allow_html=True)
            youtube_autoplay("https://youtu.be/0XTBYMfZyrM?si=yZtN5MVYe4OGoY-t", start=0, loop=True, height=400)

        with st.container(border=True):
            st.markdown('<div class="section-title">Data & Method</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card"> 
                <ul>
                    <li>Data sourced from <b>ArcGIS FeatureServer / UN SDG APIs</b> and loaded into pandas for wrangling.</li>
                    <li>Correlation analysis highlights relationships across indicators and goals; literature guides interpretation.</li>
                    <li>The app integrates <b>NLP</b> to generate dynamic plot insights.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)        

    #st.divider()


    c1, c2, c3 = st.columns([1, 4, 1], vertical_alignment="center")
    with c2:
        st.image("assets/sdg_grid.png", use_container_width=True)    


    #st.stop()


if page == "Ranking":
    st.markdown('<div class="section-title">Performance Rankings</div>', unsafe_allow_html=True)

    sub = st.segmented_control(
        "",
        ["Rankings", "Percent Change", "Bar Chart (Locale)", "Bar Chart (SDG)"], default="Rankings",
        key="ranking_view",
    )

    #tab_rank, tab_loc, tab_sdg  = st.tabs(["Rankings", "Bar Chart (Locale)", "Bar Chart(SDG)"])

    if sub == "Rankings":
        #st.subheader("get_ranking_table")
        c1, c2 = st.columns(2)
        with c1:
            years = sorted(YEARS, reverse=True)
            year = st.selectbox("Year", years, index=0, key="rnk_year")
        with c2:
            #value_col = st.selectbox("Goal/SDG or Group)", sorted(GOAL_COLS + SDG_COLS + GROUPS))
            goal_opts = ["Overall Score", *GOAL_COLS, *GROUPS, *SDG_COLS]
            sel = st.selectbox("Goal or Group", goal_opts, index=0, key="rnk_goal_choice")
            value_col = None if sel in ("", "Overall Score") else sel

        try:
            df_rank = kit.get_ranking_table(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                goal=value_col
            )
            show_plot(df=df_rank, page_key="get_ranking_table")

        except Exception as e:
            st.warning(f"Could not render: {e}")

    elif sub == "Percent Change":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            y0, y1 = st.select_slider("Start → End", options=YEARS, value=(YEARS[0], YEARS[-1]))
        with c2:
            level = st.selectbox("Level", ["goal", "sdg"], index=0)
        with c3:
            group_filter = st.multiselect("Group filter (optional)", GROUPS)
        with c4:
            entity_type = st.selectbox("Entity type", ["All", "Region", "Country"], index=0)
        with c5:
            sort_desc = st.checkbox("Sort high → low", value=False)

        entities = []
        if entity_type == "Region":
            entities = st.multiselect("Pick regions", REGIONS)
        elif entity_type == "Country":
            entities = st.multiselect("Pick countries", COUNTRIES)

        df_pct, fig = kit.pct_change_sdg(
            df_sdg=df_sdg, df_lookup=df_lookup,
            start_year=int(y0), end_year=int(y1),
            level=level,
            group_filter=group_filter if group_filter else None,
            entity_type=None if entity_type == "All" else entity_type,
            entities=entities if entities else None,
            sort_desc=sort_desc,
            return_fig=True,
        )
        show_plot(fig=fig, page_key="pct_change_sdg")




    elif sub == "Bar Chart (Locale)":
        #st.subheader("plot_sdg_ranking")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            years = sorted(YEARS, reverse=True)
            year = st.selectbox("Year", years, index=0, key="er_year")
        with c2:
            #value_col = st.selectbox("Goal/SDG or Group)", sorted(GOAL_COLS + SDG_COLS + GROUPS))
            goal_opts = ["Overall Score", *GOAL_COLS, *GROUPS, *SDG_COLS]
            sel = st.selectbox("Goal or Group", goal_opts, index=0, key="goal_choice")
            value_col = None if sel in ("", "Overall Score") else sel

        with c3:
            group_col = st.selectbox("Group column", ["Country", "Region"], index=0)
        with c4:
            top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)
        with c5:
            h = st.slider("Figure height", 400, 1200, 500, step=50, key="er_h")
        with c6:
            ascending = st.checkbox("Ascending", value=False, key="er_asc")
        try:
            fig = kit.plot_sdg_ranking(
                data=df_sdg,
                value_col=value_col if value_col else None,
                group_col=group_col,
                top_n=int(top_n),
                ascending=ascending,
                year=year,
                df_lookup=df_lookup,
                fig_height=h,
                fig_width=1100,
            )
            show_plot(fig=fig, page_key="plot_sdg_ranking")

        except Exception as e:
            st.warning(f"Could not render: {e}")


    elif sub == "Bar Chart (SDG)":
        #st.subheader("plot_goal_ranking")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="gr_year")
        with c2:
            rank_by = st.selectbox("Rank by", ["goal", "sdg", "group"], index=0, key="gr_rankby")
        with c3:
            group_filter = st.multiselect("Filter groups (optional)", GROUPS, key="gr_groups") if rank_by != "group" else []
        with c4:
            h = st.slider("Figure height", 400, 1200, 600, step=50, key="gr_h")
        with c5:
            ascending = st.checkbox("Ascending", value=True, key="gr_asc")
    

        try:
            fig = kit.plot_goal_ranking(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                rank_by=rank_by,
                group_filter=group_filter if group_filter else None,
                ascending=ascending,
                fig_height=h,
                fig_width=1100,
            )
            show_plot(fig=fig, page_key="plot_goal_ranking")

        except Exception as e:
            st.warning(f"Could not render: {e}")
    

elif page == "Trends & Timelines":
    st.markdown('<div class="section-title">Trends & Timelines</div>', unsafe_allow_html=True)

    sub = st.segmented_control(
        "",
        ["Timeline (Locale)", "Timeline (SDG)"], default="Timeline (Locale)",
        key="timeline_view",
    )

    #tab_loc, tab_sdg  = st.tabs(["Timeline (Locale)", "Timeline (SDG)"])
    
    if sub == "Timeline (Locale)":
        #st.subheader("plot_goal_entity_timeline")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            goal_opts = ["Overall Score", *GOAL_COLS, *SDG_COLS]
            sel = st.selectbox("Goal", goal_opts, index=0, key="goal_choice")
            goal = None if sel == "" else sel   # map blank to None
        with c2:
            entity_type = st.selectbox("Entity type", ["Country", "Region"], index=0)
        with c3:
            entities = REGIONS if entity_type == "Region" else COUNTRIES
            picked = st.multiselect("Entities", entities)
        with c4:
            top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)
        with c5:
            h = st.slider("Figure height", 400, 1200, 600, step=50, key="gr_h")
        try:
            fig = kit.plot_goal_entity_timeline(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                goal=goal if goal else None,
                entity_type=entity_type,
                entities=picked if picked else None,
                agg="mean",
                fig_height=h,
                top_n=top_n
            )
            show_plot(fig=fig, page_key="plot_goal_entity_timeline")

        except Exception as e:
            st.warning(f"Could not render: {e}")

    elif sub == "Timeline (SDG)":
        #st.subheader("plot_sdg_timeline")
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox("Mode", ["goal", "sdg"], index=0)
            grp_filter = st.multiselect("Group filter", GROUPS)
        with c2:
            entity_type = st.selectbox("Entity type", ["Region", "Country"], index=0)
            entities = []
            if entity_type == "Region":
                entities = st.multiselect("Regions", REGIONS)
            elif entity_type == "Country":
                entities = st.multiselect("Countries", COUNTRIES)
        with c3:
            #pool_codes = GOAL_COLS if (mode in ["auto", "goal"]) else SDG_COLS
            #items = st.multiselect("Items (codes or group names)", sorted(pool_codes + [g for g in GROUPS]))
            if mode == "goal":
                items = st.multiselect("Goal", GOAL_COLS)
            elif mode == "sdg":
                items = st.multiselect("Goal", SDG_COLS)

            h = st.slider("Figure height", 400, 1200, 600, step=50, key="pst_h")

        try:
            fig = kit.plot_sdg_timeline(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                items=items if items else None,
                mode=mode,
                p_height=h,
                group_filter=grp_filter if grp_filter else None,
                entity_type=None if entity_type == "None" else entity_type,
                entities=entities if entities else None
            )
            show_plot(fig=fig, page_key="plot_sdg_timeline")

        except Exception as e:
            st.warning(f"Could not render: {e}")


if page == "Network & Structure":
    st.markdown('<div class="section-title">Network & Structure</div>', unsafe_allow_html=True)
    sub = st.segmented_control(
        "",
        ["Network Graph", "PCA / Biplot", "Dendrogram", "Chord Diagram"], default="Network Graph",
        key="nets_view",
    )

    #tab_net, tab_pca, tab_den, tab_chd = st.tabs(["Network Graph", "PCA / Biplot", "Dendrogram", "Chord Diagram"])

    if sub == "Network Graph":
        #st.subheader("plot_sdg_network")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            level = st.selectbox("Level", ["goal", "sdg"], index=0, key="net_level")
            group_filter = []
            if level == "sdg":
                group_filter = st.multiselect(
                    "Group filter (sdg only)",
                    GROUPS,
                    default=(["economic"] if "economic" in GROUPS else None),
                    key="net_groups",
                )
        with c2:
            entity_type = st.selectbox("Entity type", ["None", "Region", "Country"], index=0, key="net_etype")
            entities = []
            if entity_type == "Region":
                entities = st.multiselect("Regions", REGIONS, key="net_regions")
            elif entity_type == "Country":
                entities = st.multiselect("Countries", COUNTRIES, key="net_countries")
        with c3:
            # --- time controls depend on selected entities ---
            single_country = (entity_type == "Country" and len(entities) == 1)

            if single_country:
                # force a years range, agg='none', min_non_null=5
                y0, y1 = st.select_slider(
                    "Years (required for a single country)",
                    options=YEARS,
                    value=(YEARS[max(0, len(YEARS) - 6)], YEARS[-1]),
                    key="net_years",
                )
                year, years = None, (int(y0), int(y1))
                #st.info("Single country selected → using a year range with agg='none' and min_non_null=5.")
            else:
                year = st.selectbox("Year", YEARS, index=len(YEARS) - 1, key="net_year")
                years = None

        with c4:
            min_abs_corr = st.slider("Min abs corr (|r|)", 0.0, 1.0, 0.10, 0.01, key="net_thr")

        with c5:
            p_scale = st.slider("Plot scale", 700, 2000, 1000, step=10, key="gr_h")

        # --- call plotter ---
        try:
            kwargs = dict(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                level=level,
                entity_type=None if entity_type == "None" else entity_type,
                entities=entities if entities else None,
                group_filter=group_filter if (level == "sdg" and group_filter) else None,
                year=year,
                years=years,
                fig_scale=p_scale,
                min_abs_corr=float(min_abs_corr),
            )
            if single_country:
                kwargs["agg"] = "none"
                kwargs["min_non_null"] = 5

            fig = kit.plot_sdg_network(**kwargs)

            show_plot(fig=fig, page_key="plot_sdg_network")

        except Exception as e:
            st.warning(f"Could not render network: {e}")

    elif sub == "PCA / Biplot":
        #st.subheader("plot_sdg_pca")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="pca_year")
        with c2:
            variable_type = st.selectbox("Variable type", ["goal", "sdg"], index=0)
        with c3:
            groups_pick_pca = st.multiselect("Groups", GROUPS, key="pca_groups")
        with c4:
            region_filter = st.selectbox("Region filter", [None] + REGIONS, index=0)
        with c5:
            kind = st.selectbox("Kind", ["biplot", "circle"], index=0)

        try:
            pca_result = kit.plot_sdg_pca(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                id_col="Country",
                year=year,
                group_filter=groups_pick_pca,
                region_filter=region_filter,
                kind=kind,
                show_var_labels=True,
                show_country_labels=True,
                color_by_group=True,
                variable_type=variable_type,
                biplot_xscale=2.5,
                biplot_yscale=3.5,
                line_width=2
            )
            if isinstance(pca_result, tuple):
                fig, load_df = pca_result
            else:
                fig, load_df = pca_result, None

            show_plot(fig=fig, page_key="plot_sdg_pca")

            if load_df is not None:
                with st.expander("Show PCA loadings table"):
                    st.dataframe(load_df)
        except Exception as e:
            st.warning(f"Could not render PCA: {e}")

    elif sub == "Dendrogram":
        #st.subheader("sdg_dendrogram")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="den_year")
        with c2:
            groups_pick = st.multiselect("Groups", GROUPS)
        with c3:
            mode = st.selectbox("Mode", ["goal", "indicator"], index=0)
        with c4:
            goal_filter = st.multiselect("Goal filter (for indicator mode)", [int(g.split("_")[1]) for g in GOAL_COLS])
        with c5:
            top_k = st.number_input("Top-k vars", 10, 200, 40, 5)

        try:
            fig = kit.sdg_dendrogram(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                mode=mode,
                groups=groups_pick if groups_pick else None,
                goal_filter=goal_filter if goal_filter else None,
                top_k=int(top_k)
            )
            show_plot(fig=fig, page_key="sdg_dendrogram")

        except Exception as e:
            st.warning(f"Could not render dendrogram: {e}")

    elif sub == "Chord Diagram":
        #st.subheader("grouped_corr_chord")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="ch_year")
        with c2:
            mode = st.selectbox("Mode", ["goal", "indicator"], index=0, key="chd_mode")
        with c3:
            agg = st.selectbox("Aggregate", ["mean", "median", "max"], index=0)
        with c4:
            min_abs = st.slider("Min |corr| to display", 0.0, 1.0, 0.3, 0.05)

        try:
            fig = kit.grouped_corr_chord(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                mode=mode,
                agg=agg,
                min_abs=float(min_abs)
            )
            show_plot(fig=fig, page_key="grouped_corr_chord")

        except Exception as e:
            st.warning(f"Could not render chord: {e}")


elif page == "Correlations":
    st.markdown('<div class="section-title">Correlations</div>', unsafe_allow_html=True)

    sub = st.segmented_control(
        "",
        ["Correlation Matrix", "Grouped Heatmaps", "Grouped Bar Chart", "Group-Pair Correlations", "Correlate with Target", "Cross-Group Summary"], default="Correlation Matrix",
        key="corr_view",
    )

    #tab_cm, tab_gch, tab_gcc, tab_gcb, tab_cwt, tab_cgs = st.tabs(["Correlation Matrix", "Grouped Corr Heatmaps", "Group-Pair Code Corr", "Grouped Corr Barchart", "Correlate With Target", "Cross-Group Corr Summary"])

    if sub == "Correlation Matrix":
        #st.subheader("Correlation Matrix") # corr_by_entity

        c1, c2, c3 = st.columns(3)
        with c1:
            entity_type = st.selectbox("Entity type", ["Region", "Country"], index=0)
            if entity_type == "Region":
                entities = st.multiselect("Entities", REGIONS)
            elif entity_type == "Country":
                entities = st.multiselect("Entities", COUNTRIES)
            else:
                entities = []
        with c2:
            mode = st.selectbox("Mode", ["within_year", "across_years"], index=0)
            year = None
            years_range = None
            if mode == "within_year":
                year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="cbe_year")
            else:
                y0, y1 = st.select_slider("Year range", options=YEARS, value=(YEARS[0], YEARS[-1]))
                years_range = (int(y0), int(y1))
        with c3:
            level = st.selectbox("Level", ["goal", "sdg"], index=0)

        try:
            # across_years requires exactly one entity
            if mode == "across_years" and (entity_type == "None" or len(entities) != 1):
                st.info("Pick exactly one Region/Country for 'across_years'.")
            else:
                corr_df, fig = kit.corr_by_entity(
                    df_sdg=df_sdg,
                    df_lookup=df_lookup,
                    level=level,
                    entity_type=None if entity_type == "None" else entity_type,
                    entities=entities if entities else None,
                    mode=mode,
                    year=year,
                    years=years_range,
                    return_fig=True,
                )
                show_plot(fig=fig, page_key="corr_by_entity")

        except Exception as e:
            st.warning(f"Could not render: {e}")        

    elif sub == "Grouped Heatmaps":
        #st.subheader("grouped_corr_heatmaps")
        c1, c2 = st.columns(2)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="gh_year")
            indicator_mode = st.checkbox("Indicator mode (sdg*)", value=False)
        with c2:
            n_cols = st.number_input("Subplot columns", 1, 4, 3)
        try:
            fig = kit.grouped_corr_heatmaps(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                indicator_mode=indicator_mode,
                n_cols=int(n_cols)
            )
            show_plot(fig=fig, page_key="grouped_corr_heatmaps")

        except Exception as e:
            st.warning(f"Could not render heatmaps: {e}")


    elif sub == "Group-Pair Correlations":
        #st.subheader("get_grouppair_code_corr")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="gp_year")
            mode = st.selectbox("Mode", ["goal", "sdg"], index=0)
        with c2:
            top_n = st.number_input("Top N", 10, 200, 50, 5)
        with c3:
            sort_by = st.selectbox("Sort by", ["abs_corr", "corr"], index=0)
            ascending = st.checkbox("Ascending", value=False)
        with c4:
            pairs_mode = st.selectbox("Pairs", ["All", "Custom"], index=0)
        pairs = None
        if pairs_mode == "Custom":
            left = st.multiselect("Left groups", GROUPS, default=["economic"])
            right = st.multiselect("Right groups", GROUPS, default=["social"]) 
            pairs = [(a, b) for a in left for b in right]
        try:
            df_pairs = kit.get_grouppair_code_corr(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                mode=mode,
                pairs=pairs,
                top_n=int(top_n),
                sort_by=sort_by,
                ascending=ascending
            )
            show_plot(df=df_pairs, page_key="get_grouppair_code_corr")

        except Exception as e:
            st.warning(f"Could not compute: {e}")

    elif sub == "Correlate with Target":
        #st.subheader("correlate_with_target")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            family = st.selectbox("Goal/SDG", ["Goal_", "sdg"], index=0)
            target_pool = GOAL_COLS if family == "Goal_" else SDG_COLS
            target = st.selectbox("Target", target_pool)
        with c2:
            #year = st.selectbox("Year", [None] + YEARS, index=0)
            year = st.selectbox("Year", [None] + YEARS, index=len(YEARS))
            top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)
        with c3:
            sign = st.selectbox("Order by", ["pos", "neg", "abs"], index=0)
        with c4:
            other_group = st.selectbox("Filter other group", [None] + GROUPS, index=0)
        try:
            df_corr = kit.correlate_with_target(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                target=target,
                year=year,
                top_n=top_n,
                sign=sign,
                other_group=other_group,
            )
            show_plot(df=df_corr, page_key="correlate_with_target")

        except Exception as e:
            st.warning(f"Could not compute: {e}")

    elif sub == "Grouped Bar Chart":
        #st.subheader("grouped_corr_barchart")
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="gb_year")
            mode = st.selectbox("Mode", ["goal", "indicator"], index=0)
        with c2:
            agg = st.selectbox("Aggregate", ["mean", "median", "max"], index=0)
            min_abs = st.slider("Min |corr| included", 0.0, 1.0, 0.0, 0.05)
        with c3:
            y_min, y_max = st.slider("Y range", -1.0, 1.0, (-0.1, 0.5), 0.05)
        try:
            fig = kit.grouped_corr_barchart(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                mode=mode,
                agg=agg,
                min_abs=float(min_abs),
                p_range=[float(y_min), float(y_max)]
            )
            show_plot(fig=fig, page_key="grouped_corr_barchart")

        except Exception as e:
            st.warning(f"Could not render grouped bar: {e}")

    elif sub == "Cross-Group Summary":
        #st.subheader("get_cross_group_corr_sym")
        year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="sym_year")
        try:
            df_sym = kit.get_cross_group_corr_sym(df_sdg=df_sdg, df_lookup=df_lookup, year=year, value_prefix="Goal_")
            show_plot(df=df_sym, page_key="get_cross_group_corr_sym")

        except Exception as e:
            st.warning(f"Could not compute summary: {e}")

elif page == "About":


    st.markdown("""
    <div class="hero">
        <h1>About</h1>
        <p>
            <b>GoalScope SDG Analytics Hub</b><br><br>
            University: <b>Massey University, New Zealand</b><br>
            Paper: <b>158888 — Information Technology Professional Project</b><br><br>
            Group Members : <b>Jesus Eric Seacor, Sai Ram Ceka</b><br>
            Teaching Team : <b>Dr. Anuradha Mathrani, Dr. Niloofar Aflaki</b><br><br>
            <i>App created by Jesus Eric Seacor; analysis and project work in collaboration with Sai Ram Ceka.</i>
        </p>
    </div>
    """, unsafe_allow_html=True)    


    st.caption("""
    AI Use Statement : 
    This app generates human-readable “NLP insights” from plots using OpenAI’s GPT models via the Responses API. 
    My contributions include: data acquisition/wrangling, correlation/analysis code, plot generation, prompt and instruction design, 
    vector-store/file-search context, UI integration, caching, and evaluation of insight quality. 
    All AI-generated text is labeled in the UI and logged for reproducibility. No model fine-tuning was performed.
    """)

    #st.caption("© Massey University GoalScope — session-based demo with AI insights")


y0, y1 = int(min(YEARS)), int(max(YEARS))
st.markdown(
    f"<div class='meta'>Data coverage: <b>{y0}–{y1}</b> • Score scale: 0–100 • Correlation ≠ causation</div>",
    unsafe_allow_html=True
)
