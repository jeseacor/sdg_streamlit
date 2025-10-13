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


def linked_image_local(img_path: str, url: str, *, width: int | None = None,
                       radius: int | str = 12, sidebar: bool = False, shadow: bool = False):
    data = Path(img_path).read_bytes()
    b64 = b64encode(data).decode("utf-8")
    ext = (Path(img_path).suffix or ".png").lstrip(".")

    # inner box width; if a width is given we’ll center that box, otherwise it can grow to 100%
    box_w = f"width:{width}px;" if width else "max-width:100%;"
    rcss = f"{int(radius)}px" if isinstance(radius, (int, float)) else str(radius)
    scss = "box-shadow:0 2px 12px rgba(0,0,0,.25);" if shadow else ""

    html = f"""
    <div style="width:100%; display:flex; justify-content:center;">  <!-- flex wrapper centers content -->
    <a href="{url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; display:block;">
        <div style="{box_w} margin:0 auto; border-radius:{rcss}; overflow:hidden; {scss} line-height:0;">
        <img src="data:image/{ext};base64,{b64}" style="width:100%; height:auto; display:block; border:0;" />
        </div>
    </a>
    </div>
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

@st.cache_resource(show_spinner=True)
def get_kit():
    return ProjectKit()

try:
    df_sdg, df_lookup = load_data()
    kit = get_kit()
    kit.set_data_for_chat(df_sdg, df_lookup)

    # If the user applied custom grouping in Settings, use it
    if "df_lookup_override" in st.session_state:
        df_lookup = st.session_state["df_lookup_override"]
        try:
            kit.set_data_for_chat(df_sdg, df_lookup)  # keep chatbot context in sync
        except Exception:
            pass

except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()


# -------------------- Lists --------------------
YEARS = sorted([int(y) for y in df_sdg["Year"].dropna().unique().tolist()])
COUNTRIES = sorted(df_sdg["Country"].dropna().unique().tolist())
REGIONS = sorted(df_sdg["Region"].dropna().unique().tolist())
GOAL_COLS = [c for c in df_sdg.columns if str(c).startswith("Goal_")]
SDG_COLS  = [c for c in df_sdg.columns if str(c).lower().startswith("sdg") and len(c) > 3]

def derive_groups(df_lookup):
    pref = ["economic", "social", "environmental"]
    if df_lookup is None or "group" not in df_lookup.columns:
        return pref
    u = (df_lookup["group"].dropna().astype(str).str.lower().unique().tolist())
    return [g for g in pref if g in u] + [g for g in u if g not in pref]

GROUPS = derive_groups(df_lookup)
st.session_state["GROUPS"] = GROUPS  # make available to all pages


# -------------------- Sidebar Navigation --------------------
PAGES = [
    "Home",
    "Performance Rankings",
    "Trends & Projections",
    "Network & Structure",
    "Correlations",
    "Settings",
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

    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home"

    page = option_menu(
        None,  # no title inside the menu
        PAGES,
        icons=["house", "trophy", "graph-up",
               "diagram-3", "link-45deg", "gear", "info-circle"],
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
    
    linked_image_local("assets/sdg_logo_b.png", "https://sdgs.un.org/goals", width=180, sidebar=True)
    
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
    font-size: 2em;
    font-weight: 700;
    margin-bottom: 0.4em;
    text-shadow: 1px 4px 12px rgba(0,0,0,0.4);
    color: #4789C8;
}
.feature-card {
    font-size: 1.3em;
    font-weight: 700;
    margin-bottom: 0.4em;
    color: #ccc;      
    background-color: #0A2240;
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
.sdg-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}
.meta { font-size:.9rem; opacity:.85; margin-top:.25rem; }
</style>
""", unsafe_allow_html=True)

#logo_path = here / "sdg_grid.png" 
# -------------------- Home --------------------

if page == "Home":
        
# Enhanced Home Page Section for GoalScope SDG Analytics Hub
# Insert this into your existing app.py where page == "Home"
    
    # Hero Section with Enhanced Branding
    st.markdown("""
    <div class="hero">
        <h1><span class="dot">GOAL</span><span class="brand">Scope </span><span class="sub">{ SDG Analytics Hub }</span></h1>
        <p>
            Welcome to Massey University GOALScope — A comprehensive platform for analyzing UN Sustainable Development Goals (SDG)
            data with AI-powered insights and interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Stats Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)

    y0, y1 = int(min(YEARS)), int(max(YEARS))
    
    with m1:
        st.metric(
            label="📊 Data Coverage",
            value=f"{y0}–{y1}",
            delta=f"{y1 - y0 + 1} years"
        )
    
    with m2:
        st.metric(
            label="🌍 Countries",
            value=len(COUNTRIES),
            delta="Global coverage"
        )
    
    with m3:
        st.metric(
            label="🎯 SDG Goals",
            value="17",
            delta="169 targets"
        )
    
    with m4:
        st.metric(
            label="📈 Indicators",
            value=len(SDG_COLS),
            delta="Tracked"
        )
    
    with m5:
        st.metric(
            label="🌏 Regions",
            value=len(REGIONS),
            delta="Analyzed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True): 
        # The 17 SDGs Visual Grid
        st.markdown('<div class="section-title" style="text-align: center;">The 17 Sustainable Development Goals</div>', unsafe_allow_html=True)
        
        # SDG badges with official colors
        sdg_data = {
            1: ("#E5243B", "No Poverty"),
            2: ("#DDA63A", "Zero Hunger"),
            3: ("#4C9F38", "Good Health"),
            4: ("#C5192D", "Quality Education"),
            5: ("#FF3A21", "Gender Equality"),
            6: ("#26BDE2", "Clean Water"),
            7: ("#FCC30B", "Clean Energy"),
            8: ("#A21942", "Decent Work"),
            9: ("#FD6925", "Innovation"),
            10: ("#DD1367", "Reduced Inequalities"),
            11: ("#FD9D24", "Sustainable Cities"),
            12: ("#BF8B2E", "Responsible Consumption"),
            13: ("#3F7E44", "Climate Action"),
            14: ("#0A97D9", "Life Below Water"),
            15: ("#56C02B", "Life on Land"),
            16: ("#00689D", "Peace & Justice"),
            17: ("#19486A", "Partnerships")
        }

        # Display SDG badges
        sdg_html = '<div style="text-align: center; margin: 2rem 0;">'
        for i, (color, name) in sdg_data.items():
            sdg_html += f'<a href="https://sdgs.un.org/goals/goal{i}" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">'
            sdg_html += f'<span class="sdg-badge" style="background-color: {color}; color: white; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; display: inline-block;" '
            sdg_html += f'onmouseover="this.style.transform=\'scale(1.05)\'; this.style.boxShadow=\'0 4px 8px rgba(0,0,0,0.3)\';" '
            sdg_html += f'onmouseout="this.style.transform=\'scale(1)\'; this.style.boxShadow=\'none\';">'
            sdg_html += f'{i}. {name}</span></a>'
        sdg_html += '</div>'

        st.markdown(sdg_html, unsafe_allow_html=True)


    # Main Content: Two Column Layout
    c1a, c2a = st.columns([1, 1], vertical_alignment="top")
    
    with c1a:
        # Video Section
        with st.container(border=True):    


            youtube_autoplay("https://youtu.be/0XTBYMfZyrM?si=yZtN5MVYe4OGoY-t", start=0, loop=True, height=400)
            st.caption("Learn about the 17 Sustainable Development Goals and their global impact.")        


        # Data & Methodology Section
        with st.container(border=True):
            st.markdown('<div class="section-title">Data Sources & Methodology</div>', unsafe_allow_html=True)
            
            col1a, col2a = st.columns([1, 1], vertical_alignment="top")
            with col1a:
                st.markdown("""
                **📥 Data Sources:**
                - UN SDG Database (official statistics)
                - ArcGIS FeatureServer APIs
                - Real-time data synchronization
                - Comprehensive country & regional coverage
                
                **🔧 Processing:**
                - Pandas-based data wrangling
                - Missing value interpolation
                - Multi-year aggregation options
                - Flexible filtering & grouping
                """)

            with col2a:
                st.markdown("""
                **📊 Analysis Methods:**
                - Correlation & regression analysis
                - Time-series forecasting (ETS)
                - Principal Component Analysis (PCA)
                - Network graph algorithms
                
                **🤖 AI Integration:**
                - OpenAI GPT models for insights
                - Context-aware analysis
                - Cached results for performance
                - Conversation history tracking
                """)               

        st.markdown("<br>", unsafe_allow_html=True)

    with c2a:

        # Features Section
        with st.container(border=True):
            st.markdown('<div class="section-title">Key Features</div>', unsafe_allow_html=True)
            st.markdown("""
                <ul>
                    <li><b>📊 Performance Rankings:</b> <br>Compare Overall Score or specific Goals (1–17) across countries and regions with multiple visualization modes.</li>
                    <li><b>📈 Trends & Projections:</b> <br>Track temporal changes and forecast future SDG performance using advanced time-series models.</li>
                    <li><b>🔗 Network Analysis:</b> <br>Visualize goal interdependencies through correlation networks, PCA biplots, and dendrograms.</li>
                    <li><b>🔍 Correlation Explorer:</b> <br>Discover relationships between indicators, goals, and country performance metrics.</li>
                    <li><b>🤖 AI Insights:</b> <br>Generate instant narrative summaries for any chart or table with one-click AI analysis.</li>
                    <li><b>💬 Interactive Chat:</b> <br>Ask questions about SDG data, methodology, and interpretation through the AI chatbot.</li>
                </ul>
            """, unsafe_allow_html=True)

        # Benefits & Use Cases
        with st.container(border=True):
            st.markdown('<div class="section-title">Benefits</div>', unsafe_allow_html=True)
            st.markdown("""
                <ul>
                    <li>📚 <b>Academic Research:</b> <br>Explore SDG data for coursework, reports, and thesis projects with robust analytical tools.</li>
                    <li>🎓 <b>Educational Tool:</b> <br>Learn about sustainable development through interactive visualizations and AI-guided insights.</li>
                    <li>📊 <b>Policy Analysis:</b> <br>Compare country performance, identify patterns, and understand goal trade-offs.</li>
                    <li>🔬 <b>Data Science:</b> <br>Apply correlation analysis, forecasting, PCA, and network analysis to real-world data.</li>
                    <li>💡 <b>Quick Insights:</b> <br>Generate professional narratives for presentations and reports with AI assistance.</li>
                </ul>
            """, unsafe_allow_html=True)

        # Getting Started Guide
        with st.expander("Getting Started Guide", expanded=False):
            st.markdown("""
            ### How to Use GoalScope
            
            **1. Explore the Navigation Menu** (left sidebar)
            - Choose from 7 main sections: Home, Rankings, Trends, Networks, Correlations, Settings, About
            - Each section offers multiple visualization and analysis modes
            
            **2. Customize Your Analysis**
            - Select years, countries, regions, and specific SDG goals
            - Apply filters to focus on economic, social, or environmental indicators
            - Adjust visualization parameters (height, colors, labels)
            
            **3. Generate AI Insights**
            - Click the 🤖 **Generate Insight** button below any chart or table
            - Insights are automatically saved and can be revisited
            - Use the chatbot for interactive Q&A about the data
            
            **4. Export & Share**
            - Right-click charts to download as PNG
            - Copy tables for use in reports and presentations
            - Share findings with the research community
            
            **5. Adjust Settings**
            - Visit the Settings page to change SDG grouping schemes
            - Preview changes before applying them globally
            - Reset to defaults anytime
            """)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    # Footer with additional info
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **📊 Current Dataset**
        - Years: {y0}–{y1}
        - Countries: {countries}
        - Regions: {regions}
        - Score Range: 0–100
        """.format(y0=y0, y1=y1, countries=len(COUNTRIES), regions=len(REGIONS)))
    
    with footer_col2:
        st.markdown("""
        **🔗 Resources**
        - [UN SDG Portal](https://sdgs.un.org/goals)
        - [SDG Indicators](https://unstats.un.org/sdgs/)
        - [Massey University](https://www.massey.ac.nz)
        """)
    
    with footer_col3:
        st.markdown("""
        **ℹ️ About This Project**
        - Course: 158888 – IT Professional Project
        - Institution: Massey University, NZ
        - Developer: J.E. Seacor
        """)


if page == "Performance Rankings":
    st.markdown('<div class="section-title">Performance Rankings</div>', unsafe_allow_html=True)

    sub = st.segmented_control(
        "",
        ["Rankings", "Percent Change", "Quadrant Scatter", "Benchmark Gaps", "Performance Wheel", "Bar Chart"], default="Rankings",
        key="ranking_view",
    )

    #tab_rank, tab_loc, tab_sdg  = st.tabs(["Rankings", "Bar Chart (Locale)", "Bar Chart(SDG)"])

    if sub == "Rankings":
        #st.subheader("get_ranking_table")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            years = sorted(YEARS, reverse=True)
            year = st.selectbox("Year", years, index=0, key="rnk_year")
        with c2:
            #value_col = st.selectbox("Goal/SDG or Group)", sorted(GOAL_COLS + SDG_COLS + GROUPS))
            goal_opts = ["Overall Score", *GOAL_COLS, *GROUPS, *SDG_COLS]
            sel = st.selectbox("Goal or Group", goal_opts, index=0, key="rnk_goal_choice")
            value_col = None if sel in ("", "Overall Score") else sel
        with c3:
            #value_col = st.selectbox("Goal/SDG or Group)", sorted(GOAL_COLS + SDG_COLS + GROUPS))
            goal_opts = ["All Regions", *REGIONS]
            sel = st.selectbox("Select Region", goal_opts, index=0, key="rnk_rgn_choice")
            value_reg = None if sel in ("", "All Regions") else sel            
        with c4:
            years_range = None
            y0, y1 = st.select_slider("Range for % Change", options=YEARS, value=(YEARS[0], YEARS[-1]))
            years_range = (int(y0), int(y1))          

        try:
            df_rank = kit.get_ranking_table(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                year=year,
                start_year=y0,
                end_year=y1,
                goal=value_col,
                region=value_reg
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
            sort_desc=False,
            return_fig=True,
        )
        show_plot(fig=fig, page_key="pct_change_sdg")

    elif sub == "Quadrant Scatter":
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1:
            #value_col = st.selectbox("Goal/SDG or Group)", sorted(GOAL_COLS + SDG_COLS + GROUPS)) 
            goal_opts = ["Overall Score", *GOAL_COLS, *GROUPS, *SDG_COLS]
            sel = st.selectbox("Goal or Group", goal_opts, index=0, key="quad_goal_choice")
            value_col = None if sel in ("", "Overall Score") else sel
        with c2:
            #value_col = st.selectbox("Goal/SDG or Group)", sorted(GOAL_COLS + SDG_COLS + GROUPS))
            goal_opts = ["All Regions", *REGIONS]
            sel = st.selectbox("Select Region", goal_opts, index=0, key="quad_rgn_choice")
            value_reg = None if sel in ("", "All Regions") else sel
        with c3:
            top_n = st.number_input("Show Top/Bottom N", min_value=5, max_value=50, value=5, step=1)
        with c4:
            y0, y1 = st.select_slider("Start → End", options=YEARS, value=(YEARS[0], YEARS[-1]))
        with c5:
            dot_sz = st.slider("Dot size", 2, 12, 4, 1)
        with c6:
            f_size = st.slider("Lable size", 8, 20, 12, 2)
        with c7:
            h = st.slider("Figure height", 200, 1200, 700, step=50, key="quad_h") 

        view3d = st.toggle("3D view", value=False, key="quad_3d")
        #label_all = st.toggle("Show labels (3D)", value=True, key="quad_labels") if view3d else False
        
        fig_scat, tbl_scat = kit.leaders_laggards_scatter(
            df_sdg=df_sdg, df_lookup=df_lookup,
            start_year=int(y0), end_year=int(y1),
            region=value_reg,
            measure=value_col,
            label_top=top_n,
            fig_height=h,
            as_3d=view3d,
            #label_all_3d=label_all,
            marker_size=dot_sz,     
            label_font_size=f_size       
        )
        show_plot(fig=fig_scat, page_key="leaders_laggards_scatter")

        st.dataframe(tbl_scat)

    elif sub == "Benchmark Gaps":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            years = sorted(YEARS, reverse=True)
            year = st.selectbox("Year", years, index=0, key="bench_year")
        with c2:
            options = sorted(df_sdg["Country"].dropna().unique().tolist())
            # default to "New Zealand" if it exists, else fall back to the first option
            default_idx = next((i for i, v in enumerate(options) if v == "New Zealand"), 0)
            country_val = st.selectbox("Country", options, index=default_idx, key="country_select")
        with c3:
            bench = st.selectbox("Benchmark", ["top_quartile", "median"], index=0)
        with c4:
            use_region_benchmark = st.toggle("Benchmark within region", value=True)
        with c5:
            h = st.slider("Figure height", 200, 1200, 500, step=50, key="bench_h")

        fig_bench, tbl_bench = kit.plot_gap_dumbbell_sdg(
            df_sdg=df_sdg,
            year=year,
            country=country_val,
            benchmark=bench,
            label_yshift=0,
            region_benchmark=use_region_benchmark,
            plot_height=h,
            line_color="#7393B3"
        )
        show_plot(fig=fig_bench, page_key="plot_gap_dumbbell_sdg")

        st.dataframe(tbl_bench)

    elif sub == "Performance Wheel":
        c1, c2, c3 = st.columns(3)
        with c1:
            years = sorted(YEARS, reverse=True)
            year = st.selectbox("Year", years, index=0, key="pw_year") 
        with c2:
            entity_type = st.selectbox("Entity type", ["All", "Region", "Country"], index=2)
        with c3:
            entities = []
            if entity_type == "Region":
                entities = st.selectbox("Pick regions", REGIONS, index=0)
            elif entity_type == "Country":
                #entities = st.selectbox("Pick countries", COUNTRIES, index=0)
                options = sorted(df_sdg["Country"].dropna().unique().tolist())
                default_idx = next((i for i, v in enumerate(options) if v == "New Zealand"), 0)
                entities = st.selectbox("Country", options, index=default_idx, key="country_select")


        fig_radial, tab_radial = kit.plot_sdg_radial(
            df_sdg=df_sdg, df_lookup=df_lookup,
            year = year,
            region_names=entities if entity_type == "Region" else None,
            country_names=entities if entity_type == "Country" else None,
            fig_height=800,
            kpi_pad=1.030
        )
        show_plot(fig=fig_radial, page_key="plot_sdg_radial")


    elif sub == "Bar Chart":
        #view_mode = st.selectbox("View Mode", ["Locale", "Goals"], index=0) 
        view_mode = st.radio(
            "View mode",
            ["Locale", "Goals"],
            index=0,
            horizontal=True,   # optional: put options on one line
            key="view_mode"
        )        

        if view_mode == "Locale":
            #st.subheader("plot_sdg_ranking")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
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
                region_view_val = "region"
                if group_col == "Region":
                    region_opts = ['All', *REGIONS]
                    region_view_val = st.selectbox("Regions", region_opts, index=0)
            with c5:
                top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)
            with c6:
                h = st.slider("Figure height", 200, 1200, 500, step=50, key="er_h")
            with c7:
                ascending = st.checkbox("Ascending", value=False, key="er_asc")
            try:
                fig = kit.plot_sdg_ranking(
                    data=df_sdg,
                    value_col=value_col if value_col else None,
                    group_col=group_col,
                    region_view=region_view_val,
                    top_n=int(top_n),
                    ascending=ascending,
                    year=year,
                    df_lookup=df_lookup,
                    fig_height=h,
                    fig_width=1100
                )
                show_plot(fig=fig, page_key="plot_sdg_ranking")

            except Exception as e:
                st.warning(f"Could not render: {e}")

        elif view_mode == "Goals":
            #st.subheader("plot_goal_ranking")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            with c1:
                year = st.selectbox("Year", YEARS, index=len(YEARS)-1, key="gr_year")
            with c2:
                rank_by = st.selectbox("Rank by", ["goal", "sdg", "group"], index=0, key="gr_rankby")
            with c3:
                group_filter = st.multiselect("Filter groups", GROUPS, key="gr_groups") if rank_by != "group" else []
            with c4:
                entity_type = st.selectbox("Entity type", ["Region", "Country"], index=0)
            with c5:
                entities = []
                if entity_type == "Region":
                    entities = st.multiselect("Pick regions", REGIONS)
                elif entity_type == "Country":
                    entities = st.multiselect("Pick countries", COUNTRIES)
            with c6:
                h = st.slider("Figure height", 400, 1200, 600, step=50, key="gr_h")
            with c7:
                ascending = st.checkbox("Ascending", value=True, key="gr_asc")

            try:
                fig = kit.plot_goal_ranking(
                    df_sdg=df_sdg,
                    df_lookup=df_lookup,
                    year=year,
                    rank_by=rank_by,
                    group_filter=group_filter if group_filter else None,
                    ascending=ascending,
                    geo_level=entity_type,
                    geo_names=entities,
                    fig_height=h,
                    fig_width=1100,
                )
                show_plot(fig=fig, page_key="plot_goal_ranking")

            except Exception as e:
                st.warning(f"Could not render: {e}")


elif page == "Trends & Projections":
    st.markdown('<div class="section-title">Trends & Projections</div>', unsafe_allow_html=True)

    sub = st.segmented_control(
        "",
        ["Trend (Locale)", "Trend (SDG)", "SDG Forecast"], default="Trend (Locale)",
        key="timeline_view",
    )

    #tab_loc, tab_sdg  = st.tabs(["Timeline (Locale)", "Timeline (SDG)"])
    
    if sub == "Trend (Locale)":
        #st.subheader("plot_goal_entity_timeline")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_value = st.selectbox("Metric mode", ["score", "percent_change"], index=0)
            sort_value = st.selectbox("Ranking type", ["top", "bottom"], index=0)
        with c2:
            goal_opts = ["Overall Score", *GOAL_COLS, *SDG_COLS]
            sel = st.selectbox("Goal", goal_opts, index=0, key="goal_choice")
            goal = None if sel == "" else sel   # map blank to None

            top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)
        with c3:
            entity_type = st.selectbox("Entity type", ["Country", "Region"], index=0)
            y0, y1 = st.select_slider("Start → End", options=YEARS, value=(YEARS[0], YEARS[-1]))
        with c4:
            entities = REGIONS if entity_type == "Region" else COUNTRIES
            picked = st.multiselect("Entities", entities)
            h = st.slider("Figure height", 400, 1200, 600, step=50, key="gr_h")
        with c5:
            region_view_val = "region"
            if entity_type == "Region":
                region_view_val = st.selectbox("Region View", ["region", "countries"], index=0)
            
        try:
            fig = kit.plot_goal_entity_timeline(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                metric_mode=metric_value,
                goal=goal if goal else None,
                entity_type=entity_type,
                entities=picked if picked else None,
                region_view=region_view_val,
                agg="mean",
                fig_height=h,
                top_mode=sort_value,
                start_year=int(y0), end_year=int(y1),
                top_n=top_n
            )
            show_plot(fig=fig, page_key="plot_goal_entity_timeline")

        except Exception as e:
            st.warning(f"Could not render: {e}")

    elif sub == "Trend (SDG)":
        #st.subheader("plot_sdg_timeline")  
        c1, c2, c3, c4 = st.columns(4)
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
            y0, y1 = st.select_slider("Start → End", options=YEARS, value=(YEARS[0], YEARS[-1]))
            if mode == "goal":
                items = st.multiselect("Goal", GOAL_COLS)
            elif mode == "sdg":
                items = st.multiselect("Goal", SDG_COLS)       
        with c4:
            h = st.slider("Figure height", 400, 1200, 600, step=50, key="pst_h")   
            as_3d = st.checkbox("3D mode", value=False)    


        try:
            fig = kit.plot_sdg_timeline(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                items=items if items else None,
                mode=mode,
                p_height=h,
                start_year=int(y0), end_year=int(y1),
                group_filter=grp_filter if grp_filter else None,
                entity_type=None if entity_type == "None" else entity_type,
                entities=entities if entities else None,
                as_3d=as_3d
            )
            show_plot(fig=fig, page_key="plot_sdg_timeline")

        except Exception as e:
            st.warning(f"Could not render: {e}")

    elif sub == "SDG Forecast":
        #st.subheader("plot_sdg_timeline") 
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            goal_opts = ["Overall Score", *GOAL_COLS]
            sel = st.selectbox("Goal", goal_opts, index=0, key="goal_choice")
            goal = None if sel == "" else sel   # map blank to None 
        with c2:
            entity_type = st.selectbox("Entity type", ["Region", "Country"], index=1)
        with c3:
            entities = REGIONS if entity_type == "Region" else COUNTRIES
            if entity_type=="Region":
                picked = st.selectbox("Region", entities, index=0) 
            else:
                #picked = st.selectbox("Country", entities)  
                options = sorted(df_sdg["Country"].dropna().unique().tolist())
                default_idx = next((i for i, v in enumerate(options) if v == "New Zealand"), 0)
                picked = st.selectbox("Country", options, index=default_idx, key="country_select")
        with c4:
            next_year = YEARS[-1] + 1 if YEARS else None   # handle empty list
            to_year = st.number_input("Year Target", min_value=next_year, max_value=2050, value=2030, step=1)
        with c5:
            h = st.slider("Figure height", 400, 1200, 600, step=50, key="gr_h")

        as_3d = st.toggle("3D forecast (ribbon)", value=False)

        if goal == "Overall Score":
            isOverAll = True
        else:
            isOverAll = False

        st.markdown(
            f"<div class='meta'>Short term ETS projection that extends the recent level with uncertainty bands; shocks or policy changes are not modeled.</div>",
            unsafe_allow_html=True
        )                    

        try: 
            df_forecast = kit.forecast_sdg_any(
                df_sdg=df_sdg,
                df_lookup=df_lookup,
                sdg=goal,
                overall=isOverAll,
                entity_level=entity_type,
                entities=picked if picked is not None else None,
                horizon_to=to_year
            )

            fig = kit.plot_forecast_from_results(df_forecast, df_sdg, df_lookup, fig_height=h, as_3d=as_3d)
            show_plot(fig=fig, page_key="forecast_sdg_any")

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
        
        net3d = st.toggle("3D network", value=False, key="net_3d")

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
                as_3d=net3d
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
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
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
        with c6:
            dot_sz = st.slider("Dot size", 2, 12, 4, 1)
        with c7:
            f_size = st.slider("Lable size", 8, 20, 12, 2)
        with c8:
            h = st.slider("Figure height", 200, 1200, 700, step=50, key="pca_h") 

        pca_3d = st.toggle("3D PCA (PC1–PC3)", value=False)


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
                line_width=2,
                as_3d=pca_3d,
                fig_height=h,
                label_font_size=f_size, 
                marker_size=dot_sz
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
            if entity_type=="Country" and entities:
                mode = st.selectbox("Mode", ["within_year", "across_years"], index=1)    
            else:
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



if page == "Settings":
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    st.subheader("Goal grouping scheme")

    st.write(
        "Choose how SDG goals are grouped into economic / social / environmental / partnership. "
        "You can preview the effect and then apply it across the app."
    )

    scheme_label = st.radio(
        "Grouping",
        [
            "Barbier & Burgess (2017, World Development)",           # class_code=2
            "Wedding cake (Stockholm Resilience ‘wedding cake’)",   # class_code=1
        ],
        index=0,
        help="This updates the 'group' column in df_lookup for all goal/indicator codes."
    )
    class_code = 1 if scheme_label.startswith("Wedding") else 2

    # Preview new grouping (does not change app state yet)
    with st.expander("Preview changes (first 25 rows)"):
        try:
            preview = get_kit().classify_groups(df_lookup, class_code=class_code)
            st.dataframe(preview.head(25), use_container_width=True, hide_index=True)
            grp_counts = preview["group"].value_counts(dropna=False).rename_axis("group").reset_index(name="n")
            st.caption("Count by group:")
            st.dataframe(grp_counts, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not preview: {e}")

    a, b, c = st.columns([1,1,1])
    with a:
        if st.button("✅ Apply grouping to app", use_container_width=True):
            updated_lookup = get_kit().classify_groups(df_lookup, class_code=class_code)
            updated_lookup = updated_lookup.drop_duplicates(subset=["code"]).reset_index(drop=True)

            st.session_state["df_lookup_override"] = updated_lookup
            st.session_state["GROUPS"] = derive_groups(updated_lookup)  # <-- add this

            try:
                kit.set_data_for_chat(df_sdg, updated_lookup)
            except Exception:
                pass

            st.success("Grouping applied. All pages will use the new scheme.")
            st.rerun()

    with b:
        if st.button("↩ Reset to default", use_container_width=True):
            st.session_state.pop("df_lookup_override", None)
            st.session_state["GROUPS"] = derive_groups(df_lookup)  # back to default groups
            try:
                kit.set_data_for_chat(df_sdg, df_lookup)  # back to default loaded values
            except Exception:
                pass
            st.info("Reverted to the default grouping from data load.")
            st.rerun()

    with c:
        if st.button("🔄 Rebuild df_lookup from source (advanced)", use_container_width=True,
                     help="Re-download ArcGIS/UN sources and rebuild lookup via ProjectKit.get_clean_data()"):
            try:
                _df_sdg, _df_lookup = load_data.clear() or (None, None)  # clear cache
            except Exception:
                pass
            try:
                _df_sdg, _df_lookup = load_data()
                st.session_state.pop("df_lookup_override", None)
                kit.set_data_for_chat(_df_sdg, _df_lookup)
                st.success("Data reloaded and default grouping restored.")
                st.rerun()
            except Exception as e:
                st.error(f"Reload failed: {e}")



elif page == "About":

    st.markdown("""
    <div class="feature-card">
        <h1>About</h1>
        <p>
            <b>GoalScope SDG Analytics Hub</b><br><br>
            University: <b>Massey University, New Zealand</b><br>
            Course: <b>158888 — Information Technology Professional Project</b><br><br>
            Student : <b>Jesus Eric Seacor [24007226]</b><br>
            Teaching Team : <b>Dr. Anuradha Mathrani, Dr. Niloofar Aflaki</b><br><br>
        </p>
    </div>
    """, unsafe_allow_html=True)    

    #<i>App created by Jesus Eric Seacor; analysis and project work in collaboration with Sai Ram Ceka.</i>

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
