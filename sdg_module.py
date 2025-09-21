import requests  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json, hashlib
import re
import itertools
import plotly.express as px
from typing import Optional
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import networkx as nx
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.io as pio
import os, base64

from dotenv import load_dotenv
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in your environment"

from openai import OpenAI
client = OpenAI()

import random
import matplotlib.colors as mcolors
import plotly.express as px
from bs4 import BeautifulSoup  
from IPython.display import HTML
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform
from sklearn.linear_model import Ridge

class ProjectKit:
    def __init__(self, df=None):
        self.df = df
        self.model = None
        self.X = None
        self.y = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_cols = None
        self.name='obj'
        self.input_cols = None
        self.target_cols = None

        # kNN parameters
        self.bestk = None
        self.bestweight = None
        self.bestmetric = None

    # region [Setters & Getters]
    def set_data(self, df):
        self.df = df

    def get_data(self):
        return self.df

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model
    # endregion

    # region [Data access, setups and helpers]

    # calls the UN SDG API and returns a dataframe of Goal codes, titles, descriptions.
    def get_sdg_goals(self):      
        url = "https://unstats.un.org/SDGAPI/v1/sdg/Goal/List"
        response = requests.get(url)
        goals = response.json()
        df_goals = pd.DataFrame(goals)
        #df_goals = df_goals[["code", "title", "description", "uri"]]
        df_goals = df_goals[["code", "title", "description"]]
        return df_goals

    # pulls SDG goals and assigns each to a high-level “economic / social / environmental / partnership” 
    # group using one of two published schemes (1 = “wedding cake”, 2 = “Barbier & Burgess”).
    def get_sdg_goals_grouped(self, class_code : int = 1):
        url = "https://unstats.un.org/SDGAPI/v1/sdg/Goal/List"
        response = requests.get(url)
        goals = response.json()
        df_goals = pd.DataFrame(goals)[["code", "title", "description"]]

        if class_code == 1: # wedding cake
            economic_codes = ["8", "9", "10", "12"]
            social_codes = ["1", "2", "3", "4", "5", "7", "11", "16"]
            env_codes = ["6", "13", "14", "15"]
        elif class_code == 2: # Barbier & Burgess
            economic_codes = ["1", "2", "3", "6", "7", "8", "9"]
            social_codes = ["4", "5", "10", "16", "17"]
            env_codes = ["11", "12", "13", "14", "15"]

        def get_group(code):
            if code in economic_codes:
                return "economic"
            elif code in social_codes:
                return "social"
            elif code in env_codes:
                return "environmental"
            return "partnership"

        df_goals["group"] = df_goals["code"].apply(get_group)
        return df_goals

    # adds a group column to the lookup table based on the chosen grouping scheme. Handy when you already have a lookup of codes → SDG numbers.
    def classify_groups(self, df_lookup : pd.DataFrame, class_code : int = 1):
        if class_code == 1:  # Wedding cake
            economic_codes = ["8", "9", "10", "12"]
            social_codes = ["1", "2", "3", "4", "5", "7", "11", "16"]
            env_codes = ["6", "13", "14", "15"]

        elif class_code == 2:  # Barbier & Burgess
            economic_codes = ["1", "2", "3", "6", "7", "8", "9"]
            social_codes = ["4", "5", "10", "16", "17"]
            env_codes = ["11", "12", "13", "14", "15"]

        else:
            raise ValueError("class_code must be 1 or 2")

        def get_group(code):
            if code in economic_codes:
                return "economic"
            elif code in social_codes:
                return "social"
            elif code in env_codes:
                return "environmental"
            return "partnership"

        df_out = df_lookup.copy()
        df_out["group"] = df_out["sdg"].astype(str).apply(get_group)

        return df_out

    # fetches the official SDG targets for a specific goal (e.g., 9) and returns goal, code, title
    def get_sdg_targets(self, goal : int = 1):
        url = "https://unstats.un.org/SDGAPI/v1/sdg/Target/List"
        response = requests.get(url)
        targets = response.json()
        df_targets = pd.DataFrame(targets)
        df_targets = df_targets[["goal", "code", "title"]]
        df_targets = df_targets[df_targets["goal"] == str(goal)]
        return df_targets     

    # returns UN geo area codes and names (as strings) for joining.
    def get_geo_areas(self):
        url = "https://unstats.un.org/SDGAPI/v1/sdg/GeoArea/List"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        df_geo = pd.DataFrame(data)[["geoAreaCode", "geoAreaName"]].copy()
        df_geo.rename(columns={"geoAreaCode": "code", "geoAreaName": "name"}, inplace=True)
        df_geo["code"] = df_geo["code"].astype(str)
        return df_geo           
    
    # generic ArcGIS downloader with pagination for the SDR layers
    def get_arcgis_data(self, dataset : str = ""):
        if dataset=="sdr_backdated":
            url = "https://services7.arcgis.com/IyvyFk20mB7Wpc95/arcgis/rest/services/SDR_2025_TIMESERIES_(Backdated_Data)/FeatureServer/0/query"
        elif dataset=="sdr2025":
            url = "https://services7.arcgis.com/IyvyFk20mB7Wpc95/ArcGIS/rest/services/Sustainable_Development_Report_2025_(with_indicators)/FeatureServer/0/query"
        elif dataset=="rawdata":
            url = "https://services7.arcgis.com/IyvyFk20mB7Wpc95/arcgis/rest/services/SDR_2025_TIMESERIES_(Raw_Trend_Data)/FeatureServer/0/query"
        elif dataset=="codebook":
            url = "https://services7.arcgis.com/IyvyFk20mB7Wpc95/arcgis/rest/services/SDR_2025_CODEBOOK/FeatureServer/0/query"

        all_records = []
        offset = 0
        page_size = 2000  # safe default on ArcGIS Online

        while True:
            params = {
                "f": "json",
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "outSR": 4326,
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
            resp = requests.get(url, params=params, timeout=60).json()

            features = resp.get("features") or []
            if not features:
                break 

            all_records.extend([f["attributes"] for f in features])
            offset += len(features)

        return pd.DataFrame(all_records)

    # the big prep step: downloads codebook + SDR datasets, builds a unified 
    # df_lookup (codes/goals/descriptions/groups) and a cleaned SDG timeseries table (df_sdg) with friendly column names. Returns (df_sdg, df_lookup).
    def get_clean_data(self):
        df_goals = self.get_sdg_goals_grouped(class_code=1)
        df_codebook = self.get_arcgis_data(dataset="codebook")
        df_sdr2025 = self.get_arcgis_data(dataset="sdr2025")
        df_backdated = self.get_arcgis_data(dataset="sdr_backdated")

        df_goals["sdg"] = df_goals["code"].astype(int)
        goals_lookup = pd.DataFrame({
            "code": df_goals["sdg"].apply(lambda x: f"Goal_{x}"),
            "sdg": df_goals["sdg"],
            "description": df_goals["title"],
            "group": df_goals["group"]
        })

        goal_group_map = df_goals.set_index("sdg")["group"]

        df_codebook_clean = df_codebook[["IndCode", "SDG", "Indicator"]].copy()
        codebook_lookup = pd.DataFrame({
            "code": df_codebook_clean["IndCode"],
            "sdg": df_codebook_clean["SDG"].astype(int),
            "description": df_codebook_clean["Indicator"]
        })

        codebook_lookup["group"] = codebook_lookup["sdg"].map(goal_group_map)
        df_lookup = pd.concat([goals_lookup, codebook_lookup], ignore_index=True)
        df_lookup = df_lookup[["code", "sdg", "group", "description"]]

        df_backdated_st = df_backdated.copy()
        df_backdated_st = df_backdated_st.dropna(subset=["indexreg_"])
        df_backdated_st = df_backdated_st.rename(columns=lambda x: f"Goal_{x[4:]}" if x.lower().startswith("goal") else x)
        df_backdated_st = df_backdated_st.rename(columns=lambda x: x.replace("n_", "") if x.startswith("n_") else x)

        df_backdated_st = df_backdated_st[
            ['id', 'Country', 'year'] +
            [col for col in df_backdated_st.columns if col.startswith('indexreg_') or col.startswith('sdg') or col.startswith('Goal_')]
        ]

        if "sdgi_s" in df_backdated_st.columns:
            df_backdated_st = df_backdated_st.drop(columns=["sdgi_s"])

        df_sdr2025_st = df_sdr2025.copy()
        df_sdr2025_st = df_sdr2025_st.rename(columns=lambda x: x.replace("Score_", "") if x.startswith("Score_") else x)
        df_sdr2025_st = df_sdr2025_st[
            ['iso3', 'Name', 'Region'] +
            [
                col for col in df_sdr2025_st.columns
                if col.startswith('Goal_') and col.endswith('_Score') or col.startswith('sdg')
            ]
        ]
        df_sdr2025_st = df_sdr2025_st.rename(columns=lambda x: x.replace("_Score", "") if x.endswith("_Score") else x)

        df_backdated_st = df_backdated_st.rename(columns={"id": "ID", "year": "Year", "indexreg_": "Region"})
        df_sdr2025_st = df_sdr2025_st.rename(columns={"iso3": "ID", "Name": "Country"})
        df_sdr2025_st.insert(2, "Year", 2025)
        df_sdg = pd.concat([df_backdated_st, df_sdr2025_st], ignore_index=True)
        df_sdg = df_sdg.sort_values(by=["Country", "Year"]).reset_index(drop=True)
        num_cols = [col for col in df_sdg.columns if "sdg" in col or "Goal" in col]
        df_sdg[num_cols] = df_sdg[num_cols].apply(pd.to_numeric, errors="coerce")
        df_sdg = df_sdg.round(0)
        region_map = {
            "E. Europe & C. Asia": "E_Euro_Asia",
            "Western Europe (non-OECD)": "W_Europe",
            "Sub-Saharan Africa": "Africa",
            "East & South Asia": "E_S_Asia"
        }
        df_sdg["Region"] = df_sdg["Region"].replace(region_map)

        return df_sdg, df_lookup

    def get_df_info(self, df=None):
        print(df.info())
        print(df.shape)
        print(df.describe())
        print(df.columns)
        print(df.isnull().sum())
        df.head()


    def _df_compact_profile(
        self,
        df: pd.DataFrame,
        max_rows: int = 8,
        max_num_cols: int = 8,
        max_cat_cols: int = 8,
    ) -> dict:
        """Small, token-friendly snapshot of a DataFrame for LLM analysis."""
        import numpy as np

        prof: dict = {}
        prof["shape"] = [int(df.shape[0]), int(df.shape[1])]
        prof["columns"] = list(map(str, df.columns))
        prof["dtypes"] = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}

        # nulls
        nulls = df.isnull().sum()
        prof["nulls_by_col"] = {str(c): int(nulls[c]) for c in df.columns}

        # numeric summary
        num_cols = df.select_dtypes(include="number").columns.tolist()[:max_num_cols]
        if num_cols:
            desc = (
                df[num_cols]
                .apply(pd.to_numeric, errors="coerce")
                .describe()
                .round(3)
                .to_dict()
            )
            prof["numeric_summary"] = desc

        # categorical summary
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()[:max_cat_cols]
        cat_info = {}
        for c in cat_cols:
            vc = df[c].astype(str).value_counts(dropna=False)
            top = vc.index[0] if len(vc) else None
            freq = int(vc.iloc[0]) if len(vc) else 0
            cat_info[str(c)] = {
                "n_unique": int(df[c].nunique(dropna=True)),
                "top": None if top is None else str(top),
                "top_freq": freq,
            }
        if cat_info:
            prof["categorical_summary"] = cat_info

        # small head sample (stringified to be safe)
        prof["sample_head"] = df.head(max_rows).astype(str).to_dict(orient="records")
        return prof


    def plotly_fig_to_data_url(self, fig, width=1200, height=800, scale=2, auto_install_chrome=True):
        try:
            png = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
        except Exception as e:
            msg = str(e).lower()
            #if auto_install_chrome and "chrome" in msg and ("not found" in msg or "install" in msg):
            #    import kaleido
            #    kaleido.get_chrome_sync()
            #    png = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
            #else:
            #    raise
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("utf-8")
        return data_url

    def generate_nlp_insight(
        self,
        fig: Optional[go.Figure] = None,
        df: Optional[pd.DataFrame] = None,
        *,
        title: Optional[str] = None,
        nlp_mod: str = "gpt-5-mini",
    ) -> str:
        """
        Generate Markdown insights for a Plotly figure OR a pandas DataFrame.
        - Pass `fig` for chart analysis (vision).
        - Pass `df` for dataset/table analysis (text profile).
        """
        assert fig is not None or df is not None, "Provide either `fig` or `df`."

        content = []
        heading = f"### NLP generated insight [Model: {nlp_mod}]"
        context_hdr = f"{heading}\n"

        if fig is not None:
            # Keep your current image flow
            data_url = self.plotly_fig_to_data_url(fig)
            prompt = (
                "You are the insights writer for an SDG analytics app. You will receive a chart "
                "image and a small JSON payload describing what is plotted. Produce the following:\n"
                f"1) A heading '### NLP generated insight [Model : {nlp_mod}]'"
                "2) Two short sentences: first says what the figure shows; second explains how to read it.\n"
                "3) minimum of five concise insights; "
            )
            if title:
                prompt = f"Title: {title}\n" + prompt

            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]

        else:
            # DataFrame mode → compact profile to keep tokens small
            profile = self._df_compact_profile(df)
            prompt = (
                "You will receive a compact JSON profile of a dataset. Write:\n"
                f"1) A heading '### NLP generated insight [Model : {nlp_mod}]'"
                "2) At least five concise bullet insights about what the data shows.\n"
                "Keep it under 180–220 words."
            )
            if title:
                prompt = f"Dataset: {title}\n" + prompt

            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_text", "text": "DATA PROFILE (JSON):"},
                {"type": "input_text", "text": json.dumps(profile, ensure_ascii=False)},
            ]

        res = client.responses.create(
            model=nlp_mod,
            instructions="You are the insights writer for an SDG analytics app. Output Markdown only.",
            input=[{"role": "user", "content": content}],
        )

        out = (res.output_text or "").strip()
        return out or "_No insight returned._"



    # --- Chatbot --------------------------------


    def set_data_for_chat(self, df_sdg: pd.DataFrame, df_lookup: pd.DataFrame) -> None:
        """Register dataset context so chat replies can be data-aware (no raw table upload)."""

        fp = (int(df_sdg.shape[0]), int(df_sdg.shape[1]), int(df_lookup.shape[0]))
        if getattr(self, "_chat_fp", None) == fp:
            return  # already primed for this data
        self._chat_fp = fp

        self._df_sdg = df_sdg
        self._df_lookup = df_lookup
        self._chat_static = ""
        self._chat_dynamic = ""
        self._retrieval_docs = []

        # Basic profile
        years = sorted(pd.to_numeric(df_sdg["Year"], errors="coerce").dropna().unique().astype(int))
        goals = [c for c in df_sdg.columns if c.startswith("Goal_")]
        sdg_codes = [c for c in df_sdg.columns if c.lower().startswith("sdg")]
        n_countries = df_sdg["Country"].nunique() if "Country" in df_sdg.columns else 0
        n_regions   = df_sdg["Region"].nunique() if "Region" in df_sdg.columns else 0
        groups = (df_lookup["group"].dropna().unique().tolist()
                if {"group"} <= set(df_lookup.columns) else [])

        self._chat_static = (
            "DATA CONTEXT\n"
            f"Years: {years[0]}–{years[-1]} ({len(years)} years)\n"
            f"Entities: {n_countries} countries • {n_regions} regions\n"
            f"Goal columns: {', '.join(goals)}\n"
            f"SDG code columns: {len(sdg_codes)} total\n"
            f"Groups: {', '.join(sorted(groups))}\n"
            "Scores generally 0–100 (higher is better unless noted). Correlation ≠ causation.\n"
        )

        # Tiny keyword retriever from df_lookup (code / group / description)
        if {"code", "group", "description"} <= set(df_lookup.columns):
            for _, r in df_lookup[["code", "group", "description"]].dropna().iterrows():
                self._retrieval_docs.append(f"{r['code']} ({r['group']}): {r['description']}")

    def _retrieve_lookup(self, question: str, k: int = 6) -> str:
        """Very small keyword retriever over df_lookup text (no external deps)."""

        if not getattr(self, "_retrieval_docs", None):
            return ""
        q_terms = set(t for t in re.findall(r"[A-Za-z0-9_]+", question.lower()) if len(t) > 2)
        scored = []
        for doc in self._retrieval_docs:
            d_terms = set(re.findall(r"[A-Za-z0-9_]+", doc.lower()))
            score = len(q_terms & d_terms)
            scored.append((score, doc))
        top = [d for s, d in sorted(scored, key=lambda x: x[0], reverse=True)[:k] if s > 0]
        return "\n".join(top)
    

    def _view_fingerprint(self, fig=None, df=None):
        h = hashlib.sha256()
        if fig is not None:
            j = fig.to_plotly_json()
            # prune: keep only elements that define the view identity
            pruned = {
                "layout_title": j.get("layout", {}).get("title", {}).get("text"),
                "series": [t.get("name") for t in j.get("data", [])[:30]],
                "x": j.get("layout", {}).get("xaxis", {}).get("title", {}).get("text"),
                "y": j.get("layout", {}).get("yaxis", {}).get("title", {}).get("text"),
            }
            h.update(json.dumps(pruned, sort_keys=True, default=str).encode())
        if df is not None:
            prof = self._df_compact_profile(df, max_rows=4, max_num_cols=6, max_cat_cols=4)
            h.update(json.dumps(prof, sort_keys=True, default=str).encode())
        return h.hexdigest()    

    def update_plot_context(
        self,
        *,
        fig=None,
        df: "pd.DataFrame | None" = None,
        title: str | None = None,
    ) -> None:
        """
        Make the chatbot 'view-aware' using ONLY fig/df.
        - fig -> snapshot as PNG data-URL (small)
        - df  -> compact JSON profile (few rows/cols + schema/stats)
        """

        fp = self._view_fingerprint(fig=fig, df=df)
        if getattr(self, "_chat_view_fp", None) == fp:
            # Same view; just refresh the CURRENT VIEW title text and return
            if title is None and fig is not None:
                try: title = fig.layout.title.text or "Current view"
                except Exception: title = "Current view"
            self._chat_dynamic = f"CURRENT VIEW\nTitle: {title or 'Current view'}\n"
            return

        # 1) Title
        if title is None and fig is not None:
            try:
                t = fig.layout.title.text
                title = str(t) if t else None
            except Exception:
                title = None
        if title is None:
            title = "Current view"

        # 2) Build a tiny, helpful text context (also tells you if it saw a table)
        parts = []
        if df is not None and hasattr(df, "shape"):
            try:
                r, c = df.shape
                cols = [str(x) for x in df.columns[:8]]
                parts.append(f"Table: {r}×{c} cols={', '.join(cols)}{'…' if c>8 else ''}")
                if "Year" in df.columns:
                    yrs = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).unique()
                    if len(yrs) == 1:
                        parts.append(f"Year={int(yrs[0])}")
                    elif len(yrs) > 1:
                        parts.append(f"Years={int(np.min(yrs))}–{int(np.max(yrs))}")
            except Exception:
                pass
        if fig is not None:
            try:
                xlab = fig.layout.xaxis.title.text if fig.layout.xaxis.title.text else None
                ylab = fig.layout.yaxis.title.text if fig.layout.yaxis.title.text else None
                if xlab and ylab:
                    parts.append(f"Axes: x={xlab}, y={ylab}")
            except Exception:
                pass

        info_line = " • ".join(parts) if parts else "—"
        self._chat_dynamic = f"CURRENT VIEW\nTitle: {title}\nInfo: {info_line}\n"

        # 3) Attach lightweight payloads for the chat call
        self._chat_view = {}
        # figure snapshot
        if fig is not None:
            try:
                self._chat_view["image"] = self.plotly_fig_to_data_url(fig, width=900, height=600, scale=1)
            except Exception:
                self._chat_view["image"] = None
        # dataframe compact profile
        if df is not None and hasattr(df, "head"):
            try:
                self._chat_view["profile"] = self._df_compact_profile(
                    df, max_rows=6, max_num_cols=8, max_cat_cols=6
                )
            except Exception:
                self._chat_view["profile"] = None

        self._chat_view_fp = fp


    def chat(
        self,
        prompt: str,
        history: list[dict] | None = None,
        model: str = "gpt-5-mini",
        system: str = (
            "You are the assistant for an SDG analytics app. Be concise and precise. "
            "You only answer questions related to SDG. "
            "Use the provided DATA CONTEXT and CURRENT VIEW when answering. "
            "Do not recommend actions not in SDG data context and current view. "
        ),
    ) -> str:
        """Chat that automatically includes dataset context + current view + small lookup snippets."""
        history = history or []
        retrieved = self._retrieve_lookup(prompt, k=6)
        context = (getattr(self, "_chat_static", "") or "") + "\n" + (getattr(self, "_chat_dynamic", "") or "")
        if retrieved:
            context += "\nLOOKUP NOTES\n" + retrieved

        # pack brief history + user message
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history[-8:]]
        lines.append(f"User: {prompt}")
        packed = f"{context}\n\nConversation so far:\n" + "\n".join(lines)

        # NEW: include view payload (image/profile) if present
        content = [{"type": "input_text", "text": packed}]
        view = getattr(self, "_chat_view", None)
        if view:
            if view.get("image"):
                content.append({"type": "input_image", "image_url": view["image"]})
            if view.get("profile"):
                import json as _json
                content.append({"type": "input_text", "text": "DATA PROFILE (JSON):"})
                content.append({"type": "input_text", "text": _json.dumps(view["profile"], ensure_ascii=False)})

        res = client.responses.create(
            model=model,
            instructions=system,
            input=[{"role": "user", "content": content}],  # now supports text+image+profile
        )
        return (res.output_text or "").strip()




    # endregion

    # region [Ranking and timelines]

    # Ranks and plots (horizontal bars) the Goals or SDG indicators within a selected SDG group for a chosen year.
    def plot_group_ranking(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int,
        group: str, # Social, Economic, Environmental, Partnership
        level: str = "goal",     # "goal" or "sdg"
        template: str = "plotly_dark",
        ascending: bool = False, # False = highest first
        fig_height: int = 500,
        fig_width: int = 900
    ):
        df_year = df_sdg[df_sdg["Year"] == year].copy()
        group_norm = group.strip().lower()
        look = df_lookup[df_lookup["group"].str.lower() == group_norm].copy()

        if level == "goal":
            value_cols = [c for c in df_year.columns if c.startswith("Goal_") and c in look["code"].values]
            title = f"Ranking of Goals in {group} group • Year {year}"
            y_name = "Goal"
            if not value_cols:
                raise ValueError("No goal columns for that group")
            means = df_year[value_cols].mean(numeric_only=True).reset_index()
            means.columns = ["code", "value"]
            lab = means.merge(look[["code", "description"]], on="code", how="left")
            lab["y_label"] = lab["code"]
            out = lab.sort_values("value", ascending=ascending)

        elif level == "sdg":
            goal_codes = look["code"].unique()
            goal_nums = [re.findall(r"(\d+)", c)[0] for c in goal_codes if re.findall(r"(\d+)", c)]
            value_cols = [c for c in df_year.columns if c.lower().startswith("sdg") 
                        and any(c.startswith(f"sdg{n}") for n in goal_nums)]
            title = f"Ranking of SDG indicators in {group} group • Year {year}"
            y_name = "Indicator"
            if not value_cols:
                raise ValueError("No SDG indicator columns for that group")
            means = df_year[value_cols].mean(numeric_only=True).reset_index()
            means.columns = ["code", "value"]
            look_slim = look[["code", "sdg", "description"]] if "sdg" in look.columns else look.assign(sdg=None)[["code","sdg","description"]]
            lab = means.merge(look_slim, on="code", how="left")
            lab["y_label"] = lab["code"]
            out = lab.sort_values("value", ascending=ascending)

        else:
            raise ValueError("level must be 'goal' or 'sdg'")

        fig = px.bar(
            out,
            x="value",
            y="y_label",
            orientation="h",
            text="value",
            template=template,
            title=title,
            custom_data=["description"]
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Score: %{x:.2f}<extra></extra>"
        )
        fig.update_layout(yaxis_title=y_name, xaxis_title="Mean Score", margin=dict(l=10, r=10, t=60, b=10), height=fig_height, width=fig_width)
        return fig


    # horizontal bar chart ranking of goals/indicators/groups for a selected year; supports 
    # filtering by group or by goal number and sorts per your ascending choice.
    def plot_goal_ranking(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int,
        template: str = "plotly_dark",
        rank_by: str = "goal", # "goal", "sdg", or "group"
        group_filter=None,
        goal_filter=None, 
        ascending: bool = True,
        fig_height: int = 500,
        fig_width: int = 900
    ):
        df_year = df_sdg[df_sdg["Year"] == year].copy()

        if rank_by.lower() == "goal":
            value_cols = [c for c in df_year.columns if c.startswith("Goal_")]
            chart_title = f"Ranking of SDG Goals • Year {year}"

        elif rank_by.lower() == "sdg":
            value_cols = [c for c in df_year.columns if c.lower().startswith("sdg") and len(c) > 7]
            chart_title = f"Ranking of SDG Indicators • Year {year}"

        elif rank_by.lower() == "group":
            value_cols = [c for c in df_year.columns if c.startswith("Goal_")]
            chart_title = f"Ranking of SDG Groups • Year {year}"

        else:
            raise ValueError("rank_by must be 'goal', 'sdg', or 'group'")

        if not value_cols:
            raise ValueError("No matching columns found for the selected mode.")

        means = df_year[value_cols].mean(numeric_only=True).reset_index()
        means.columns = ["code", "value"]

        meta = df_lookup[["code", "description", "group", "sdg"]].copy()
        merged = means.merge(meta, on="code", how="left")

        if group_filter:
            groups = [group_filter] if isinstance(group_filter, str) else list(group_filter)
            groups = [g.lower() for g in groups]
            merged = merged[merged["group"].str.lower().isin(groups)]

        if rank_by.lower() == "sdg" and goal_filter:
            goals = [goal_filter] if isinstance(goal_filter, str) else list(goal_filter)
            wanted_nums = {re.findall(r"(\d+)", g)[0] for g in goals}
            merged = merged[merged["sdg"].astype(str).isin(wanted_nums)]

        if merged.empty:
            raise ValueError("No data left after filtering. Check your filters.")

        if rank_by.lower() == "group":
            out = (
                merged.groupby("group", as_index=False)
                    .agg(value=("value", "mean"), goals_in_group=("code", "count"))
            )
            out["y_label"] = out["group"].str.capitalize()
        else:
            if rank_by.lower() == "goal":
                merged["y_label"] = merged["code"] + " (" + merged["group"].fillna("").str.capitalize() + ")"
            else: 
                merged["y_label"] = merged.apply(
                    lambda r: f"{r['code']}" + (f" (Goal {int(r['sdg'])})" if pd.notna(r['sdg']) else ""),
                    axis=1
                )
            out = merged

        out = out.sort_values("value", ascending=ascending)

        fig = px.bar(
            out,
            x="value",
            y="y_label",
            orientation="h",
            text="value",
            template=template,
            title=chart_title,
            custom_data=[c for c in ["code","description","group","sdg","goals_in_group"] if c in out.columns]
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )
        fig.update_layout(
            yaxis_title=rank_by.capitalize(),
            xaxis_title="Mean Score",
            margin=dict(l=10, r=10, t=60, b=10),
            height=fig_height,
            width=fig_width
        )

        return fig


    # horizontal bar ranking for countries/regions by a chosen SDG measure (mean across selected columns)
    def plot_sdg_ranking(
        self,
        data: pd.DataFrame,
        value_col: str | None = None,           # None => Overall Score
        group_col: str = "Country",             # "Region" or "Country"
        top_n: int = 5,
        ascending: bool = False,
        year: int | None = None,
        aggfunc: str = "mean",                  # "mean","median","max","min"
        title: str | None = None,
        plot_theme: str = "plotly_dark",
        df_lookup: pd.DataFrame | None = None,
        fig_height: int = 400,
        fig_width: int = 1000
    ):
        d = data.copy()
        if year is not None:
            d = d[d["Year"] == year]

        # ---- dynamic groups from df_lookup (no hard-coding) ----
        groups_lower: set[str] = set()
        group_label_map: dict[str, str] = {}
        if df_lookup is not None and "group" in df_lookup.columns:
            _groups = df_lookup["group"].dropna().astype(str).str.strip()
            groups_lower = set(_groups.str.lower().unique())
            # map lower->original casing for nice labels
            group_label_map = {g.lower(): g for g in _groups.unique()}

        # default: Overall Score = mean of Goal_* that exist
        if value_col is None:
            goal_cols = [c for c in d.columns if c.startswith("Goal_")]
            if not goal_cols:
                raise ValueError("No Goal_* columns found to compute Overall Score.")
            d = d.copy()
            d["__overall__"] = d[goal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            value_key = "__overall__"
            label_for_x = "Overall Score"
            val_lower = "overall"
        else:
            val_lower = str(value_col).strip().lower()
            label_for_x = value_col
            value_key = value_col

            # handle SDG group via df_lookup (dynamic set)
            if val_lower in groups_lower:
                if df_lookup is None or not {"group", "code"}.issubset(df_lookup.columns):
                    raise ValueError("df_lookup with 'group' and 'code' columns is required when value_col is a group name.")
                codes_in_group = (
                    df_lookup.loc[df_lookup["group"].astype(str).str.strip().str.lower() == val_lower, "code"]
                    .dropna().astype(str).tolist()
                )
                cols = [c for c in codes_in_group if c in d.columns]
                if not cols:
                    raise ValueError(f"No columns for group '{value_col}' were found in the data.")
                d = d.copy()
                d["__group_metric__"] = d[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
                value_key = "__group_metric__"
                label_for_x = group_label_map.get(val_lower, value_col.title())

        # If ranking countries and Region exists, show "Country (Region)"
        if group_col == "Country" and "Region" in d.columns:
            d[group_col] = d["Country"] + " (" + d["Region"] + ")"

        # aggregate
        if aggfunc == "median":
            g = d.groupby(group_col, as_index=False)[value_key].median()
        elif aggfunc == "max":
            g = d.groupby(group_col, as_index=False)[value_key].max()
        elif aggfunc == "min":
            g = d.groupby(group_col, as_index=False)[value_key].min()
        else:
            g = d.groupby(group_col, as_index=False)[value_key].mean()

        g = g.sort_values(by=value_key, ascending=ascending).head(top_n)

        # auto title
        if title is None:
            rank_word = "Bottom" if ascending else "Top"
            if value_col is None:
                title_val = "Overall Score"
            elif val_lower in groups_lower:
                title_val = f"{label_for_x} group"
            else:
                title_val = value_col
            title = f"{rank_word} {top_n} {group_col} by {title_val}"
            if year is not None:
                title += f" • Year {year}"

        # plot
        fig = px.bar(
            g, x=value_key, y=group_col, orientation="h",
            text=value_key, title=title, template=plot_theme
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            xaxis_title=label_for_x,
            yaxis_title=group_col if group_col != "Country" else "Country (Region)",
            margin=dict(l=80, r=40, t=60, b=40),
            bargap=0.2,
            height=fig_height,
            width=fig_width,
        )
        return fig


    def get_ranking_table(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        *,
        year: int | None = None,
        goal: str | None = None,     # None => Overall; or "Goal_#", "sdg*", or a group name from df_lookup["group"]
        agg: str = "mean",
        decimals: int = 2,
    ) -> pd.DataFrame:
        d = df_sdg.copy()
        if year is not None:
            d = d[d["Year"] == int(year)]

        # ---- derive group names dynamically from df_lookup ----
        group_names_lower: set[str] = set()
        group_label_lookup: dict[str, str] = {}
        if df_lookup is not None and "group" in df_lookup.columns:
            groups_series = df_lookup["group"].dropna().astype(str).str.strip()
            group_names_lower = set(groups_series.str.lower().unique())
            # map lowercased -> original (for nicer label casing)
            group_label_lookup = {g.lower(): g for g in groups_series.unique()}

        # Label helper
        def _score_label_from_goal(g: str | None) -> str:
            if g is None:
                return "Overall Score"
            g_str = str(g).strip()
            gl = g_str.lower()
            if g_str.startswith("Goal_"):
                return f"Goal {g_str.split('_', 1)[1]}"
            if gl in group_names_lower:
                return group_label_lookup.get(gl, g_str.title())
            return g_str  # e.g., "sdg9_uni" or any literal column

        score_src_label = _score_label_from_goal(goal)

        # ---- build score series ----
        if goal is None or str(goal).strip().lower() in {"overall", "overall score", "overall scores"}:
            goal_cols = [c for c in d.columns if c.startswith("Goal_")]
            if not goal_cols:
                raise ValueError("No Goal_* columns found to compute Overall Score.")
            score = d[goal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        else:
            g = str(goal).strip()
            gl = g.lower()
            if gl in group_names_lower:
                # need df_lookup with group->code mapping
                if df_lookup is None or not {"group", "code"}.issubset(df_lookup.columns):
                    raise ValueError("df_lookup with 'group' and 'code' columns is required when goal is a group name.")
                codes = (
                    df_lookup.loc[df_lookup["group"].astype(str).str.strip().str.lower() == gl, "code"]
                    .dropna().astype(str).tolist()
                )
                cols = [c for c in codes if c in d.columns]
                if not cols:
                    raise ValueError(f"No df_sdg columns matched the '{g}' group.")
                score = d[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            else:
                # treat as a literal column (Goal_* or sdg*)
                if g not in d.columns:
                    raise ValueError(f"Column not found: {g}")
                score = pd.to_numeric(d[g], errors="coerce")

        out = d[["Country", "Region"]].copy()
        out["Score"] = score

        # aggregate across rows per entity if year not fixed
        if year is None:
            if agg == "median":
                out = out.groupby(["Country", "Region"], as_index=False)["Score"].median()
            elif agg == "max":
                out = out.groupby(["Country", "Region"], as_index=False)["Score"].max()
            elif agg == "min":
                out = out.groupby(["Country", "Region"], as_index=False)["Score"].min()
            else:
                out = out.groupby(["Country", "Region"], as_index=False)["Score"].mean()

        # order, rank, round, and label columns
        out = (out.dropna(subset=["Score"])
                .sort_values("Score", ascending=False)
                .reset_index(drop=True))

        out["Score"] = out["Score"].round(decimals)

        rank_col_name = f"Rank ({int(year)})" if year is not None else "Rank"
        score_col_name = f"Score ({score_src_label})"

        out.insert(0, rank_col_name, range(1, len(out) + 1))
        out = out.rename(columns={"Score": score_col_name})

        return out



    # line chart over time for Goals or SDG indicators
    def plot_sdg_timeline(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        items: str | list[str] | None = None, # "Goal_9", "sdg9_uni", ["Goal_1","Goal_9"], "economic", None
        mode: str = "auto", # "goal", "sdg", or "auto"
        group_filter: str | list[str] | None = None,  # "economic" | ["social","environmental"] | None
        entity_type: str | None = None, # None, "Region", or "Country"
        entities: str | list[str] | None = None, # a name or list to filter when entity_type is set
        agg: str = "mean", # "mean" or "median"
        template: str = "plotly_dark",
        p_height: int = 600,
        title_prefix: str = "SDG timeline"
    ):
        def _norm_list(x):
            if x is None: return None
            return [x] if isinstance(x, str) else list(x)

        def _is_group_name(s: str) -> bool:
            return s.lower() in set(df_lookup["group"].str.lower().unique())

        def _prefix_for(m):
            return "Goal_" if m == "goal" else "sdg"

        items_list = _norm_list(items)
        if mode not in {"goal", "sdg", "auto"}:
            raise ValueError("mode must be 'goal', 'sdg', or 'auto'.")

        if mode == "auto":
            if items_list and any(str(x).lower().startswith("sdg") for x in items_list):
                mode = "sdg"
            else:
                mode = "goal"

        prefix = _prefix_for(mode)

        look = df_lookup[df_lookup["code"].str.startswith(prefix)][["code", "group", "description"]].drop_duplicates()

        groups = _norm_list(group_filter)
        if groups:
            groups_lc = [g.lower() for g in groups]
            look = look[look["group"].str.lower().isin(groups_lc)]

        if items_list is None:
            codes = look["code"].tolist()
        else:
            expanded: list[str] = []
            for it in items_list:
                it_str = str(it)
                if _is_group_name(it_str):
                    expanded += look[look["group"].str.lower() == it_str.lower()]["code"].tolist()
                else:
                    if it_str.startswith(prefix):
                        expanded.append(it_str)
            codes = list(dict.fromkeys(expanded))

        codes = [c for c in codes if c in df_sdg.columns]

        if not codes:
            raise ValueError("No matching columns found in df_sdg for the given inputs (mode, groups, items).")

        df = df_sdg.copy()
        if entity_type and entities:
            ents = _norm_list(entities)
            df = df[df[entity_type].isin(ents)]

        df_long = df[["Year"] + codes].melt("Year", var_name="code", value_name="value")

        if agg == "median":
            df_plot = df_long.groupby(["Year", "code"], as_index=False)["value"].median()
        else:
            df_plot = df_long.groupby(["Year", "code"], as_index=False)["value"].mean()

        df_plot = df_plot.merge(look, on="code", how="left")
        df_plot["label"] = df_plot["code"] + " (" + df_plot["group"].str.capitalize().fillna("") + ")"

        scope_txt = ""
        if entity_type and entities:
            ents = _norm_list(entities)
            scope_txt = f" • {entity_type}: {', '.join(ents)}"

        if groups:
            group_txt = f" • Groups: {', '.join(groups)}"
        else:
            group_txt = ""

        sel_txt = ", ".join(codes[:5]) + (" …" if len(codes) > 5 else "")
        mode_txt = "Goals" if mode == "goal" else "SDG indicators"

        title = f"{title_prefix} • {mode_txt}{group_txt}{scope_txt}<br><sup>Showing: {sel_txt}</sup>"

        fig = px.line(
            df_plot, x="Year", y="value", color="label", markers=True,
            template=template, title=title,
            hover_data={"code": True, "group": True, "description": True, "value": ":.2f"},
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Score: %{y:.2f}<br>"
                "Group: %{customdata[1]}<br>"
                "%{customdata[2]}"
            )
        )

        fig.update_layout(
            xaxis_title="Year", yaxis_title="Score",
            legend_title="Series", height=p_height,
            margin=dict(l=12, r=12, t=70, b=10),
        )
        return fig


    # compare one Goal_* across multiple countries or regions over time; chooses mean/median and decorates titles/hover from df_lookup
    def plot_goal_entity_timeline(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        goal: str | None = None,       
        entity_type: str = "Country",
        entities=None,
        agg: str = "mean",
        template: str = "plotly_dark",
        *,
        fig_height: int = 600,
        top_n: int | None = None,
        top_mode: str = "top",
        rank_year: int | None = None
    ):
        import plotly.express as px

        # --- resolve the series to plot ---
        df = df_sdg.copy()

        is_overall = (goal is None) or (
            isinstance(goal, str) and goal.strip().lower() == "overall score"
        )

        goal_cols = [c for c in df.columns if c.startswith("Goal_")]
        overall_col = None
        goal_label = "Overall Score" if is_overall else str(goal)

        if is_overall:
            if "Overall_Score" in df.columns:
                overall_col = "Overall_Score"
            else:
                overall_col = "_overall_tmp_"
                df[overall_col] = df[goal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            target_col = overall_col
        else:
            target_col = goal_label
            if target_col not in df.columns:
                raise ValueError(f"{goal_label} is not a valid column (looked for '{target_col}')")

        # --- entity filtering (unchanged) ---
        if entities is None:
            entities = df[entity_type].dropna().unique().tolist()
        elif isinstance(entities, str):
            entities = [entities]

        df = df[df[entity_type].isin(entities)].copy()
        if df.empty:
            raise ValueError("No rows found for the given entities and entity_type")

        # --- metadata / titles ---
        if overall_col:
            goal_group, goal_desc = "All Goals", "Mean of Goal_1 … Goal_17 per row"
        else:
            meta = df_lookup[df_lookup["code"] == target_col][["code", "group", "description"]].drop_duplicates()
            if meta.empty:
                goal_group, goal_desc = "", ""
            else:
                goal_group = meta["group"].iloc[0]
                goal_desc  = meta["description"].iloc[0]

        # --- aggregation & ranking (same logic as before, just using target_col) ---
        group_keys = ["Year", entity_type]
        if agg == "median":
            df_plot = df.groupby(group_keys, as_index=False)[target_col].median()
        else:
            df_plot = df.groupby(group_keys, as_index=False)[target_col].mean()

        df_plot.rename(columns={target_col: "value", entity_type: "Entity"}, inplace=True)

        ryear = int(df_plot["Year"].max()) if rank_year is None else int(rank_year)
        df_rank = df_plot[df_plot["Year"] == ryear].dropna(subset=["value"]).copy()
        if df_rank.empty:
            raise ValueError(f"No data available for ranking in year {ryear}")

        if top_n is not None and top_n > 0:
            if top_mode.lower() == "bottom":
                df_rank = df_rank.sort_values("value", ascending=True).head(top_n)
                order_entities = df_rank.sort_values("value", ascending=True)["Entity"].tolist()
                ttl_prefix = f"Bottom {top_n}"
            else:
                df_rank = df_rank.sort_values("value", ascending=False).head(top_n)
                order_entities = df_rank.sort_values("value", ascending=False)["Entity"].tolist()
                ttl_prefix = f"Top {top_n}"
            df_plot = df_plot[df_plot["Entity"].isin(order_entities)].copy()
        else:
            order_entities = sorted(df_plot["Entity"].unique())
            ttl_prefix = "All"

        ents_txt = ", ".join(order_entities) if len(order_entities) < 10 else f"{len(order_entities)} entities"
        title_main = f"{goal_label} ({goal_group})" if goal_group else f"{goal_label}"
        title_sub  = f"{goal_desc}<br><sup>{ttl_prefix} — ranked on {ryear} • {entity_type}: {ents_txt}</sup>"

        fig = px.line(
            df_plot, x="Year", y="value", color="Entity", markers=True,
            template=template,
            title=title_main + "<br>" + title_sub,
            hover_data={"value":":.2f", "Year": True, "Entity": True},
            category_orders={"Entity": order_entities},
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title=goal_label,   # ← use label
            height = fig_height,
            legend_title=entity_type,
            margin=dict(l=10, r=10, t=80, b=10),
        )

        fig.update_xaxes(dtick=5)

        if overall_col == "_overall_tmp_":
            df.drop(columns=[overall_col], inplace=True, errors="ignore")


        return fig



    # endregion

    # region [Correlation and structure]

    # computes a correlation table (year/mode filters) and draws a heatmap with labels and hover
    def plot_corr_matrix(
            self,
            df,
            year=2025,
            group_col=None, # "Country" or "Region"
            group_val=None,# "New Zealand, Canada" or "OECD"
            mode="auto", # "auto", "across_years", "across_countries"
            years=None, # (2010, 2025) or [2012,2015,2020]
            width=1000,
            height=700,
            decimals=2,
            square=False,
            dark=True,
            min_rows=3
        ):
        if mode == "auto":
            if group_col == "Country":
                mode_effective = "across_years"
            elif group_col == "Region":
                mode_effective = "across_countries"
            else:
                mode_effective = "across_countries"
        else:
            mode_effective = mode

        df_work = df.copy()

        title_suffix = ""
        if mode_effective == "across_countries":
            df_work = df_work[df_work["Year"] == year]
            if group_col and group_val:
                if group_col not in df_work.columns:
                    raise ValueError(f"{group_col} not found in dataframe")
                df_work = df_work[df_work[group_col] == group_val]
                title_suffix = f"{group_col}: {group_val}  Year: {year}"
            else:
                title_suffix = f"All Countries  Year: {year}"

        elif mode_effective == "across_years":
            if group_col != "Country":
                raise ValueError("across_years mode requires group_col='Country'")
            if not group_val:
                raise ValueError("across_years mode requires a specific Country as group_val")
            df_work = df_work[df_work["Country"] == group_val]

            if years is not None:
                if isinstance(years, tuple) and len(years) == 2:
                    y0, y1 = years
                    df_work = df_work[(df_work["Year"] >= y0) & (df_work["Year"] <= y1)]
                    title_suffix = f"Country: {group_val}  Years: {y0}-{y1}"
                else:
                    years_set = set(years)
                    df_work = df_work[df_work["Year"].isin(years_set)]
                    title_suffix = f"Country: {group_val}  Years: {sorted(list(years_set))}"
            else:
                yrs = sorted(df_work["Year"].unique().tolist())
                title_suffix = f"Country: {group_val}  Years: {yrs[0]}-{yrs[-1]}" if len(yrs)>1 else f"Country: {group_val}"

        sdg_cols = [c for c in df_work.columns if c.startswith("Goal_")]
        df_scores = df_work[sdg_cols].copy()

        nunique = df_scores.nunique(dropna=True)
        keep_cols = [c for c in sdg_cols if nunique.get(c, 0) > 1]
        df_scores = df_scores[keep_cols]

        if df_scores.shape[0] < min_rows or df_scores.shape[1] < 2:
            msg_lines = []
            msg_lines.append("Not enough data to compute correlations.")
            msg_lines.append(f"Rows available: {df_scores.shape[0]}  Columns with variance: {df_scores.shape[1]}")
            if mode_effective == "across_years":
                msg_lines.append("Tip: widen the years range for this country.")
            else:
                msg_lines.append("Tip: choose a region with more countries or pick a different year.")
            text_msg = "<br>".join(msg_lines)

            fig = px.imshow(np.array([[np.nan]]), text_auto=False, zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
            fig.update_layout(
                width=width,
                height=height,
                title=f"Correlation Matrix unavailable  {title_suffix}",
                template="plotly_dark" if dark else "plotly",
                annotations=[dict(text=text_msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
            )
            return fig

        corr = df_scores.corr()

        fmt = f".{decimals}f"
        fig = px.imshow(
            corr,
            text_auto=fmt,
            aspect="equal" if square else "auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig.update_layout(
            width=width,
            height=height,
            title=f"Correlation Matrix of SDG Scores  {title_suffix}",
            template="plotly_dark" if dark else "plotly"
        )
        fig.update_xaxes(tickangle=-90)
        return fig


    def corr_by_entity(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        *,
        level: str = "goal", # "goal" or "sdg"
        entity_type: str | None = None, # None, "Region", or "Country"
        entities: str | list[str] | None = None,
        mode: str = "within_year", # "within_year" or "across_years"
        year: int | None = 2025, # used when mode == "within_year"
        years: tuple[int, int] | list[int] | None = None, # used when mode == "across_years"
        agg: str = "none", # "none", "mean", "median"
        template: str = "plotly_dark",
        decimals: int = 2,
        min_rows: int = 3,
        zmin: float = -1.0,
        zmax: float = 1.0,
        fig_width: int = 1000,
        fig_height: int = 700,
        return_fig: bool = True,
        p_aspect: str = "auto" # auto or equal
    ):

        if level.lower() == "goal":
            value_cols = [c for c in df_sdg.columns if isinstance(c, str) and c.startswith("Goal_")]
        elif level.lower() == "sdg":
            value_cols = [c for c in df_sdg.columns if isinstance(c, str) and c.lower().startswith("sdg")]
        else:
            raise ValueError("level must be 'goal' or 'sdg'")

        if not value_cols:
            raise ValueError("No value columns found for the selected level")

        d = df_sdg.copy()

        def _to_list(x):
            if x is None:
                return None
            return [x] if isinstance(x, str) else list(x)

        ents = _to_list(entities)

        if entity_type is not None:
            if entity_type not in {"Region", "Country"}:
                raise ValueError("entity_type must be None, 'Region', or 'Country'")
            if ents is not None:
                d = d[d[entity_type].isin(ents)]

        title_suffix = ""
        if mode == "within_year":
            if year is None:
                raise ValueError("year must be provided for mode='within_year'")
            d = d[d["Year"] == int(year)].copy()
            title_bits = [f"Year {year}"]
            if entity_type and ents:
                title_bits.append(f"{entity_type}: {', '.join(ents)}")
            elif entity_type:
                title_bits.append(f"{entity_type}: ALL")
            title_suffix = "  ".join(title_bits)

            if agg.lower() in {"mean", "median"} and entity_type is not None:
                grp_keys = []
                if entity_type == "Region":
                    grp_keys = ["Year", entity_type]
                elif entity_type == "Country":
                    grp_keys = ["Year", entity_type]
                if grp_keys:
                    if agg.lower() == "median":
                        d = d.groupby(grp_keys, as_index=False)[value_cols].median(numeric_only=True)
                    else:
                        d = d.groupby(grp_keys, as_index=False)[value_cols].mean(numeric_only=True)

        elif mode == "across_years":
            if entity_type is None or ents is None or len(ents) != 1:
                raise ValueError("across_years mode requires entity_type and exactly one entity in 'entities'")
            d = d[d[entity_type] == ents[0]].copy()

            if years is None:
                pass
            elif isinstance(years, tuple) and len(years) == 2:
                y0, y1 = int(years[0]), int(years[1])
                d = d[(d["Year"] >= y0) & (d["Year"] <= y1)]
            else:
                year_set = set(int(y) for y in years)
                d = d[d["Year"].isin(year_set)]

            title_bits = [f"{entity_type}: {ents[0]}"]
            yr_list = sorted(d["Year"].unique().tolist())
            if yr_list:
                title_bits.append(f"Years {yr_list[0]} to {yr_list[-1]}" if len(yr_list) > 1 else f"Year {yr_list[0]}")
            title_suffix = "  ".join(title_bits)

            if agg.lower() in {"mean", "median"}:
                if agg.lower() == "median":
                    d = d.groupby(["Year"], as_index=False)[value_cols].median(numeric_only=True)
                else:
                    d = d.groupby(["Year"], as_index=False)[value_cols].mean(numeric_only=True)
        else:
            raise ValueError("mode must be 'within_year' or 'across_years'")

        X = d[value_cols].apply(pd.to_numeric, errors="coerce")
        nunique = X.nunique(dropna=True)
        keep = [c for c in X.columns if nunique.get(c, 0) > 1]
        X = X[keep].dropna(how="all")

        if X.shape[0] < min_rows or X.shape[1] < 2:
            if return_fig:
                fig = px.imshow(
                    np.array([[np.nan]]),
                    text_auto=False,
                    zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r",
                    aspect=p_aspect
                )
                fig.update_layout(
                    width=fig_width, height=fig_height,
                    title=f"Correlation not available  {title_suffix}",
                    template=template,
                    annotations=[dict(
                        text=f"Not enough data  rows {X.shape[0]}  vars {X.shape[1]}",
                        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
                    )]
                )
                return pd.DataFrame(), fig
            return pd.DataFrame(), None

        corr_df = X.corr().round(decimals)

        fig = None
        if return_fig:
            fig = px.imshow(
                corr_df,
                text_auto=f".{decimals}f",
                color_continuous_scale="RdBu_r",
                zmin=zmin, zmax=zmax,
                aspect=p_aspect
            )
            fig.update_layout(
                width=fig_width,
                height=fig_height,
                title=f"Correlation Matrix  {title_suffix}",
                template=template
            )
            fig.update_xaxes(tickangle=-90)

        return corr_df, fig


    # builds a grid of heatmaps showing correlations within/among groups (economic/social/environmental), with flexible layout and annotations.
    def grouped_corr_heatmaps(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int = 2025,
        group_pairs = (("economic","social"), ("economic","environmental"), ("social","environmental")),
        template: str = "plotly_dark",
        zmin: float = -1.0,
        zmax: float = 1.0,
        indicator_mode: bool = False, # False → use Goal_*; True → use sdg* indicators
        title: str = "Cross-Group SDG Correlations",
        n_cols: int = 3, # number of subplot columns
        fig_height: int = 400,
        fig_width: int = 360
    ):

        d = df_sdg[df_sdg["Year"] == year].copy()

        if indicator_mode:
            value_cols = [c for c in d.columns if isinstance(c, str) and c.lower().startswith("sdg")]
            kind_name = "Indicators"
        else:
            value_cols = [c for c in d.columns if isinstance(c, str) and c.startswith("Goal_")]
            kind_name = "Goals"

        if not value_cols:
            raise ValueError("No value columns found for the selected mode.")

        meta = df_lookup[["code", "group"]].dropna().copy()
        meta["group"] = meta["group"].str.lower()
        code_to_group = dict(zip(meta["code"], meta["group"]))

        value_cols = [c for c in value_cols if code_to_group.get(c, None) in {"economic","social","environmental","partnership"}]
        if not value_cols:
            raise ValueError("No columns matched to (economic/social/environmental/partnership) groups.")

        corr = d[value_cols].corr(method="pearson")

        n = len(group_pairs)
        rows = -(-n // n_cols)
        cols = min(n_cols, n)

        fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.08, vertical_spacing=0.12,
                            subplot_titles=[f"{a.title()} ↔ {b.title()}" for (a,b) in group_pairs])

        coloraxis_name = "coloraxis"

        for i, (ga, gb) in enumerate(group_pairs):
            r, c = divmod(i, n_cols)
            r, c = r + 1, c + 1

            codes_a = [c for c in value_cols if code_to_group.get(c) == ga]
            codes_b = [c for c in value_cols if code_to_group.get(c) == gb]

            if not codes_a or not codes_b:
                fig.add_trace(
                    go.Heatmap(z=[[np.nan]], x=["—"], y=["—"], showscale=False),
                    row=r, col=c
                )
                continue

            sub_z = corr.loc[codes_a, codes_b].values
            fig.add_trace(
                go.Heatmap(
                    z=sub_z,
                    x=codes_b,
                    y=codes_a,
                    zmin=zmin, zmax=zmax,
                    coloraxis=coloraxis_name,
                    hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>corr: %{z:.2f}<extra></extra>"
                ),
                row=r, col=c
            )
            fig.update_xaxes(showgrid=False, tickangle=-90, row=r, col=c)
            fig.update_yaxes(showgrid=False, autorange="reversed", row=r, col=c)

        fig.update_layout(
            template=template,
            coloraxis=dict(colorscale="RdBu_r", cmin=zmin, cmax=zmax),
            title=dict(text=f"{title} • {kind_name} • {year}", x=0.5),
            height=fig_height * rows,
            width=fig_width * cols,
            margin=dict(l=60, r=30, t=80, b=60)
        )
        return fig

    # hierarchical clustering dendrogram of SDG variables to visualize similarity structure
    def sdg_dendrogram(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int = 2025,
        mode: str = "goal", # "goal" or "indicator"
        groups=None, # "economic" | "social" | "environmental" | list
        goal_filter=None,
        top_k: int = 20,
        method: str = "ward", # linkage method: "ward", "average", "complete", etc.
        metric: str = "euclidean", # distance metric applied to correlation-based space
        template: str = "plotly_dark",
        title: str = "SDG Hierarchical Clustering (Dendrogram)"
    ):

        d = df_sdg[df_sdg["Year"] == year].copy()
        if d.empty:
            raise ValueError(f"No rows for year {year}")

        if mode.lower() == "goal":
            value_cols = [c for c in d.columns if c.startswith("Goal_")]
            kind = "Goals"
        elif mode.lower() == "indicator":
            value_cols = [c for c in d.columns if str(c).lower().startswith("sdg")]
            kind = "Indicators"
        else:
            raise ValueError("mode must be 'goal' or 'indicator'")

        if not value_cols:
            raise ValueError("No columns found for selected mode")

        meta = df_lookup[["code","group","sdg","description"]].drop_duplicates().copy()
        meta["group"] = meta["group"].str.lower()
        lookup = meta.set_index("code")

        if groups:
            groups = [groups] if isinstance(groups, str) else groups
            groups = [g.lower() for g in groups]
            value_cols = [c for c in value_cols if c in lookup.index and lookup.loc[c,"group"] in groups]

        if mode.lower() == "indicator" and goal_filter:
            gf = [goal_filter] if isinstance(goal_filter,(int,str)) else goal_filter
            gf_nums = {str(g).replace("Goal_","") for g in gf}
            value_cols = [c for c in value_cols if str(lookup.loc[c,"sdg"]) in gf_nums]

        if len(value_cols) < 2:
            raise ValueError("Not enough variables after filtering")

        var = d[value_cols].var(numeric_only=True).sort_values(ascending=False)
        dims = var.index.tolist()[:min(top_k, len(var))]

        X = d[dims].dropna(axis=1, how="all").fillna(0).values
        labels = dims

        corr = np.corrcoef(X.T)
        dist = 1 - corr 
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)

        Z = linkage(condensed, method=method)

        fig = ff.create_dendrogram(
            X.T, orientation="left", labels=labels, linkagefun=lambda _: Z, color_threshold=0.7
        )
        fig.update_layout(
            template=template,
            title=f"{title} • {kind} • {year}",
            width=900,
            height=25*len(labels) + 200,
            margin=dict(l=200, t=80, r=40, b=40)
        )
        return fig

    # bar chart summarizing cross-group correlation strengths for selected pairs
    def grouped_corr_barchart(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int = 2025,
        mode: str = "goal", # "goal" or "indicator"
        agg: str = "mean", # "mean", "median", or "max"
        min_abs: float = 0.0,
        p_range = [-1,1],
        p_height=600,
        p_width=1000,
        template: str = "plotly_dark",
        title: str = "Cross-Group Correlation Summary"
    ):

        d = df_sdg[df_sdg["Year"] == year].copy()
        if d.empty:
            raise ValueError(f"No rows found for year {year}")

        if mode.lower() == "goal":
            value_cols = [c for c in d.columns if str(c).startswith("Goal_")]
            kind = "Goals"
        elif mode.lower() == "indicator":
            value_cols = [c for c in d.columns if str(c).lower().startswith("sdg")]
            kind = "Indicators"
        else:
            raise ValueError("mode must be 'goal' or 'indicator'")

        if not value_cols:
            raise ValueError("No value columns found")

        meta = df_lookup[["code","group"]].drop_duplicates().copy()
        meta["group"] = meta["group"].str.lower()
        lookup = dict(zip(meta["code"], meta["group"]))

        value_cols = [c for c in value_cols if lookup.get(c) in {"economic","social","environmental","partnership"}]
        if len(value_cols) < 2:
            raise ValueError("Not enough variables after filtering")

        corr = d[value_cols].corr()

        records = []
        for i, a in enumerate(value_cols):
            for j, b in enumerate(value_cols):
                if j <= i:
                    continue
                ga, gb = lookup.get(a), lookup.get(b)
                if ga and gb and ga != gb:
                    r = corr.loc[a, b]
                    if abs(r) >= min_abs:
                        records.append({"group_a":ga, "group_b":gb, "corr":r})

        if not records:
            raise ValueError("No cross-group correlations above threshold")

        df_pairs = pd.DataFrame(records)

        df_pairs["pair"] = df_pairs.apply(lambda r: " ↔ ".join(sorted([r["group_a"], r["group_b"]])), axis=1)

        if agg == "mean":
            summary = df_pairs.groupby("pair")["corr"].mean().reset_index()
        elif agg == "median":
            summary = df_pairs.groupby("pair")["corr"].median().reset_index()
        elif agg == "max":
            summary = df_pairs.groupby("pair")["corr"].max().reset_index()
        else:
            raise ValueError("agg must be mean/median/max")

        fig = px.bar(
            summary,
            x="pair",
            y="corr",
            color="pair",
            text=summary["corr"].round(2),
            template=template,
            title=f"{title} • {kind} • {year} ({agg})"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis=dict(range=p_range), height=p_height, width=p_width)

        return fig

    # chord diagram of between-group correlations (e.g., economic ↔ social) for a quick gestalt read of where links are strongest
    def grouped_corr_chord(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int = 2025,
        mode: str = "goal", # "goal" or "indicator"
        agg: str = "mean",
        min_abs: float = 0.3,
        title: str = "Cross-Group SDG Correlations (Chord Diagram)",
        plot_template: str = "plotly_dark"
    ):

        d = df_sdg[df_sdg["Year"] == year].copy()
        if d.empty:
            raise ValueError(f"No rows for year {year}")

        if mode.lower() == "goal":
            value_cols = [c for c in d.columns if str(c).startswith("Goal_")]
        else:
            value_cols = [c for c in d.columns if str(c).lower().startswith("sdg")]

        meta = df_lookup[["code","group"]].dropna().copy()
        meta["group"] = meta["group"].str.lower()
        lookup = dict(zip(meta["code"], meta["group"]))
        groups = ["economic","social","environmental","partnership"]

        value_cols = [c for c in value_cols if lookup.get(c) in groups]

        corr = d[value_cols].corr()

        records = []
        for i, a in enumerate(value_cols):
            for j, b in enumerate(value_cols):
                if j <= i: continue
                ga, gb = lookup.get(a), lookup.get(b)
                if ga and gb and ga != gb:
                    r = corr.loc[a,b]
                    if abs(r) >= min_abs:
                        records.append({"a":ga,"b":gb,"corr":r})

        df_pairs = pd.DataFrame(records)
        if df_pairs.empty:
            raise ValueError("No cross-group correlations above threshold")

        df_pairs["pair"] = df_pairs.apply(lambda r: tuple(sorted([r["a"], r["b"]])), axis=1)
        if agg=="mean":
            summary = df_pairs.groupby("pair")["corr"].mean()
        elif agg=="median":
            summary = df_pairs.groupby("pair")["corr"].median()
        else:
            summary = df_pairs.groupby("pair")["corr"].max()
        summary = summary.reset_index()

        nodes = groups
        color_map = {
            "economic":"#1f77b4",
            "social":"#ff7f0e",
            "environmental":"#2ca02c",
            "partnership":"#d62728"
        }

        node_idx = {g:i for i,g in enumerate(nodes)}
        links = {
            "source":[],
            "target":[],
            "value":[],
            "color":[]
        }

        for (ga,gb), val in zip(summary["pair"], summary["corr"]):
            links["source"].append(node_idx[ga])
            links["target"].append(node_idx[gb])
            links["value"].append(abs(val))
            # color by sign: green=positive, red=negative
            links["color"].append("rgba(0,200,100,0.6)" if val>=0 else "rgba(220,70,70,0.6)")

        fig = go.Figure(go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[g.title() for g in nodes],
                color=[color_map[g] for g in nodes]
            ),
            link=links
        ))

        fig.update_layout(title_text=title, font_size=12, template=plot_template)
        return fig

    # computes a symmetric matrix of cross-group correlations, suitable for chord/heatmap inputs.
    def get_cross_group_corr_sym(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int = 2025,
        value_prefix: str = "Goal_",
    ) -> pd.DataFrame:

        df_year = df_sdg.loc[df_sdg["Year"] == year].copy()
        goal_cols = [c for c in df_year.columns if c.startswith(value_prefix)]
        df_year = df_year[goal_cols].copy()

        if len(goal_cols) < 2:
            return pd.DataFrame(columns=["group_a","group_b","corr_mean","corr_abs_mean","n_pairs","pair"])

        corr = df_year.corr()

        code_to_group = dict(zip(df_lookup["code"], df_lookup["group"].str.lower()))

        rows = []
        for i, g1 in enumerate(goal_cols):
            grp1 = code_to_group.get(g1, "unknown")
            for g2 in goal_cols[i+1:]:
                grp2 = code_to_group.get(g2, "unknown")
                if grp1 == grp2:
                    continue
                ga, gb = sorted([grp1, grp2])
                rows.append({
                    "group_a": ga,
                    "group_b": gb,
                    "corr":   corr.loc[g1, g2]
                })

        if not rows:
            return pd.DataFrame(columns=["group_a","group_b","corr_mean","corr_abs_mean","n_pairs","pair"])

        df_pairs = pd.DataFrame(rows)

        out = (
            df_pairs
            .groupby(["group_a","group_b"], as_index=False)
            .agg(corr_mean=("corr","mean"),
                corr_abs_mean=("corr", lambda x: x.abs().mean()),
                n_pairs=("corr","size"))
        )

        out["corr_mean"] = out["corr_mean"].round(2)
        out["corr_abs_mean"] = out["corr_abs_mean"].round(2)        

        out["pair"] = out["group_a"] + " ↔ " + out["group_b"]

        out = out.sort_values("corr_abs_mean", ascending=False).reset_index(drop=True)
        return out

    # returns a tidy table of correlations for specific code pairs drawn from two groups (e.g., “economic vs environmental”), including labels for plotting.
    def get_grouppair_code_corr(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        year: int = 2025,
        mode: str = "goal", # "goal" or "sdg"
        pairs: list[tuple[str, str]] | None = None,  # if None -> all pairs incl. within-group (A,A) and cross-group (A,B)
        min_abs: float = 0.0,
        top_n: int | None = None,
        sort_by: str = "abs_corr", # "abs_corr" (default) or "corr"
        ascending: bool = False
    ) -> pd.DataFrame:

        if mode.lower() == "sdg":
            val_cols = [c for c in df_sdg.columns if c.lower().startswith("sdg")]
        else:
            val_cols = [c for c in df_sdg.columns if c.startswith("Goal_")]

        val_cols = [c for c in val_cols if c in df_sdg.columns]
        if not val_cols:
            return pd.DataFrame(columns=[
                "code_a","code_b","group_a","group_b","corr","abs_corr","sdg_a","sdg_b","desc_a","desc_b"
            ])

        df_year = df_sdg.loc[df_sdg["Year"] == year, val_cols]
        corr = df_year.corr()

        meta = (
            df_lookup[["code","group","sdg","description"]]
            .drop_duplicates()
            .set_index("code")
        )
        meta = meta.loc[meta.index.intersection(val_cols)]

        all_groups = meta["group"].dropna().str.lower().unique().tolist()
        if pairs is None:
            pairs = list(itertools.combinations_with_replacement(sorted(all_groups), 2))
        else:
            pairs = [(a.lower(), b.lower()) for a, b in pairs]

        codes_by_group = {g: meta.index[meta["group"].str.lower() == g].tolist() for g in all_groups}

        rows = []

        for gA, gB in pairs:
            codesA = codes_by_group.get(gA, [])
            codesB = codes_by_group.get(gB, [])
            if not codesA or not codesB:
                continue

            if gA == gB:
                for a, b in itertools.combinations(codesA, 2):
                    if a in corr.index and b in corr.columns:
                        r = corr.loc[a, b]
                        if pd.notna(r):
                            rows.append({
                                "code_a": a, "code_b": b,
                                "group_a": gA, "group_b": gB,
                                "corr": float(r),
                                "abs_corr": float(abs(r)),
                                "sdg_a": int(meta.loc[a, "sdg"]) if pd.notna(meta.loc[a, "sdg"]) else None,
                                "sdg_b": int(meta.loc[b, "sdg"]) if pd.notna(meta.loc[b, "sdg"]) else None,
                                "desc_a": meta.loc[a, "description"] if "description" in meta.columns else None,
                                "desc_b": meta.loc[b, "description"] if "description" in meta.columns else None,
                            })
            else:
                for a in codesA:
                    for b in codesB:
                        if a in corr.index and b in corr.columns:
                            r = corr.loc[a, b]
                            if pd.notna(r):
                                rows.append({
                                    "code_a": a, "code_b": b,
                                    "group_a": gA, "group_b": gB,
                                    "corr": float(r),
                                    "abs_corr": float(abs(r)),
                                    "sdg_a": int(meta.loc[a, "sdg"]) if pd.notna(meta.loc[a, "sdg"]) else None,
                                    "sdg_b": int(meta.loc[b, "sdg"]) if pd.notna(meta.loc[b, "sdg"]) else None,
                                    "desc_a": meta.loc[a, "description"] if "description" in meta.columns else None,
                                    "desc_b": meta.loc[b, "description"] if "description" in meta.columns else None,
                                })

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out = out.loc[out["abs_corr"] >= float(min_abs)].copy()
        out["corr"] = out["corr"].round(2)
        out["abs_corr"] = out["abs_corr"].round(2)

        if sort_by not in ["corr","abs_corr"]:
            sort_by = "abs_corr"
        out.sort_values(sort_by, ascending=ascending, inplace=True)

        if top_n is not None:
            out = out.head(int(top_n)).reset_index(drop=True)

        return out

    # calculates correlations of a chosen target metric against other SDG variables (with cleaning/ties handling), returning a tidy table you can sort by strength
    def correlate_with_target(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        target: str,
        year: int | None = None,
        top_n: int = 5,
        sign: str = "pos", # "pos" | "neg" | "abs"
        other_group: str | None = None  # social, economic etc.
    ) -> pd.DataFrame:

        df = df_sdg.copy()
        if year is not None and "Year" in df.columns:
            df = df.loc[df["Year"] == year].copy()

        target = str(target)
        is_goal = target.lower().startswith("goal_")
        is_sdg  = target.lower().startswith("sdg")
        if not (is_goal or is_sdg):
            raise ValueError("`target` must start with 'Goal_' or 'sdg'.")

        if is_goal:
            value_cols = [c for c in df.columns if c.startswith("Goal_")]
        else:
            value_cols = [c for c in df.columns if c.startswith("sdg")]

        if target not in value_cols:
            raise KeyError(
                f"{target} not found in data columns. "
                f"Available example: {value_cols[:5]} … (total {len(value_cols)})"
            )

        lookup = df_lookup[["code","group","sdg","description"]].copy()
        lookup["group"] = lookup["group"].str.lower()
        code_to_group = dict(zip(lookup["code"], lookup["group"]))
        code_to_desc  = dict(zip(lookup["code"], lookup["description"]))
        code_to_sdg   = dict(zip(lookup["code"], lookup["sdg"]))

        if other_group is not None:
            og = other_group.lower()
            keep_cols = [c for c in value_cols if code_to_group.get(c, None) == og or c == target]
        else:
            keep_cols = value_cols

        corr = df[keep_cols].corr()

        s = corr[target].drop(labels=[target], errors="ignore").dropna()

        if sign == "pos":
            s = s.sort_values(ascending=False)
        elif sign == "neg":
            s = s.sort_values(ascending=True)
        elif sign == "abs":
            s = s.abs().sort_values(ascending=False) * s.apply(lambda x: 1 if x>=0 else -1)
        else:
            raise ValueError("`sign` must be 'pos', 'neg', or 'abs'.")

        s = s.head(top_n)

        out = (
            s.rename("corr")
            .reset_index()
            .rename(columns={"index": "code_b"})
        )
        out.insert(0, "code_a", target)
        out["abs_corr"] = out["corr"].abs()
        out["group_a"]  = out["code_a"].map(code_to_group)
        out["group_b"]  = out["code_b"].map(code_to_group)
        out["sdg_a"]    = out["code_a"].map(code_to_sdg)
        out["sdg_b"]    = out["code_b"].map(code_to_sdg)
        out["desc_a"]   = out["code_a"].map(code_to_desc)
        out["desc_b"]   = out["code_b"].map(code_to_desc)

        cols = ["code_a","code_b","group_a","group_b","corr","abs_corr","sdg_a","sdg_b","desc_a","desc_b"]
        return out[cols]


    def format_corr_table(self, df: pd.DataFrame) -> None: # this is for the correlate_with_target function
        if df.empty:
            print("No results.")
            return

        target_desc = df["desc_a"].iloc[0]
        target_code = df["code_a"].iloc[0]

        print(f"\n### Correlations with {target_code}: {target_desc}\n")
        
        display_cols = ["code_b", "desc_b", "corr", "abs_corr"]
        display_df = df[display_cols].rename(
            columns={
                "code_b": "Code",
                "desc_b": "Description",
                "corr": "Correlation",
                "abs_corr": "Abs.Correlation"
            }
        )

        from IPython.display import display
        display(
            display_df.style
            .format({"Correlation": "{:.2f}", "Abs.Correlation": "{:.2f}"})
            .set_properties(subset=["Description", "Code"], **{"text-align": "left"})
        )


    def plot_sdg_pca(
        self,
        df_sdg: pd.DataFrame,
        df_lookup: pd.DataFrame,
        sdg_cols: list | None = None, # optional explicit list to plot
        id_col: str = "Country",
        year: int | None = None,
        region_filter: str | None = None,
        kind: str = "biplot", # "biplot" or "circle"
        show_var_labels: bool = True,
        show_country_labels: bool = True,
        color_by_group: bool = True,
        variable_type: str = "goal", # "goal" or "sdg"
        group_filter: str | list | None = None, 
        line_width: float = 1.5, # arrow thickness
        circle_scale: float | None = None, # correlation-circle radius; None = auto by longest arrow
        biplot_scale: float | None = None,
        biplot_xscale: float | None = None,
        biplot_yscale: float | None = None,
        p_template: str = "plotly_dark"
    ):
        lu = df_lookup.copy()
        if {"code","group"} - set(lu.columns):
            raise ValueError("df_lookup must have columns: code, group")
        lu["code"] = lu["code"].astype(str)
        lu["group_norm"] = lu["group"].astype(str).str.strip().str.lower()

        if isinstance(group_filter, str):
            gf = group_filter.strip().lower()
            group_filter_norm = None if gf == "all" else [gf]
        elif isinstance(group_filter, list):
            group_filter_norm = [g.strip().lower() for g in group_filter]
        else:
            group_filter_norm = None

        df = df_sdg.copy()
        if year is not None and "Year" in df.columns:
            df = df[df["Year"] == year].copy()
        if region_filter is not None and "Region" in df.columns:
            df = df[df["Region"] == region_filter].copy()

        if sdg_cols is None:
            if variable_type.lower() == "goal":
                cand = [c for c in df.columns if str(c).lower().startswith("goal_")]
                if group_filter_norm:
                    allowed = set(lu.loc[lu["group_norm"].isin(group_filter_norm), "code"])
                    cand = [c for c in cand if c in allowed]
                sdg_cols = cand
            else:
                exist_codes = [c for c in lu["code"].unique().tolist()
                            if c in df.columns and not str(c).lower().startswith("goal_")]
                if group_filter_norm:
                    allowed = set(lu.loc[lu["group_norm"].isin(group_filter_norm), "code"])
                    exist_codes = [c for c in exist_codes if c in allowed]
                sdg_cols = exist_codes
        else:
            if variable_type.lower() == "goal":
                sdg_cols = [c for c in sdg_cols if str(c).lower().startswith("goal_")]
            else:
                sdg_cols = [c for c in sdg_cols if not str(c).lower().startswith("goal_")]
            if group_filter_norm:
                allowed = set(lu.loc[lu["group_norm"].isin(group_filter_norm), "code"])
                sdg_cols = [c for c in sdg_cols if c in allowed]

        sdg_cols = [c for c in sdg_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not sdg_cols:
            raise ValueError("No columns left after applying variable_type/group_filter. Check df_lookup['group'] and codes.")

        X = df[sdg_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
        X = X.apply(lambda s: s.fillna(s.mean()), axis=0)

        ids = None
        if id_col in df.columns and kind.lower() == "biplot":
            ids = df.loc[X.index, id_col]

        Z = StandardScaler().fit_transform(X.values)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(Z) 
        loadings = pca.components_.T 
        var_pct = pca.explained_variance_ratio_ * 100

        code_to_group = lu.set_index("code")["group"].to_dict()
        groups = [code_to_group.get(c, "Other") for c in sdg_cols]

        if color_by_group:
            uniq = pd.unique(pd.Series(groups))
            palette = px.colors.qualitative.Set2 + px.colors.qualitative.Bold + px.colors.qualitative.Pastel
            group_color = {g: palette[i % len(palette)] for i, g in enumerate(uniq)}
            var_colors = [group_color[g] for g in groups]
        else:
            palette = px.colors.qualitative.Plotly
            var_colors = [palette[i % len(palette)] for i in range(len(sdg_cols))]
            group_color = {}

        fig = go.Figure()

        if kind.lower() == "circle":
            vec_lengths = np.sqrt(np.sum(loadings[:, :2]**2, axis=1))
            max_len = float(np.max(vec_lengths)) if vec_lengths.size else 1.0
            circle_radius = (circle_scale if circle_scale is not None else max_len * 1.1)

            theta = np.linspace(0, 2*np.pi, 400)
            fig.add_trace(go.Scatter(
                x=circle_radius * np.cos(theta),
                y=circle_radius * np.sin(theta),
                mode="lines",
                line=dict(color="rgba(120,120,120,0.6)", width=1),
                showlegend=False, hoverinfo="skip"
            ))

            fig.add_hline(y=0, line=dict(width=1, dash="dot", color="rgba(100,100,100,0.6)"))
            fig.add_vline(x=0, line=dict(width=1, dash="dot", color="rgba(100,100,100,0.6)"))

            for i, col in enumerate(sdg_cols):
                x1, y1 = float(loadings[i, 0]), float(loadings[i, 1])
                c = var_colors[i]
                fig.add_trace(go.Scatter(x=[0, x1], y=[0, y1], mode="lines",
                                        line=dict(width=line_width, color=c),
                                        showlegend=False, hoverinfo="skip"))
                fig.add_annotation(
                    x=x1, y=y1, ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.3,
                    arrowwidth=line_width,
                    arrowcolor=c
                )   
                if show_var_labels:
                    lab = col.replace("_Score","").replace("Goal_","SDG ")
                    fig.add_annotation(x=x1, y=y1, text=lab, showarrow=False,
                                    font=dict(size=12, color=c),
                                    xanchor="left", yanchor="bottom")

            fig.update_xaxes(range=[-circle_radius, circle_radius], zeroline=False,
                            scaleanchor="y", scaleratio=1)
            fig.update_yaxes(range=[-circle_radius, circle_radius], zeroline=False)
            title = "PCA correlation circle"

        
        else: # BIPLOT
            score_range = max(np.ptp(scores[:,0]) or 1.0, np.ptp(scores[:,1]) or 1.0)
            load_range  = max(np.ptp(loadings[:,0]) or 1.0, np.ptp(loadings[:,1]) or 1.0)
            arrow_scale = 0.35 * (score_range / load_range)

            country_color = "rgba(50, 100, 200, 0.85)"
            fig.add_trace(go.Scatter(
                x=scores[:,0], y=scores[:,1],
                mode="markers+text" if show_country_labels else "markers",
                text=ids if show_country_labels else None,
                textposition="top center",
                textfont=dict(color=country_color, size=12),
                marker=dict(size=6, color=country_color,
                            line=dict(width=0.3, color="rgba(50,50,50,0.5)")),
                name="Countries",
                hovertext=ids,
                hovertemplate="%{hovertext}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>"
            ))

            fig.add_hline(y=0, line=dict(width=1, dash="dot", color="rgba(100,100,100,0.6)"))
            fig.add_vline(x=0, line=dict(width=1, dash="dot", color="rgba(100,100,100,0.6)"))

            arrow_end_x = loadings[:, 0] * arrow_scale
            arrow_end_y = loadings[:, 1] * arrow_scale

            for i, col in enumerate(sdg_cols):
                x1 = float(arrow_end_x[i])
                y1 = float(arrow_end_y[i])
                c = var_colors[i]
                fig.add_trace(go.Scatter(x=[0, x1], y=[0, y1], mode="lines",
                                        line=dict(width=line_width, color=c),
                                        showlegend=False, hoverinfo="skip"))
                #fig.add_annotation(x=x1, y=y1, ax=0, ay=0, showarrow=True,
                #                arrowhead=3, arrowsize=1, arrowwidth=line_width, arrowcolor=c)
                fig.add_annotation(
                    x=x1, y=y1, ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.3,
                    arrowwidth=line_width,
                    arrowcolor=c
                )                
                if show_var_labels:
                    lab = col.replace("_Score","").replace("Goal_","SDG ")
                    fig.add_annotation(x=x1, y=y1, text=lab, showarrow=False,
                                    font=dict(size=14, color=c),
                                    xanchor="left", yanchor="bottom")


            if (biplot_xscale is None) or (biplot_yscale is None) or (biplot_scale is not None and biplot_xscale is None and biplot_yscale is None):
                max_scores = np.max(np.abs(scores[:, :2])) if scores.size else 1.0
                max_arrows = max(np.max(np.abs(arrow_end_x)), np.max(np.abs(arrow_end_y))) if arrow_end_x.size else 1.0
                auto_lim = float(max(max_scores, max_arrows) * 1.1)
            else:
                auto_lim = None 

            xlim = (-float(biplot_xscale), float(biplot_xscale)) if biplot_xscale is not None \
                else ((-float(biplot_scale), float(biplot_scale)) if biplot_scale is not None else (-auto_lim, auto_lim))
            ylim = (-float(biplot_yscale), float(biplot_yscale)) if biplot_yscale is not None \
                else ((-float(biplot_scale), float(biplot_scale)) if biplot_scale is not None else (-auto_lim, auto_lim))

            lock_aspect = (abs(xlim[0]) == abs(xlim[1]) and abs(ylim[0]) == abs(ylim[1]) and (abs(xlim[1] - ylim[1]) < 1e-9))

            if lock_aspect:
                fig.update_xaxes(range=list(xlim), zeroline=False, scaleanchor="y", scaleratio=1)
                fig.update_yaxes(range=list(ylim), zeroline=False)
            else:
                fig.update_xaxes(range=list(xlim), zeroline=False)
                fig.update_yaxes(range=list(ylim), zeroline=False)

            title = "PCA biplot"

        if color_by_group:
            for g, colr in group_color.items():
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                        marker=dict(size=10, color=colr), name=g))

        fig.update_layout(
            title=title,
            xaxis_title=f"Dim1 ({var_pct[0]:.1f}%)",
            yaxis_title=f"Dim2 ({var_pct[1]:.1f}%)",
            width=900, height=700,
            template=p_template,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        load_df = pd.DataFrame({
            "code": sdg_cols,
            "group": [lu.set_index("code")["group"].to_dict().get(c, "Other") for c in sdg_cols],
            "loading_x": loadings[:, 0],
            "loading_y": loadings[:, 1],
            "var_dim1_pct": var_pct[0],
            "var_dim2_pct": var_pct[1],
        })
        return fig, load_df


    def plot_sdg_network(
        self,
        df_sdg: "pd.DataFrame",
        df_lookup: "pd.DataFrame",
        level: str = "goal",                     # "goal" or "sdg"
        entity_type: str | None = None,          # None | "Region" | "Country"
        entities: list[str] | None = None,       # optional subset of regions/countries
        year: int | None = None,                 # single year (within-year)
        years: tuple[int, int] | None = None,    # inclusive range (across-years)
        agg: str = "mean",                       # "mean" | "median" when years is provided
        group_filter: list[str] | None = None,   # filter sdg dimensions (economic, environmental, social)
        corr_method: str = "pearson",            # "pearson" | "spearman" | "kendall"
        min_abs_corr: float = 0.1,               # keep edges with |r| >= threshold
        top_k_edges: int | None = None,          # keep strongest |r| edges (after threshold)
        min_non_null: int = 10,                  # min rows required per series
        node_size_mode: str = "centrality",      # "degree" | "centrality" | "mean_score"
        node_size_min: float = 12.0,
        node_size_max: float = 25.0,
        label_strategy: str = "auto",            # "auto" | "all" | "none"
        label_top_n: int = 25,
        label_font_size: int = 12,
        edge_width_scale: float = 3.0,           # multiplies |r| for line width
        edge_opacity: float = 0.6,
        positive_color: str = "red",
        negative_color: str = "#4A90E2",
        template: str = "plotly_dark",
        fig_scale: int = 900,                    # <- NEW: sets width & height equally
        hide_axes: bool = True,
        title: str | None = None,
    ):
        df = df_sdg.copy()

        if entity_type in ("Region", "Country") and entities:
            df = df[df[entity_type].isin(entities)]

        if (year is not None) and (years is not None):
            raise ValueError("Provide either year or years, not both.")

        if year is not None:
            df = df[df["Year"] == int(year)]
        elif years is not None:
            y0, y1 = map(int, years)
            df = df[(df["Year"] >= y0) & (df["Year"] <= y1)]
            if agg in ("mean", "median"):
                group_cols = [c for c in ["Region", "Country"] if c in df.columns]
                if group_cols:
                    agg_fn = np.nanmean if agg == "mean" else np.nanmedian
                    df = df.groupby(group_cols, dropna=False).agg(agg_fn).reset_index()

        if level == "goal":
            value_cols = [c for c in df.columns if c.startswith("Goal_")]
            if group_filter:
                allowed = set(df_lookup.loc[df_lookup["group"].isin(group_filter), "code"].astype(str))
                value_cols = [c for c in value_cols if c in allowed]
        elif level == "sdg":
            value_cols = [c for c in df.columns if c.startswith("sdg")]
            if group_filter:
                allowed = set(df_lookup.loc[df_lookup["group"].isin(group_filter), "code"].astype(str))
                value_cols = [c for c in value_cols if c in allowed]
        else:
            raise ValueError("level must be 'goal' or 'sdg'")

        numeric = df[value_cols].apply(pd.to_numeric, errors="coerce")
        good_cols = [c for c in numeric.columns if numeric[c].count() >= int(min_non_null)]
        numeric = numeric[good_cols]
        if numeric.shape[1] < 2:
            raise ValueError("Not enough valid series to build a network (after filtering).")

        corr = numeric.corr(method=corr_method)

        edges = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= float(min_abs_corr):
                    edges.append((cols[i], cols[j], float(r)))
        if not edges:
            raise ValueError("No edges passed the correlation threshold; relax min_abs_corr or filters.")
        if top_k_edges is not None and top_k_edges > 0 and len(edges) > top_k_edges:
            edges = sorted(edges, key=lambda e: abs(e[2]), reverse=True)[:top_k_edges]

        code_to_group, code_to_desc = {}, {}
        if {"code", "group", "description"}.issubset(set(df_lookup.columns)):
            code_to_group = df_lookup.set_index("code")["group"].astype(str).to_dict()
            code_to_desc  = df_lookup.set_index("code")["description"].astype(str).to_dict()

        goal_to_group = {}
        if level == "goal" and {"sdg", "group"}.issubset(set(df_lookup.columns)):
            for gcol in [c for c in good_cols if c.startswith("Goal_")]:
                try:
                    gnum = int(gcol.split("_", 1)[1])
                    grp_counts = Counter(df_lookup.loc[df_lookup["sdg"] == gnum, "group"].dropna().astype(str))
                    goal_to_group[gcol] = (grp_counts.most_common(1)[0][0] if grp_counts else "Unassigned")
                except Exception:
                    goal_to_group[gcol] = "Unassigned"

        group_palette = {
            "Environmental": "#2ca02c",
            "Economic":      "#ff7f0e",
            "Social":        "#1f77b4",
            "Partnership":   "#9467bd",
            "Unassigned":    "#7f7f7f",
        }

        nodes = []
        series_means = numeric.mean(axis=0, skipna=True).to_dict()
        for name in good_cols:
            if level == "sdg":
                grp = code_to_group.get(name, "Unassigned")
                desc = code_to_desc.get(name, "")
            else:
                grp = goal_to_group.get(name, "Unassigned")
                desc = f"{name} (goal-level)"
            nodes.append({"name": name, "group": grp, "description": desc,
                        "mean_score": float(series_means.get(name, np.nan))})

        G = nx.Graph()
        for n in nodes:
            G.add_node(n["name"], **n)
        for a, b, r in edges:
            G.add_edge(a, b, weight=abs(r), corr=r)

        # ---- circular layout ONLY ----
        pos = nx.circular_layout(G)

        # Node sizes
        def scale(vals, vmin, vmax, tmin, tmax):
            vals = np.asarray(vals, dtype=float)
            if vmax <= vmin or np.allclose(vmax, vmin):
                return np.full_like(vals, (tmin + tmax) / 2.0)
            return tmin + (vals - vmin) * (tmax - tmin) / (vmax - vmin)

        if node_size_mode == "degree":
            vals = [G.degree(n) for n in G.nodes()]
        elif node_size_mode == "centrality":
            cent = nx.degree_centrality(G)
            vals = [cent[n] for n in G.nodes()]
        elif node_size_mode == "mean_score":
            vals = [G.nodes[n].get("mean_score", np.nan) for n in G.nodes()]
            if all(pd.isna(v) for v in vals):
                vals = [G.degree(n) for n in G.nodes()]
        else:
            raise ValueError("node_size_mode must be 'degree', 'centrality', or 'mean_score'")

        sizes = scale(vals, np.nanmin(vals), np.nanmax(vals), node_size_min, node_size_max)

        node_colors = [group_palette.get(G.nodes[n].get("group", "Unassigned"), "#7f7f7f") for n in G.nodes()]

        fig = go.Figure()
        fig.update_layout(template=template, width=fig_scale, height=fig_scale)  # <- use fig_scale

        # edges
        for a, b, r in edges:
            x0, y0 = pos[a]; x1, y1 = pos[b]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=max(0.5, edge_width_scale * abs(r)),
                        color=positive_color if r >= 0 else negative_color),
                opacity=edge_opacity, hoverinfo="text",
                text=[f"{a} — {b}<br>corr={r:.2f}", f"{a} — {b}<br>corr={r:.2f}"],
                showlegend=False
            ))

        names = list(G.nodes())
        xs = [pos[n][0] for n in names]
        ys = [pos[n][1] for n in names]
        texts = []
        for n in names:
            grp = G.nodes[n].get("group", "Unassigned")
            desc = G.nodes[n].get("description", "")
            texts.append(f"{n}<br>group={grp}{'<br>'+desc if desc else ''}")

        if label_strategy == "all":
            text_vals = names
        elif label_strategy == "none":
            text_vals = ["" for _ in names]
        else:
            order = np.argsort(-np.asarray(vals))
            show_set = set([names[i] for i in order[:int(label_top_n)]])
            text_vals = [n if n in show_set else "" for n in names]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if label_strategy != "none" else "markers",
            text=text_vals, textposition="top center",
            textfont=dict(size=label_font_size),
            marker=dict(size=sizes, color=node_colors, line=dict(width=1, color="#333")),
            hoverinfo="text", hovertext=texts, showlegend=False,
        ))

        if title is None:
            scope = f"{entity_type}={','.join(entities)}" if entity_type and entities else "All entities"
            when = f"Year {year}" if year is not None else (f"Years {years[0]}–{years[1]}" if years else "All years")
            lvl = "Goals" if level == "goal" else "SDG indicators"
            title = f"Network of correlations among {lvl}  {scope}  {when}  (|r| ≥ {min_abs_corr:.2f})"

        fig.update_layout(title=None, margin=dict(t=110))
        left = 0.0
        fig.update_layout(
            title=dict(text=title, x=left, xanchor="left", pad=dict(l=0, t=6)),
            margin=dict(t=100)
        )
        fig.add_annotation(
            x=left, y=1.02, xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            text="Red = positive correlation • Blue = negative correlation • Edge width ∝ |r|",
            showarrow=False, align="left", font=dict(size=12, color="#bbbbbb")
        )

        # keep plot square & hide axes if desired
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False, constrain="domain")
        fig.update_yaxes(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
        if hide_axes:
            fig.update_xaxes(visible=False, showline=False, ticks="")
            fig.update_yaxes(visible=False, showline=False, ticks="")

        return fig





    # endregion

    # region [Composition and polish]

    def dedupe_shared_legend(self, fig):
        seen = set()
        for tr in fig.data:
            name = getattr(tr, "name", None)
            if not name:
                tr.showlegend = False
                continue
            if name in seen:
                tr.showlegend = False
            else:
                tr.showlegend = True
                seen.add(name)

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                y=1, yanchor="top",
                x=1.02, xanchor="left",
                tracegroupgap=8
            )
        )
        return fig


    def combine_figs(self, figs, rows=1, cols=None, main_title=None, height=600, width=1000, template='plotly_dark', legend_mode='shared'):    
        n = len(figs)
        if cols is None:
            cols = -(-n // rows)

        combined = make_subplots(rows=rows, cols=cols)

        for i, fig in enumerate(figs):
            r = i // cols + 1
            c = i % cols + 1
            for trace in fig.data:
                combined.add_trace(trace, row=r, col=c)
            if fig.layout.xaxis.title.text:
                combined.update_xaxes(title_text=fig.layout.xaxis.title.text, row=r, col=c)
            if fig.layout.yaxis.title.text:
                combined.update_yaxes(title_text=fig.layout.yaxis.title.text, row=r, col=c)

        combined.update_layout(
            height=height,
            width=width,
            template=template,
            title=dict(
                text=main_title if main_title else "",
                x=0.5, xanchor="center",
                yanchor="top"
            )
        )

        if legend_mode == "shared":
            self.dedupe_shared_legend(combined)
        elif legend_mode == "none":
            combined.update_layout(showlegend=False)
            
        return combined

    # endregion

    # region [Model Evaluation]

    # endregion