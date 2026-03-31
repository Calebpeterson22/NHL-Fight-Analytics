import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from plotnine import *
import io
import os
from databricks import sql  # or whatever client you use

# --- Setup Databricks credentials ---
import os

# Attempt to get secrets from environment variables first (Railway)
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID")


# Verify required secrets exist
required_vars = {
    "DATABRICKS_HOST": DATABRICKS_HOST,
    "DATABRICKS_HTTP_PATH": DATABRICKS_HTTP_PATH,
    "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
    "DATABRICKS_WAREHOUSE_ID": DATABRICKS_WAREHOUSE_ID
}

missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
# Example check
if DATABRICKS_HOST is None:
    raise ValueError("Missing DATABRICKS_HOST environment variable")

# --- Connect to Databricks ---
try:
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
except Exception as e:
    st.error("Could not connect to Databricks: " + str(e))
    st.stop()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NHL Fight Dashboard",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

        /* ── PALETTE ──────────────────────────────────────────
           Blue  primary:  #1d4ed8   dark: #1e3a8a
           Orange accent:  #ea580c   hover: #c2410c
           Bg:             #dde5f7   (deeper slate-blue)
           Surface:        #ffffff
           Text primary:   #0a0a20
           Text muted:     #4a5580
        ────────────────────────────────────────────────────── */

        html, body, [data-testid="stAppViewContainer"] {
            background: #dde5f7 !important;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent !important; display: none; }

        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 3px solid #ea580c !important;
            min-width: 320px !important;
            max-width: 420px !important;
            width: 360px !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            height: auto !important;
        }
        [data-testid="stSidebar"] [role="listbox"] {
            min-width: 360px !important;
            max-width: 500px !important;
            white-space: normal !important;
        }
        [data-testid="stSidebar"] [role="option"] {
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.5rem; }

        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button,
        section[data-testid="stSidebar"] + div button,
        .stSidebarCollapseButton {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        .stSelectbox label {
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.7rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: #4a5580 !important;
            margin-bottom: 0.2rem !important;
        }
        .stSelectbox > div > div {
            background: #edf1fc !important;
            border: 1.5px solid #7a90d8 !important;
            border-radius: 4px !important;
            color: #0a0a20 !important;
        }
        .stSelectbox > div > div:hover { border-color: #1d4ed8 !important; }

        .stDateInput label {
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.7rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: #4a5580 !important;
        }
        .stDateInput input {
            background: #edf1fc !important;
            border: 1.5px solid #7a90d8 !important;
            border-radius: 4px !important;
            color: #0a0a20 !important;
        }

        /* Primary button — orange */
        .stButton > button[kind="primary"] {
            background: #ea580c !important;
            border: 2px solid #ea580c !important;
            border-radius: 4px !important;
            font-family: 'Bebas Neue', sans-serif !important;
            letter-spacing: 0.15em !important;
            font-size: 1rem !important;
            color: #ffffff !important;
            padding: 0.6rem 1rem !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #c2410c !important;
            border-color: #c2410c !important;
        }

        .stButton > button:not([kind="primary"]) {
            background: #ffffff !important;
            border: 1.5px solid #93a8e8 !important;
            border-radius: 4px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.08em !important;
            color: #4a5580 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:not([kind="primary"]):hover {
            border-color: #ea580c !important;
            color: #ea580c !important;
        }

        .main-header {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3.2rem;
            letter-spacing: 0.08em;
            color: #1e3a8a;
            line-height: 1;
            margin-bottom: 0.25rem;
        }
        .main-subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.8rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #ea580c;
            margin-bottom: 2.5rem;
        }

        .stat-card {
            background: #ffffff;
            border: 1.5px solid #a8bae8;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.75rem;
        }
        .stat-card-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.65rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #4a5580;
            margin-bottom: 0.4rem;
        }
        .stat-card-value {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2rem;
            letter-spacing: 0.05em;
            color: #0a0a20;
            line-height: 1;
        }
        .stat-card-delta-pos {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.75rem;
            color: #148a3a;
            margin-top: 0.2rem;
        }
        .stat-card-delta-neg {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.75rem;
            color: #b01830;
            margin-top: 0.2rem;
        }
        .stat-card-delta-neu {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.75rem;
            color: #4a5580;
            margin-top: 0.2rem;
        }

        /* Section headers — orange underline accent */
        .section-header {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.4rem;
            letter-spacing: 0.12em;
            color: #1e3a8a;
            margin: 2rem 0 0.75rem 0;
            border-bottom: 3px solid #ea580c;
            padding-bottom: 0.4rem;
        }

        hr { border-color: #a8bae8 !important; margin: 1.5rem 0 !important; }

        .stAlert {
            background: #ffffff !important;
            border: 1.5px solid #a8bae8 !important;
            border-radius: 4px !important;
            font-family: 'DM Sans', sans-serif !important;
            color: #1e3a8a !important;
        }

        .stSpinner > div { border-top-color: #ea580c !important; }
        #MainMenu, footer, header { visibility: hidden; }
        .stImage img { border-radius: 6px; }

        .stTabs [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 2px solid #a8bae8 !important;
            gap: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #4a5580 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
        }
        .stTabs [aria-selected="true"] {
            color: #ea580c !important;
            border-bottom: 3px solid #ea580c !important;
        }

        [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #a8bae8 !important;
            margin-bottom: 1rem !important;
        }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
            font-size: 0.65rem !important;
            padding: 0.4rem 0.8rem !important;
            color: #4a5580 !important;
        }
        [data-testid="stSidebar"] .stTabs [aria-selected="true"] {
            color: #ea580c !important;
            border-bottom: 3px solid #ea580c !important;
        }
    </style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────
# DATABRICKS REST API
# ─────────────────────────────────────────────

def run_query(sql: str) -> pd.DataFrame:
    host         = os.environ["DATABRICKS_HOST"]
    token        = os.environ["DATABRICKS_TOKEN"]
    warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]

    response = requests.post(
        f"https://{host}/api/2.0/sql/statements",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "warehouse_id": warehouse_id,
            "statement": sql,
            "wait_timeout": "30s",
        }
    )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    data = response.json()
    statement_id = data.get("statement_id")

    while data.get("status", {}).get("state") in ("PENDING", "RUNNING"):
        poll = requests.get(
            f"https://{host}/api/2.0/sql/statements/{statement_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = poll.json()

    state = data.get("status", {}).get("state")
    if state != "SUCCEEDED":
        raise Exception(f"Query failed: {data.get('status', {}).get('error', {}).get('message', 'Unknown error')}")

    cols = [c["name"] for c in data["manifest"]["schema"]["columns"]]
    result = data.get("result", {})
    data_array = result.get("data_array", None)
    data_typed_array = result.get("data_typed_array", None)

    if data_array is not None:
        return pd.DataFrame(data_array, columns=cols)
    elif data_typed_array:
        rows = [[v.get("str", None) for v in row["values"]] for row in data_typed_array]
        return pd.DataFrame(rows, columns=cols)
    else:
        return pd.DataFrame(columns=cols)

# ─────────────────────────────────────────────
# QUERIES
# ─────────────────────────────────────────────
@st.cache_data(ttl=0)
def load_game_data(home_alias: str, away_alias: str, game_date: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT
            game_id, event_id, home_name, away_name,
            date,
            description, full_game_time, period_number_fixed,
            home_points, away_points,
            home_primary_color, away_primary_color
        FROM senior_project.default.selectgame
        WHERE home_alias = '{home_alias}'
          AND away_alias = '{away_alias}'
          AND date = '{game_date}'
    """)

@st.cache_data(ttl=300)
def load_all_fight_titles() -> pd.DataFrame:
    df = run_query("""
        SELECT DISTINCT
            home_alias,
            away_alias,
            date,
            fight_time,
            voted_winner,
            home_fighter,
            away_fighter
        FROM senior_project.default.nhl_full_fight_effects
        ORDER BY date DESC, fight_time
    """)
    if not df.empty:
        df["fight_time_num"] = pd.to_numeric(df["fight_time"], errors="coerce").fillna(0).astype(int)
        df["date_fmt"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%m-%d-%Y").fillna(df["date"].astype(str))
        df["fight_label"] = (
            df["away_fighter"].fillna(df["voted_winner"].fillna("Unknown"))
            + " vs "
            + df["home_fighter"].fillna("Unknown")
            + " | "
            + df["away_alias"] + " @ " + df["home_alias"]
            + " | "
            + df["date_fmt"]
        )
    return df

@st.cache_data(ttl=300)
def load_top5_fights() -> pd.DataFrame:
    df = run_query("""
        SELECT
            home_alias,
            away_alias,
            date,
            fight_time,
            voted_winner,
            home_fighter,
            away_fighter,
            goals_delta
        FROM senior_project.default.nhl_full_fight_effects
        ORDER BY CAST(goals_delta AS DOUBLE) DESC
        LIMIT 5
    """)
    if not df.empty:
        df["goals_delta"] = pd.to_numeric(df["goals_delta"], errors="coerce")
        df["fight_time_num"] = pd.to_numeric(df["fight_time"], errors="coerce").fillna(0).astype(int)
        df["date_fmt"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%m-%d-%Y").fillna(df["date"].astype(str))
        df["fight_label"] = (
            df["away_fighter"].fillna(df["voted_winner"].fillna("Unknown"))
            + " vs "
            + df["home_fighter"].fillna("Unknown")
            + " | "
            + df["away_alias"] + " @ " + df["home_alias"]
            + " | "
            + df["date_fmt"]
        )
    return df

@st.cache_data(ttl=0)
def load_fight_effects_by_game(home_alias: str, away_alias: str, game_date: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT
            game_id, date, home_alias, away_alias, fight_time,
            attrib_team_full_name, is_home, fight_winning_team, voted_winner,
            home_fighter, away_fighter,
            corsi_before, corsi_after, corsi_before_per60, corsi_after_per60, corsi_delta,
            xg_before, xg_after, xg_before_per60, xg_after_per60, xg_delta,
            goals_before, goals_after, goals_before_per60, goals_after_per60, goals_delta,
            before_seconds, after_seconds, did_you_win_the_fight
        FROM senior_project.default.nhl_full_fight_effects
        WHERE date = '{game_date}'
          AND home_alias = '{home_alias}'
          AND away_alias = '{away_alias}'
        ORDER BY fight_time
    """)

@st.cache_data(ttl=0)
def load_fight_coords(home_alias: str, away_alias: str, game_date: str) -> pd.DataFrame:
    try:
        return run_query(f"""
            SELECT DISTINCT fight_time, coord_x, coord_y
            FROM senior_project.default.nhl_full_fight_effects
            WHERE date = '{game_date}'
              AND home_alias = '{home_alias}'
              AND away_alias = '{away_alias}'
        """)
    except Exception:
        return pd.DataFrame(columns=["fight_time", "coord_x", "coord_y"])

@st.cache_data(ttl=300)
# Remove @st.cache_data here — caching empty results hides failures
def load_fight_video(home_alias: str, away_alias: str, game_date: str, home_fighter: str, away_fighter: str) -> str:
    try:
        home_last = home_fighter.split()[-1] if home_fighter else ""
        away_last = away_fighter.split()[-1] if away_fighter else ""

        if not home_last and not away_last:
            return ""

        sql = """
            SELECT video_url
            FROM senior_project.default.hockeyfights_website_data
            WHERE date = '{date}'
              AND (
                home_player LIKE '%{hl}'
                OR away_player LIKE '%{al}'
                OR home_player LIKE '%{al}'
                OR away_player LIKE '%{hl}'
              )
            LIMIT 1
        """.format(date=game_date, hl=home_last, al=away_last)

        df = run_query(sql)
        if not df.empty and df["video_url"].iloc[0]:
            return str(df["video_url"].iloc[0])

    except Exception as e:
        st.write(f"[DEBUG] load_fight_video error: {e}")

    return ""

@st.cache_data(ttl=300)
def load_player_profile(full_name: str) -> pd.DataFrame:
    # Guard against None / empty / "nan" values arriving from missing DB data
    if not full_name or str(full_name).strip() in ("", "nan", "None", "NaN"):
        return pd.DataFrame()
    safe_name = full_name.replace("'", "''")
    try:
        return run_query(f"""
            SELECT full_name, primary_position, jersey_number,
                   height, weight, handedness, experience, salary,
                   headshot_url, home_name
            FROM senior_project.default.full_player_data
            WHERE full_name = '{safe_name}'
            LIMIT 1
        """)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_position_influence() -> pd.DataFrame:
    try:
        return run_query("""
            SELECT
                p.primary_position,
                AVG(CAST(e.goals_delta AS DOUBLE)) as avg_goals_delta,
                COUNT(*) as fight_count
            FROM senior_project.default.nhl_full_fight_effects e
            JOIN senior_project.default.full_player_data p
                ON p.full_name = e.voted_winner
            WHERE e.goals_delta IS NOT NULL
              AND p.primary_position IS NOT NULL
            GROUP BY p.primary_position
            ORDER BY avg_goals_delta DESC
        """)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_fight_stats() -> pd.DataFrame:
    try:
        return run_query("""
            SELECT
                home_player, away_player, home_team, away_team,
                voted_winner, winner_pct, voted_rating, vote_count,
                date, video_url
            FROM senior_project.default.hockeyfights_website_data
            WHERE voted_winner IS NOT NULL
        """)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_fight_timing() -> pd.DataFrame:
    try:
        return run_query("""
            SELECT DISTINCT game_id, fight_time
            FROM senior_project.default.nhl_full_fight_effects
            WHERE fight_time IS NOT NULL
        """)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_jersey_stats() -> pd.DataFrame:
    try:
        return run_query("""
            SELECT p.full_name, p.jersey_number, p.primary_position, p.home_name as team
            FROM senior_project.default.full_player_data p
            WHERE p.jersey_number IS NOT NULL
        """)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _clean_fighter_name(raw) -> "str | None":
    """Return a clean fighter name string, or None if the value is empty/null."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "nan", "None", "NaN"):
        return None
    return s

def time_to_seconds(t: str) -> float:
    try:
        parts = str(t).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0

def seconds_to_mmss(s) -> str:
    s = int(float(s))
    return f"{s // 60:02d}:{s % 60:02d}"

def delta_color(val: float) -> str:
    if val > 0.05:
        return "stat-card-delta-pos"
    elif val < -0.05:
        return "stat-card-delta-neg"
    return "stat-card-delta-neu"

def delta_arrow(val: float) -> str:
    if val > 0.05:
        return f"▲ +{val:.2f}"
    elif val < -0.05:
        return f"▼ {val:.2f}"
    return f"→ {val:.2f}"

# ─────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df["game_seconds"] = df["full_game_time"].apply(time_to_seconds)

    df["is_fight"] = df["description"].str.contains(r'(?i)\bfighting\b', regex=True, na=False).astype(int)
    df = df.drop_duplicates(subset=["game_id", "event_id", "game_seconds", "is_fight"])
    df = df.sort_values("game_seconds")
    df["game_fight_number"] = df.groupby("game_id")["is_fight"].cumsum() / 2

    id_cols = [
        "game_id", "period_number_fixed", "full_game_time", "game_seconds", "description",
        "date", "home_name", "away_name", "is_fight", "game_fight_number",
        "home_primary_color", "away_primary_color",
    ]
    df = df.melt(
        id_vars=id_cols,
        value_vars=["home_points", "away_points"],
        var_name="team_type",
        value_name="points",
    )

    df["home_fighter"] = df["description"].str.extract(r"to\s+(.+?)\s+5", expand=False)
    df["away_fighter"] = df["description"].str.extract(r"\bby\s+([^)]+)", expand=False)
    df = df.sort_values("game_seconds")

    return df

def prepare_fight_effects(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    numeric_cols = [
        "fight_time", "corsi_before", "corsi_after", "corsi_before_per60",
        "corsi_after_per60", "corsi_delta", "xg_before", "xg_after",
        "xg_before_per60", "xg_after_per60", "xg_delta", "goals_before",
        "goals_after", "goals_before_per60", "goals_after_per60", "goals_delta",
        "before_seconds", "after_seconds",
    ]
    for coord_col in ["coord_x", "coord_y"]:
        if coord_col not in df.columns:
            df[coord_col] = None
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def build_goal_chart(df: pd.DataFrame, selected_fight_time: float = None) -> ggplot:
    home       = df["home_name"].iloc[0]
    away       = df["away_name"].iloc[0]
    home_color = df["home_primary_color"].iloc[0]
    away_color = df["away_primary_color"].iloc[0]

    all_fight_times = df[df["is_fight"] == 1].drop_duplicates("game_seconds").copy()
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    tick_seconds = [20*60, 40*60, 60*60]
    tick_labels  = ["20:00", "40:00", "60:00"]

    if selected_fight_time is not None:
        selected_fights = all_fight_times[all_fight_times["game_seconds"] == selected_fight_time].copy()
        other_fights    = all_fight_times[all_fight_times["game_seconds"] != selected_fight_time].copy()
    else:
        selected_fights = all_fight_times.copy()
        other_fights    = pd.DataFrame(columns=all_fight_times.columns)

    selected_fights["fight_label"] = "Selected Fight"
    other_fights["fight_label"]    = "Other Fight"

    p = (
        ggplot(df, aes(x="game_seconds", y="points", color="team_type", group="team_type"))
        + geom_step(size=1.5)
        + scale_color_manual(
            values={"away_points": away_color, "home_points": home_color},
            labels={"away_points": away, "home_points": home},
        )
        + scale_x_continuous(breaks=tick_seconds, labels=tick_labels)
        + labs(
            title="Game Progression",
            subtitle="",
            x="Game Time", y="Goals", color="",
        )
        + theme_minimal()
        + theme(
            legend_position="bottom",
            axis_text_x=element_text(angle=45, hjust=1),
            axis_line_y=element_blank(),
            panel_grid_minor=element_blank(),
            plot_background=element_rect(fill="#f0f4ff", color="#f0f4ff"),
            panel_background=element_rect(fill="#ffffff"),
            text=element_text(color="#0d0d26"),
            plot_title=element_text(face="bold", size=22),
            plot_subtitle=element_text(size=13, color="#6b7db8"),
            axis_text=element_text(color="#5a6899", size=12),
            axis_title=element_text(color="#5a6899", size=13),
            legend_background=element_rect(fill="#f7f8fc"),
            legend_text=element_text(color="#1e2a60", size=16),
            panel_grid_major=element_line(color="#e8eaf2"),
        )
    )

    if not selected_fights.empty:
        p = p + geom_vline(
            data=selected_fights,
            mapping=aes(xintercept="game_seconds", linetype="fight_label"),
            color="#0d0d26", alpha=0.9, size=1.1,
        )

    if not other_fights.empty:
        p = p + geom_vline(
            data=other_fights,
            mapping=aes(xintercept="game_seconds", linetype="fight_label"),
            color="#7a88bb", alpha=0.85, size=1.0,
        )

    p = p + scale_linetype_manual(
        values={"Selected Fight": "dashed", "Other Fight": "dotted"},
        name=""
    ) + guides(linetype=guide_legend(title=""), color=guide_legend(title=""))

    return p

def build_before_after_chart(effects_df: pd.DataFrame, metric: str, label: str) -> plt.Figure:
    before_col = f"{metric}_before_per60"
    after_col  = f"{metric}_after_per60"

    plot_df = effects_df[["attrib_team_full_name", "fight_time", before_col, after_col]].dropna().copy()
    plot_df["fight_label"] = plot_df["fight_time"].apply(lambda x: f"Fight @ {seconds_to_mmss(x)}")

    fights = plot_df["fight_label"].unique()
    teams  = plot_df["attrib_team_full_name"].unique()
    n_fights = len(fights)

    fig, axes = plt.subplots(1, n_fights, figsize=(5 * n_fights, 5), squeeze=False)
    fig.patch.set_facecolor("#f0f4ff")

    colors = ["#3a3aff", "#ff4466", "#44dd88", "#ffaa33"]

    for i, fight in enumerate(fights):
        ax = axes[0][i]
        ax.set_facecolor("#ffffff")

        fight_data = plot_df[plot_df["fight_label"] == fight]
        x = np.arange(len(fight_data))
        width = 0.35

        bars_before = ax.bar(x - width/2, fight_data[before_col], width,
                             label="Before", color="#b0c4e8", alpha=0.95, zorder=3)
        bars_after  = ax.bar(x + width/2, fight_data[after_col],  width,
                             label="After",  color="#0d0d26", alpha=0.90, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(fight_data["attrib_team_full_name"].str.split().str[-1],
                           color="#888899", fontsize=8)
        ax.set_title(fight, color="#1e2a60", fontsize=9, fontfamily="sans-serif", pad=8)
        ax.set_ylabel(f"{label} per 60", color="#5a6899", fontsize=8)
        ax.tick_params(colors="#5a6899")
        ax.spines[:].set_color("#e8eaf2")
        ax.yaxis.label.set_color("#5a6899")
        ax.grid(axis="y", color="#e8eaf2", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        for bar in list(bars_before) + list(bars_after):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    color="#1e2a60", fontsize=7)

        if i == 0:
            ax.legend(facecolor="#ffffff", edgecolor="#e8eaf2",
                      labelcolor="#1e2a60", fontsize=8)

    fig.suptitle(f"{label} Per 60 — Before vs After Each Fight",
                 color="#0d0d26", fontsize=12, fontfamily="sans-serif", y=1.02)
    plt.tight_layout()
    return fig

def build_delta_bar(effects_df: pd.DataFrame) -> plt.Figure:
    metrics = [
        ("xg_delta",    "xG Delta"),
        ("corsi_delta", "Corsi Delta"),
        ("goals_delta", "Goals Delta"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(effects_df) * 0.5 + 2)))
    fig.patch.set_facecolor("#f0f4ff")

    for ax, (col, title) in zip(axes, metrics):
        ax.set_facecolor("#ffffff")
        plot_df = effects_df[["attrib_team_full_name", "fight_time", col]].dropna().copy()
        plot_df["label"] = (
            plot_df["attrib_team_full_name"].str.split().str[-1]
            + " @ "
            + plot_df["fight_time"].apply(seconds_to_mmss)
        )
        plot_df = plot_df.sort_values(col)

        bar_colors = ["#1a8a4a" if v >= 0 else "#c0233a" for v in plot_df[col]]
        ax.barh(plot_df["label"], plot_df[col], color=bar_colors, alpha=0.85, zorder=3)
        ax.axvline(0, color="#6b7db8", linewidth=0.8, linestyle="--")
        ax.set_title(title, color="#1e2a60", fontsize=10, pad=8)
        ax.tick_params(colors="#5a6899", labelsize=7)
        ax.spines[:].set_color("#e8eaf2")
        ax.grid(axis="x", color="#e8eaf2", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Per-60 Rate Change After Each Fight",
                 color="#0d0d26", fontsize=13, fontfamily="sans-serif")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# RENDER STAT CARDS FOR A FIGHT ROW
# ─────────────────────────────────────────────
def render_fight_cards(row: pd.Series):
    team   = row["attrib_team_full_name"]
    winner = row["fight_winning_team"]
    voted  = row["voted_winner"] if pd.notna(row.get("voted_winner")) else "N/A"
    won    = str(row.get("did_you_win_the_fight", "")).lower() == "true"

    win_badge = (
        '<span style="background:#44dd8822;border:1px solid #44dd88;color:#44dd88;'
        'border-radius:3px;padding:1px 8px;font-size:0.65rem;letter-spacing:0.1em;">WON</span>'
        if won else
        '<span style="background:#ff446622;border:1px solid #ff4466;color:#ff4466;'
        'border-radius:3px;padding:1px 8px;font-size:0.65rem;letter-spacing:0.1em;">LOST</span>'
    )

    st.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.8rem;color:#1e3a8a;'
        f'margin-bottom:0.5rem;">'
        f'<b style="color:#0a0a20">{team}</b> &nbsp;{win_badge}&nbsp; '
        f'<span style="color:#444466;font-size:0.7rem;">Voted winner: {voted}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    for col_widget, metric, label in [
        (c1, "xg",    "xG"),
        (c2, "corsi", "Corsi"),
        (c3, "goals", "Goals"),
    ]:
        before = row.get(f"{metric}_before_per60", 0) or 0
        after  = row.get(f"{metric}_after_per60",  0) or 0
        delta  = row.get(f"{metric}_delta", 0) or 0
        css    = delta_color(delta)
        arrow  = delta_arrow(delta)
        with col_widget:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-card-title">{label} / 60</div>'
                f'<div class="stat-card-value">{after:.2f}</div>'
                f'<div class="{css}">{arrow} vs {before:.2f} before</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# PLAYER CARD
# ─────────────────────────────────────────────
def render_player_card(player_df: pd.DataFrame, fighter_name: str, is_winner: bool, align: str = "left"):
    lbl = '<span style="color:#1d4ed8;text-transform:uppercase;font-size:0.78rem;letter-spacing:0.1em;font-weight:500;">'

    if player_df.empty:
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #b8c8f8;border-radius:14px;"
            "box-shadow:0 2px 16px rgba(37,99,235,0.10);padding:2rem 1.5rem;text-align:center;height:100%;'>"
            "<div style='width:150px;height:150px;border-radius:50%;background:#f4f7ff;"
            "border:3px solid #ea580c;margin:0 auto 16px;display:flex;align-items:center;"
            "justify-content:center;font-size:2.5rem;'>🏒</div>"
            "<div style='font-size:1.5rem;font-weight:600;color:#0a0a20;margin-bottom:6px;'>" + str(fighter_name) + "</div>"
            "<div style='font-size:0.9rem;color:#4a5580;margin-bottom:20px;'>Profile not available</div>"
            "<div style='background:#f4f7ff;border-radius:10px;padding:1rem 1.2rem;text-align:center;'>"
            "<div style='font-size:0.85rem;color:#4a5580;letter-spacing:0.08em;'>Stats unavailable</div>"
            "</div></div>",
            unsafe_allow_html=True
        )
        return

    p = player_df.iloc[0]
    name = str(p.get("full_name", fighter_name) or fighter_name)
    pos  = str(p.get("primary_position", "") or "—")
    num  = str(p.get("jersey_number", "") or "—")
    ht   = p.get("height", "")
    wt   = str(p.get("weight", "") or "—")
    hand = str(p.get("handedness", "") or "—")
    exp  = str(p.get("experience", "") or "—")
    sal  = p.get("salary", None)
    team = str(p.get("home_name", "") or "—")
    img  = str(p.get("headshot_url", "") or "")

    try:
        sal_str = "${:,}".format(int(float(sal)))
    except Exception:
        sal_str = "—"

    try:
        inches = int(float(ht))
        ht_str = str(inches // 12) + "'" + str(inches % 12) + '"'
    except Exception:
        ht_str = str(ht) if ht else "—"

    winner_badge = (
        "<span style='background:#2563eb;color:#fff;font-size:0.6rem;letter-spacing:0.1em;"
        "padding:2px 8px;border-radius:3px;margin-left:6px;text-transform:uppercase;'>Winner</span>"
        if is_winner else ""
    )

    img_tag = (
        "<img src='" + img + "' style='width:150px;height:150px;border-radius:50%;"
        "object-fit:cover;border:3px solid #ea580c;margin-bottom:16px;'>"
        if img else ""
    )

    html = (
        "<div style='background:#ffffff;border:1px solid #b8c8f8;border-radius:14px;"
        "box-shadow:0 2px 16px rgba(37,99,235,0.10);padding:2rem 1.5rem;text-align:center;height:100%;'>"
        + img_tag
        + "<div style='font-size:1.5rem;font-weight:600;color:#0a0a20;margin-bottom:6px;'>" + name + winner_badge + "</div>"
        + "<div style='font-size:0.9rem;color:#4a5580;margin-bottom:20px;letter-spacing:0.04em;'>" + team + "</div>"
        + "<div style='background:#f4f7ff;border-radius:10px;padding:1rem 1.2rem;text-align:left;'>"
        + "<div style='font-size:0.95rem;color:#1e3a8a;line-height:1;'>"
        + "<div style='margin-bottom:11px;display:flex;justify-content:space-between;'>" + lbl + "Position</span><span style='color:#0a0a20;font-weight:500;'>" + pos + "</span></div>"
        + "<div style='margin-bottom:11px;display:flex;justify-content:space-between;'>" + lbl + "Jersey</span><span style='color:#0a0a20;font-weight:500;'>#" + num + "</span></div>"
        + "<div style='margin-bottom:11px;display:flex;justify-content:space-between;'>" + lbl + "Height</span><span style='color:#0a0a20;font-weight:500;'>" + ht_str + "</span></div>"
        + "<div style='margin-bottom:11px;display:flex;justify-content:space-between;'>" + lbl + "Weight</span><span style='color:#0a0a20;font-weight:500;'>" + wt + " lbs</span></div>"
        + "<div style='margin-bottom:11px;display:flex;justify-content:space-between;'>" + lbl + "Shoots</span><span style='color:#0a0a20;font-weight:500;'>" + hand + "</span></div>"
        + "<div style='margin-bottom:11px;display:flex;justify-content:space-between;'>" + lbl + "Experience</span><span style='color:#0a0a20;font-weight:500;'>" + exp + " yrs</span></div>"
        + "<div style='display:flex;justify-content:space-between;'>" + lbl + "Salary</span><span style='color:#0a0a20;font-weight:500;'>" + sal_str + "</span></div>"
        + "</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RINK MAP
# ─────────────────────────────────────────────
RINK_URL = "https://raw.githubusercontent.com/Calebpeterson22/Senior_Project_NHL_Injury_risk_predicitons/main/hockey_rink_2.jpg"

def build_rink_map(coords_df: pd.DataFrame, selected_fight_time: float = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f0f4ff")
    ax.set_facecolor("#f7f8fc")

    try:
        resp = requests.get(RINK_URL, timeout=10)
        resp.raise_for_status()
        rink_img = plt.imread(io.BytesIO(resp.content), format="jpg")
        ax.imshow(rink_img, extent=[0, 2400, 0, 1020], aspect="auto", alpha=0.85)
    except Exception:
        ax.set_xlim(0, 2400)
        ax.set_ylim(0, 1020)
        ax.set_facecolor("#dff0f7")
        ax.axvline(1200, color="#e63560", linewidth=2, alpha=0.7)
        ax.axvline(600,  color="#3a7bd5", linewidth=2, alpha=0.7)
        ax.axvline(1800, color="#3a7bd5", linewidth=2, alpha=0.7)
        ax.text(1200, 510, "Rink image unavailable", ha="center", va="center",
                color="#5a6899", fontsize=11)

    plot_df = coords_df.copy()
    plot_df["coord_x"] = pd.to_numeric(plot_df["coord_x"], errors="coerce")
    plot_df["coord_y"] = pd.to_numeric(plot_df["coord_y"], errors="coerce")
    plot_df["fight_time"] = pd.to_numeric(plot_df["fight_time"], errors="coerce")
    plot_df = plot_df.dropna(subset=["coord_x", "coord_y"]).drop_duplicates(subset=["fight_time"])

    if not plot_df.empty:
        selected = plot_df[plot_df["fight_time"] == selected_fight_time] if selected_fight_time is not None else pd.DataFrame()
        other    = plot_df[plot_df["fight_time"] != selected_fight_time] if selected_fight_time is not None else plot_df

        if not other.empty:
            ax.scatter(
                other["coord_x"], other["coord_y"],
                color="#0d0d26", s=2000, marker="*", zorder=5,
                edgecolors="#ffffff", linewidths=0.8, alpha=0.75,
            )

        if not selected.empty:
            ax.scatter(
                selected["coord_x"], selected["coord_y"],
                color="#f5c518", s=2000, marker="*", zorder=6,
                edgecolors="#0d0d26", linewidths=1.2, alpha=1.0,
            )

        for _, row in plot_df.iterrows():
            ax.annotate(
                seconds_to_mmss(row["fight_time"]),
                (row["coord_x"], row["coord_y"]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, color="#0d0d26",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#e8eaf2", alpha=0.9)
            )
    else:
        ax.text(1200, 510, "No coordinate data for this fight", ha="center", va="center",
                color="#5a6899", fontsize=11)

    ax.set_xlabel("Rink X", color="#5a6899", fontsize=10)
    ax.set_ylabel("Rink Y", color="#5a6899", fontsize=10)
    ax.set_title("Fight Location on Ice", color="#0d0d26", fontsize=18, fontweight="normal", pad=14)
    ax.tick_params(colors="#5a6899")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8eaf2")

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">NHL Fight Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Goal progression · Fight analytics</p>', unsafe_allow_html=True)

main_tab1, main_tab2, main_tab3 = st.tabs(["Fight Explorer", "Fight Statistics", "Most Dominant Fights"])

# ── Load fight data ──
try:
    fights_index = load_all_fight_titles()
except Exception as e:
    fights_index = pd.DataFrame()

try:
    top5_index = load_top5_fights()
except Exception as e:
    top5_index = pd.DataFrame()

# ── Build combined index ──
def _has_fight_label(df):
    return not df.empty and "fight_label" in df.columns

if _has_fight_label(fights_index) and _has_fight_label(top5_index):
    combined_index = pd.concat([fights_index, top5_index], ignore_index=True).drop_duplicates(subset=["fight_label"])
elif _has_fight_label(fights_index):
    combined_index = fights_index
elif _has_fight_label(top5_index):
    combined_index = top5_index
else:
    combined_index = pd.DataFrame()

with main_tab1:
    all_fight_labels = fights_index["fight_label"].tolist() if _has_fight_label(fights_index) else []

    # Top 5 section
    st.markdown(
        "<div style='font-family:Bebas Neue,sans-serif;font-size:1.1rem;letter-spacing:0.14em;"
        "color:#1e3a8a;border-bottom:3px solid #ea580c;padding-bottom:0.3rem;margin-bottom:0.8rem;'>Influential Fights Top 5 </div>",
        unsafe_allow_html=True
    )
    if not top5_index.empty:
        top5_cols = st.columns(5)
        for i, (_, row) in enumerate(top5_index.iterrows()):
            delta_str = f"+{row['goals_delta']:.2f}" if row['goals_delta'] >= 0 else f"{row['goals_delta']:.2f}"
            home_f = str(row.get("home_fighter", "") or "").strip()
            away_f = str(row.get("away_fighter", "") or "").strip()
            fighters_line = f"{away_f} vs {home_f}" if away_f and home_f else row.get("voted_winner", "Unknown")
            matchup_line  = f"{row['away_alias']} @ {row['home_alias']}"
            date_obj = pd.to_datetime(row["date"], errors="coerce")
            date_line = date_obj.strftime("%m-%d-%Y") if not pd.isnull(date_obj) else str(row["date"])
            btn_label = fighters_line + "\n" + matchup_line + "\n" + date_line
            with top5_cols[i]:
                if st.button(btn_label, use_container_width=True, key=f"top5_{row['fight_label']}"):
                    st.session_state["selected_fight_label"] = row["fight_label"]
                    st.session_state["team_filter"] = "All Teams"
                    st.session_state["player_filter"] = "All Players"

    st.markdown("<hr style='border-color:#b8c8f8;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Browse controls ──
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 2, 2, 4])

    with filter_col1:
        if st.button("🎲 Random", type="primary", use_container_width=True):
            if all_fight_labels:
                st.session_state["selected_fight_label"] = all_fight_labels[random.randint(0, len(all_fight_labels) - 1)]
                st.session_state["team_filter"] = "All Teams"
                st.session_state["player_filter"] = "All Players"

    with filter_col2:
        all_aliases = sorted(pd.concat([
            fights_index["home_alias"], fights_index["away_alias"]
        ]).dropna().unique().tolist()) if not fights_index.empty else []
        team_options = ["All Teams"] + all_aliases
        team_filter_default = st.session_state.get("team_filter", "All Teams")
        team_filter_idx = team_options.index(team_filter_default) if team_filter_default in team_options else 0
        selected_team = st.selectbox("Filter by Team", team_options, index=team_filter_idx, label_visibility="collapsed")
        if selected_team != st.session_state.get("team_filter"):
            st.session_state["team_filter"] = selected_team
            st.session_state["player_filter"] = "All Players"
            st.session_state["player_filter_idx"] = 0
            st.rerun()

    with filter_col3:
        if selected_team == "All Teams":
            team_fights = fights_index
            home_fighters = team_fights["home_fighter"].dropna().tolist() if "home_fighter" in team_fights.columns else []
            away_fighters = team_fights["away_fighter"].dropna().tolist() if "away_fighter" in team_fights.columns else []
            all_players = sorted(set(home_fighters + away_fighters))
        else:
            home_team_fighters = fights_index[fights_index["home_alias"] == selected_team]["home_fighter"].dropna().tolist() if "home_fighter" in fights_index.columns else []
            away_team_fighters = fights_index[fights_index["away_alias"] == selected_team]["away_fighter"].dropna().tolist() if "away_fighter" in fights_index.columns else []
            all_players = sorted(set(home_team_fighters + away_team_fighters))

        player_options = ["All Players"] + [p for p in all_players if p.strip()]
        player_filter_idx = st.session_state.get("player_filter_idx", 0)
        selected_player = st.selectbox("Filter by Player", player_options, index=player_filter_idx, label_visibility="collapsed")
        if selected_player != st.session_state.get("player_filter"):
            st.session_state["player_filter"] = selected_player
            st.session_state["player_filter_idx"] = player_options.index(selected_player)

    with filter_col4:
        # Apply both team and player filters
        if selected_team == "All Teams":
            filtered_fights = fights_index
        else:
            filtered_fights = fights_index[
                (fights_index["home_alias"] == selected_team) |
                (fights_index["away_alias"] == selected_team)
            ]

        if selected_player != "All Players":
            if "home_fighter" in filtered_fights.columns and "away_fighter" in filtered_fights.columns:
                filtered_fights = filtered_fights[
                    (filtered_fights["home_fighter"] == selected_player) |
                    (filtered_fights["away_fighter"] == selected_player)
                ]

        fight_labels = filtered_fights["fight_label"].tolist() if "fight_label" in filtered_fights.columns else []
        display_options = ["— Select a fight —"] + fight_labels
        current_selected = st.session_state.get("selected_fight_label", None)
        select_index = fight_labels.index(current_selected) + 1 if current_selected and current_selected in fight_labels else 0
        chosen = st.selectbox("Fight", display_options, index=select_index, label_visibility="collapsed")
        if chosen != "— Select a fight —":
            st.session_state["selected_fight_label"] = chosen

    selected_fight_label = st.session_state.get("selected_fight_label", None)
    if selected_fight_label == "— Select a fight —":
        selected_fight_label = None

    st.markdown("<hr style='border-color:#b8c8f8;margin:1rem 0;'>", unsafe_allow_html=True)

with main_tab1:
        if selected_fight_label is not None and not combined_index.empty and selected_fight_label in combined_index["fight_label"].values:

            fight_meta = combined_index[combined_index["fight_label"] == selected_fight_label].iloc[0]
            home_alias    = fight_meta["home_alias"]
            away_alias    = fight_meta["away_alias"]
            game_date_str = str(fight_meta["date"])

            with st.spinner("Loading fight effects..."):
                try:
                    effects_raw = load_fight_effects_by_game(home_alias, away_alias, game_date_str)
                    effects_df  = prepare_fight_effects(effects_raw)
                except Exception as e:
                    st.error(f"Failed to load fight effects: {e}")
                    st.stop()

            if effects_df.empty:
                st.warning("No fight effects data found for this fight.")
                st.stop()

            with st.spinner(f"Loading game data for {away_alias} @ {home_alias} on {game_date_str}..."):
                try:
                    raw_df = load_game_data(home_alias, away_alias, game_date_str)
                except Exception as e:
                    st.error(f"Failed to load game data: {e}")
                    st.stop()

            if raw_df.empty:
                st.warning(f"No game data found for {away_alias} @ {home_alias} on {game_date_str}.")
                st.stop()

            filtered_df = transform(raw_df)

            fight_winner = str(effects_df["fight_winning_team"].iloc[0]).lower() if not effects_df.empty else ""
            home_name = filtered_df["home_name"].iloc[0] if not filtered_df.empty else home_alias
            away_name = filtered_df["away_name"].iloc[0] if not filtered_df.empty else away_alias
            voted = fight_meta.get("voted_winner", "Unknown") or "Unknown"

            winner_rows = effects_df[effects_df["voted_winner"] == voted] if not effects_df.empty else pd.DataFrame()
            winning_team_full = winner_rows["attrib_team_full_name"].iloc[0] if not winner_rows.empty else None

            if winning_team_full and winning_team_full == home_name:
                title_html = f'{away_name} @ <b style="color:#0a0a20">{home_name}</b>'
            elif winning_team_full and winning_team_full == away_name:
                title_html = f'<b style="color:#0a0a20">{away_name}</b> @ {home_name}'
            else:
                title_html = f'{away_name} @ {home_name}'

            fight_time_str = seconds_to_mmss(float(fight_meta["fight_time"])) if "fight_time" in fight_meta else ""

            st.markdown(
                f'<div style="margin-bottom:1.2rem;text-align:center;">'
                f'<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.8rem;letter-spacing:0.06em;color:#0a0a20;line-height:1.1;">'
                f'{title_html}'
                f'</div>'
                f'<div style="font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:#4a5580;margin-top:0.3rem;">'
                f'{game_date_str} &nbsp;·&nbsp; Fight at {fight_time_str} &nbsp;·&nbsp; Winner: {voted}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            selected_ft = float(fight_meta["fight_time"]) if "fight_time" in fight_meta else None

            selected_fight_row = effects_df[effects_df["fight_time"] == selected_ft] if selected_ft is not None else effects_df
            if selected_fight_row.empty:
                selected_fight_row = effects_df
            home_fighter_name = _clean_fighter_name(
                selected_fight_row["home_fighter"].iloc[0]
                if "home_fighter" in selected_fight_row.columns and not selected_fight_row.empty
                else None
            )
            away_fighter_name = _clean_fighter_name(
                selected_fight_row["away_fighter"].iloc[0]
                if "away_fighter" in selected_fight_row.columns and not selected_fight_row.empty
                else None
            )

            home_player_df = pd.DataFrame()
            away_player_df = pd.DataFrame()
            if home_fighter_name:
                try:
                    home_player_df = load_player_profile(home_fighter_name)
                except Exception:
                    home_player_df = pd.DataFrame()
            if away_fighter_name:
                try:
                    away_player_df = load_player_profile(away_fighter_name)
                except Exception:
                    away_player_df = pd.DataFrame()

            col_away, col_chart, col_home = st.columns([1.4, 4, 1.4])
            with col_away:
                render_player_card(away_player_df, away_fighter_name or "Away Fighter",
                                   is_winner=(bool(away_fighter_name) and voted == away_fighter_name), align="left")
            with col_chart:
                chart = build_goal_chart(filtered_df, selected_fight_time=selected_ft)
                fig = chart.draw()
                fig.set_size_inches(14, 7)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="#f0f4ff")
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(fig)
            with col_home:
                render_player_card(home_player_df, home_fighter_name or "Home Fighter",
                                   is_winner=(bool(home_fighter_name) and voted == home_fighter_name), align="left")

            st.markdown("<hr style='border-color:#b8c8f8;margin:1.2rem 0;'>", unsafe_allow_html=True)

            with st.spinner("Loading fight location..."):
                try:
                    coords_df = load_fight_coords(home_alias, away_alias, game_date_str)
                except Exception:
                    coords_df = pd.DataFrame(columns=["fight_time", "coord_x", "coord_y"])

            video_url = load_fight_video(home_alias, away_alias, game_date_str,
                                         home_fighter_name or "", away_fighter_name or "")

            if video_url:
                col_rink, col_vid = st.columns(2)
                with col_rink:
                    rink_fig = build_rink_map(coords_df, selected_fight_time=selected_ft)
                    buf = io.BytesIO()
                    rink_fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="#f0f4ff")
                    buf.seek(0)
                    st.image(buf, use_container_width=True)
                    plt.close(rink_fig)
                with col_vid:
                    st.video(video_url)
            else:
                rink_fig = build_rink_map(coords_df, selected_fight_time=selected_ft)
                buf = io.BytesIO()
                rink_fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="#f0f4ff")
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(rink_fig)

        else:
            st.markdown(
                '<p style="color:#4a5580; font-family:\'DM Sans\',sans-serif; font-size:0.85rem; '
                'letter-spacing:0.1em; text-transform:uppercase;">Select a fight above to load the dashboard</p>',
                unsafe_allow_html=True
            )
            
with main_tab2:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.spinner("Loading fight statistics..."):
        fight_df = load_fight_stats()
        jersey_df = load_jersey_stats()

    if fight_df.empty:
        st.info("No fight statistics available.")
    else:
        all_fighters = []
        for _, row in fight_df.iterrows():
            winner = str(row["voted_winner"]).strip()
            home_p = str(row["home_player"]).strip()
            away_p = str(row["away_player"]).strip()
            home_t = str(row["home_team"]).strip()
            away_t = str(row["away_team"]).strip()

            winner_last = winner.split()[-1].upper() if winner else ""
            home_last = home_p.split(".")[-1].strip().upper() if "." in home_p else home_p.split()[-1].upper()
            away_last = away_p.split(".")[-1].strip().upper() if "." in away_p else away_p.split()[-1].upper()

            home_won = winner_last == home_last and winner_last != ""
            away_won = winner_last == away_last and winner_last != ""

            all_fighters.append({"player": home_p, "team": home_t, "won": home_won})
            all_fighters.append({"player": away_p, "team": away_t, "won": away_won})

        fighters_df = pd.DataFrame(all_fighters)
        fighters_df = fighters_df[fighters_df["player"].notna() & (fighters_df["player"] != "nan")]

        record_df = fighters_df.groupby(["player", "team"]).agg(
            fights=("won", "count"),
            wins=("won", "sum")
        ).reset_index()
        record_df["losses"] = record_df["fights"] - record_df["wins"]
        record_df["win_pct"] = (record_df["wins"] / record_df["fights"] * 100).round(1)
        record_df = record_df.sort_values(["wins", "losses"], ascending=[False, True])

        home_fights = fight_df.groupby("home_team").size().reset_index(name="fights")
        away_fights = fight_df.groupby("away_team").size().reset_index(name="fights")
        home_fights.columns = ["team", "fights"]
        away_fights.columns = ["team", "fights"]
        team_fights = pd.concat([home_fights, away_fights]).groupby("team")["fights"].sum().reset_index()

        winner_teams = []
        for _, row in fight_df.iterrows():
            winner = str(row["voted_winner"]).strip()
            home_p = str(row["home_player"]).strip()
            away_p  = str(row["away_player"]).strip()
            winner_last = winner.split()[-1].upper() if winner else ""
            home_last = home_p.split(".")[-1].strip().upper() if "." in home_p else home_p.split()[-1].upper()
            away_last = away_p.split(".")[-1].strip().upper() if "." in away_p else away_p.split()[-1].upper()
            if winner_last == home_last:
                winner_teams.append(str(row["home_team"]))
            elif winner_last == away_last:
                winner_teams.append(str(row["away_team"]))

        team_wins_s = pd.Series(winner_teams).value_counts().reset_index()
        team_wins_s.columns = ["team", "wins"]
        team_stats = team_fights.merge(team_wins_s, on="team", how="left").fillna(0)
        team_stats["wins"] = team_stats["wins"].astype(int)
        team_stats["losses"] = team_stats["fights"] - team_stats["wins"]
        team_stats = team_stats.sort_values("fights", ascending=False).head(10)

        all_fighter_names = pd.concat([
            fight_df[["home_player"]].rename(columns={"home_player": "player"}),
            fight_df[["away_player"]].rename(columns={"away_player": "player"})
        ]).drop_duplicates()

        # ── shared chart style helper ──────────────────────────────────────
        def _card(title):
            return (
                f"<div style='background:#ffffff;border-radius:12px;"
                f"box-shadow:0 2px 12px rgba(30,58,138,0.10);padding:1rem 1.1rem 0.6rem;margin-bottom:1rem;'>"
                f"<div style='font-family:Bebas Neue,sans-serif;font-size:1rem;letter-spacing:0.12em;"
                f"color:#1e3a8a;border-bottom:3px solid #ea580c;padding-bottom:0.25rem;margin-bottom:0.7rem;'>"
                f"{title}</div>"
            )
        CARD_CLOSE = "</div>"

        def _style_ax(ax, ylabel=None):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#a8bae8")
            ax.spines["bottom"].set_color("#a8bae8")
            ax.grid(axis="y", color="#c8d4f0", linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(colors="#1e2a60", labelsize=8)
            if ylabel:
                ax.set_ylabel(ylabel, color="#4a65c0", fontsize=8)

        NHL_COLORS = {
            "ANA":"#F47A38","BOS":"#FFB81C","BUF":"#003087","CGY":"#C8102E",
            "CAR":"#CC0000","CHI":"#CF0A2C","COL":"#6F263D","CBJ":"#002654",
            "DAL":"#006847","DET":"#CE1126","EDM":"#FF4C00","FLA":"#C8102E",
            "LAK":"#111111","MIN":"#154734","MTL":"#AF1E2D","NSH":"#FFB81C",
            "NJD":"#CE1126","NYI":"#00539B","NYR":"#0038A8","OTT":"#C52032",
            "PHI":"#F74902","PIT":"#FCB514","SJS":"#006D75","SEA":"#001628",
            "STL":"#002F87","TBL":"#002868","TOR":"#003E7E","UTA":"#69B3E7",
            "VAN":"#00205B","VGK":"#B4975A","WSH":"#C8102E","WPG":"#041E42",
        }
        POS_COLORS = {"C":"#2563eb","LW":"#1a8a4a","RW":"#c07a00","D":"#c0233a","G":"#4a65c0"}

        # ── 3-COLUMN DASHBOARD ─────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        # ── COL 1: Top Fighters  +  Jersey Numbers ─────────────────────────
        with col1:
            # Chart 1 — Top 10 Fighters by Wins
                        # Chart 1 — Fighter Scatter Plot
            st.markdown(_card("Fighters: Win Rate vs Total Fights"), unsafe_allow_html=True)
            scatter_df = record_df[record_df["fights"] >= 0].copy()  # filter noise
            fig1, ax1 = plt.subplots(figsize=(4.5, 4))
            fig1.patch.set_facecolor("#ffffff")
            ax1.set_facecolor("#f0f3fb")

            ax1.scatter(
                scatter_df["win_pct"],
                scatter_df["fights"],
                color="#2563eb",
                alpha=0.75,
                s=60,
                zorder=3,
                edgecolors="#1e3a8a",
                linewidths=0.5,
            )

            for _, r in scatter_df.iterrows():
                last = r["player"].split(".")[-1].strip() if "." in r["player"] else r["player"].split()[-1]
                ax1.annotate(
                    last,
                    (r["win_pct"], r["fights"]),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=5.5,
                    color="#1e3a8a",
                )

            ax1.set_xlabel("Win %", color="#4a65c0", fontsize=8)
            _style_ax(ax1, ylabel="Total Fights")
            ax1.axvline(50, color="#ea580c", linewidth=0.8, linestyle="--", alpha=0.6)  # 50% reference line
            plt.tight_layout(pad=0.4)
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
            buf1.seek(0)
            st.image(buf1, use_container_width=True)
            plt.close(fig1)
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

            # Chart 3 — Jersey Numbers
            st.markdown(_card("Most Common Jersey #s"), unsafe_allow_html=True)
            if not jersey_df.empty:
                jersey_df["last_name"] = jersey_df["full_name"].str.split().str[-1].str.upper()
                all_fighter_names["last_name"] = all_fighter_names["player"].str.split(".").str[-1].str.strip().str.upper()
                merged = all_fighter_names.merge(jersey_df[["last_name", "jersey_number"]], on="last_name", how="inner")
                jersey_counts = merged["jersey_number"].astype(str).value_counts().head(10).reset_index()
                jersey_counts.columns = ["jersey", "count"]
                jc = jersey_counts.sort_values("count", ascending=False)
                fig3, ax3 = plt.subplots(figsize=(4.5, 3.2))
                fig3.patch.set_facecolor("#ffffff")
                ax3.set_facecolor("#f0f3fb")
                x3 = np.arange(len(jc))
                bars3 = ax3.bar(x3, jc["count"], color="#2563eb", alpha=0.85, width=0.6)
                for bar in bars3:
                    h = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                             ha="center", va="bottom", fontsize=7, color="#1d4ed8", fontweight="bold")
                ax3.set_xticks(x3)
                ax3.set_xticklabels(["#" + j for j in jc["jersey"]], fontsize=7.5, color="#0d0d26")
                _style_ax(ax3, ylabel="Fighters")
                plt.tight_layout(pad=0.4)
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
                buf3.seek(0)
                st.image(buf3, use_container_width=True)
                plt.close(fig3)
            else:
                st.info("Jersey data unavailable.")
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

            # Chart 6 — Fight Timing Histogram
            st.markdown(_card("When Do Fights Happen?"), unsafe_allow_html=True)
            with st.spinner(""):
                timing_df = load_fight_timing()
            if not timing_df.empty:
                timing_df["fight_time"] = pd.to_numeric(timing_df["fight_time"], errors="coerce")
                timing_df = timing_df.dropna(subset=["fight_time"])
                # Convert seconds to game minute (1-60 for regulation, cap OT at 65+)
                timing_df["minute"] = (timing_df["fight_time"] // 60).astype(int)
                # Build per-minute counts across all 60 min buckets
                max_min = int(timing_df["minute"].max()) + 1
                minute_counts = timing_df.groupby("minute").size().reset_index(name="count")
                all_minutes = pd.DataFrame({"minute": range(0, max_min)})
                minute_counts = all_minutes.merge(minute_counts, on="minute", how="left").fillna(0)

                # Color bars by period
                def _period_color(m):
                    if m < 20:   return "#1d4ed8"   # P1 — blue
                    elif m < 40: return "#ea580c"   # P2 — orange
                    elif m < 60: return "#1a8a4a"   # P3 — green
                    else:        return "#9333ea"   # OT — purple

                bar_colors = [_period_color(m) for m in minute_counts["minute"]]

                fig6, ax6 = plt.subplots(figsize=(4.5, 3.2))
                fig6.patch.set_facecolor("#ffffff")
                ax6.set_facecolor("#f0f3fb")
                ax6.bar(minute_counts["minute"], minute_counts["count"],
                        color=bar_colors, alpha=0.9, width=0.85)

                # Period dividers
                for xline, label in [(20, "P2"), (40, "P3"), (60, "OT")]:
                    if xline < max_min:
                        ax6.axvline(xline, color="#a8bae8", linewidth=1, linestyle="--", zorder=3)
                        ax6.text(xline + 0.4, ax6.get_ylim()[1] * 0.92,
                                 label, fontsize=6.5, color="#4a5580", va="top")

                ax6.set_xlabel("Game Minute", color="#4a5580", fontsize=8)
                _style_ax(ax6, ylabel="Fights")
                # Tick every 5 min
                ax6.set_xticks(range(0, max_min, 5))
                ax6.set_xticklabels([str(m) for m in range(0, max_min, 5)], fontsize=7)

                # Legend patches
                import matplotlib.patches as mpatches
                legend_patches = [
                    mpatches.Patch(color="#1d4ed8", label="P1"),
                    mpatches.Patch(color="#ea580c", label="P2"),
                    mpatches.Patch(color="#1a8a4a", label="P3"),
                ]
                if max_min > 60:
                    legend_patches.append(mpatches.Patch(color="#9333ea", label="OT"))
                ax6.legend(handles=legend_patches, fontsize=6.5, facecolor="#ffffff",
                           edgecolor="#c8d4f0", loc="upper right", ncol=len(legend_patches))

                plt.tight_layout(pad=0.4)
                buf6 = io.BytesIO()
                fig6.savefig(buf6, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
                buf6.seek(0)
                st.image(buf6, use_container_width=True)
                plt.close(fig6)
            else:
                st.info("Fight timing data unavailable.")
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

        # ── COL 2: Fight-Prone Teams  +  Position Volume ───────────────────
        with col2:
            # Chart 2 — Most Fight-Prone Teams
            st.markdown(_card("Most Fight-Prone Teams"), unsafe_allow_html=True)
            ts = team_stats.sort_values("fights", ascending=False)
            fig2, ax2 = plt.subplots(figsize=(4.5, 4))
            fig2.patch.set_facecolor("#ffffff")
            ax2.set_facecolor("#f0f3fb")
            x2 = np.arange(len(ts))
            colors2 = [NHL_COLORS.get(str(t), "#2563eb") for t in ts["team"]]
            ax2.bar(x2, ts["fights"], color=colors2, alpha=0.9, width=0.6)
            for i, v in enumerate(ts["fights"]):
                ax2.text(x2[i], v + 0.1, str(int(v)), ha="center", va="bottom",
                         fontsize=6.5, color="#0d0d26", fontweight="bold")
            ax2.set_xticks(x2)
            ax2.set_xticklabels(ts["team"], fontsize=7, color="#0d0d26", rotation=35, ha="right")
            _style_ax(ax2, ylabel="Total Fights")
            plt.tight_layout(pad=0.4)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
            buf2.seek(0)
            st.image(buf2, use_container_width=True)
            plt.close(fig2)
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

            # Chart 4 — Fight Volume by Position
            st.markdown(_card("Fights by Position"), unsafe_allow_html=True)
            if not jersey_df.empty:
                all_fighter_names2 = pd.concat([
                    fight_df[["home_player"]].rename(columns={"home_player": "player"}),
                    fight_df[["away_player"]].rename(columns={"away_player": "player"})
                ])
                jersey_df["last_name"] = jersey_df["full_name"].str.split().str[-1].str.upper()
                all_fighter_names2["last_name"] = all_fighter_names2["player"].str.split(".").str[-1].str.strip().str.upper()
                pos_merged = all_fighter_names2.merge(
                    jersey_df[["last_name", "primary_position"]].dropna(), on="last_name", how="inner"
                )
                pos_counts = pos_merged["primary_position"].value_counts().reset_index()
                pos_counts.columns = ["position", "count"]
                pos_counts = pos_counts.sort_values("count", ascending=False)
                colors_pos = [POS_COLORS.get(str(p), "#5a6899") for p in pos_counts["position"]]
                fig4, ax4 = plt.subplots(figsize=(4.5, 3.2))
                fig4.patch.set_facecolor("#ffffff")
                ax4.set_facecolor("#f0f3fb")
                x4 = np.arange(len(pos_counts))
                bars4 = ax4.bar(x4, pos_counts["count"], color=colors_pos, alpha=0.9, width=0.5)
                for bar in bars4:
                    h = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                             ha="center", va="bottom", fontsize=8, color="#0d0d26", fontweight="bold")
                ax4.set_xticks(x4)
                ax4.set_xticklabels(pos_counts["position"], fontsize=9, color="#0d0d26", fontweight="600")
                _style_ax(ax4, ylabel="Fights")
                plt.tight_layout(pad=0.4)
                buf4 = io.BytesIO()
                fig4.savefig(buf4, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
                buf4.seek(0)
                st.image(buf4, use_container_width=True)
                plt.close(fig4)
            else:
                st.info("Position data unavailable.")
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

        # ── COL 3: Team Bruisers table  +  Goals Delta chart ──────────────
        with col3:
            # Table — Team Bruisers
            st.markdown(_card("Team Bruisers"), unsafe_allow_html=True)
            bruisers = record_df.loc[record_df.groupby("team")["fights"].idxmax()].sort_values("team")
            rows_b = ""
            for _, row in bruisers.iterrows():
                rows_b += (
                    f"<tr style='border-bottom:1px solid #eef1fb;'>"
                    f"<td style='padding:4px 7px;font-weight:700;color:#1d4ed8;font-size:0.72rem;'>{row['team']}</td>"
                    f"<td style='padding:4px 7px;color:#0a0a20;font-size:0.75rem;'>{row['player']}</td>"
                    f"<td style='padding:4px 7px;text-align:center;color:#4a5580;font-size:0.75rem;'>{int(row['fights'])}</td>"
                    f"<td style='padding:4px 7px;text-align:center;color:#1a8a4a;font-weight:700;font-size:0.75rem;'>{int(row['wins'])}W</td>"
                    f"</tr>"
                )
            st.markdown(
                "<div style='background:#fafbff;border:1px solid #c8d4f0;border-radius:8px;overflow:hidden;'>"
                "<table style='width:100%;border-collapse:collapse;'>"
                "<thead><tr style='background:#eef2fc;'>"
                "<th style='padding:5px 7px;text-align:left;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#1d4ed8;'>Team</th>"
                "<th style='padding:5px 7px;text-align:left;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#1d4ed8;'>Fighter</th>"
                "<th style='padding:5px 7px;text-align:center;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#1d4ed8;'>F</th>"
                "<th style='padding:5px 7px;text-align:center;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#1d4ed8;'>W</th>"
                "</tr></thead><tbody>" + rows_b + "</tbody></table></div>",
                unsafe_allow_html=True
            )
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

            # Chart 5 — Avg Goals Delta by Position
            st.markdown(_card("Avg Goals Delta by Position"), unsafe_allow_html=True)
            with st.spinner(""):
                pos_inf_df = load_position_influence()
            if not pos_inf_df.empty:
                pos_inf_df["avg_goals_delta"] = pd.to_numeric(pos_inf_df["avg_goals_delta"], errors="coerce")
                pos_inf_df["fight_count"] = pd.to_numeric(pos_inf_df["fight_count"], errors="coerce")
                pos_inf_df = pos_inf_df.sort_values("avg_goals_delta", ascending=False)
                colors_inf = [POS_COLORS.get(str(p), "#5a6899") for p in pos_inf_df["primary_position"]]
                fig5, ax5 = plt.subplots(figsize=(4.5, 3.2))
                fig5.patch.set_facecolor("#ffffff")
                ax5.set_facecolor("#f0f3fb")
                x5 = np.arange(len(pos_inf_df))
                bars5 = ax5.bar(x5, pos_inf_df["avg_goals_delta"], color=colors_inf, alpha=0.9, width=0.5)
                for bar in bars5:
                    h = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2,
                             h + (0.003 if h >= 0 else -0.015),
                             f"{h:.2f}",
                             ha="center", va="bottom" if h >= 0 else "top",
                             fontsize=7.5, color="#0d0d26", fontweight="bold")
                ax5.axhline(0, color="#a8bae8", linewidth=1)
                ax5.set_xticks(x5)
                ax5.set_xticklabels(pos_inf_df["primary_position"], fontsize=9, color="#0d0d26", fontweight="600")
                _style_ax(ax5, ylabel="Avg Δ Goals/60")
                for i, (_, row) in enumerate(pos_inf_df.iterrows()):
                    ax5.text(x5[i], ax5.get_ylim()[0] * 0.92,
                             f"n={int(row['fight_count'])}",
                             ha="center", va="top", fontsize=6.5, color="#4a5580")
                plt.tight_layout(pad=0.4)
                buf5 = io.BytesIO()
                fig5.savefig(buf5, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
                buf5.seek(0)
                st.image(buf5, use_container_width=True)
                plt.close(fig5)
            else:
                st.info("Position influence data not available.")
            st.markdown(CARD_CLOSE, unsafe_allow_html=True)

with main_tab3:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.spinner("Loading dominant fights..."):
        dom_df = load_fight_stats()

    if dom_df.empty:
        st.info("No fight data available.")
    else:
        dom_df["winner_pct"] = pd.to_numeric(dom_df["winner_pct"], errors="coerce")
        dom_df["voted_rating"] = pd.to_numeric(dom_df["voted_rating"], errors="coerce")
        dom_df["vote_count"] = pd.to_numeric(dom_df["vote_count"], errors="coerce")

        dominant = (
            dom_df[dom_df["vote_count"] >= 10]
            .sort_values(["winner_pct", "voted_rating"], ascending=[False, False])
            .head(20)
            .reset_index(drop=True)
        )

        if "dom_video_url" not in st.session_state:
            st.session_state["dom_video_url"] = None

        st.markdown(
            "<div style='font-family:Bebas Neue,sans-serif;font-size:1.4rem;letter-spacing:0.12em;"
            "color:#1e3a8a;border-bottom:3px solid #ea580c;padding-bottom:0.4rem;margin-bottom:0.4rem;'>"
            "Top 20 Most Dominant Fights</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-size:0.78rem;color:#4a5580;margin-bottom:1.5rem;'>"
            "Ranked by vote % for winner · minimum 10 votes · click Watch to load video</div>",
            unsafe_allow_html=True
        )

        RANK_COLORS = {
            0: ("#FFD700", "#7a6000"),
            1: ("#C0C0C0", "#555555"),
            2: ("#CD7F32", "#5a3000"),
        }

        for i, row in dominant.iterrows():
            winner_last = str(row["voted_winner"]).split()[-1] if pd.notna(row["voted_winner"]) else ""
            home_last = str(row["home_player"]).split(".")[-1].strip() if "." in str(row["home_player"]) else str(row["home_player"]).split()[-1]
            away_last = str(row["away_player"]).split(".")[-1].strip() if "." in str(row["away_player"]) else str(row["away_player"]).split()[-1]
            loser = row["away_player"] if winner_last.upper() == home_last.upper() else row["home_player"]

            date_str = pd.to_datetime(row["date"], errors="coerce").strftime("%m-%d-%Y") if pd.notna(row.get("date")) else ""
            video = str(row.get("video_url", "") or "")

            pct = int(row["winner_pct"]) if pd.notna(row["winner_pct"]) else 0
            rating = round(float(row["voted_rating"]), 1) if pd.notna(row["voted_rating"]) else 0
            votes = int(row["vote_count"]) if pd.notna(row["vote_count"]) else 0

            pct_color = "#1a8a4a" if pct >= 80 else "#c07a00" if pct >= 60 else "#5a6899"
            rank_bg, rank_text = RANK_COLORS.get(i, ("#dde4f8", "#5a6899"))

            rank_badge = (
                f"<div style='width:44px;height:44px;border-radius:50%;background:{rank_bg};"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-family:Bebas Neue,sans-serif;font-size:1.3rem;color:{rank_text};"
                f"flex-shrink:0;box-shadow:0 2px 6px rgba(0,0,0,0.10);'>{i+1}</div>"
            )

            card_html = (
                f"<div style='background:#ffffff;border:1.5px solid #a8bae8;border-radius:14px;"
                f"box-shadow:0 2px 16px rgba(37,99,235,0.10);padding:1.1rem 1.4rem;"
                f"margin-bottom:0.75rem;display:flex;align-items:center;gap:1.2rem;'>"

                # Rank badge
                f"{rank_badge}"

                # Fighter info
                f"<div style='flex:1;min-width:0;'>"
                f"<div style='font-size:1.05rem;font-weight:700;color:#0a0a20;'>{row['voted_winner']}"
                f"<span style='font-weight:400;color:#4a5580;font-size:0.9rem;'> def. {loser}</span></div>"
                f"<div style='font-size:0.75rem;color:#1d4ed8;margin-top:2px;letter-spacing:0.04em;'>"
                f"{row['away_team']} @ {row['home_team']}"
                f"{'  ·  ' + date_str if date_str else ''}</div>"
                f"</div>"

                # Win %
                f"<div style='text-align:center;min-width:80px;'>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{pct_color};line-height:1;'>{pct}%</div>"
                f"<div style='font-size:0.6rem;color:#4a5580;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;'>Win Vote</div>"
                f"</div>"

                # Rating
                f"<div style='text-align:center;min-width:70px;'>"
                f"<div style='font-size:1.4rem;font-weight:800;color:#2563eb;line-height:1;'>{rating}</div>"
                f"<div style='font-size:0.6rem;color:#4a5580;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;'>Rating</div>"
                f"</div>"

                # Votes
                f"<div style='text-align:center;min-width:60px;'>"
                f"<div style='font-size:1.1rem;font-weight:600;color:#4a5580;line-height:1;'>{votes}</div>"
                f"<div style='font-size:0.6rem;color:#4a5580;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;'>Votes</div>"
                f"</div>"

                f"</div>"
            )

            col_card, col_btn = st.columns([11, 1.2])
            with col_card:
                st.markdown(card_html, unsafe_allow_html=True)
            with col_btn:
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                if video:
                    if st.button("▶ Watch", key=f"dom_watch_{i}", use_container_width=True):
                        st.session_state["dom_video_url"] = video if st.session_state.get("dom_video_url") != video else None

            if st.session_state.get("dom_video_url") == video and video:
                st.video(video)
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)