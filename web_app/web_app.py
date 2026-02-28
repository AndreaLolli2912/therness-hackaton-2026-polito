"""
Weld Analysis Platform
──────────────────────
Part 1 — Single CSV time-series viewer  (sidebar controls + media viewer + truth label display)
Part 2 — Multi-file statistics by material / type
"""

import re
import io
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════
# Page config — must be first Streamlit call
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Weld Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
# Global CSS — dark industrial theme
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:    #0a0c10;
    --bg-panel:   #111318;
    --bg-card:    #181c24;
    --bg-hover:   #1e2330;
    --border:     #2a3040;
    --border-glow:#3d7fff44;
    --accent:     #3d7fff;
    --accent2:    #00e5ff;
    --accent3:    #ff6b35;
    --warn:       #ffb800;
    --success:    #00d68f;
    --danger:     #ff3d71;
    --text-primary: #e8ecf4;
    --text-secondary: #7a8499;
    --text-dim:   #4a5568;
}

/* ── App background ── */
.stApp {
    background: var(--bg-deep);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, #1a2744 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, #1a2030 39px, #1a2030 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, #1a2030 39px, #1a2030 40px);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 100% !important; }

/* ── Custom header banner ── */
.weld-header {
    background: linear-gradient(135deg, #0d1526 0%, #111827 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--accent);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 4px 40px #3d7fff18, 0 1px 0 #3d7fff33 inset;
}
.weld-header-icon {
    font-size: 2.8rem;
    filter: drop-shadow(0 0 12px #3d7fff88);
}
.weld-header-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}
.weld-header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    min-width: 140px;
    flex: 1;
    box-shadow: 0 2px 16px #00000040;
}
.metric-card-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent2);
    line-height: 1;
}
.metric-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
}

/* ── Label card (truth label display) ── */
.label-card {
    background: linear-gradient(135deg, #111827 0%, #0f1923 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent3);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 4px 24px #ff6b3510;
}
.label-result {
    background: #0a1628;
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 1.4rem 1.8rem;
    margin-top: 1rem;
    box-shadow: 0 0 20px #3d7fff18;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.label-badge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    line-height: 1;
}
.label-meta-block {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.label-meta-item {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}
.label-meta-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    min-width: 90px;
}
.label-meta-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-secondary);
}
.label-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    vertical-align: middle;
    box-shadow: 0 0 8px currentColor;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.9rem 1.8rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent2) !important;
    border-bottom: 2px solid var(--accent2) !important;
    background: var(--bg-hover) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stTextInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
.sidebar-section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px #3d7fff44 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

/* ── Plot background ── */
.stPlot { border-radius: 8px; overflow: hidden; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 8px !important; border-left-width: 3px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Matplotlib dark theme
# ══════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#111318",
    "axes.facecolor":    "#0e1219",
    "axes.edgecolor":    "#2a3040",
    "axes.labelcolor":   "#7a8499",
    "axes.titlecolor":   "#e8ecf4",
    "xtick.color":       "#4a5568",
    "ytick.color":       "#4a5568",
    "grid.color":        "#1e2330",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#e8ecf4",
    "figure.dpi":        110,
})

PLOT_COLORS = ["#3d7fff", "#00e5ff", "#ff6b35", "#00d68f", "#ffb800", "#c77dff", "#ff3d71", "#74c0fc"]

# ══════════════════════════════════════════
# Constants / helpers
# ══════════════════════════════════════════
SEPARATOR_RE   = re.compile(r"[_\-]+")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE   = re.compile(r"[^a-z0-9\s]+")

# ── Ground-truth label map ─────────────────────────────────────────────────────
# Maps the last two digits of a sample filename → (display label, hex color)
# e.g.  10-11-22-0021-02.csv  →  suffix "02"  →  "Burn Through"
SUFFIX_LABEL_MAP: dict[str, tuple[str, str]] = {
    "00": ("Good Weld",             "#00d68f"),
    "01": ("Excessive Penetration", "#74c0fc"),
    "02": ("Burn Through",          "#ff3d71"),
    "06": ("Overlap",               "#ffb800"),
    "07": ("Lack of Fusion",        "#c77dff"),
    "08": ("Excessive Convexity",   "#ff6b35"),
    "11": ("Crater Cracks",         "#3d7fff"),
}


def _normalize(text: str) -> str:
    s = SEPARATOR_RE.sub(" ", text.lower())
    s = NON_ALNUM_RE.sub(" ", s)
    return MULTI_SPACE_RE.sub(" ", s).strip()


def infer_metadata(path: Path):
    folder_text = _normalize(" ".join(path.parts[:-1]))
    material = (
        "BSK46"   if re.search(r"\bbsk\s*46\b", folder_text) else
        "Fe410"   if re.search(r"\bfe\s*410\b",  folder_text) else
        "unknown"
    )
    weld_type = (
        "butt"        if re.search(r"\bbutt\b", folder_text) else
        "plane_plate" if (
            re.search(r"\bplane\b", folder_text)
            and re.search(r"\bplate\b", folder_text)
        ) else
        "unknown"
    )
    return material, weld_type


def extract_truth_label(csv_path: str, experiment_folder: str) -> dict:
    """
    Extract the ground-truth defect label from the filename suffix.

    Sample pattern:  <date>-<NNNN>-<XX>.csv
      e.g.  10-11-22-0021-02.csv
                               ^^— class code looked up in SUFFIX_LABEL_MAP

    Returns dict with keys:
        defect_type   – human-readable label, e.g. "Burn Through"
        color         – hex colour for UI highlight
        sample_id     – 4-digit numeric ID parsed from filename
        sample_suffix – the two-digit class code (e.g. "02")
        experiment    – raw experiment folder name
        known         – True if the suffix was found in SUFFIX_LABEL_MAP
    """
    p    = Path(csv_path)
    stem = p.stem  # e.g. "10-11-22-0021-02"

    # ── Parse class code (last two digits) from filename ─────────────────
    suffix_match  = re.search(r'-(\d{2})$', stem)
    sample_suffix = suffix_match.group(1) if suffix_match else "?"

    id_match  = re.search(r'-(\d{4})-', stem)
    sample_id = id_match.group(1) if id_match else "?"

    # ── Look up label ─────────────────────────────────────────────────────
    if sample_suffix in SUFFIX_LABEL_MAP:
        defect_type, color = SUFFIX_LABEL_MAP[sample_suffix]
        known = True
    else:
        defect_type = f"Unknown (code {sample_suffix})"
        color       = "#4a5568"
        known       = False

    return {
        "defect_type":   defect_type,
        "color":         color,
        "sample_id":     sample_id,
        "sample_suffix": sample_suffix,
        "experiment":    experiment_folder,
        "known":         known,
    }


# ══════════════════════════════════════════
# CSV discovery & reading
# ══════════════════════════════════════════
@st.cache_data(show_spinner="Scanning dataset…")
def discover_csvs(root_dir: str) -> pd.DataFrame:
    rows = [
        {
            "csv_path":   str(p),
            "label":      f"{p.parent.name}  ({mat} / {wtype})",
            "material":   mat,
            "type":       wtype,
            "experiment": p.parent.parent.name,   # the named experiment folder
            "folder":     str(p.parent),
        }
        for p in sorted(Path(root_dir).rglob("*.csv"))
        for mat, wtype in [infer_metadata(p)]
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def read_csv_parsed(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    time_col = next((c for c in df.columns if c.lower() == "time"), None)
    if date_col and time_col:
        df["datetime"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        df = df.drop(columns=[date_col, time_col])
    elif date_col:
        df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.drop(columns=[date_col])
    drop_cols = [c for c in df.columns
                 if c.lower().replace(" ", "").replace("_", "") in {"partno", "part", "id", "source"}]
    df = df.drop(columns=drop_cols, errors="ignore")
    for col in [c for c in df.columns if c != "datetime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(axis=1, how="all")


def get_value_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "datetime"]


# ══════════════════════════════════════════
# ffmpeg / video conversion
# ══════════════════════════════════════════
def find_ffmpeg() -> str | None:
    import shutil, os
    if os.name == "nt":
        try:
            live_path = subprocess.run(["cmd", "/c", "echo %PATH%"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            os.environ["PATH"] = live_path
        except Exception:
            pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    if os.name == "nt":
        try:
            result = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                first = result.stdout.strip().splitlines()[0]
                if Path(first).exists():
                    return first
        except Exception:
            pass
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]
    local_app = __import__("os").environ.get("LOCALAPPDATA", "")
    if local_app:
        winget_base = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if winget_base.exists():
            for exe in winget_base.rglob("ffmpeg.exe"):
                candidates.append(str(exe))
    for c in candidates:
        if Path(c).exists():
            return c
    return None


_FFMPEG_NOT_FOUND_MSG = (
    "**ffmpeg not found.** Install it then restart Streamlit:\n\n"
    "```powershell\nwinget install Gyan.FFmpeg\n```\n"
    "or download from https://www.gyan.dev/ffmpeg/builds/ and extract to `C:\\ffmpeg`."
)


def ensure_mp4(avi_path: Path) -> tuple[Path | None, str | None]:
    mp4_path = avi_path.with_suffix(".mp4")
    if mp4_path.exists():
        return mp4_path, None
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return None, _FFMPEG_NOT_FOUND_MSG
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(avi_path),
             "-vcodec", "libx264", "-acodec", "aac",
             "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
             str(mp4_path)],
            capture_output=True, timeout=300)
    except FileNotFoundError:
        return None, _FFMPEG_NOT_FOUND_MSG
    except subprocess.TimeoutExpired:
        return None, "ffmpeg timed out."
    if result.returncode == 0 and mp4_path.exists():
        return mp4_path, None
    return None, f"ffmpeg failed:\n```\n{result.stderr.decode(errors='ignore')[-600:]}\n```"


# ══════════════════════════════════════════
# Media discovery
# ══════════════════════════════════════════
def find_media(folder: str):
    p = Path(folder)
    avi_files = sorted(p.glob("*.avi"))
    mp4_only  = [f for f in sorted(p.glob("*.mp4")) if not f.with_suffix(".avi").exists()]
    return {
        "video":  avi_files + mp4_only,
        "audio":  sorted(p.glob("*.flac")) + sorted(p.glob("*.wav")) + sorted(p.glob("*.mp3")),
        "images": sorted((p / "images").glob("*")) if (p / "images").is_dir()
                  else sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) + sorted(p.glob("*.jpeg")),
    }


# ══════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════
def plot_single_column(df: pd.DataFrame, col: str, color: str) -> plt.Figure:
    use_dt = "datetime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["datetime"])
    x = df["datetime"] if use_dt else np.arange(len(df))
    y = df[col].values

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.plot(x, y, linewidth=1.2, color=color, zorder=3)
    ax.fill_between(x, y, alpha=0.10, color=color, zorder=2)

    ax.yaxis.grid(True, zorder=1)
    ax.set_axisbelow(True)

    ax.set_title(col, fontsize=10, fontweight="600", pad=6)
    ax.set_ylabel("", fontsize=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(labelsize=7, length=0)

    if use_dt:
        fig.autofmt_xdate(rotation=20, ha="right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.tight_layout(pad=0.8)
    return fig


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


def render_metric_cards(metrics: dict):
    cards_html = "".join(
        f'<div class="metric-card">'
        f'<div class="metric-card-value">{v}</div>'
        f'<div class="metric-card-label">{k}</div>'
        f'</div>'
        for k, v in metrics.items()
    )
    st.markdown(f'<div class="metric-row">{cards_html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# Header
# ══════════════════════════════════════════
st.markdown("""
<div class="weld-header">
  <div class="weld-header-icon">⚡</div>
  <div>
    <div class="weld-header-title">Weld Analysis Platform</div>
    <div class="weld-header-sub">Sensor · Time-series · Truth Label Inspector · Statistics</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">⚙ Configuration</div>', unsafe_allow_html=True)
    root_dir = st.text_input("Dataset root folder", value="data", label_visibility="visible")

    index_df = discover_csvs(root_dir)
    if index_df.empty:
        st.error(f"No CSV files found under **{root_dir}**")
        st.stop()

    # ── ffmpeg diagnostics ────────────────
    with st.expander("🔧 ffmpeg diagnostics", expanded=False):
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            st.success(f"✅ Found:\n`{ffmpeg_path}`")
            try:
                ver = subprocess.run([ffmpeg_path, "-version"],
                    capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
                st.caption(ver)
            except Exception:
                pass
        else:
            st.error("❌ ffmpeg not found.")
            st.markdown(_FFMPEG_NOT_FOUND_MSG)

    st.divider()

    # ── Part 1 controls ───────────────────
    st.markdown('<div class="sidebar-section-title">📈 Single File</div>', unsafe_allow_html=True)

    label_to_row = {row["label"]: row for _, row in index_df.iterrows()}
    chosen_label = st.selectbox("Experiment", list(label_to_row.keys()), label_visibility="visible")
    chosen_row    = label_to_row[chosen_label]
    chosen_path   = chosen_row["csv_path"]
    chosen_folder = chosen_row["folder"]

    df_single    = read_csv_parsed(chosen_path)
    vcols_single = get_value_cols(df_single)

    selected_cols = st.multiselect("Columns to plot", vcols_single, default=vcols_single)
    plots_per_row = st.radio("Plots per row", [1, 2], index=1, horizontal=True)

    st.divider()

    # ── Part 2 controls ───────────────────
    st.markdown('<div class="sidebar-section-title">📊 Multi-File Stats</div>', unsafe_allow_html=True)

    materials = ["All"] + sorted(index_df["material"].unique())
    types     = ["All"] + sorted(index_df["type"].unique())
    mat_choice  = st.selectbox("Material", materials, key="p2_mat")
    type_choice = st.selectbox("Type",     types,     key="p2_type")

    filtered = index_df.copy()
    if mat_choice  != "All": filtered = filtered[filtered["material"] == mat_choice]
    if type_choice != "All": filtered = filtered[filtered["type"]     == type_choice]

    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.75rem;'
        f'color:#00e5ff;padding:0.4rem 0;">'
        f'▸ {len(filtered)} files selected</div>',
        unsafe_allow_html=True
    )


def display_type(t: str) -> str:
    return t.replace("butt", "butt-joint").replace("_", " ")


# ══════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════
tab1, tab2 = st.tabs(["  📈  SINGLE FILE  ", "  📊  STATISTICS  "])


# ══════════════════════════════════════════
#  PART 1
# ══════════════════════════════════════════
with tab1:

    # ── Experiment info bar ───────────────
    exp_meta = label_to_row[chosen_label]
    render_metric_cards({
        "Material":    exp_meta["material"],
        "Weld Type":   exp_meta["type"].replace("butt", "butt-joint").replace("_", " "),
        "Experiment":  exp_meta["experiment"],
    })

    # ── Truth Label Inspector ─────────────
    st.markdown('<div class="section-label">🏷️ Truth Label</div>', unsafe_allow_html=True)

    st.markdown('<div class="label-card">', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family:Inter;font-size:0.9rem;color:#7a8499;margin:0;'>"
        "Click the button to reveal the ground-truth defect label for this sample, "
        "extracted directly from the dataset folder and filename convention.</p>",
        unsafe_allow_html=True
    )

    reveal_clicked = st.button("🏷️ Reveal Truth Label", type="primary", use_container_width=False)

    if reveal_clicked:
        truth = extract_truth_label(chosen_path, exp_meta["experiment"])
        c     = truth["color"]

        unknown_warning = (
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'color:#ffb800;margin-top:0.6rem;">⚠ Code <b>{truth["sample_suffix"]}</b> '
            f'is not in the label map. Add it to SUFFIX_LABEL_MAP in app.py.</div>'
        ) if not truth["known"] else ""

        st.markdown(
            f"""
            <div class="label-result">
                <div style="display:flex;flex-direction:column;align-items:center;
                            background:#0d1526;border:1px solid {c}33;border-radius:8px;
                            padding:0.8rem 1.2rem;min-width:80px;text-align:center;">
                    <div style="font-family:JetBrains Mono,monospace;font-size:1.6rem;
                                font-weight:700;color:{c};line-height:1;">{truth['sample_suffix']}</div>
                    <div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;
                                color:#4a5568;text-transform:uppercase;letter-spacing:0.12em;
                                margin-top:4px;">class code</div>
                </div>
                <div style="font-size:1.6rem;color:#2a3040;font-weight:300;">→</div>
                <div>
                    <span class="label-dot" style="background:{c};color:{c};"></span>
                    <span class="label-badge" style="color:{c};">{truth['defect_type']}</span>
                    <div class="label-meta-block" style="margin-top:0.6rem;">
                        <div class="label-meta-item">
                            <span class="label-meta-key">Sample ID</span>
                            <span class="label-meta-val">{truth['sample_id']}</span>
                        </div>
                        <div class="label-meta-item">
                            <span class="label-meta-key">Experiment</span>
                            <span class="label-meta-val">{truth['experiment']}</span>
                        </div>
                        <div class="label-meta-item">
                            <span class="label-meta-key">File</span>
                            <span class="label-meta-val" style="font-size:0.68rem;color:#4a5568;">
                                {Path(chosen_path).name}</span>
                        </div>
                    </div>
                </div>
            </div>
            {unknown_warning}
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Media viewer ─────────────────────
    st.markdown('<div class="section-label">🎬 Media</div>', unsafe_allow_html=True)
    media     = find_media(chosen_folder)
    has_media = any(media[k] for k in media)

    with st.expander("Open Media Viewer", expanded=False):
        if not has_media:
            st.info("No media files found in this experiment folder.")
        else:
            media_tab_labels = []
            if media["video"]:  media_tab_labels.append("🎥 Video")
            if media["audio"]:  media_tab_labels.append("🔊 Audio")
            if media["images"]: media_tab_labels.append("🖼️ Images")

            media_tabs = st.tabs(media_tab_labels)
            tab_idx = 0

            if media["video"]:
                with media_tabs[tab_idx]:
                    for vf in media["video"]:
                        if vf.suffix.lower() == ".avi":
                            mp4_exists = vf.with_suffix(".mp4").exists()
                            if not mp4_exists:
                                with st.spinner(f"Converting {vf.name} → MP4…"):
                                    mp4_path, err = ensure_mp4(vf)
                            else:
                                mp4_path, err = vf.with_suffix(".mp4"), None
                            if err:
                                st.error(err)
                                with open(vf, "rb") as f:
                                    st.download_button(f"⬇️ {vf.name}", f.read(),
                                        vf.name, "video/x-msvideo", key=f"vid_orig_{vf.name}")
                            else:
                                st.caption(f"{vf.name}  →  MP4")
                                st.video(str(mp4_path))
                                with open(mp4_path, "rb") as f:
                                    st.download_button(f"⬇️ {mp4_path.name}", f.read(),
                                        mp4_path.name, "video/mp4", key=f"vid_{vf.name}")
                        else:
                            st.video(str(vf))
                            with open(vf, "rb") as f:
                                st.download_button(f"⬇️ {vf.name}", f.read(),
                                    vf.name, "video/mp4", key=f"vid_{vf.name}")
                tab_idx += 1

            if media["audio"]:
                with media_tabs[tab_idx]:
                    for af in media["audio"]:
                        st.caption(af.name)
                        try:
                            st.audio(str(af))
                        except Exception:
                            st.warning(f"Could not load {af.name}")
                        with open(af, "rb") as f:
                            st.download_button(f"⬇️ {af.name}", f.read(),
                                af.name, "audio/flac", key=f"aud_{af.name}")
                tab_idx += 1

            if media["images"]:
                with media_tabs[tab_idx]:
                    img_files = [f for f in media["images"]
                                 if f.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tiff",".webp"}]
                    if not img_files:
                        st.info("No displayable images found.")
                    else:
                        for i in range(0, len(img_files), 3):
                            row_imgs = img_files[i:i+3]
                            cols = st.columns(len(row_imgs))
                            for col, img in zip(cols, row_imgs):
                                col.image(str(img), caption=img.name, use_container_width=True)

    # ── Summary stats ─────────────────────
    with st.expander("📋 Summary statistics", expanded=False):
        if selected_cols:
            st.dataframe(df_single[selected_cols].describe().T.round(4), use_container_width=True)
        st.download_button("⬇️ Raw data CSV", df_single.to_csv(index=False).encode(),
            f"{Path(chosen_path).stem}.csv", "text/csv")

    # ── Time-series plots ─────────────────
    st.markdown('<div class="section-label">📡 Sensor Time-Series</div>', unsafe_allow_html=True)

    if not selected_cols:
        st.info("Select at least one column in the sidebar.")
    else:
        pairs = [selected_cols[i:i+plots_per_row] for i in range(0, len(selected_cols), plots_per_row)]
        for pair in pairs:
            ui_cols = st.columns(len(pair))
            for ui_col, col_name in zip(ui_cols, pair):
                color = PLOT_COLORS[selected_cols.index(col_name) % len(PLOT_COLORS)]
                fig   = plot_single_column(df_single, col_name, color)
                ui_col.pyplot(fig, use_container_width=True)
                ui_col.download_button(
                    f"⬇️ {col_name}.png", fig_to_png(fig),
                    f"{col_name.replace(' ','_')}.png", "image/png",
                    key=f"p1_dl_{col_name}"
                )
                plt.close(fig)


# ══════════════════════════════════════════
#  PART 2
# ══════════════════════════════════════════
with tab2:

    render_metric_cards({
        "Material":  mat_choice,
        "Type":      type_choice.replace("butt", "butt-joint").replace("_", " "),
        "Files":     str(len(filtered)),
    })

    if filtered.empty:
        st.warning("No files match the current filters.")
        st.stop()

    with st.expander("Files included", expanded=False):
        st.dataframe(filtered[["experiment","material","type","csv_path"]], use_container_width=True)

    @st.cache_data(show_spinner="Loading files…")
    def load_all_values(paths: tuple) -> pd.DataFrame:
        frames = []
        for p in paths:
            try:
                df = read_csv_parsed(p)
                vdf = df[get_value_cols(df)].copy()
                vdf["__file__"] = Path(p).name
                frames.append(vdf)
            except Exception as e:
                st.warning(f"Skipped {Path(p).name}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    paths_tuple = tuple(filtered["csv_path"].tolist())
    all_data    = load_all_values(paths_tuple)

    if all_data.empty:
        st.warning("No numeric data could be loaded.")
        st.stop()

    vcols_multi = [c for c in all_data.columns if c != "__file__"]

    # ── Global stats ──────────────────────
    st.markdown('<div class="section-label">📐 Descriptive Statistics</div>', unsafe_allow_html=True)
    global_stats = all_data[vcols_multi].describe().T.round(4)
    global_stats.index.name = "column"
    st.dataframe(global_stats, use_container_width=True)
    st.download_button("⬇️ Download statistics CSV", global_stats.to_csv().encode(),
        "statistics.csv", "text/csv", key="p2_stats_dl")

    st.divider()

    # ── Per-file means ────────────────────
    st.markdown('<div class="section-label">📁 Per-File Column Means</div>', unsafe_allow_html=True)
    per_file = (all_data.groupby("__file__")[vcols_multi]
                .mean().round(4).rename_axis("file").reset_index())
    st.dataframe(per_file, use_container_width=True)
    st.download_button("⬇️ Download per-file means", per_file.to_csv(index=False).encode(),
        "per_file_means.csv", "text/csv", key="p2_perfile_dl")

    st.divider()

    # ── Visualisations ────────────────────
    st.markdown('<div class="section-label">📊 Visual Comparisons</div>', unsafe_allow_html=True)

    viz_col_choice = st.multiselect("Columns to visualise", vcols_multi, default=vcols_multi, key="p2_viz")
    viz_type = st.radio("Chart type",
        ["Box plot", "Bar chart (mean ± std)", "Correlation heatmap"],
        horizontal=True, key="p2_viz_type")

    if not viz_col_choice:
        st.info("Select at least one column.")
        st.stop()

    if viz_type == "Box plot":
        n    = len(viz_col_choice)
        ncol = min(3, n)
        nrow = -(-n // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.5, nrow * 3.2))
        axes_flat = [axes] if n == 1 else list(axes.flatten())
        for i, col in enumerate(viz_col_choice):
            ax   = axes_flat[i]
            data = all_data[col].dropna()
            bp   = ax.boxplot(data, patch_artist=True, widths=0.5,
                              medianprops=dict(color="#e8ecf4", linewidth=2),
                              whiskerprops=dict(color="#4a5568"),
                              capprops=dict(color="#4a5568"),
                              flierprops=dict(marker="o", color=PLOT_COLORS[i%len(PLOT_COLORS)],
                                             markersize=3, alpha=0.5))
            bp["boxes"][0].set_facecolor(PLOT_COLORS[i % len(PLOT_COLORS)])
            bp["boxes"][0].set_alpha(0.75)
            ax.set_title(col, fontsize=9, fontweight="600")
            ax.set_xticks([])
            ax.spines[["top","right","bottom","left"]].set_visible(False)
            ax.tick_params(labelsize=7, length=0)
            ax.yaxis.grid(True, alpha=0.4)
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle(f"{mat_choice} / {display_type(type_choice)}  —  {len(filtered)} file(s)",
                     fontsize=10, color="#7a8499", y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download plot", fig_to_png(fig), "boxplots.png", "image/png", key="p2_box_dl")
        plt.close(fig)

    elif viz_type == "Bar chart (mean ± std)":
        means = all_data[viz_col_choice].mean()
        stds  = all_data[viz_col_choice].std()
        fig, ax = plt.subplots(figsize=(max(7, len(viz_col_choice)*1.1), 4))
        x = range(len(viz_col_choice))
        ax.bar(x, means, yerr=stds, capsize=4,
               color=PLOT_COLORS[:len(viz_col_choice)], edgecolor="#0e1219",
               linewidth=0.5, alpha=0.85,
               error_kw=dict(elinewidth=1.2, ecolor="#4a5568"))
        ax.set_xticks(list(x))
        ax.set_xticklabels(viz_col_choice, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Value", fontsize=8)
        ax.set_title(f"Mean ± Std  —  {mat_choice} / {display_type(type_choice)}", fontsize=10)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download plot", fig_to_png(fig), "bar_mean_std.png", "image/png", key="p2_bar_dl")
        plt.close(fig)

    else:
        corr = all_data[viz_col_choice].corr()
        sz   = max(5, len(viz_col_choice) * 0.9)
        fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        ax.set_xticks(range(len(viz_col_choice)))
        ax.set_yticks(range(len(viz_col_choice)))
        ax.set_xticklabels(viz_col_choice, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(viz_col_choice, fontsize=8)
        for i in range(len(viz_col_choice)):
            for j in range(len(viz_col_choice)):
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="#e8ecf4" if abs(v) > 0.5 else "#4a5568")
        ax.set_title(f"Correlation  —  {mat_choice} / {display_type(type_choice)}", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download plot", fig_to_png(fig), "correlation.png", "image/png", key="p2_corr_dl")
        plt.close(fig)