from __future__ import annotations

import os
import re
import base64
import textwrap
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import requests
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CVR Capstone Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# BASE_DIR: works locally (Desktop/Capstone Dashboard) AND on Streamlit Cloud (next to app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_local = os.path.join(os.path.expanduser("~"), "Desktop", "Capstone Dashboard")
if os.path.isdir(_local):
    BASE_DIR = _local

def p(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)

def first_existing(candidates: List[str]) -> Optional[str]:
    for name in candidates:
        full = p(name)
        if os.path.exists(full):
            return full
    return None

C = {
    "white": "#FFFFFF", "bg": "#FFFFFF", "text": "#141414", "muted": "#59536B",
    "purple": "#b86ce0", "orchid": "#b86ce0", "deep": "#201436", "border": "#EDE0FA",
    "good": "#2CB67D", "warn": "#ffa600", "bad": "#ff5c83", "blue": "#7678ed",
    "teal": "#7678ed", "gold": "#ffa600", "orange": "#ff8a38", "pink": "#ff5c83",
    "indigo": "#7678ed", "panel_bg": "#FCFBFE",
}

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}

def inject_css() -> None:
    st.markdown(f"""
    <style>
    .stApp {{ background: {C["bg"]}; }}
    .block-container {{ max-width: 1460px; padding-top: 2.2rem; padding-bottom: 2rem; }}
    .block-container > div {{ animation: fadeSlideIn 0.4s cubic-bezier(0.16,1,0.3,1); }}
    @keyframes fadeSlideIn {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .block-container > div > div:nth-child(1) {{ animation-delay: 0.05s; }}
    .block-container > div > div:nth-child(2) {{ animation-delay: 0.10s; }}
    .block-container > div > div:nth-child(3) {{ animation-delay: 0.15s; }}
    .js-plotly-plot {{ animation: fadeSlideIn 0.45s cubic-bezier(0.16,1,0.3,1); }}
    .section-panel {{
        transition: box-shadow 0.22s ease, transform 0.22s ease;
        animation: fadeSlideIn 0.4s cubic-bezier(0.16,1,0.3,1);
    }}
    .section-panel:hover {{
        box-shadow: 0 8px 28px rgba(139,47,201,0.14);
        transform: translateY(-1px);
    }}
    .kpi-card {{ transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.2s ease; }}
    .kpi-card:hover {{ transform: translateY(-3px) scale(1.01); box-shadow: 0 10px 28px rgba(139,47,201,0.20); }}
    .stSelectbox > div, .stNumberInput > div, .stSlider > div, .stRadio > div {{
        transition: all 0.18s ease;
    }}
    .stTabs [data-baseweb="tab"] {{ transition: all 0.18s ease; }}
    .stTabs [aria-selected="true"] {{ transition: all 0.18s ease; }}
    .streamlit-expanderHeader {{ transition: all 0.18s ease; }}
    .stButton > button {{ transition: all 0.18s cubic-bezier(0.34,1.56,0.64,1); }}
    .stButton > button:hover {{ transform: translateY(-1px); }}
    .file-link-card {{ transition: transform 0.18s ease, box-shadow 0.18s ease; }}
    .file-link-card:hover {{ transform: translateX(3px); box-shadow: 0 4px 16px rgba(139,47,201,0.10); }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #F8F2FE 0%, #FCFBFE 100%);
        border-right: 1px solid {C["border"]};
    }}
    h1, h2, h3 {{ color: {C["deep"]}; }}
    .hero-wrap {{
        position: relative; width: 100%; min-height: 320px; border-radius: 20px;
        overflow: hidden; margin-bottom: 1.2rem; border: 1px solid {C["border"]};
        box-shadow: 0 14px 34px rgba(32,20,54,0.14); background: linear-gradient(135deg, #201436 0%, #b86ce0 100%);
    }}
    .hero-overlay {{
        position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0.08) 0%, rgba(0,0,0,0.55) 100%);
        display: flex; align-items: flex-end; padding: 2rem 2.2rem;
    }}
    .hero-tag {{
        display: inline-block; padding: 0.36rem 0.72rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: white; background: rgba(79,38,131,0.76); border: 1px solid rgba(255,255,255,0.22);
        margin-bottom: 0.75rem;
    }}
    .hero-title {{
        margin: 0 0 0.3rem 0; color: white !important;
        font-size: clamp(1.4rem, 2.5vw, 2.2rem); font-weight: 800; line-height: 1.1;
        text-shadow: 0 2px 18px rgba(0,0,0,0.48);
        animation: fadeSlideIn 0.5s cubic-bezier(0.16,1,0.3,1);
    }}
    .hero-sub {{ margin: 0; color: rgba(255,255,255,0.92); font-size: 1rem; line-height: 1.5; }}
    .section-title {{
        font-size: 2rem; font-weight: 800; color: {C["deep"]};
        margin: 0.65rem 0 0.2rem 0; line-height: 1.1;
        animation: fadeSlideIn 0.4s cubic-bezier(0.16,1,0.3,1);
    }}
    .section-sub {{ color: {C["muted"]}; font-size: 0.98rem; margin-bottom: 0.9rem; }}
    .kpi-card {{
        background: #fff; border: 1px solid {C["border"]}; border-radius: 18px;
        padding: 1rem 1rem 0.85rem; box-shadow: 0 8px 18px rgba(32,20,54,0.06); margin-bottom: 0.45rem;
        animation: fadeSlideIn 0.4s cubic-bezier(0.16,1,0.3,1);
    }}
    .kpi-label {{ font-size: 0.76rem; color: {C["purple"]}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.28rem; }}
    .kpi-value {{ font-size: 1.85rem; font-weight: 800; color: {C["deep"]}; line-height: 1; margin-bottom: 0.15rem; }}
    .kpi-sub {{ font-size: 0.82rem; color: {C["muted"]}; }}
    .section-panel {{
        background: {C["panel_bg"]}; border: 1px solid {C["border"]}; border-radius: 18px;
        padding: 1rem 1.15rem; box-shadow: 0 8px 18px rgba(32,20,54,0.05); margin-bottom: 0.95rem;
    }}
    .section-panel h3 {{ margin-top: 0; color: {C["purple"]}; font-size: 1.18rem; font-weight: 700; letter-spacing: -0.01em; }}
    .section-panel p {{ margin: 0 0 0.85rem 0; color: {C["muted"]}; line-height: 1.7; font-size: 0.96rem; }}
    .section-panel p:last-child {{ margin-bottom: 0; }}
    .analysis-box {{
        background: #FAF7FE; border: 1px solid {C["border"]}; border-radius: 14px;
        padding: 0.8rem 0.95rem; margin-top: 0.35rem; margin-bottom: 0.85rem;
        color: {C["muted"]}; font-size: 0.9rem; line-height: 1.55;
        animation: fadeSlideIn 0.4s cubic-bezier(0.16,1,0.3,1);
    }}
    .term-box {{
        background: #FAF7FE; border: 1px solid {C["border"]}; border-radius: 16px;
        padding: 0.9rem 1rem; margin-bottom: 0.8rem;
    }}
    .term-box b {{ color: {C["deep"]}; }}
    .term-box p {{ margin: 0.35rem 0; color: {C["muted"]}; line-height: 1.55; font-size: 0.92rem; }}
    .result-card {{
        background: #FFFFFF; border: 1px solid {C["border"]}; border-radius: 18px;
        padding: 1rem 1.1rem; margin-bottom: 0.8rem; box-shadow: 0 8px 18px rgba(32,20,54,0.05);
    }}
    .result-card h3 {{ margin: 0 0 0.55rem 0; color: {C["purple"]}; }}
    .result-card p {{ margin: 0; color: {C["muted"]}; line-height: 1.6; }}
    .file-link-card {{
        background: #FAF7FE; border: 1px solid {C["border"]}; border-radius: 14px;
        padding: 0.85rem 1.1rem; margin-bottom: 0.55rem; display: flex; align-items: center; gap: 0.7rem;
    }}
    .file-link-card a {{ color: {C["purple"]}; font-weight: 700; text-decoration: none; font-size: 0.95rem; }}
    .file-link-card a:hover {{ text-decoration: underline; }}
    .file-link-desc {{ color: {C["muted"]}; font-size: 0.85rem; margin-top: 0.15rem; }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ── HELPERS ──────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def panel(title: str, body_html: str) -> None:
    st.markdown(f"""
    <div class="section-panel">
        <h3>{title}</h3>
        {textwrap.dedent(body_html).strip()}
    </div>""", unsafe_allow_html=True)

def analysis_box(html: str) -> None:
    st.markdown(f'<div class="analysis-box">{textwrap.dedent(html).strip()}</div>', unsafe_allow_html=True)

def section_heading(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div class="section-title">{title}</div>
    <div class="section-sub">{subtitle}</div>""", unsafe_allow_html=True)

def show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ── DATA LOADING ─────────────────────────────────────────────
def make_unique_columns(cols) -> List[str]:
    out, seen = [], {}
    for idx, col in enumerate(cols):
        if col is None or (isinstance(col, float) and pd.isna(col)):
            base = f"Unnamed_{idx}"
        else:
            base = re.sub(r"\s+", " ", str(col).strip()) or f"Unnamed_{idx}"
        if base not in seen:
            seen[base] = 0; out.append(base)
        else:
            seen[base] += 1; out.append(f"{base}_{seen[base]}")
    return out

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = make_unique_columns(out.columns)
    return out.dropna(axis=1, how="all")

def read_xlsx_via_openpyxl(path: str, sheet_name=0) -> pd.DataFrame:
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb.worksheets[sheet_name] if isinstance(sheet_name, int) else wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    header_row_idx = next((i for i, r in enumerate(rows[:20]) if r and any(v is not None and str(v).strip() for v in r)), 0)
    header = make_unique_columns(rows[header_row_idx])
    return clean_table(pd.DataFrame(rows[header_row_idx + 1:], columns=header))

def read_sheet_raw_rows(path: str, sheet_name=0) -> List[List[Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True, read_only=True)
        ws = wb.worksheets[sheet_name] if isinstance(sheet_name, int) else wb[sheet_name]
        return [list(r) for r in ws.iter_rows(values_only=True)]
    if ext == ".xls":
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        return df.where(pd.notna(df), None).values.tolist()
    if ext == ".csv":
        df = pd.read_csv(path, header=None)
        return df.where(pd.notna(df), None).values.tolist()
    raise ValueError(f"Unsupported file type: {path}")

def read_table(path: str, sheet_name=0) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return clean_table(pd.read_csv(path))
    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return read_xlsx_via_openpyxl(path, sheet_name=sheet_name)
    if ext == ".xls":
        return clean_table(pd.read_excel(path, sheet_name=sheet_name))
    raise ValueError(f"Unsupported file type: {path}")

def resolve_files() -> Dict[str, Optional[str]]:
    return {
        "constz":       first_existing(["ConstantZLoad (Consolidated data).xlsx",
                                        "ConstantZLoad(Consolidated data).xlsx",
                                        "ConstantZLoad (Consolidated Data).xlsx",
                                        "ConstantZLoad (Consolidated data).csv",
                                        "ConstantZLoad.xlsx", "ConstantZLoad.csv"]),
        "consti":       first_existing(["ConstantILoad (Consolidated Data).xlsx",
                                        "ConstantILoad(Consolidated Data).xlsx",
                                        "ConstantILoad (Consolidated data).xlsx",
                                        "ConstantILoad (Consolidated Data).csv",
                                        "ConstantILoad.xlsx", "ConstantILoad.csv"]),
        "zip_main":     first_existing(["ZIPLoad.xlsx", "ZIPLoad.csv"]),
        "zip_analysis": first_existing(["ZIPLoad(Analysis).xlsx", "ZIPLoad(Analysis).csv"]),
        "ieee":         first_existing(["IEEE14busresults.xlsx"]),
        "cost_dx":      first_existing(["Final Cost Savings Analysis.xlsx",
                                        "Final Cost Savings Analysis OLD(Dx Feeder Cost Savings).xlsx"]),
        "cost_full":    first_existing(["Final Cost Savings Analysis.xlsx",
                                        "Final Cost Savings Analysis OLD.xlsx"]),
        "solar_farm":   first_existing(["Solar Farm Data(Tx Connected Solar Farms).xlsx",
                                        "Solar Farm Data(Tx Connected Solar Farms).csv",
                                        "Solar Farm Data (Tx Connected Solar Farms).xlsx",
                                        "Solar Farm Data (Tx Connected Solar Farms).csv"]),
        "proto":        first_existing([
                            "Capstone Prototype Data(Sheet1).xlsx",
                            "Capstone Prototype Data(Sheet1).csv",
                            "Capstone Prototype Data (Sheet1).xlsx",
                            "Capstone Prototype Data (Sheet1).csv",
                            "Capstone Prototype Data(Sheet1).xls",
                            "Capstone Prototype Data.xlsx",
                            "Capstone Prototype Data.csv",
                            "Capstone Prototype Data.xls",
                        ]),
        "video":        first_existing(["solar-energy-2026-01-21-12-26-38-utc.mp4"]),
        "img_dx":       first_existing(["Dx_Feeder_Image.png", "Dx_Feeder_Image.jpg",
                                        "Dx_Feeder_Image.jpeg", "Dx_Feeder_Image"]),
        "img_ieee":     first_existing(["IEEE14_Image.png", "IEEE14_Image.jpg",
                                        "IEEE14_Image.jpeg", "IEEE14_Image"]),
        "img_tx_moved": first_existing(["TransformerMoved.png", "TransformerMoved.jpg",
                                        "TransformerMoved.jpeg", "TransformerMoved"]),
    }

FILES = resolve_files()

@st.cache_data(show_spinner=False)
def load_data():
    required = ["constz", "zip_main", "ieee", "cost_dx", "cost_full"]
    missing = [k for k in required if FILES.get(k) is None]
    if missing:
        raise FileNotFoundError(
            f"Missing required files: {', '.join(missing)}\n"
            f"Expected in: {BASE_DIR}"
        )
    constz       = read_table(FILES["constz"])
    zip_df       = read_table(FILES["zip_main"])
    zip_analysis = read_table(FILES["zip_analysis"]) if FILES.get("zip_analysis") else pd.DataFrame()
    ieee         = read_table(FILES["ieee"])
    cost_dx      = read_table(FILES["cost_dx"])
    cost_full    = read_table(FILES["cost_full"])
    consti_raw   = read_table(FILES["consti"]) if FILES.get("consti") else pd.DataFrame()
    return constz, zip_df, zip_analysis, ieee, cost_dx, cost_full, consti_raw

# ── DATA PARSING ─────────────────────────────────────────────
def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")

def norm_text(series: pd.Series, fallback: str = "unknown") -> pd.Series:
    return series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).fillna(fallback)

def find_existing_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = name.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None

def prepare_constz(df: pd.DataFrame) -> pd.DataFrame:
    raw = clean_table(df)
    hour_col     = find_existing_col(raw, ["hour"])
    no_cvr_col   = find_existing_col(raw, ["load_mw_no_cvr", "mw no cvr", "load mw no cvr"])
    with_cvr_col = find_existing_col(raw, ["load_mw_with_cvr", "mw with cvr", "load mw with cvr"])
    red_col      = find_existing_col(raw, ["reduction_pct", "reduction %"])
    pv_bus_col   = find_existing_col(raw, ["pv_bus", "pv bus", "bus"])
    pf_col       = find_existing_col(raw, ["pf", "power factor"])
    v_no_col     = find_existing_col(raw, ["load_bus_v_no_cvr_pu", "voltage no cvr", "load bus v no cvr pu"])
    v_with_col   = find_existing_col(raw, ["load_bus_v_with_cvr_pu", "voltage with cvr", "load bus v with cvr pu"])
    pv_size_col  = find_existing_col(raw, ["pv_size_mva", "pv size mva", "pv size"])
    sun_col      = find_existing_col(raw, ["sun_rating", "sun rating"])

    if hour_col is None and raw.shape[1] >= 23:
        tmp = raw.copy()
        tmp.columns = [f"c{i}" for i in range(tmp.shape[1])]
        tmp = tmp.iloc[2:].reset_index(drop=True)
        out = pd.DataFrame({
            "hour": to_num(tmp["c0"]), "load_mw_no_cvr": to_num(tmp["c1"]), "pf": to_num(tmp["c3"]),
            "sun_rating": norm_text(tmp["c5"], "unknown"), "pv_size_mva": to_num(tmp["c6"]),
            "pv_bus": to_num(tmp["c8"]), "load_bus_v_no_cvr_pu": to_num(tmp["c11"]),
            "load_bus_v_with_cvr_pu": to_num(tmp["c15"]), "load_mw_with_cvr": to_num(tmp["c19"]),
            "reduction_pct": to_num(tmp["c22"]),
        })
        out = out.dropna(subset=["hour"])
    else:
        out = pd.DataFrame()
        if hour_col:     out["hour"]                 = to_num(raw[hour_col])
        if no_cvr_col:   out["load_mw_no_cvr"]       = to_num(raw[no_cvr_col])
        if with_cvr_col: out["load_mw_with_cvr"]     = to_num(raw[with_cvr_col])
        if red_col:      out["reduction_pct"]         = to_num(raw[red_col])
        if pv_bus_col:   out["pv_bus"]               = to_num(raw[pv_bus_col])
        if pf_col:       out["pf"]                   = to_num(raw[pf_col])
        if v_no_col:     out["load_bus_v_no_cvr_pu"] = to_num(raw[v_no_col])
        if v_with_col:   out["load_bus_v_with_cvr_pu"]= to_num(raw[v_with_col])
        if pv_size_col:  out["pv_size_mva"]           = to_num(raw[pv_size_col])
        if sun_col:      out["sun_rating"]            = norm_text(raw[sun_col], "unknown")

    if "reduction_pct" not in out.columns and {"load_mw_no_cvr","load_mw_with_cvr"}.issubset(out.columns):
        out["reduction_pct"] = np.where(
            out["load_mw_no_cvr"] != 0,
            100*(out["load_mw_no_cvr"]-out["load_mw_with_cvr"])/out["load_mw_no_cvr"], np.nan)

    if "pv_size_mva" not in out.columns: out["pv_size_mva"] = 1.0
    if "sun_rating"  not in out.columns: out["sun_rating"]  = "unknown"

    out = out.dropna(how="all")
    for col in ["hour","load_mw_no_cvr","load_mw_with_cvr","reduction_pct","pf","pv_bus","pv_size_mva"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sun_rating"] = norm_text(out["sun_rating"], "unknown")
    out = out.dropna(subset=[c for c in ["hour","load_mw_no_cvr","load_mw_with_cvr","pf","pv_bus"] if c in out.columns])
    if "hour"   in out.columns: out["hour"]   = out["hour"].astype(int)
    if "pv_bus" in out.columns: out["pv_bus"] = out["pv_bus"].astype(int)
    return out.reset_index(drop=True)

# ── BASE CHART LAYOUT ─────────────────────────────────────────
def base_layout(title: str, height: int = 320) -> Dict[str, Any]:
    return dict(
        title=f"<b>{title}</b>", template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
        height=height, margin=dict(l=20, r=20, t=58, b=24),
        font=dict(size=12, color=C["text"]),
        legend=dict(orientation="h", y=1.10, x=0),
    )

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED AI MODULE — physics-informed ML for CVR forecasting
# Trains on all 5,184 PSSE rows; predicts any PF 0.90-0.98 continuously
# ═══════════════════════════════════════════════════════════════════════════

# ── AI Constants ─────────────────────────────────────────────
AI_TOU_RATES: Dict[int, float] = {}
for _h in list(range(1, 8)) + list(range(20, 25)):
    AI_TOU_RATES[_h] = 9.8
for _h in list(range(8, 12)) + list(range(18, 20)):
    AI_TOU_RATES[_h] = 20.3
for _h in range(12, 18):
    AI_TOU_RATES[_h] = 15.7

AI_IESO_PCT = [
    75.47, 73.20, 72.01, 71.82, 73.25, 77.21, 83.31, 88.27,
    90.17, 90.96, 91.71, 92.45, 92.58, 92.51, 92.80, 94.69,
    98.03, 100.0, 99.69, 98.02, 95.25, 90.38, 84.31, 78.95,
]

AI_ANNUAL_SAVINGS: Dict[str, int] = {
    "Z": 484568, "I": 247210, "ZIP - Res": 257078, "ZIP - Comm": 243466,
}

AI_SUN_MAP  = {"very sunny": 0.0, "moderate sun": 0.5, "cloudy": 1.0}
AI_LT_MAP   = {"Z": 0, "I": 1, "ZIP - Res": 2, "ZIP - Comm": 3}
AI_LT_LABEL = {
    "Z":          "Constant-Z  (load ∝ V²)",
    "I":          "Constant-I  (load ∝ V)",
    "ZIP - Res":  "ZIP-Residential  (55% const-I)",
    "ZIP - Comm": "ZIP-Commercial   (33% each Z/I/P)",
}

AI_FEATURE_COLS = [
    "hour", "hsin", "hcos",
    "pf", "qaf", "qcap",
    "pv_bus", "pv_size",
    "sun_num", "lt_enc",
    "v_no", "mw_no",
]
AI_VNO_FEATURE_COLS = [
    "hour", "hsin", "hcos", "pf", "qaf", "pv_bus", "pv_size", "lt_enc", "mw_no",
]
AI_VNO_IDX = [AI_FEATURE_COLS.index(c) for c in AI_VNO_FEATURE_COLS]

AI_ALL_LOAD_TYPES  = ["Z", "I", "ZIP - Res", "ZIP - Comm"]
AI_ALL_PF_STUDY    = [0.90, 0.95, 0.98]
AI_ALL_PV_BUSES    = [3, 4, 5]
AI_ALL_PV_SIZES    = [5.263, 10.526]
AI_ALL_SUN_RATINGS = ["very sunny", "moderate sun", "cloudy"]

AI_ET_WEIGHT   = 0.60
AI_RF_WEIGHT   = 0.40
AI_RAND_STATE  = 42
AI_MIN_RED_PCT = 2.0
AI_MIN_V_PU    = 0.95
AI_MAX_V_PU    = 1.05
AI_STUDY_PEAK  = 10.0
AI_WEATHER_LAT = 42.9849
AI_WEATHER_LON = -81.2453
AI_WEATHER_TO  = 20

WEATHER_CODE_MAP = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 51: "Light drizzle", 53: "Moderate drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 80: "Slight showers",
    81: "Moderate showers", 95: "Thunderstorm",
}

# ── AI Physics helpers ────────────────────────────────────────
def _ai_qaf(pf: np.ndarray) -> np.ndarray:
    pf = np.clip(np.asarray(pf, dtype=float), 0.90, 0.98)
    return np.sqrt(np.clip(1.0 - pf**2, 0.0, 1.0)) / pf

def _ai_qcap(pf: np.ndarray, pv_size: np.ndarray) -> np.ndarray:
    pf = np.clip(np.asarray(pf, dtype=float), 0.90, 0.98)
    return np.asarray(pv_size, dtype=float) * np.sqrt(np.clip(1.0 - pf**2, 0.0, 1.0))

def _ai_add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pf        = df["pf"].clip(0.90, 0.98).values
    df["qaf"] = _ai_qaf(pf)
    df["qcap"]= _ai_qcap(pf, df["pv_size"].values)
    h         = df["hour"].values
    df["hsin"]= np.sin(2*np.pi*h/24.0)
    df["hcos"]= np.cos(2*np.pi*h/24.0)
    return df

# ── AI Data loading ───────────────────────────────────────────
def _ai_find_file(names: List[str]) -> Optional[str]:
    for n in names:
        fp = os.path.join(BASE_DIR, n)
        if os.path.exists(fp):
            return fp
    return None

def ai_load_training_dataframe() -> pd.DataFrame:
    path_all = _ai_find_file(["AllResults.xlsx"])
    path_td  = _ai_find_file(["TrainingData.xlsx"])

    if path_all:
        from openpyxl import load_workbook as _lw
        wb  = _lw(path_all, read_only=True, data_only=True)
        ws  = wb["AllLoadData"]
        raw = list(ws.iter_rows(values_only=True))
        df  = pd.DataFrame(raw[1:], columns=raw[0]).dropna(subset=["hour"])
        col_map = {
            "hour": "hour", "load MW": "mw_no", "load MW CVR": "mw_cvr",
            "PF": "pf", "PV_size (MVA)": "pv_size",
            "Load bus (5) pu (no CVR)": "v_no", "Load bus (5) pu (CVR)": "v_cvr",
        }
        for src, dst in col_map.items():
            df[dst] = pd.to_numeric(df[src], errors="coerce")
        df["pv_bus"]     = pd.to_numeric(df["PV bus #"], errors="coerce").astype(int)
        df["load_type"]  = df["load type"].astype(str).str.strip()
        df["sun_rating"] = df["sun rating"].astype(str).str.strip()

    elif path_td:
        from openpyxl import load_workbook as _lw
        wb  = _lw(path_td, read_only=True, data_only=True)
        ws  = wb.worksheets[0]
        raw = list(ws.iter_rows(values_only=True))
        df  = pd.DataFrame(raw[1:], columns=raw[0]).dropna(subset=["hour","load MW","load type"])
        col_map = {
            "hour":"hour","load MW":"mw_no","load MW CVR":"mw_cvr","PF":"pf",
            "PV_size (MVA)":"pv_size",
            "Load bus (5) pu (no CVR)":"v_no","Load bus (5) pu (CVR)":"v_cvr",
        }
        for src, dst in col_map.items():
            df[dst] = pd.to_numeric(df[src], errors="coerce")
        df["pv_bus"]    = pd.to_numeric(df["PV bus #"], errors="coerce").astype(int)
        df["load_type"] = df["load type"].astype(str).str.strip()
        df["sun_rating"]= df["sun rating"].astype(str).str.strip()
    else:
        raise FileNotFoundError(
            f"AllResults.xlsx or TrainingData.xlsx not found in: {BASE_DIR}"
        )

    df["hour"] = df["hour"].astype(int)
    df["reduction_pct"] = np.where(
        df["mw_no"] > 0, (df["mw_no"]-df["mw_cvr"])/df["mw_no"]*100.0, 0.0)
    df["sun_num"]  = df["sun_rating"].map(AI_SUN_MAP).fillna(0.5)
    df["lt_enc"]   = df["load_type"].map(AI_LT_MAP).fillna(1)
    df["group_id"] = (
        df["load_type"] + "_" + df["pf"].round(3).astype(str) + "_" +
        df["pv_bus"].astype(str) + "_" + df["pv_size"].round(3).astype(str) +
        "_" + df["sun_rating"]
    )
    df = _ai_add_derived(df)
    return df.dropna(subset=["mw_no","mw_cvr","pf","v_no"]).reset_index(drop=True)

# ── AI Model training ─────────────────────────────────────────
def ai_train_models(df: pd.DataFrame) -> Dict[str, Any]:
    X      = df[AI_FEATURE_COLS].values
    y_red  = df["reduction_pct"].values
    y_vcvr = df["v_cvr"].values
    groups = df["group_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=AI_RAND_STATE)
    tr, te = next(gss.split(X, y_red, groups))

    et_r = ExtraTreesRegressor(n_estimators=300, max_depth=16, min_samples_leaf=1,
                                random_state=AI_RAND_STATE, n_jobs=-1)
    rf_r = RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=2,
                                  random_state=AI_RAND_STATE, n_jobs=-1)
    et_v = ExtraTreesRegressor(n_estimators=300, max_depth=16, min_samples_leaf=1,
                                random_state=AI_RAND_STATE, n_jobs=-1)
    rf_v = RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=2,
                                  random_state=AI_RAND_STATE, n_jobs=-1)

    et_r.fit(X[tr], y_red[tr]);   rf_r.fit(X[tr], y_red[tr])
    et_v.fit(X[tr], y_vcvr[tr]);  rf_v.fit(X[tr], y_vcvr[tr])

    p_r = AI_ET_WEIGHT*et_r.predict(X[te]) + AI_RF_WEIGHT*rf_r.predict(X[te])
    p_v = AI_ET_WEIGHT*et_v.predict(X[te]) + AI_RF_WEIGHT*rf_v.predict(X[te])

    metrics = {
        "r2_reduction":   float(r2_score(y_red[te],  p_r)),
        "mae_reduction":  float(mean_absolute_error(y_red[te],  p_r)),
        "rmse_reduction": float(np.sqrt(mean_squared_error(y_red[te], p_r))),
        "r2_vcvr":        float(r2_score(y_vcvr[te], p_v)),
        "mae_vcvr":       float(mean_absolute_error(y_vcvr[te], p_v)),
        "n_total":        int(len(df)),
        "n_groups_train": int(len(set(groups[tr]))),
        "n_groups_test":  int(len(set(groups[te]))),
    }

    # Retrain on full data
    et_r.fit(X, y_red);   rf_r.fit(X, y_red)
    et_v.fit(X, y_vcvr);  rf_v.fit(X, y_vcvr)

    # v_no sub-model
    Xvno  = df[AI_VNO_FEATURE_COLS].values
    et_vno = ExtraTreesRegressor(n_estimators=150, max_depth=14,
                                  random_state=AI_RAND_STATE, n_jobs=-1)
    et_vno.fit(Xvno, df["v_no"].values)

    return {"et_r": et_r, "rf_r": rf_r, "et_v": et_v, "rf_v": rf_v,
            "et_vno": et_vno, "metrics": metrics}

# ── AI Batch vectorized prediction ───────────────────────────
def _ai_mw_base(peak_mw: float, temp_arr: Optional[np.ndarray] = None) -> np.ndarray:
    ieso = np.array(AI_IESO_PCT)/100.0
    if temp_arr is not None:
        dev   = np.maximum(0.0, np.abs(temp_arr-15.0)-5.0)/20.0
        scale = 1.0 + 0.04*dev
    else:
        scale = np.ones(24)
    return peak_mw*ieso*scale

def _ai_build_batch(scenarios: List[Tuple], peak_mw: float,
                    temp_arr: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mw_base = _ai_mw_base(peak_mw, temp_arr)
    hours   = np.arange(1, 25)
    hsin    = np.sin(2*np.pi*hours/24.0)
    hcos    = np.cos(2*np.pi*hours/24.0)
    n_feats = len(AI_FEATURE_COLS)
    Xbig    = np.zeros((len(scenarios)*24, n_feats), dtype=np.float64)
    mw_all  = np.zeros(len(scenarios)*24, dtype=np.float64)

    fi = {c: AI_FEATURE_COLS.index(c) for c in AI_FEATURE_COLS}
    for i, (lt, pf, pv_bus, pv_size, sun_rating) in enumerate(scenarios):
        pf_c  = float(np.clip(pf, 0.90, 0.98))
        qaf_v = float(_ai_qaf(np.array([pf_c]))[0])
        qcap_v= float(_ai_qcap(np.array([pf_c]), np.array([pv_size]))[0])
        lt_e  = float(AI_LT_MAP.get(lt, 1))
        sun_n = float(AI_SUN_MAP.get(sun_rating, 0.5))
        sl    = slice(i*24, (i+1)*24)
        Xbig[sl, fi["hour"]]   = hours
        Xbig[sl, fi["hsin"]]   = hsin
        Xbig[sl, fi["hcos"]]   = hcos
        Xbig[sl, fi["pf"]]     = pf_c
        Xbig[sl, fi["qaf"]]    = qaf_v
        Xbig[sl, fi["qcap"]]   = qcap_v
        Xbig[sl, fi["pv_bus"]] = float(pv_bus)
        Xbig[sl, fi["pv_size"]]= pv_size
        Xbig[sl, fi["sun_num"]]= sun_n
        Xbig[sl, fi["lt_enc"]] = lt_e
        Xbig[sl, fi["mw_no"]]  = mw_base
        mw_all[sl]              = mw_base
    return Xbig, mw_all

def ai_batch_predict_rank(models: Dict, scenarios: List[Tuple],
                           peak_mw: float = 10.0,
                           temp_arr: Optional[np.ndarray] = None) -> pd.DataFrame:
    Xbig, mw_all = _ai_build_batch(scenarios, peak_mw, temp_arr)
    Xvno         = Xbig[:, AI_VNO_IDX]
    v_no_pred    = np.clip(models["et_vno"].predict(Xvno), 0.95, 1.06)
    Xbig[:, AI_FEATURE_COLS.index("v_no")] = v_no_pred

    pred_red  = np.clip(AI_ET_WEIGHT*models["et_r"].predict(Xbig) +
                        AI_RF_WEIGHT*models["rf_r"].predict(Xbig), 0.0, 15.0)
    pred_vcvr = np.clip(AI_ET_WEIGHT*models["et_v"].predict(Xbig) +
                        AI_RF_WEIGHT*models["rf_v"].predict(Xbig), 0.93, 1.06)

    tou_arr = np.array([AI_TOU_RATES.get(h, 9.8) for h in range(1, 25)])
    results = []
    for i, (lt, pf, pv_bus, pv_size, sun_rating) in enumerate(scenarios):
        sl       = slice(i*24, (i+1)*24)
        mw_b     = mw_all[sl]
        red_sc   = pred_red[sl]
        vcvr_sc  = pred_vcvr[sl]
        mw_saved = np.clip(mw_b*red_sc/100.0, 0.0, mw_b*0.15)
        d_base   = float(mw_b.sum())
        d_saved  = float(mw_saved.sum())
        d_pct    = 100.0*d_saved/d_base if d_base > 0 else 0.0
        min_v    = float(vcvr_sc.min())
        max_v    = float(vcvr_sc.max())
        d_cost   = float((mw_saved*1000.0*tou_arr/100.0).sum())
        feasible = (d_pct >= AI_MIN_RED_PCT and min_v >= AI_MIN_V_PU and max_v <= AI_MAX_V_PU)
        results.append({
            "load_type": lt, "pf": pf, "pv_bus": int(pv_bus),
            "pv_size_mva": pv_size, "sun_rating": sun_rating,
            "daily_base_mwh": d_base, "daily_saved_mwh": d_saved,
            "daily_reduction_pct": d_pct, "min_v_cvr_pu": min_v,
            "max_v_cvr_pu": max_v, "daily_cost_usd": d_cost, "feasible": feasible,
        })
    df = pd.DataFrame(results)
    def _norm(s):
        lo, hi = s.min(), s.max()
        return (s-lo)/(hi-lo+1e-9)
    df["score"] = (0.50*_norm(df["daily_saved_mwh"]) +
                   0.30*_norm(df["daily_reduction_pct"]) +
                   0.20*_norm(df["min_v_cvr_pu"]))
    df.loc[~df["feasible"], "score"] -= 100.0
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def ai_predict_24h(models: Dict, pf: float, pv_bus: int, pv_size: float,
                   load_type: str, sun_rating: str, peak_mw: float = 10.0,
                   forecast_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    temp_arr = np.full(24, 15.0)
    if forecast_df is not None and "temperature_c" in forecast_df.columns:
        for _, row in forecast_df.iterrows():
            h = int(row.get("hour", 0))
            if 1 <= h <= 24:
                temp_arr[h-1] = float(row.get("temperature_c", 15.0) or 15.0)

    scenario = [(load_type, pf, pv_bus, pv_size, sun_rating)]
    Xbig, mw_all = _ai_build_batch(scenario, peak_mw, temp_arr)
    Xvno  = Xbig[:, AI_VNO_IDX]
    v_no_pred = np.clip(models["et_vno"].predict(Xvno), 0.95, 1.06)
    Xbig[:, AI_FEATURE_COLS.index("v_no")] = v_no_pred

    pred_red  = np.clip(AI_ET_WEIGHT*models["et_r"].predict(Xbig) +
                        AI_RF_WEIGHT*models["rf_r"].predict(Xbig), 0.0, 15.0)
    pred_vcvr = np.clip(AI_ET_WEIGHT*models["et_v"].predict(Xbig) +
                        AI_RF_WEIGHT*models["rf_v"].predict(Xbig), 0.93, 1.06)

    mw_saved = np.clip(mw_all*pred_red/100.0, 0.0, mw_all*0.15)
    mw_cvr   = mw_all - mw_saved
    tou_arr  = np.array([AI_TOU_RATES.get(h, 9.8) for h in range(1, 25)])
    cost_arr = mw_saved*1000.0*tou_arr/100.0

    return pd.DataFrame({
        "hour":          np.arange(1, 25),
        "mw_no":         mw_all,
        "mw_cvr":        mw_cvr,
        "mw_reduction":  mw_saved,
        "reduction_pct": pred_red,
        "v_no":          v_no_pred,
        "v_cvr":         pred_vcvr,
        "cost_saved_usd":cost_arr,
        "tou_rate":      tou_arr,
    })

# ── AI Weather API ────────────────────────────────────────────
def _ai_cloud_to_sun(cloud_pct: float) -> str:
    if pd.isna(cloud_pct): return "moderate sun"
    c = float(cloud_pct)
    if c < 25:  return "very sunny"
    if c < 65:  return "moderate sun"
    return "cloudy"

@st.cache_data(show_spinner=False, ttl=3600)
def ai_fetch_weather() -> pd.DataFrame:
    params = {
        "latitude": AI_WEATHER_LAT, "longitude": AI_WEATHER_LON,
        "hourly": ",".join(["temperature_2m","relative_humidity_2m","precipitation",
                             "cloud_cover","wind_speed_10m","weather_code"]),
        "forecast_days": 3, "timezone": "auto",
        "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    try:
        resp   = requests.get("https://api.open-meteo.com/v1/forecast",
                              params=params, timeout=AI_WEATHER_TO)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
    except Exception as exc:
        st.warning(f"Weather API unavailable ({exc}). Using default moderate-day forecast.")
        return _ai_synthetic_weather()

    df = pd.DataFrame({
        "time":            hourly.get("time", []),
        "temperature_c":   hourly.get("temperature_2m", []),
        "humidity_pct":    hourly.get("relative_humidity_2m", []),
        "precip_mm":       hourly.get("precipitation", []),
        "cloud_cover_pct": hourly.get("cloud_cover", []),
        "wind_speed_kph":  hourly.get("wind_speed_10m", []),
        "weather_code":    hourly.get("weather_code", []),
    })
    if df.empty:
        return _ai_synthetic_weather()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["date"] = df["time"].dt.date
    dates       = sorted(df["date"].unique())
    target      = dates[1] if len(dates) >= 2 else dates[0]
    df          = df[df["date"] == target].copy()
    df["hour"]  = df["time"].dt.hour + 1
    df["weather_condition"] = df["weather_code"].apply(
        lambda c: WEATHER_CODE_MAP.get(int(c), "Unknown") if pd.notna(c) else "Unknown"
    )
    df["sun_rating"]    = df["cloud_cover_pct"].apply(_ai_cloud_to_sun)
    df["forecast_date"] = str(target)

    if len(df) != 24:
        full = pd.DataFrame({"hour": range(1, 25)})
        df   = full.merge(df, on="hour", how="left")
        for col in ["temperature_c","humidity_pct","cloud_cover_pct","wind_speed_kph"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").interpolate().bfill().ffill()
        df["precip_mm"]     = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0)
        df["weather_code"]  = pd.to_numeric(df["weather_code"], errors="coerce").bfill().ffill().fillna(0).astype(int)
        df["weather_condition"] = df["weather_code"].apply(
            lambda c: WEATHER_CODE_MAP.get(int(c), "Unknown"))
        df["sun_rating"]    = df["cloud_cover_pct"].apply(_ai_cloud_to_sun)
        df["forecast_date"] = str(target)
    return df.sort_values("hour").reset_index(drop=True)

def _ai_synthetic_weather() -> pd.DataFrame:
    return pd.DataFrame({
        "hour":             range(1, 25),
        "temperature_c":    [8,7,7,6,6,7,9,11,13,15,16,17,17,17,16,15,14,13,12,11,10,10,9,8],
        "humidity_pct":     [75]*24,
        "precip_mm":        [0.0]*24,
        "cloud_cover_pct":  [40]*24,
        "wind_speed_kph":   [15]*24,
        "weather_code":     [2]*24,
        "weather_condition":["Partly cloudy"]*24,
        "sun_rating":       ["moderate sun"]*24,
        "forecast_date":    ["N/A"]*24,
    })

# ── AI Main pipeline ──────────────────────────────────────────
@st.cache_data(show_spinner="Training CVR model on 5,184 PSSE cases…", ttl=3600)
def ai_build_predictions(peak_mw: float = 10.0):
    train_df    = ai_load_training_dataframe()
    models      = ai_train_models(train_df)
    forecast_df = ai_fetch_weather()

    daytime     = forecast_df[forecast_df["hour"].between(8, 18)]
    dominant_sun = (daytime["sun_rating"].mode().iloc[0]
                    if not daytime.empty else "moderate sun")

    temp_arr = np.full(24, 15.0)
    if "temperature_c" in forecast_df.columns:
        for _, row in forecast_df.iterrows():
            h = int(row["hour"])
            if 1 <= h <= 24:
                temp_arr[h-1] = float(row.get("temperature_c", 15.0) or 15.0)

    all_scenarios = [
        (lt, pf, bus, sz, sun)
        for lt  in AI_ALL_LOAD_TYPES
        for pf  in AI_ALL_PF_STUDY
        for bus in AI_ALL_PV_BUSES
        for sz  in AI_ALL_PV_SIZES
        for sun in AI_ALL_SUN_RATINGS
    ]
    summary_df = ai_batch_predict_rank(models, all_scenarios, peak_mw, temp_arr)

    feas      = summary_df[summary_df["feasible"]]
    best_row  = feas.iloc[0] if not feas.empty else summary_df.iloc[0]
    best_pred = ai_predict_24h(
        models, pf=float(best_row["pf"]), pv_bus=int(best_row["pv_bus"]),
        pv_size=float(best_row["pv_size_mva"]),
        load_type=str(best_row["load_type"]), sun_rating=str(best_row["sun_rating"]),
        peak_mw=peak_mw, forecast_df=forecast_df,
    )
    return forecast_df, train_df, models, best_pred, summary_df, dominant_sun

# ── AI Charts ─────────────────────────────────────────────────
def _ai_bl(title: str, height: int = 320) -> Dict:
    return dict(
        title=f"<b>{title}</b>", template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
        height=height, margin=dict(l=20, r=20, t=58, b=24),
        font=dict(size=12, color=C["text"]),
        legend=dict(orientation="h", y=1.10, x=0),
    )

def ai_ch_load(df24: pd.DataFrame, title: str = "Feeder Load With / Without CVR") -> go.Figure:
    f = go.Figure()
    f.add_trace(go.Scatter(x=df24["hour"], y=df24["mw_no"], name="Without CVR",
        mode="lines+markers", line=dict(color=C["blue"], width=3.5), marker=dict(size=6)))
    f.add_trace(go.Scatter(x=df24["hour"], y=df24["mw_cvr"], name="With CVR",
        mode="lines+markers", line=dict(color=C["good"], width=3.5, dash="dash"),
        marker=dict(size=6), fill="tonexty", fillcolor="rgba(44,182,125,0.12)"))
    f.update_layout(**_ai_bl(title))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="MW")
    return f

def ai_ch_voltage(df24: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    f.add_trace(go.Scatter(x=df24["hour"], y=df24["v_cvr"], name="With-CVR Bus Voltage",
        mode="lines+markers", line=dict(color=C["gold"], width=3), marker=dict(size=5)))
    for val, lbl, col in [(1.05,"Max 1.05 pu",C["warn"]),(0.97,"Target 0.97 pu",C["warn"]),(0.95,"Min 0.95 pu",C["bad"])]:
        f.add_hline(y=val, line_dash="dot", line_color=col,
                    annotation_text=lbl, annotation_font_size=10)
    lay = _ai_bl("Predicted With-CVR Bus Voltage")
    lay["yaxis"] = {"range": [0.93, 1.08], "title": "Voltage (pu)"}
    f.update_layout(**lay)
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    return f

def ai_ch_reduction(df24: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    f.add_trace(go.Bar(x=df24["hour"], y=df24["reduction_pct"],
        marker_color=C["purple"], opacity=0.88, name="% Reduction"))
    f.add_hline(y=2.0, line_dash="dot", line_color=C["gold"],
                annotation_text="2% target", annotation_font_size=10)
    f.update_layout(**_ai_bl("Hourly CVR % MW Reduction"))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="% Reduction")
    return f

def ai_ch_mw_saved(df24: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    f.add_trace(go.Bar(x=df24["hour"], y=df24["mw_reduction"],
        marker_color=C["teal"], opacity=0.85, name="MW Saved"))
    f.update_layout(**_ai_bl("Hourly MW Saved by CVR"))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="MW Saved")
    return f

def ai_ch_cost(df24: pd.DataFrame) -> go.Figure:
    f = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = [C["bad"] if r==20.3 else C["gold"] if r==15.7 else C["blue"]
                  for r in df24["tou_rate"]]
    f.add_trace(go.Bar(x=df24["hour"], y=df24["cost_saved_usd"], marker_color=bar_colors,
        opacity=0.87, name="$/hr Saved",
        text=[f"${v:.1f}" for v in df24["cost_saved_usd"]],
        textposition="outside", textfont=dict(size=8)), secondary_y=False)
    f.add_trace(go.Scatter(x=df24["hour"], y=df24["tou_rate"], name="TOU Rate (¢/kWh)",
        mode="lines+markers", line=dict(color=C["deep"], width=2, dash="dot"),
        marker=dict(size=4)), secondary_y=True)
    f.update_layout(**_ai_bl("Hourly Cost Savings (Ontario TOU)", height=360))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="$/hr Saved", secondary_y=False)
    f.update_yaxes(title="TOU Rate (¢/kWh)", secondary_y=True, showgrid=False)
    return f

def ai_ch_cumul_cost(df24: pd.DataFrame) -> go.Figure:
    cumul = df24["cost_saved_usd"].cumsum()
    f = go.Figure()
    f.add_trace(go.Scatter(x=df24["hour"], y=cumul, mode="lines+markers",
        line=dict(color=C["purple"], width=3.5), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(184,108,224,0.09)", name="Cumulative $ Saved"))
    for h0, h1 in [(8,11),(18,19)]:
        f.add_vrect(x0=h0-0.5, x1=h1+0.5, fillcolor="rgba(230,57,70,0.07)", line_width=0)
    f.update_layout(**_ai_bl("Cumulative Daily Cost Savings", height=360))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="Cumulative $ Saved")
    return f

def ai_ch_pf_curve(models: Dict, pv_bus: int, pv_size: float,
                   load_type: str, sun_rating: str, peak_mw: float) -> go.Figure:
    pf_range = np.round(np.linspace(0.90, 0.98, 33), 4)
    avgs, h18s = [], []
    for pf_t in pf_range:
        d = ai_predict_24h(models, pf=pf_t, pv_bus=pv_bus, pv_size=pv_size,
                           load_type=load_type, sun_rating=sun_rating, peak_mw=peak_mw)
        avgs.append(float(d["reduction_pct"].mean()))
        h18s.append(float(d[d["hour"]==18]["reduction_pct"].values[0]))
    f = go.Figure()
    f.add_trace(go.Scatter(x=pf_range, y=avgs, mode="lines+markers",
        line=dict(color=C["blue"], width=2.5), marker=dict(size=5), name="Daily Avg %"))
    f.add_trace(go.Scatter(x=pf_range, y=h18s, mode="lines+markers",
        line=dict(color=C["purple"], width=2.5, dash="dash"), marker=dict(size=5), name="Hour-18 %"))
    for pf_s in AI_ALL_PF_STUDY:
        f.add_vline(x=pf_s, line_dash="dot", line_color=C["muted"],
                    annotation_text=f"PF {pf_s}", annotation_font_size=9)
    f.add_hline(y=2.0, line_dash="dot", line_color=C["gold"],
                annotation_text="2% target", annotation_font_size=10)
    f.update_layout(**_ai_bl("CVR % Reduction vs Power Factor — Continuous Interpolation", height=340))
    f.update_xaxes(title="Power Factor", tickformat=".2f", dtick=0.01)
    f.update_yaxes(title="Predicted % Reduction")
    return f

def ai_ch_ranking(summary_df: pd.DataFrame) -> go.Figure:
    top    = summary_df.head(30)
    labels = [f"{r['load_type']}<br>PF{r['pf']:.2f}/B{int(r['pv_bus'])}/{r['sun_rating'][:3]}"
              for _, r in top.iterrows()]
    colors = [C["purple"] if bool(v) else C["bad"] for v in top["feasible"]]
    y_max  = top["daily_saved_mwh"].max()*1.30 if len(top) > 0 else 20
    f      = go.Figure()
    f.add_trace(go.Bar(x=labels, y=top["daily_saved_mwh"], marker_color=colors, opacity=0.88,
        text=[f"{v:.2f}" for v in top["daily_saved_mwh"]],
        textposition="outside", textfont=dict(size=8), name="Daily MWh Saved"))
    lay = _ai_bl("Top-30 Scenario Ranking — All 216 Evaluated", height=420)
    lay["yaxis"] = {"title": "Daily MWh Saved", "range": [0, y_max]}
    f.update_layout(**lay)
    f.update_xaxes(title="Load Type · PF · Bus · Sun")
    return f

def ai_ch_weather(fdf: pd.DataFrame) -> go.Figure:
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=fdf["hour"], y=fdf["precip_mm"],
        name="Precipitation (mm)", marker_color=C["blue"], opacity=0.65), secondary_y=False)
    f.add_trace(go.Scatter(x=fdf["hour"], y=fdf["temperature_c"],
        name="Temperature (°C)", mode="lines+markers",
        line=dict(color=C["orange"], width=3), marker=dict(size=5)), secondary_y=True)
    f.update_layout(**_ai_bl("London, ON — Tomorrow's Forecast"))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="Precipitation (mm)", secondary_y=False)
    f.update_yaxes(title="Temperature (°C)", secondary_y=True)
    return f

def ai_ch_cloud_wind(fdf: pd.DataFrame) -> go.Figure:
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=fdf["hour"], y=fdf["cloud_cover_pct"],
        name="Cloud Cover (%)", marker_color=C["purple"], opacity=0.75), secondary_y=False)
    f.add_trace(go.Scatter(x=fdf["hour"], y=fdf["wind_speed_kph"],
        name="Wind Speed (km/h)", mode="lines+markers",
        line=dict(color=C["teal"], width=3), marker=dict(size=5)), secondary_y=True)
    for t, l in [(25,"< 25% → Very Sunny"),(65,"< 65% → Moderate Sun")]:
        f.add_hline(y=t, line_dash="dot", line_color=C["gold"],
                    annotation_text=l, annotation_font_size=9, secondary_y=False)
    f.update_layout(**_ai_bl("Cloud Cover & Wind Speed"))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="Cloud Cover (%)", secondary_y=False)
    f.update_yaxes(title="Wind Speed (km/h)", secondary_y=True)
    return f

# ═══════════════════════════════════════════════════════════════════════════
# IEEE 14-BUS HARDCODED SCENARIO DATA
# ═══════════════════════════════════════════════════════════════════════════
IEEE_HOURS = list(range(1, 25))
_SA_BUS4  = [2.9615,2.9586,2.9472,2.9462,2.9605,2.9633,3.0685,3.1915,3.1334,3.0021,2.8604,2.7859,2.8326,2.8802,3.0317,3.2332,3.4477,3.5502,3.5528,3.5139,3.4272,3.254,3.0429,2.9637]
_SA_BUS9  = [2.7708,2.7696,2.765,2.7646,2.7704,2.7714,2.8146,2.8512,2.7767,2.6486,2.5144,2.4412,2.4782,2.5189,2.6449,2.7951,2.9387,3.0109,3.0195,3.0036,2.9671,2.8941,2.8049,2.7715]
_SA_BUS14 = [3.5294,3.5266,3.5227,3.5223,3.5276,3.5323,3.5647,3.5756,3.4755,3.3161,3.1506,3.0593,3.1018,3.15,3.2977,3.4653,3.6168,3.6952,3.7083,3.6978,3.6724,3.6214,3.5591,3.533]
_SB_BUS4  = [4.7016,4.769,4.7959,4.7981,4.7547,4.6063,4.3737,4.2106,4.1638,4.1449,4.095,3.9056,3.962,4.0534,4.0773,4.0011,3.8851,3.8374,3.8513,3.8956,3.9907,4.18,4.403,4.5846]
_SB_BUS9  = [1.8307,1.8611,1.8729,1.8739,1.8549,1.7914,1.6911,1.6212,1.6012,1.5923,1.5721,1.4926,1.5161,1.5544,1.5636,1.5317,1.4823,1.462,1.4679,1.4868,1.5273,1.6081,1.7037,1.782]
_SB_BUS14 = [1.1416,1.1622,1.1699,1.1705,1.1582,1.1186,1.0548,1.0104,0.9977,0.9912,0.9794,0.9298,0.9444,0.9682,0.9731,0.9536,0.9223,0.9095,0.9132,0.9252,0.9508,1.0021,1.0628,1.1127]
_SC_BUS4  = [1.0568,1.0553,1.0447,1.0438,1.0569,1.0561,1.0539,1.0428,1.0131,0.9721,0.9303,0.9066,0.9133,0.9249,0.9572,0.9928,1.0355,1.1025,1.0965,1.0576,1.0511,1.0534,1.0548,1.056]
_SC_BUS9  = [1.9706,1.9695,1.965,1.9646,1.9702,1.9708,1.9693,1.9429,1.8647,1.7568,1.6464,1.5839,1.6017,1.6325,1.7181,1.8164,1.9098,1.9813,1.9898,1.9741,1.9709,1.9712,1.9713,1.9708]
_SC_BUS14 = [3.0217,3.0187,3.0148,3.0144,3.0196,3.0248,3.0291,2.9926,2.8729,2.7067,2.5364,2.4403,2.4681,2.5157,2.6483,2.8025,2.9464,3.0401,3.057,3.0467,3.0421,3.0371,3.0313,3.0256]

def _ieee_chart(title, bus4, bus4_label, bus9, bus9_label, bus14, bus14_label):
    f = go.Figure()
    f.add_trace(go.Scatter(x=IEEE_HOURS, y=bus14, name=bus14_label, mode="lines",
        line=dict(color=C["pink"], width=2.8), showlegend=False))
    f.add_annotation(x=24, y=bus14[-1], text=f"<b>{bus14_label}</b>",
        xanchor="left", showarrow=False, font=dict(color=C["pink"], size=11), xshift=6)
    f.add_trace(go.Scatter(x=IEEE_HOURS, y=bus4, name=bus4_label, mode="lines",
        line=dict(color=C["purple"], width=2.8), showlegend=False))
    f.add_annotation(x=24, y=bus4[-1], text=f"<b>{bus4_label}</b>",
        xanchor="left", showarrow=False, font=dict(color=C["purple"], size=11), xshift=6)
    f.add_trace(go.Scatter(x=IEEE_HOURS, y=bus9, name=bus9_label, mode="lines",
        line=dict(color=C["blue"], width=2.8), showlegend=False))
    f.add_annotation(x=24, y=bus9[-1], text=f"<b>{bus9_label}</b>",
        xanchor="left", showarrow=False, font=dict(color=C["blue"], size=11), xshift=6)
    f.add_shape(type="line", x0=1, x1=24, y0=2.0, y1=2.0,
        line=dict(color=C["gold"], width=1.5, dash="dot"))
    f.add_annotation(x=12, y=2.0, text="<b>2% target</b>",
        showarrow=False, yanchor="bottom", yshift=4, font=dict(color=C["gold"], size=11))
    f.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)", height=360, showlegend=False,
        margin=dict(l=20, r=120, t=90, b=40), font=dict(size=12, color=C["text"]),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                   font=dict(size=15, color=C["deep"])))
    f.update_xaxes(title="Hour of Day", tickvals=list(range(1, 25, 2)))
    f.update_yaxes(title="% MW Reduction")
    return f

def chart_ieee_scenario1():
    return _ieee_chart(
        "% MW Reduction in Loads — One Medium & 2 Small PV Farms<br>"
        "<sup>PV Farm Sizes: Bus 4 = 52.632 MVAR, Bus 9 = 10.526 MVAR, Bus 14 = 10.526 MVAR</sup>",
        _SA_BUS4,"Bus 4 Load",_SA_BUS9,"Bus 9 Load",_SA_BUS14,"Bus 14 Load")

def chart_ieee_scenario2():
    return _ieee_chart(
        "% MW Reduction in Loads — One Large PV Farm<br>"
        "<sup>PV Farm Sizes: Bus 4 = 105.263 MVAR</sup>",
        _SB_BUS4,"Bus 4 Load",_SB_BUS9,"Bus 9 Load",_SB_BUS14,"Bus 14 Load")

def chart_ieee_scenario3():
    return _ieee_chart(
        "% MW Reduction in Loads — Three Small PV Farms<br>"
        "<sup>PV Farm Sizes: Bus 4 = 10.526 MVAR, Bus 9 = 10.526 MVAR, Bus 14 = 10.526 MVAR</sup>",
        _SC_BUS4,"Bus 4 Load",_SC_BUS9,"Bus 9 Load",_SC_BUS14,"Bus 14 Load")

# ── Image helper ──────────────────────────────────────────────
def render_image(file_key: str, caption: str = "", max_width: str = "100%") -> None:
    path = FILES.get(file_key)
    if path is None or not os.path.exists(str(path)):
        base_candidates = {
            "img_dx":       "Dx_Feeder_Image",
            "img_ieee":     "IEEE14_Image",
            "img_tx_moved": "TransformerMoved",
        }
        base = base_candidates.get(file_key, "")
        if base:
            for ext in [".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"]:
                candidate = p(base+ext)
                if os.path.exists(candidate):
                    path = candidate; break
            if path is None or not os.path.exists(str(path)):
                no_ext = p(base)
                if os.path.exists(no_ext):
                    path = no_ext
    if path is None or not os.path.exists(str(path)):
        return
    ext  = os.path.splitext(str(path))[1].lower()
    mime = "image/jpeg" if ext in {".jpg",".jpeg"} else "image/png"
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        cap_html = f'<div style="font-size:0.82rem;color:{C["muted"]};text-align:center;margin-top:0.4rem;">{caption}</div>' if caption else ""
        st.markdown(f"""
        <div style="text-align:center;margin:0.6rem 0 0.2rem 0;">
            <img src="data:{mime};base64,{b64}"
                 style="max-width:{max_width};border-radius:14px;border:1px solid {C["border"]};
                        box-shadow:0 6px 18px rgba(32,20,54,0.10);" />
            {cap_html}
        </div>""", unsafe_allow_html=True)
    except Exception:
        pass

# ── Prototype helpers ─────────────────────────────────────────
def resolve_prototype_file() -> Optional[str]:
    cached = FILES.get("proto")
    if cached and os.path.exists(cached):
        return cached
    candidates = [
        "Capstone Prototype Data(Sheet1).xlsx","Capstone Prototype Data(Sheet1).xls",
        "Capstone Prototype Data (Sheet1).xlsx","Capstone Prototype Data (Sheet1).xls",
        "Capstone Prototype Data.xlsx","Capstone Prototype Data.xls","Capstone Prototype Data.csv",
    ]
    for name in candidates:
        full = p(name)
        if os.path.exists(full):
            return full
    try:
        import glob
        matches = glob.glob(os.path.join(BASE_DIR, "Capstone Prototype*"))
        if matches: return matches[0]
    except Exception:
        pass
    return None

def load_prototype_data() -> pd.DataFrame:
    proto_path = resolve_prototype_file()
    if proto_path is None:
        raise FileNotFoundError(f"Prototype file not found. Searched in: {BASE_DIR}.")
    rows = read_sheet_raw_rows(proto_path)
    if not rows:
        raise ValueError("Prototype file is empty.")
    parsed_rows, current_scenario, current_pv_location = [], None, None
    for row in rows:
        if not row: continue
        cells = ["" if v is None else str(v).strip() for v in row]
        if not any(cells): continue
        first = cells[0]
        if "prototype" in first.lower():
            sm = re.search(r"(\d+\s*V)", first, flags=re.I)
            lm = re.search(r"\((.*?)\)", first)
            current_scenario   = sm.group(1).upper().replace("  "," ") if sm else None
            current_pv_location= lm.group(1).strip() if lm else None
            continue
        lowered = [c.lower() for c in cells[:4]]
        if len(lowered) >= 4 and lowered[0] == "component" and lowered[1] == "value": continue
        component = first
        if component == "": continue
        value   = pd.to_numeric(cells[1] if len(cells)>1 else np.nan, errors="coerce")
        current = pd.to_numeric(cells[2] if len(cells)>2 else np.nan, errors="coerce")
        wr      = cells[3] if len(cells)>3 else np.nan
        wattage = np.nan if isinstance(wr,str) and wr.strip().lower()=="na" else pd.to_numeric(wr, errors="coerce")
        parsed_rows.append({"scenario":current_scenario,"pv_location":current_pv_location,
            "component":component,"value":value,"current":current,"wattage":wattage})
    out = pd.DataFrame(parsed_rows)
    if out.empty or out["component"].dropna().empty:
        raise ValueError("Prototype file could not be parsed.")
    out = out.dropna(subset=["scenario","pv_location"], how="any")
    out["component"] = out["component"].astype(str).str.strip()
    return out.reset_index(drop=True)

def chart_prototype_load_power(proto_df: pd.DataFrame) -> go.Figure:
    df = proto_df[proto_df["component"].str.lower()=="load r"].copy()
    df["case"] = df["scenario"]+" · "+df["pv_location"]
    f = go.Figure()
    palette = [C["purple"],C["orchid"],C["blue"],C["teal"]]
    vals = [v for v in df["wattage"] if pd.notna(v)]
    y_max = max(vals)*1.35 if vals else 1000
    f.add_trace(go.Bar(x=df["case"], y=df["wattage"], name="Load Power (W)",
        marker_color=palette[:len(df)],
        text=[f"{v:.2f} W" if pd.notna(v) else "" for v in df["wattage"]],
        textposition="outside", textfont=dict(size=11, color=C["text"])))
    lay = base_layout("Prototype Load Power by Configuration")
    lay["yaxis"] = {"title":"Load Power (W)","range":[0,y_max]}
    f.update_layout(**lay)
    f.update_xaxes(title="Prototype Case")
    return f

def chart_prototype_current_comparison(proto_df: pd.DataFrame) -> go.Figure:
    df = proto_df[proto_df["component"].isin(["Solar Farm R","Load R"])].copy()
    df["case"] = df["scenario"]+" · "+df["pv_location"]
    f = go.Figure()
    for comp, color in [("Solar Farm R",C["gold"]),("Load R",C["indigo"])]:
        sub = df[df["component"]==comp]
        f.add_trace(go.Bar(x=sub["case"], y=sub["current"], name=comp, marker_color=color))
    f.update_layout(**base_layout("Prototype Current Comparison"), barmode="group")
    f.update_xaxes(title="Prototype Case"); f.update_yaxes(title="Current (A)")
    return f

def chart_prototype_line_losses(proto_df: pd.DataFrame) -> go.Figure:
    df = proto_df[proto_df["component"].isin(["Tx Line 1 R","Tx Line 2 R"])].copy()
    df["case"] = df["scenario"]+" · "+df["pv_location"]
    f = go.Figure()
    for comp, color in [("Tx Line 1 R",C["orange"]),("Tx Line 2 R",C["pink"])]:
        sub = df[df["component"]==comp]
        f.add_trace(go.Bar(x=sub["case"], y=sub["wattage"], name=comp, marker_color=color))
    f.update_layout(**base_layout("Transformer-Line Resistive Losses"), barmode="group")
    f.update_xaxes(title="Prototype Case"); f.update_yaxes(title="Wattage (W)")
    return f

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HERO / ABOUT
# ═══════════════════════════════════════════════════════════════════════════
def render_hero() -> None:
    video_path = FILES.get("video")
    if video_path and os.path.exists(video_path):
        try:
            with open(video_path, "rb") as vf:
                video_b64 = base64.b64encode(vf.read()).decode()
            st.markdown(f"""
            <div class="hero-wrap">
                <video autoplay muted loop playsinline style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;">
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
                <div class="hero-overlay">
                    <div>
                        <div class="hero-tag">ECE 4416 Capstone · Group 4 · Western University</div>
                        <h1 class="hero-title">Implementation of Conservation Voltage Reduction (CVR) with PV Farm Inverters as a Strategy to Lower Demand</h1>
                        <p class="hero-sub">Using reactive power from PV farm inverters to reduce feeder demand — ECE 4416 Capstone, Group 4</p>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            return
        except Exception:
            pass
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-overlay">
            <div>
                <div class="hero-tag">ECE 4416 Capstone · Group 4 · Western University</div>
                <h1 class="hero-title">Implementation of Conservation Voltage Reduction (CVR) with PV Farm Inverters as a Strategy to Lower Demand</h1>
                <p class="hero-sub">Using reactive power from PV farm inverters to reduce feeder demand — ECE 4416 Capstone, Group 4</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def page_about():
    render_hero()
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""<div class="result-card"><h3>Problem</h3>
        <p>Ontario electricity demand is growing, creating pressure on the grid during peak periods.
        Solar PV farms connected to the grid are typically used only for active power generation —
        their reactive power capability is largely unused. This project investigates whether PV
        inverter reactive power can implement <b>Conservation Voltage Reduction (CVR)</b>
        to safely reduce peak demand.</p></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="result-card"><h3>Solution</h3>
        <p>PV farm inverters absorb reactive power to lower bus voltage toward <b>0.97 pu</b>
        while staying within 0.95–1.05 pu (ANSI C84.1). When load voltage drops, load power
        drops — no hardware upgrades needed, just smart inverter control. Studied on a modified
        Dx distribution feeder and the IEEE 14-bus transmission system. Validated with an AI
        surrogate model and a hardware bench prototype.</p></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div class="result-card"><h3>Results</h3>
        <p>Dx feeder: <b>2.94% average demand reduction</b> across 5,184 simulation cases.<br>
        IEEE 14-bus: <b>2.44% average reduction</b> across 168 cases.<br>
        Both exceed the 2% design requirement. All cases maintained voltage within the safe
        operating band. At 10 MW peak, 2.94% = <b>294 kW saved</b> — equivalent to
        ~275 Ontario homes at peak.</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    panel("Why Does This Matter?", f"""
    <p><b>CVR is one of the cheapest demand-reduction tools available</b> — no new customer equipment,
    no behaviour change, no service interruption. Lower the voltage slightly and most loads use less power.</p>
    <p>Solar farms already being built in Ontario have <b>spare inverter capacity on cloudy days</b>
    that currently goes unused. This project shows that capacity can be used for CVR — for free.</p>
    <p><b>At scale:</b> If 10% of Ontario feeders near solar farms adopted CVR at 2.94%:
    ~<b>140 MW</b> of peak demand reduction — equivalent to a peaking gas plant.
    At the feeder level: up to <b>$484,568/year</b> in savings (Constant-Z, 10 MW peak, Ontario TOU rates).</p>
    """)
    panel("Faculty Advisors and Course", """
    <p>This project was completed as part of <b>ECE 4416 Electrical Engineering Design Project</b>
    at the Department of Electrical and Computer Engineering, Western University,
    London, Ontario, Canada, March 2026.</p>
    <p>Faculty Advisors: <b>Dr. Varma, P.Eng.</b> and <b>Dr. Michaelson</b>.</p>
    <p>The simulation platform used was <b>PSSE</b> (Power System Simulation for Engineering),
    with Python automation for running all 5,184 Dx and 168 IEEE 14-bus cases.
    The AI surrogate model and this dashboard were built using Python —
    scikit-learn (ML), Streamlit (web app), Plotly (charts), and Open-Meteo (weather API).</p>
    <p><b>References:</b> ANSI C84.1 (voltage standards) · IEEE Standard 2800-2022 (inverter requirements)
    · IESO 2024 hourly demand data · Mahendru &amp; Varma (2019) IEEE CVR paper
    · Ontario Energy Board TOU rate schedule.</p>
    """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DX FEEDER RESULTS
# ═══════════════════════════════════════════════════════════════════════════
def page_dx_results(constz_raw, constz, cost_dx, cost_full):
    section_heading(
        "Dx Distribution Feeder — Study Results",
        "5,184 PSSE simulation cases — 4 load types × 3 PF × 3 PV buses × 2 PV sizes × 3 sun ratings × 24 hours. "
        "10 MW peak load, CVR target 0.97 pu, safe band 0.95–1.05 pu (ANSI C84.1)."
    )
    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Average CVR Reduction","2.94%","Grand average across all 5,184 cases")
    with k2: kpi("Total Cases","5,184","All parameter combinations × 24 hours")
    with k3: kpi("CVR Target Voltage","0.97 pu","0.02 pu safety margin above 0.95 pu minimum")
    with k4: kpi("Best Case","7.96%","Constant-Z · PF 0.98 · Bus 5 · 10.526 MVA · Cloudy")
    render_image("img_dx","Modified Dx Feeder: Bus 5 = load bus, PV farm connected at Bus 3, 4, or 5",max_width="90%")
    panel("Key Takeaways — Dx Feeder", f"""
    <p>The Dx feeder study confirmed: <b>CVR using PV inverter reactive power is effective and safe</b> across all 5,184 tested cases.</p>
    <p><b>Key findings:</b><br>
    &#8226; Average CVR reduction: <b>2.94%</b> — exceeds the 2% design requirement.<br>
    &#8226; Best case: <b>7.96%</b> (Constant-Z, PF 0.98, Bus 5, 10.526 MVA, Cloudy).<br>
    &#8226; PV at Bus 5 (load bus) always outperforms Bus 3 or Bus 4.<br>
    &#8226; Larger 10.526 MVA inverter consistently achieves greater reduction.<br>
    &#8226; Higher PF (0.98) gives nearly double the CVR benefit of PF 0.90.<br>
    &#8226; Cloudy conditions provide the most reactive headroom.<br>
    &#8226; All 5,184 cases maintained voltage within 0.95–1.05 pu.</p>
    """)
    panel("What These Results Show", """
    <p>The PV farm inverter absorbs reactive power to lower Bus 5 voltage to <b>0.97 pu</b>.
    Lower voltage → lower power consumption for most load types. Average reduction: <b>2.94%</b> — exceeding the 2% target.
    At 10 MW peak, 2.94% = <b>294 kW saved</b> continuously.</p>
    <p><b>Glossary:</b> <em>pu</em> = fraction of nominal voltage (1.0 pu = normal).
    <em>MVA</em> = apparent power capacity. <em>Power factor</em> = real/apparent power ratio.
    <em>Reactive power</em> = power needed to maintain voltage — no real work done, but essential.</p>
    """)

    _H = list(range(1,25))
    _LT_Z    = [5.4,5.833,5.867,5.87,5.818,5.315,5.105,4.573,4.474,4.27,4.015,3.947,3.91,3.93,3.974,3.952,3.875,3.843,3.861,3.905,4.0,4.557,5.133,5.295]
    _LT_I    = [2.789,3.02,3.038,3.039,2.813,2.746,2.636,2.364,2.13,2.086,2.05,2.014,2.022,2.035,2.051,2.042,1.995,1.981,1.988,2.011,2.061,2.296,2.653,2.737]
    _LT_RES  = [2.872,3.098,3.118,3.119,3.09,2.825,2.711,2.426,2.372,2.26,2.127,2.09,2.092,2.079,2.105,2.093,2.053,2.036,2.046,2.07,2.121,2.419,2.724,2.814]
    _LT_COMM = [2.738,2.957,2.975,2.977,2.949,2.692,2.58,2.308,2.203,2.07,2.024,1.967,1.969,1.979,2.001,1.99,1.949,1.933,1.942,1.965,2.014,2.3,2.598,2.682]
    _PV_B3   = [2.328,2.475,2.475,2.475,2.444,2.329,2.325,2.044,1.968,1.897,1.825,1.761,1.742,1.746,1.81,1.853,1.889,1.91,1.914,1.914,1.914,2.061,2.33,2.329]
    _PV_B4   = [3.386,3.946,3.946,3.945,3.828,3.388,3.393,2.975,2.788,2.559,2.341,2.314,2.323,2.336,2.368,2.394,2.417,2.427,2.428,2.428,2.427,2.945,3.394,3.389]
    _PV_B5   = [4.635,4.759,4.827,4.833,4.731,4.466,4.056,3.735,3.629,3.559,3.496,3.438,3.43,3.435,3.42,3.311,3.099,3.008,3.035,3.12,3.305,3.673,4.108,4.428]
    _PV_S    = [3.088,3.225,3.244,3.246,3.194,3.04,2.923,2.628,2.535,2.439,2.323,2.261,2.273,2.295,2.343,2.35,2.308,2.292,2.304,2.332,2.394,2.618,2.94,3.03]
    _PV_L    = [3.812,4.228,4.254,4.257,4.142,3.749,3.593,3.208,3.055,2.904,2.785,2.748,2.724,2.716,2.722,2.689,2.628,2.605,2.615,2.643,2.704,3.168,3.614,3.734]
    _PF_90   = [2.703,3.269,3.302,3.304,3.146,2.622,2.429,1.613,1.575,1.558,1.544,1.528,1.52,1.518,1.503,1.443,1.35,1.313,1.323,1.359,1.436,1.587,2.453,2.604]
    _PF_95   = [3.613,3.861,3.886,3.889,3.811,3.55,3.397,3.281,3.11,2.984,2.741,2.668,2.643,2.645,2.676,2.639,2.576,2.551,2.561,2.588,2.648,3.274,3.417,3.536]
    _PF_98   = [4.034,4.051,4.06,4.061,4.047,4.011,3.948,3.86,3.699,3.472,3.377,3.318,3.332,3.354,3.419,3.476,3.479,3.481,3.493,3.516,3.563,3.818,3.961,4.005]
    _SUN_C   = [3.45,3.727,3.749,3.751,3.668,3.394,3.26,2.927,2.834,2.756,2.675,2.656,2.63,2.616,2.601,2.554,2.481,2.451,2.459,2.488,2.549,2.893,3.277,3.382]
    _SUN_M   = [3.45,3.727,3.749,3.751,3.668,3.394,3.26,2.927,2.824,2.729,2.638,2.614,2.601,2.592,2.589,2.55,2.481,2.451,2.459,2.488,2.549,2.893,3.277,3.382]
    _SUN_V   = [3.45,3.727,3.749,3.751,3.668,3.394,3.254,2.899,2.727,2.53,2.349,2.244,2.264,2.309,2.408,2.454,2.443,2.443,2.459,2.488,2.549,2.893,3.277,3.382]
    _BC      = [4.5845,4.6756,4.7258,4.7299,4.6547,4.4601,4.1611,3.9556,3.8972,3.8737,3.8589,3.8439,3.8295,3.8218,3.79,3.6981,3.5574,3.5,3.5167,3.5701,3.6855,3.9173,4.1982,4.4317]
    _BR      = [4.8506,4.9455,4.9977,5.002,4.9237,4.721,4.4091,4.1943,4.1333,4.1087,4.0931,4.0775,4.0624,4.0543,4.021,3.9248,3.7772,3.7171,3.7346,3.7906,3.9115,4.1542,4.4478,4.6914]
    _WC      = [1.281,1.2805,1.2812,1.2813,1.2803,1.2777,1.271,0.9547,0.9543,0.9541,0.8656,0.7875,0.8108,0.8484,0.9535,0.9529,0.9519,0.9515,0.9516,0.952,0.9528,0.9544,1.2741,1.2773]
    _WR      = [1.3117,1.3115,1.3123,1.3124,1.3111,1.3077,1.2996,0.9935,0.9932,0.9931,0.8895,0.8075,0.8319,0.8714,0.9843,0.9922,0.9914,0.9911,0.9912,0.9915,0.9921,0.9933,1.3031,1.3072]
    _MW_NO   = [7.716,7.517,7.407,7.398,7.563,7.986,8.626,9.059,9.181,9.23,9.261,9.292,9.322,9.338,9.404,9.594,9.883,10.0,9.966,9.857,9.62,9.139,8.547,8.047]
    _MW_CVR  = [7.4498,7.2368,7.1293,7.1205,7.2856,7.7149,8.345,8.7947,8.9244,8.9834,9.0245,9.0593,9.0891,9.104,9.1658,9.3523,9.6391,9.7552,9.7209,9.6118,9.3748,8.8746,8.2669,7.7749]
    _V_NO    = [1.0082,1.0093,1.01,1.01,1.0091,1.0065,1.0025,0.9999,0.9992,0.9988,0.9985,0.9984,0.9983,0.9982,0.9977,0.9966,0.9947,0.9939,0.9942,0.9949,0.9963,0.9994,1.0031,1.0061]
    _V_CVR   = [0.9804,0.9793,0.9798,0.9798,0.9796,0.9792,0.9765,0.9765,0.9769,0.9776,0.9782,0.9785,0.9783,0.9782,0.9775,0.9764,0.975,0.9744,0.9746,0.9751,0.976,0.9763,0.9768,0.9789]

    def _lc(title, traces, h=320, target=2.0):
        f = go.Figure()
        pal = [C["purple"],C["blue"],C["orange"],C["gold"],C["pink"],C["good"]]
        for i,(name,vals) in enumerate(traces):
            f.add_trace(go.Scatter(x=_H, y=vals, name=name, mode="lines",
                line=dict(color=pal[i%len(pal)], width=2.5), showlegend=True))
        if target:
            f.add_hline(y=target, line_dash="dot", line_color=C["warn"],
                annotation_text="2% target", annotation_font_size=9,
                annotation_position="bottom right")
        lay = base_layout(title, height=h)
        lay["legend"] = dict(orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=C["border"], borderwidth=1)
        lay["margin"] = dict(l=20,r=20,t=65,b=40)
        f.update_layout(**lay)
        f.update_xaxes(title="Hour of Day", tickvals=list(range(1,25,3)))
        f.update_yaxes(title="% Reduction")
        return f

    section_heading("Average of All Load Types","Grand average across all 5,184 cases.")
    da1,da2 = st.columns(2)
    with da1:
        fa = go.Figure()
        fa.add_trace(go.Scatter(x=_H,y=_MW_NO,name="Without CVR",mode="lines+markers",line=dict(color=C["blue"],width=3),marker=dict(size=5)))
        fa.add_trace(go.Scatter(x=_H,y=_MW_CVR,name="With CVR",mode="lines+markers",line=dict(color=C["purple"],width=3,dash="dash"),marker=dict(size=5),fill="tonexty",fillcolor="rgba(184,108,224,0.10)"))
        lay_a=base_layout("Feeder Load · With and Without CVR",height=320)
        lay_a["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        lay_a["margin"]=dict(l=20,r=20,t=65,b=40)
        fa.update_layout(**lay_a); fa.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2))); fa.update_yaxes(title="MW")
        show_chart(fa)
        analysis_box("Solid = no CVR. Dashed = with CVR. Shaded area = energy saved.")
    with da2:
        fv=go.Figure()
        fv.add_trace(go.Scatter(x=_H,y=_V_NO,name="Without CVR",mode="lines+markers",line=dict(color=C["orange"],width=3),marker=dict(size=5)))
        fv.add_trace(go.Scatter(x=_H,y=_V_CVR,name="With CVR",mode="lines+markers",line=dict(color=C["purple"],width=3,dash="dash"),marker=dict(size=5)))
        fv.add_hline(y=1.05,line_dash="dot",line_color=C["warn"],annotation_text="Max 1.05 pu",annotation_font_size=9)
        fv.add_hline(y=0.97,line_dash="dot",line_color=C["warn"],annotation_text="Target 0.97 pu",annotation_font_size=9)
        fv.add_hline(y=0.95,line_dash="dot",line_color=C["bad"],annotation_text="Min 0.95 pu",annotation_font_size=9)
        lay_v=base_layout("Load-Bus Voltage Compliance",height=320)
        lay_v["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        lay_v["margin"]=dict(l=20,r=20,t=65,b=40)
        lay_v["yaxis"]=dict(title="Voltage (pu)",range=[0.93,1.08])
        fv.update_layout(**lay_v); fv.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2)))
        show_chart(fv)
        analysis_box("Without CVR: voltage near 1.0 pu. With CVR: held at 0.97 pu target. <b>No voltage violations in any of the 5,184 cases.</b>")

    section_heading("PV Location and Size Impact (Controllable Parameters)","Parameters the utility can choose.")
    _db_l1,_db_r1 = st.columns([3,2])
    with _db_l1:
        show_chart(_lc("Hourly % Reduction by PV Bus Location",[("PV Bus 3",_PV_B3),("PV Bus 4",_PV_B4),("PV Bus 5",_PV_B5)]))
        analysis_box("<b>PV Bus 5 (at load)</b> gives the highest reduction. Reactive power is most effective when injected close to the load.")
    with _db_r1:
        _b3a,_b4a,_b5a=round(sum(_PV_B3)/24,2),round(sum(_PV_B4)/24,2),round(sum(_PV_B5)/24,2)
        _fb_bus=go.Figure()
        _fb_bus.add_trace(go.Bar(x=["Bus 3","Bus 4","Bus 5"],y=[_b3a,_b4a,_b5a],marker_color=[C["purple"],C["blue"],C["orange"]],text=[f"{v:.2f}%" for v in [_b3a,_b4a,_b5a]],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        _fb_bus.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay_bus=base_layout("Daily Avg by PV Bus",height=340)
        _lay_bus["yaxis"]=dict(title="Avg % Reduction",range=[0,max([_b3a,_b4a,_b5a])*1.35])
        _lay_bus["margin"]=dict(l=20,r=20,t=65,b=40)
        _fb_bus.update_layout(**_lay_bus); show_chart(_fb_bus)

    _db_l2,_db_r2 = st.columns([3,2])
    with _db_l2:
        show_chart(_lc("Hourly % Reduction by PV Inverter Size",[("5.263 MVA — Small",_PV_S),("10.526 MVA — Large",_PV_L)]))
        analysis_box("<b>Larger inverter</b> provides more reactive power → stronger voltage pull-down → higher % reduction.")
    with _db_r2:
        _s_a,_l_a=round(sum(_PV_S)/24,2),round(sum(_PV_L)/24,2)
        _fb_sz=go.Figure()
        _fb_sz.add_trace(go.Bar(x=["5.263 MVA","10.526 MVA"],y=[_s_a,_l_a],marker_color=[C["purple"],C["blue"]],text=[f"{v:.2f}%" for v in [_s_a,_l_a]],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        _fb_sz.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay_sz=base_layout("Daily Avg by PV Size",height=340)
        _lay_sz["yaxis"]=dict(title="Avg % Reduction",range=[0,max([_s_a,_l_a])*1.35])
        _lay_sz["margin"]=dict(l=20,r=20,t=65,b=40)
        _fb_sz.update_layout(**_lay_sz); show_chart(_fb_sz)

    section_heading("Power Factor and Sun Condition Impact (Uncontrollable Parameters)","Parameters that depend on the feeder and weather.")
    _dc_l1,_dc_r1 = st.columns([3,2])
    with _dc_l1:
        show_chart(_lc("Hourly % Reduction by Power Factor",[("PF = 0.90",_PF_90),("PF = 0.95",_PF_95),("PF = 0.98",_PF_98)]))
        analysis_box("<b>Higher PF = more reactive headroom</b> for CVR. PF 0.98 achieves nearly double the reduction of PF 0.90 during peak hours.")
    with _dc_r1:
        _pf_avgs=[round(sum(_PF_90)/24,2),round(sum(_PF_95)/24,2),round(sum(_PF_98)/24,2)]
        _fpf=go.Figure()
        _fpf.add_trace(go.Bar(x=["PF 0.90","PF 0.95","PF 0.98"],y=_pf_avgs,marker_color=[C["purple"],C["blue"],C["orange"]],text=[f"{v:.2f}%" for v in _pf_avgs],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        _fpf.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay_pf=base_layout("Daily Avg by Power Factor",height=340)
        _lay_pf["yaxis"]=dict(title="Avg % Reduction",range=[0,max(_pf_avgs)*1.35])
        _lay_pf["margin"]=dict(l=20,r=20,t=65,b=40)
        _fpf.update_layout(**_lay_pf); show_chart(_fpf)

    _dc_l2,_dc_r2 = st.columns([3,2])
    with _dc_l2:
        _f_sun=go.Figure()
        for _nm,_vl,_cl in [("Cloudy",_SUN_C,C["purple"]),("Moderate Sun",_SUN_M,C["blue"]),("Very Sunny",_SUN_V,C["gold"])]:
            _f_sun.add_trace(go.Scatter(x=list(range(1,25)),y=_vl,name=_nm,mode="lines",line=dict(color=_cl,width=2.5),showlegend=True))
        _f_sun.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9,annotation_position="bottom right")
        _lay_sun2=base_layout("Hourly % Reduction by Sun Condition",height=320)
        _lay_sun2["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        _lay_sun2["margin"]=dict(l=20,r=20,t=65,b=40)
        _f_sun.update_layout(**_lay_sun2); _f_sun.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,3))); _f_sun.update_yaxes(title="% Reduction")
        show_chart(_f_sun)
        analysis_box("<b>Cloudy days</b> give the best CVR — the PV inverter produces less active power, freeing reactive capacity (Q) for voltage control.")
    with _dc_r2:
        _sun_avgs=[round(sum(_SUN_C)/24,2),round(sum(_SUN_M)/24,2),round(sum(_SUN_V)/24,2)]
        _fsun=go.Figure()
        _fsun.add_trace(go.Bar(x=["Cloudy","Moderate Sun","Very Sunny"],y=_sun_avgs,marker_color=[C["purple"],C["blue"],C["gold"]],text=[f"{v:.2f}%" for v in _sun_avgs],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        _fsun.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay_sun=base_layout("Daily Avg by Sun Condition",height=340)
        _lay_sun["yaxis"]=dict(title="Avg % Reduction",range=[0,max(_sun_avgs)*1.35])
        _lay_sun["margin"]=dict(l=20,r=20,t=65,b=40)
        _fsun.update_layout(**_lay_sun); show_chart(_fsun)

    section_heading("Load Type Impact","How the type of electrical load affects CVR response.")
    dd1,dd2 = st.columns([2,1])
    with dd1:
        show_chart(_lc("Hourly % Reduction in Load MW by Load Type",[("Z-Load (Constant-Z)",_LT_Z),("I-Load (Constant-I)",_LT_I),("Residential (ZIP Mix)",_LT_RES),("Commercial (ZIP Mix)",_LT_COMM)]))
        analysis_box("<b>Constant-Z</b> (resistive loads) responds most. <b>Constant-I</b> responds linearly. <b>ZIP loads</b> sit in between.")
    with dd2:
        lt_avgs=[round(sum(_LT_Z)/24,2),round(sum(_LT_I)/24,2),round(sum(_LT_RES)/24,2),round(sum(_LT_COMM)/24,2)]
        fl=go.Figure()
        fl.add_trace(go.Bar(x=["Z","I","ZIP-Res","ZIP-Comm"],y=lt_avgs,marker_color=[C["purple"],C["blue"],C["orange"],C["gold"]],text=[f"{v:.2f}%" for v in lt_avgs],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        fl.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        lay_lt=base_layout("Daily Average by Load Type",height=320)
        lay_lt["yaxis"]=dict(title="Avg % Reduction",range=[0,max(lt_avgs)*1.35])
        lay_lt["margin"]=dict(l=20,r=20,t=65,b=40)
        fl.update_layout(**lay_lt); show_chart(fl)

    section_heading("Most and Least Effective Conditions","Best vs worst configurations.")
    de1,de2 = st.columns([2,1])
    with de1:
        fb=go.Figure()
        fb.add_trace(go.Scatter(x=_H,y=_BR,name="Residential — Most Effective",mode="lines",line=dict(color=C["purple"],width=2.5),showlegend=True))
        fb.add_trace(go.Scatter(x=_H,y=_BC,name="Commercial — Most Effective",mode="lines",line=dict(color=C["blue"],width=2.5),showlegend=True))
        fb.add_trace(go.Scatter(x=_H,y=_WR,name="Residential — Least Effective",mode="lines",line=dict(color=C["orange"],width=2,dash="dot"),showlegend=True))
        fb.add_trace(go.Scatter(x=_H,y=_WC,name="Commercial — Least Effective",mode="lines",line=dict(color=C["gold"],width=2,dash="dot"),showlegend=True))
        fb.add_hline(y=2.0,line_dash="dot",line_color=C["bad"],annotation_text="2% requirement",annotation_font_size=9)
        avg_diff=round(sum(_BR)/24-sum(_WR)/24,2)
        fb.add_annotation(x=12,y=3.0,text=f"<b>Avg Difference = {avg_diff:.1f}%</b>",showarrow=False,bgcolor="rgba(255,255,255,0.8)",bordercolor=C["purple"],borderwidth=1,font=dict(size=11,color=C["purple"]))
        lay_bw=base_layout("Most vs Least Effective Conditions for CVR in ZIP Mixture Load Types",height=360)
        lay_bw["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=9),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        lay_bw["margin"]=dict(l=20,r=20,t=80,b=40)
        lay_bw["yaxis"]=dict(title="% Reduction",range=[0,5.5])
        fb.update_layout(**lay_bw); fb.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,3)))
        show_chart(fb)
    with de2:
        analysis_box(f"""<b>Most Effective:</b> Cloudy · PF 0.98 · 10.526 MVA · Bus 5<br><br>
        <b>Least Effective:</b> Very Sunny · PF 0.90 · 5.263 MVA · Bus 3<br><br>
        Average difference: <b>{avg_diff:.1f}%</b>.""")

    section_heading("Cost Savings — Dx Feeder","Ontario TOU pricing applied to CVR energy savings at 10 MW peak.")
    _DX_TOU={1:9.8,2:9.8,3:9.8,4:9.8,5:9.8,6:9.8,7:9.8,8:20.3,9:20.3,10:20.3,11:20.3,12:15.7,13:15.7,14:15.7,15:15.7,16:15.7,17:15.7,18:20.3,19:20.3,20:9.8,21:9.8,22:9.8,23:9.8,24:9.8}
    _DX_CS={1:(40.84,21.09,21.72,20.70,26.09),2:(42.95,22.25,22.82,21.78,27.45),3:(42.59,22.05,22.63,21.60,27.22),4:(42.55,22.03,22.61,21.58,27.19),5:(43.13,20.85,22.90,21.86,27.19),6:(41.58,21.49,22.11,21.07,26.56),7:(43.15,22.28,22.92,21.81,27.54),8:(84.07,43.47,44.62,42.44,53.65),9:(83.37,39.70,44.21,41.07,52.09),10:(80.01,39.09,42.35,38.79,50.06),11:(75.48,38.54,39.99,38.04,48.01),12:(57.59,29.38,30.50,28.69,36.54),13:(57.23,29.60,30.62,28.82,36.57),14:(57.62,29.84,30.47,29.01,36.74),15:(58.67,30.27,31.08,29.55,37.39),16:(59.52,30.75,31.53,29.98,37.95),17:(60.12,30.96,31.86,30.25,38.30),18:(78.00,40.21,41.33,39.24,49.69),19:(78.11,40.22,41.38,39.29,49.75),20:(37.72,19.43,19.99,18.98,24.03),21:(37.70,19.43,20.00,18.99,24.03),22:(40.81,20.56,21.66,20.60,25.91),23:(42.99,22.22,22.82,21.76,27.45),24:(41.76,21.59,22.19,21.15,26.67)}
    _DX_ANNUAL={"Constant-Z":484568,"Constant-I":247210,"ZIP-Residential":257078,"ZIP-Commercial":243466,"All Avg":308081}
    _DX_LT_LABELS=["Constant-Z","Constant-I","ZIP-Residential","ZIP-Commercial"]
    _DX_HOURS=list(range(1,25))
    _tou_arr=np.array([_DX_TOU[h] for h in _DX_HOURS])
    _dx_daily_all=[sum(_DX_CS[h][i] for h in _DX_HOURS) for i in range(4)]
    _dx_annual_all=[_DX_ANNUAL[lt] for lt in _DX_LT_LABELS]
    _dx_best_lt=_DX_LT_LABELS[int(np.argmax(_dx_daily_all))]
    ck1,ck2,ck3,ck4=st.columns(4)
    with ck1: kpi("Best Load Type",_dx_best_lt,f"${max(_dx_daily_all):,.2f}/day at 10 MW peak")
    with ck2: kpi("Best Annual Savings",f"${max(_dx_annual_all):,}",f"{_dx_best_lt} · ×365 days")
    with ck3: kpi("All-Type Average Daily",f"${sum(_DX_CS[h][4] for h in _DX_HOURS):,.2f}","Average across all 4 load types")
    with ck4: kpi("TOU Rate Range","9.8 – 20.3 ¢/kWh","Ontario Off-Peak → On-Peak")
    panel("About These Cost Savings","<p>Cost savings are calculated by multiplying hourly MW reduction by the Ontario TOU electricity rate. Values are taken directly from <b>Final Cost Savings Analysis.xlsx</b> (Dx Feeder sheet). Rates: <b>Off-Peak 9.8 ¢/kWh</b> · <b>Mid-Peak 15.7 ¢/kWh</b> · <b>On-Peak 20.3 ¢/kWh</b>.</p>")
    _lt_colors=[C["purple"],C["blue"],C["orange"],C["gold"]]
    _cf1,_cf2=st.columns(2)
    with _cf1:
        _fc=make_subplots(specs=[[{"secondary_y":True}]])
        for _i,(_lt,_col) in enumerate(zip(_DX_LT_LABELS,_lt_colors)):
            _fc.add_trace(go.Bar(x=_DX_HOURS,y=[_DX_CS[h][_i] for h in _DX_HOURS],name=_lt,marker_color=_col,opacity=0.82),secondary_y=False)
        _fc.add_trace(go.Scatter(x=_DX_HOURS,y=_tou_arr,name="TOU Rate (¢/kWh)",mode="lines",line=dict(color=C["deep"],width=2,dash="dot")),secondary_y=True)
        _lay_cf=base_layout("Hourly Cost Savings by Load Type",height=340)
        _lay_cf["barmode"]="group"
        _lay_cf["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=9),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        _lay_cf["margin"]=dict(l=20,r=20,t=65,b=40)
        _fc.update_layout(**_lay_cf); _fc.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2)))
        _fc.update_yaxes(title="$/hr Saved",secondary_y=False); _fc.update_yaxes(title="TOU Rate (¢/kWh)",secondary_y=True,showgrid=False)
        show_chart(_fc)
        analysis_box("On-peak hours 8–11 and 18–19 generate the highest savings. Constant-Z saves the most.")
    with _cf2:
        _fb_cost=go.Figure()
        _fb_cost.add_trace(go.Bar(x=_DX_LT_LABELS,y=_dx_daily_all,marker_color=_lt_colors,text=[f"${v:,.0f}" for v in _dx_daily_all],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        _lay_fb=base_layout("Daily Cost Savings by Load Type (10 MW Peak)",height=340)
        _lay_fb["yaxis"]=dict(title="Daily $ Saved",range=[0,max(_dx_daily_all)*1.35])
        _lay_fb["margin"]=dict(l=20,r=20,t=65,b=40)
        _fb_cost.update_layout(**_lay_fb); show_chart(_fb_cost)
        analysis_box(f"Constant-Z: <b>${_dx_daily_all[0]:,.2f}/day</b> · Constant-I: <b>${_dx_daily_all[1]:,.2f}/day</b> · ZIP-Res: <b>${_dx_daily_all[2]:,.2f}/day</b> · ZIP-Comm: <b>${_dx_daily_all[3]:,.2f}/day</b>.")
    _cf3,_cf4=st.columns(2)
    with _cf3:
        _fa_ann=go.Figure()
        _fa_ann.add_trace(go.Bar(x=_DX_LT_LABELS,y=_dx_annual_all,marker_color=_lt_colors,text=[f"${v/1000:.0f}k" for v in _dx_annual_all],textposition="outside",textfont=dict(size=11,color=C["deep"])))
        _lay_ann=base_layout("Annual Cost Savings by Load Type (×365)",height=340)
        _lay_ann["yaxis"]=dict(title="Annual $ Saved",range=[0,max(_dx_annual_all)*1.30])
        _lay_ann["margin"]=dict(l=20,r=20,t=65,b=40)
        _fa_ann.update_layout(**_lay_ann); show_chart(_fa_ann)
        analysis_box(f"Constant-Z: <b>${_DX_ANNUAL['Constant-Z']:,}/yr</b> · Constant-I: <b>${_DX_ANNUAL['Constant-I']:,}/yr</b> · ZIP-Res: <b>${_DX_ANNUAL['ZIP-Residential']:,}/yr</b> · ZIP-Comm: <b>${_DX_ANNUAL['ZIP-Commercial']:,}/yr</b>.")
    with _cf4:
        _cumul_z=np.cumsum([_DX_CS[h][0] for h in _DX_HOURS])
        _cumul_i=np.cumsum([_DX_CS[h][1] for h in _DX_HOURS])
        _cumul_avg=np.cumsum([_DX_CS[h][4] for h in _DX_HOURS])
        _fc4=go.Figure()
        _fc4.add_trace(go.Scatter(x=_DX_HOURS,y=_cumul_z,name="Constant-Z",mode="lines",line=dict(color=C["purple"],width=2.5)))
        _fc4.add_trace(go.Scatter(x=_DX_HOURS,y=_cumul_i,name="Constant-I",mode="lines",line=dict(color=C["blue"],width=2.5)))
        _fc4.add_trace(go.Scatter(x=_DX_HOURS,y=_cumul_avg,name="All-Type Avg",mode="lines",line=dict(color=C["gold"],width=2.5,dash="dot"),fill="tozeroy",fillcolor="rgba(255,166,0,0.08)"))
        for _hs,_he in [(8,11),(18,19)]:
            _fc4.add_vrect(x0=_hs-0.5,x1=_he+0.5,fillcolor="rgba(230,57,70,0.07)",line_width=0)
        _lay_c4=base_layout("Cumulative Daily Savings ($)",height=340)
        _lay_c4["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        _lay_c4["margin"]=dict(l=20,r=20,t=65,b=40)
        _fc4.update_layout(**_lay_c4); _fc4.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2))); _fc4.update_yaxes(title="Cumulative $ Saved")
        show_chart(_fc4)
        analysis_box(f"Savings accelerate during on-peak windows (shaded red). Constant-Z reaches <b>${float(_cumul_z[-1]):,.2f}</b> by end of day.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: IEEE 14-BUS RESULTS
# ═══════════════════════════════════════════════════════════════════════════
def page_ieee_results(ieee):
    section_heading("IEEE 14-Bus System — Study Results",
        "Load flow studies on the IEEE 14-bus standard test network across 168 cases. "
        "Residential and commercial loads at buses 4, 9, and 14 — average 2.44% CVR reduction.")
    k1,k2,k3,k4=st.columns(4)
    with k1: kpi("Average CVR Reduction","2.44%","Average across all 168 IEEE 14-bus cases")
    with k2: kpi("Total Simulation Cases (IEEE)","168","7 PV farm size combinations × 3 focus buses × 24 hours")
    with k3: kpi("Key Load Buses","4, 9, 14","Three largest non-industrial loads")
    with k4: kpi("PV Farm Sizes Tested","10.526 / 52.632 / 105.263 MVA","Small, Medium, Large")
    panel("Key Takeaways — IEEE 14-Bus", f"""
    <p>The IEEE 14-bus study confirmed: <b>CVR extends effectively to meshed transmission/sub-transmission networks.</b></p>
    <p><b>Key findings:</b><br>
    &#8226; Average CVR reduction: <b>2.44%</b> — exceeds the 2% design requirement.<br>
    &#8226; Best configuration: Medium PV at Bus 4 + small PV at buses 9 and 14 — all buses exceed 2%.<br>
    &#8226; Bus 14 (most downstream, residential) consistently achieves the highest % reduction (~3.5%).<br>
    &#8226; Concentrating reactive power at one bus (Bus 4 only) leaves Bus 14 below 2% — not sufficient.<br>
    &#8226; Equal sizing across all three buses is suboptimal — electrically strong buses need less support.<br>
    &#8226; All 168 cases maintained voltage within the 0.95–1.05 pu safe band.</p>
    """)
    render_image("img_ieee","IEEE 14-Bus System in PSSE — key load buses 4, 9, and 14 highlighted",max_width="90%")
    section_heading("Three Key PV Farm Scenarios","MW reduction at each load bus.")
    _s1a,_s1b=st.columns(2)
    with _s1a: show_chart(chart_ieee_scenario1())
    with _s1b:
        _s1_avgs={"Bus 4":2.8,"Bus 9":2.3,"Bus 14":3.5}
        _fs1=go.Figure()
        _fs1.add_trace(go.Bar(x=list(_s1_avgs.keys()),y=list(_s1_avgs.values()),marker_color=[C["purple"],C["blue"],C["pink"]],text=[f"{v:.1f}%" for v in _s1_avgs.values()],textposition="outside",textfont=dict(size=12,color=C["deep"])))
        _fs1.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay1=base_layout("Avg % Reduction — Scenario 1",height=300)
        _lay1["yaxis"]=dict(title="Avg % Reduction",range=[0,5.0])
        _lay1["margin"]=dict(l=20,r=20,t=65,b=40)
        _fs1.update_layout(**_lay1); show_chart(_fs1)
    analysis_box("<b>Scenario 1 — Medium at Bus 4 + 2 Small PV Farms:</b> Bus 14 achieves the highest reduction (~3.5%). All three buses exceed the 2% target. Best overall configuration.")
    _s2a,_s2b=st.columns(2)
    with _s2a: show_chart(chart_ieee_scenario2())
    with _s2b:
        _s2_avgs={"Bus 4":4.5,"Bus 9":1.4,"Bus 14":1.0}
        _fs2=go.Figure()
        _fs2.add_trace(go.Bar(x=list(_s2_avgs.keys()),y=list(_s2_avgs.values()),marker_color=[C["purple"],C["blue"],C["pink"]],text=[f"{v:.1f}%" for v in _s2_avgs.values()],textposition="outside",textfont=dict(size=12,color=C["deep"])))
        _fs2.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay2=base_layout("Avg % Reduction — Scenario 2",height=300)
        _lay2["yaxis"]=dict(title="Avg % Reduction",range=[0,6.0])
        _lay2["margin"]=dict(l=20,r=20,t=65,b=40)
        _fs2.update_layout(**_lay2); show_chart(_fs2)
    analysis_box("<b>Scenario 2 — One Large PV Farm at Bus 4 Only:</b> Bus 4 achieves ~4.5% but Bus 14 drops to ~1.0% — below the 2% requirement. Centralising reactive power at one bus fails.")
    _s3a,_s3b=st.columns(2)
    with _s3a: show_chart(chart_ieee_scenario3())
    with _s3b:
        _s3_avgs={"Bus 4":1.1,"Bus 9":1.8,"Bus 14":3.2}
        _fs3=go.Figure()
        _fs3.add_trace(go.Bar(x=list(_s3_avgs.keys()),y=list(_s3_avgs.values()),marker_color=[C["purple"],C["blue"],C["pink"]],text=[f"{v:.1f}%" for v in _s3_avgs.values()],textposition="outside",textfont=dict(size=12,color=C["deep"])))
        _fs3.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=9)
        _lay3=base_layout("Avg % Reduction — Scenario 3",height=300)
        _lay3["yaxis"]=dict(title="Avg % Reduction",range=[0,5.0])
        _lay3["margin"]=dict(l=20,r=20,t=65,b=40)
        _fs3.update_layout(**_lay3); show_chart(_fs3)
    analysis_box("<b>Scenario 3 — Three Equal Small PV Farms:</b> Bus 14 leads (~3.2%) but Bus 4 drops to ~1.1%. Equal sizing is suboptimal.")
    section_heading("Load Classification Table","Bus-by-bus load type assignment.")
    st.markdown("""
    | Bus | Load Type | Behaviour | CVR Sensitivity |
    |-----|-----------|-----------|----------------|
    | 2 | Industrial | Constant power | Minimal |
    | 3 | Industrial | Constant power | Minimal |
    | 4 | Commercial | Mixed Z/I/P | Moderate |
    | 5 | Commercial | Mixed Z/I/P | Moderate |
    | 6 | Residential | Primarily constant current | Good |
    | 9 | Residential | Primarily constant current | Good |
    | 10 | Residential | Primarily constant current | Good |
    | 11 | Residential | Primarily constant current | Good |
    | 12 | Residential | Primarily constant current | Good |
    | 13 | Residential | Primarily constant current | Good |
    | 14 | Residential | Primarily constant current | Best — most downstream |
    """)
    section_heading("Cost Savings — IEEE 14-Bus System","Ontario TOU rates applied to MW reduction at each focus bus.")
    _IEEE_BUS_LOAD={"Bus 4":47.8,"Bus 9":29.5,"Bus 14":14.9}
    _IEEE_BUS_RED={"Bus 4":2.8,"Bus 9":2.3,"Bus 14":3.5}
    _IEEE_TOU={1:9.8,2:9.8,3:9.8,4:9.8,5:9.8,6:9.8,7:9.8,8:20.3,9:20.3,10:20.3,11:20.3,12:15.7,13:15.7,14:15.7,15:15.7,16:15.7,17:15.7,18:20.3,19:20.3,20:9.8,21:9.8,22:9.8,23:9.8,24:9.8}
    _IEEE_HOURS=list(range(1,25))
    _IEEE_IESO_PCT=[75.47,73.20,72.01,71.82,73.25,77.21,83.31,88.27,90.17,90.96,91.71,92.45,92.58,92.51,92.80,94.69,98.03,100.0,99.69,98.02,95.25,90.38,84.31,78.95]
    _ieee_bus_savings={}
    for _bus,_mw in _IEEE_BUS_LOAD.items():
        _red=_IEEE_BUS_RED[_bus]/100.0
        _hourly=[]
        for _i,_h in enumerate(_IEEE_HOURS):
            _load_h=_mw*_IEEE_IESO_PCT[_i]/100.0
            _hourly.append(_load_h*_red*1000.0*_IEEE_TOU[_h]/100.0)
        _ieee_bus_savings[_bus]=_hourly
    _ieee_daily={b:sum(v) for b,v in _ieee_bus_savings.items()}
    _ieee_total_daily=sum(_ieee_daily.values())
    _ieee_total_annual=_ieee_total_daily*365
    _ik1,_ik2,_ik3,_ik4=st.columns(4)
    with _ik1: kpi("Best Bus (Bus 14)",f"${_ieee_daily['Bus 14']:,.2f}/day","3.5% avg reduction · 14.9 MW load")
    with _ik2: kpi("Total Daily Savings",f"${_ieee_total_daily:,.2f}","All 3 focus buses combined")
    with _ik3: kpi("Annual Projection",f"${_ieee_total_annual:,.0f}","3 buses × 365 days")
    with _ik4: kpi("Avg CVR Reduction","2.44%","Average across all 168 IEEE cases")
    _bus_colors={"Bus 4":C["purple"],"Bus 9":C["blue"],"Bus 14":C["pink"]}
    _ic1,_ic2=st.columns(2)
    with _ic1:
        _tou_line=[_IEEE_TOU[h] for h in _IEEE_HOURS]
        _fic2=make_subplots(specs=[[{"secondary_y":True}]])
        for _bus,_hvec in _ieee_bus_savings.items():
            _fic2.add_trace(go.Bar(x=_IEEE_HOURS,y=_hvec,name=_bus,marker_color=_bus_colors[_bus],opacity=0.85),secondary_y=False)
        _fic2.add_trace(go.Scatter(x=_IEEE_HOURS,y=_tou_line,name="TOU Rate (¢/kWh)",mode="lines",line=dict(color=C["deep"],width=2,dash="dot")),secondary_y=True)
        _lay_ic=base_layout("Hourly Cost Savings by Bus (Best Scenario)",height=340)
        _lay_ic["barmode"]="stack"
        _lay_ic["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        _lay_ic["margin"]=dict(l=20,r=20,t=65,b=40)
        _fic2.update_layout(**_lay_ic); _fic2.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2)))
        _fic2.update_yaxes(title="$/hr Saved (stacked)",secondary_y=False); _fic2.update_yaxes(title="TOU Rate (¢/kWh)",secondary_y=True,showgrid=False)
        show_chart(_fic2)
        analysis_box("Bus 4 contributes most in absolute $/hr. Bus 14 achieves highest % reduction per MW. On-peak hours generate highest savings.")
    with _ic2:
        _fid=go.Figure()
        _fid.add_trace(go.Bar(x=list(_ieee_daily.keys()),y=list(_ieee_daily.values()),marker_color=[_bus_colors[b] for b in _ieee_daily],text=[f"${v:,.0f}" for v in _ieee_daily.values()],textposition="outside",textfont=dict(size=12,color=C["deep"])))
        _lay_id=base_layout("Daily Savings per Bus",height=340)
        _lay_id["yaxis"]=dict(title="Daily $ Saved",range=[0,max(_ieee_daily.values())*1.35])
        _lay_id["margin"]=dict(l=20,r=20,t=65,b=40)
        _fid.update_layout(**_lay_id); show_chart(_fid)
        analysis_box(f"Bus 4: <b>${_ieee_daily['Bus 4']:,.2f}/day</b> · Bus 9: <b>${_ieee_daily['Bus 9']:,.2f}/day</b> · Bus 14: <b>${_ieee_daily['Bus 14']:,.2f}/day</b>.<br><b>Combined: ${_ieee_total_daily:,.2f}/day · ${_ieee_total_annual:,.0f}/yr</b>.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OPTIMIZED AI FORECASTING  (replaces page_ai)
# ═══════════════════════════════════════════════════════════════════════════
def page_ai(constz_raw=None, consti_raw=None, zip_raw=None):
    """Optimized AI forecasting page — continuous PF interpolation, all 5,184 rows."""
    section_heading(
        "Forecasting Model — Optimized AI",
        "Physics-informed ET+RF ensemble trained on all 5,184 PSSE simulation cases. "
        "Any PF from 0.90–0.98 predicted with continuous interpolation (no nearest-neighbour snapping). "
        "Live weather from Open-Meteo adjusts load shape and sun rating automatically.",
    )

    # ── Load / train ──────────────────────────────────────────────────────
    try:
        (forecast_df, train_df, models, best_pred_df,
         summary_df, dominant_sun) = ai_build_predictions()
    except Exception as exc:
        st.error(f"Failed to build AI forecast: {exc}")
        return

    m = models["metrics"]

    # ── Performance banner ────────────────────────────────────────────────
    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1: kpi("Grouped Test R² (Reduction)",f"{m['r2_reduction']:.4f}",f"{m['n_groups_test']} unseen scenario groups")
    with mc2: kpi("MAE (Reduction)",f"{m['mae_reduction']:.4f}%","Mean absolute error")
    with mc3: kpi("R² (v_cvr)",f"{m['r2_vcvr']:.4f}","CVR voltage prediction")
    with mc4: kpi("Training Rows",f"{m['n_total']:,}",f"{m['n_groups_train']} grouped scenarios")
    st.markdown("---")

    # ── Configuration panel ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-panel" style="border:2px solid {C['purple']};">
      <h3 style="margin-top:0;">⚙️ Configure Your CVR Scenario</h3>
      <p><b>Any power factor 0.90–0.98</b> is predicted with continuous interpolation using
      physics-based reactive-power features — no nearest-neighbour snapping.
      Peak load scales all results to your feeder size.</p>
    </div>""", unsafe_allow_html=True)

    ca,cb = st.columns(2)
    with ca:
        sel_lt = st.selectbox("Load Type", list(AI_LT_LABEL.keys()),
            format_func=lambda k: f"{k}  —  {AI_LT_LABEL[k]}", index=0, key="opt_lt")
        sel_pf = float(st.slider("Power Factor  (0.90 – 0.98, any value)",
            min_value=0.90, max_value=0.98, value=0.95, step=0.01, key="opt_pf",
            help="Smooth continuous prediction — no snapping to 0.90/0.95/0.98."))
        sel_peak = float(st.number_input("Peak Feeder Load (MW)",
            min_value=1.0, max_value=100.0, value=10.0, step=0.5, key="opt_peak"))
    with cb:
        sel_bus = int(st.selectbox("PV Farm Bus Location", [3,4,5],
            format_func=lambda b: {3:"Bus 3 — near substation  (weakest CVR)",
                                   4:"Bus 4 — mid-feeder",
                                   5:"Bus 5 — at load bus  (strongest CVR)"}[b],
            index=2, key="opt_bus"))
        sel_size = float(st.selectbox("PV Farm Size (MVA)", [5.263, 10.526],
            format_func=lambda s: f"{s:.3f} MVA  ({'Small 5 MW' if s<6 else 'Large 10 MW'})",
            index=1, key="opt_size"))
        sun_idx = ["very sunny","moderate sun","cloudy"].index(dominant_sun)
        sel_sun = st.selectbox("Sun Rating  (auto-set from weather; override here)",
            ["very sunny","moderate sun","cloudy"], index=sun_idx, key="opt_sun",
            help="Open-Meteo cloud cover sets this automatically.")

    is_interp = sel_pf not in AI_ALL_PF_STUDY
    if is_interp:
        st.info(
            f"ℹ️  PF {sel_pf:.2f} is between study training values (0.90 / 0.95 / 0.98). "
            "The model interpolates using physics-based Q-capacity features — no snapping.",
            icon=None,
        )

    # ── Predict selected scenario ─────────────────────────────────────────
    df24 = ai_predict_24h(models, pf=sel_pf, pv_bus=sel_bus, pv_size=sel_size,
                          load_type=sel_lt, sun_rating=sel_sun,
                          peak_mw=sel_peak, forecast_df=forecast_df)

    d_base   = float(df24["mw_no"].sum())
    d_saved  = float(df24["mw_reduction"].sum())
    d_pct    = 100.0*d_saved/d_base if d_base > 0 else 0.0
    d_cost   = float(df24["cost_saved_usd"].sum())
    avg_red  = float(df24["reduction_pct"].mean())
    min_v    = float(df24["v_cvr"].min())
    max_v    = float(df24["v_cvr"].max())
    scale_f  = min(sel_peak/AI_STUDY_PEAK, 1.0)
    annual_s = int(AI_ANNUAL_SAVINGS.get(sel_lt, 308081)*scale_f)
    feasible = (d_pct >= AI_MIN_RED_PCT and min_v >= AI_MIN_V_PU and max_v <= AI_MAX_V_PU)
    feas_str = "✅ Feasible" if feasible else "⚠️ Check limits"
    scale_note = (f"Scaled {scale_f:.2f}× from {AI_STUDY_PEAK:.0f} MW study reference."
                  if sel_peak < AI_STUDY_PEAK else f"At/above {AI_STUDY_PEAK:.0f} MW study reference.")

    # ── KPIs ──────────────────────────────────────────────────────────────
    section_heading("Scenario Results",
        f"{AI_LT_LABEL[sel_lt]} · PF {sel_pf:.2f}{'  ★' if is_interp else ''} · "
        f"Bus {sel_bus} · {sel_size:.3f} MVA · {sel_sun.title()} · {sel_peak:.0f} MW")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: kpi("Status",feas_str,"≥ 2% + 0.95–1.05 pu")
    with k2: kpi("Avg % Reduction",f"{avg_red:.2f}%","Daily average 24 hours")
    with k3: kpi("Daily Energy",f"{d_saved:.2f} MWh","Total MW saved × hours")
    with k4: kpi("Daily Cost",f"${d_cost:,.2f}","Ontario TOU rates")
    with k5: kpi("Annual Savings",f"${annual_s:,}","×365 · scaled to peak MW")
    with k6: kpi("Min CVR Voltage",f"{min_v:.4f} pu","Must be ≥ 0.95 pu")

    # ── 24-Hour Profile ───────────────────────────────────────────────────
    section_heading("24-Hour CVR Profile")
    r1a,r1b = st.columns(2)
    with r1a:
        show_chart(ai_ch_load(df24))
        analysis_box(f"Shaded gap = MW saved each hour. Total: <b>{d_saved:.2f} MWh</b>. {scale_note}")
    with r1b:
        show_chart(ai_ch_voltage(df24))
        v_ok = "✓ All hours within safe band" if min_v>=AI_MIN_V_PU and max_v<=AI_MAX_V_PU else "⚠ Voltage limit exceeded"
        analysis_box(f"Target 0.97 pu. Min: <b>{min_v:.4f} pu</b>. <b style='color:{'green' if min_v>=0.95 else 'red'};'>{v_ok}</b>")
    r2a,r2b = st.columns(2)
    with r2a:
        show_chart(ai_ch_reduction(df24))
        analysis_box(f"Average: <b>{avg_red:.2f}%</b>. Higher % at low-load early hours where baseline voltage is highest.")
    with r2b:
        show_chart(ai_ch_mw_saved(df24))
        analysis_box(f"MW saved = baseline × reduction% / 100. Total daily: <b>{d_saved:.2f} MWh</b>.")

    # ── PF Sensitivity ────────────────────────────────────────────────────
    section_heading("Power Factor Sensitivity — Continuous Interpolation",
        "Smooth prediction for any PF 0.90–0.98.  Vertical lines = study training PF values.  ★ = your selection.")
    pfa,pfb = st.columns(2)
    with pfa:
        show_chart(ai_ch_pf_curve(models, sel_bus, sel_size, sel_lt, sel_sun, sel_peak))
        analysis_box("Curve is smooth because the model uses <b>q_available_factor</b> and "
                     "<b>pv_q_max_mvar</b> — physics-derived features continuous in PF.")
    with pfb:
        pf_vals = np.round(np.arange(0.90, 0.99, 0.01), 2)
        tbl = []
        for pf_t in pf_vals:
            d = ai_predict_24h(models, pf=pf_t, pv_bus=sel_bus, pv_size=sel_size,
                               load_type=sel_lt, sun_rating=sel_sun, peak_mw=sel_peak)
            tbl.append({
                "PF":f"{pf_t:.2f}",
                "Hr-18 Red%":f"{float(d[d['hour']==18]['reduction_pct'].values[0]):.3f}",
                "Daily Avg Red%":f"{float(d['reduction_pct'].mean()):.3f}",
                "Daily Cost $":f"${float(d['cost_saved_usd'].sum()):,.0f}",
                "Source":"Training" if pf_t in AI_ALL_PF_STUDY else "★ Interpolated",
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        analysis_box("★ = interpolated. Monotonic increase 0.90→0.98 = physically consistent.")

    # ── Cost Savings ──────────────────────────────────────────────────────
    section_heading("Cost Savings — Ontario TOU Pricing",
        "Blue = Off-Peak 9.8¢/kWh · Gold = Mid-Peak 15.7¢/kWh · Red = On-Peak 20.3¢/kWh")
    panel("Rate Structure", f"""
    <p><b>Off-Peak (9.8¢/kWh)</b> — hours 1–7 and 20–24.<br>
    <b>Mid-Peak (15.7¢/kWh)</b> — hours 12–17.<br>
    <b>On-Peak (20.3¢/kWh)</b> — hours 8–11 and 18–19.<br>
    <em>{scale_note}  Annual savings from Final Cost Savings Analysis.xlsx scaled to {sel_peak:.0f} MW.</em></p>
    """)
    cs1,cs2 = st.columns(2)
    with cs1:
        show_chart(ai_ch_cost(df24))
        analysis_box(f"Daily: <b>${d_cost:,.2f}</b>. Annual ×365: <b>${annual_s:,}</b>.")
    with cs2:
        show_chart(ai_ch_cumul_cost(df24))
        analysis_box("Savings fastest in on-peak windows (shaded red).")

    # ── Weather ───────────────────────────────────────────────────────────
    section_heading("Tomorrow's Weather — London, Ontario",
        "Open-Meteo API · Cloud → sun_rating · Temperature → load scale")
    fdate     = forecast_df["forecast_date"].iloc[0] if "forecast_date" in forecast_df.columns else "N/A"
    avg_cloud = float(forecast_df["cloud_cover_pct"].mean()) if "cloud_cover_pct" in forecast_df.columns else 50.0
    avg_temp  = float(forecast_df["temperature_c"].mean())   if "temperature_c"   in forecast_df.columns else 10.0
    wk1,wk2,wk3 = st.columns(3)
    with wk1: kpi("Forecast Date",str(fdate),"London, Ontario")
    with wk2: kpi("Avg Cloud Cover",f"{avg_cloud:.0f}%",f"→ {dominant_sun.title()}")
    with wk3: kpi("Avg Temperature",f"{avg_temp:.1f}°C","Adjusts heating/cooling load")
    wg1,wg2 = st.columns(2)
    with wg1: show_chart(ai_ch_weather(forecast_df))
    with wg2: show_chart(ai_ch_cloud_wind(forecast_df))

    # ── AI Best Case ──────────────────────────────────────────────────────
    section_heading("AI-Recommended Best Case for Tomorrow",
        "Highest composite score across all 216 standard scenarios.")
    feas_df  = summary_df[summary_df["feasible"]]
    best_row = feas_df.iloc[0] if not feas_df.empty else summary_df.iloc[0]
    bk1,bk2,bk3,bk4 = st.columns(4)
    with bk1: kpi("Best Load Type",str(best_row["load_type"]),f"PF {best_row['pf']:.2f}")
    with bk2: kpi("Best Config",f"Bus {int(best_row['pv_bus'])} · {best_row['pv_size_mva']:.3f} MVA",str(best_row["sun_rating"]).title())
    with bk3: kpi("Predicted Daily",f"{best_row['daily_saved_mwh']:.2f} MWh",f"{best_row['daily_reduction_pct']:.2f}% avg")
    with bk4: kpi("Min Voltage",f"{best_row['min_v_cvr_pu']:.4f} pu","✅ Feasible" if bool(best_row["feasible"]) else "⚠ Infeasible")
    bg1,bg2 = st.columns(2)
    with bg1: show_chart(ai_ch_load(best_pred_df,"AI Best Case — 24h Load Profile"))
    with bg2: show_chart(ai_ch_reduction(best_pred_df))

    # ── Full Ranking ──────────────────────────────────────────────────────
    section_heading("Full Scenario Ranking — All 216 Cases",
        "Purple = feasible (≥ 2% + 0.95–1.05 pu).  Red = infeasible.")
    sr1,sr2 = st.columns([2,1])
    with sr1: show_chart(ai_ch_ranking(summary_df))
    with sr2:
        disp = [c for c in ["load_type","pf","pv_bus","pv_size_mva","sun_rating",
                             "daily_saved_mwh","daily_reduction_pct","min_v_cvr_pu",
                             "daily_cost_usd","feasible","score"] if c in summary_df.columns]
        st.dataframe(summary_df[disp].round(3), use_container_width=True, hide_index=True)

    # ── Design Explainers ─────────────────────────────────────────────────
    section_heading("AI Model Design Decisions","")
    with st.expander("Why does continuous PF interpolation work?  (Physics-based features)"):
        st.markdown("""
**Problem with the original code (nearest-neighbour snapping):**
PF was snapped to the nearest study value (0.90 / 0.95 / 0.98). PF 0.93 produced the same result as PF 0.90 — physically wrong.

**Fix — two physics-derived features replace the raw PF scalar:**

| Feature | Formula | Physical meaning |
|---|---|---|
| `q_available_factor` | `√(1 − PF²) / PF` | Reactive power per unit MW |
| `pv_q_max_mvar` | `PV_size × √(1 − PF²)` | Max inverter Q absorption (MVAr) |

Both are **continuous and monotonic** in PF. ExtraTrees interpolates smoothly between training points.
**Result:** Monotonically increasing reduction curve from 0.90 → 0.98 with no discontinuities.
        """)
    with st.expander("Why GroupShuffleSplit?  (No data leakage)"):
        st.markdown("""
Each of the 216 unique scenarios produces exactly 24 hourly rows.

With a **random split**, 23 of those 24 hours could land in training — the model memorises the scenario
and the held-out test hour appears artificially accurate.

`GroupShuffleSplit` guarantees all 24 hours of every scenario go entirely to one split. ~44 grouped
scenarios (20%) form the test set. The reported R² ≈ 0.993 reflects true generalisation to never-seen scenarios.
        """)
    with st.expander("Why vectorized batch prediction?  (Speed)"):
        st.markdown("""
The original ranking code called `predict_scenario()` 216 × 24 = 5,184 times in a Python loop.

The optimised version builds a single **(5,184 × 12) matrix** and calls `model.predict()` once per estimator.
Scikit-learn's tree implementations are heavily vectorised — ranking drops from ~8 s to **< 0.5 s** after training.
        """)
    with st.expander("Why a separate v_no model?  (Pre-CVR voltage)"):
        st.markdown("""
`v_no` (pre-CVR load bus voltage) is an input feature to both the reduction and v_cvr models.
At inference time for an arbitrary PF, there is no PSSE run to look up the actual `v_no`.

A dedicated ExtraTrees sub-model predicts `v_no` from `(hour, PF, pv_bus, pv_size, load_type, mw_no)`.
This replaces the original approximation `v_no ≈ 1.0 + constant` with a proper physics-consistent prediction.
        """)
    with st.expander("How does weather adjust the load forecast?"):
        st.markdown("""
**Cloud cover → sun_rating** (automatic):
- Cloud < 25 % → `very sunny` · 25–65 % → `moderate sun` · ≥ 65 % → `cloudy`

**Temperature → load scaling** (per hour):
```python
temp_deviation  = max(0,  |temp_c − 15| − 5) / 20      # 0 → 1 scale
load_scale      = 1.0 + 0.04 × temp_deviation           # up to +4 %
mw_no(h)        = peak_MW × IESO_pct(h) × load_scale(h)
```
A −5 °C winter morning gets ~4 % more load than a 15 °C spring morning.
        """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PROTOTYPE
# ═══════════════════════════════════════════════════════════════════════════
def page_prototype():
    try:
        proto = load_prototype_data()
    except FileNotFoundError:
        st.error(
            "**Prototype file not found.**\n\n"
            f"Looking in: `{BASE_DIR}`\n\n"
            "Expected: `Capstone Prototype Data(Sheet1).xlsx`"
        )
        return
    except Exception as e:
        st.error(f"Failed to load prototype data: {e}"); return

    load_rows     = proto[proto["component"].str.lower()=="load r"].copy()
    solar_r_rows  = proto[proto["component"].str.lower()=="solar farm r"].copy()
    num_cases     = proto[["scenario","pv_location"]].drop_duplicates().shape[0]
    avg_load_power= load_rows["wattage"].mean() if not load_rows.empty else np.nan
    max_load_power= load_rows["wattage"].max()  if not load_rows.empty else np.nan
    avg_solar_cur = solar_r_rows["current"].mean() if not solar_r_rows.empty else np.nan

    section_heading("Hardware Prototype",
        "A bench-scale circuit demonstrating voltage reduction via an inductor simulating a solar farm. "
        "Tested at 120 V and 30 V with PV connected at the load bus and midline bus.")
    k1,k2,k3,k4=st.columns(4)
    with k1: kpi("Prototype Cases",f"{num_cases}","2 voltage levels × 2 PV locations")
    with k2: kpi("Avg Load Power",f"{avg_load_power:.2f} W" if pd.notna(avg_load_power) else "N/A","Across all measured configurations")
    with k3: kpi("Max Load Power",f"{max_load_power:.2f} W" if pd.notna(max_load_power) else "N/A","Highest measured load wattage")
    with k4: kpi("Avg Solar Current",f"{avg_solar_cur:.3f} A" if pd.notna(avg_solar_cur) else "N/A","Average Solar Farm R branch current")
    panel("Circuit Design and Purpose", """
    <p><b>Tx Line 1</b> (1 Ω, 1 mH) — long distribution line with resistance and inductance.
    <b>Tx Line 2</b> (0.5 Ω) — shorter line section closer to the load.
    <b>Load</b> (20 Ω resistor) — models a resistive load; CVR works best at reducing resistive loads.
    <b>Simulated PV Farm</b> (10 mH inductor + 5 Ω series resistor) — the inductor draws reactive power
    from the source, which lowers the bus voltage — mimicking what a real PV inverter does when absorbing VArs.
    The <b>dual throw switch</b> connects the PV branch to either the midline bus or the load bus.</p>
    <p><b>Measurement points:</b> Point 1 = source voltage; Point 2 = midline bus voltage; Point 3 = load bus voltage.
    Tested at 120 V AC and 30 V AC. <b>Why 30 V?</b> Redesigned from 120 V to meet safety and cost constraints.</p>
    """)
    g1,g2=st.columns(2)
    with g1:
        show_chart(chart_prototype_load_power(proto))
        analysis_box("At <b>120 V</b>, both PV placements produce the same load power (921.04 W). At <b>30 V</b>, PV location matters — midline-bus PV increases load power from 62.10 W to 64.31 W.")
    with g2:
        show_chart(chart_prototype_current_comparison(proto))
        analysis_box("At 30 V, the solar-farm branch current becomes a larger fraction of total circuit current — showing higher sensitivity to PV placement at lower voltages.")
    g3,g4=st.columns(2)
    with g3:
        show_chart(chart_prototype_line_losses(proto))
        analysis_box("At 120 V, losses are negligible (< 0.15 W). At 30 V, Tx Line 1 losses are more noticeable with midline PV.")
    with g4:
        load_rows2=proto[proto["component"].str.lower()=="load r"].copy()
        load_rows2["case"]=load_rows2["scenario"]+" · "+load_rows2["pv_location"]
        st.dataframe(load_rows2[["case","value","current","wattage"]].rename(columns={"value":"Load R (Ω)","current":"Load Current (A)","wattage":"Load Power (W)"}),use_container_width=True,hide_index=True)
        analysis_box("120 V cases show identical results for both PV locations; 30 V cases show 3.6% increase when PV moves to midline bus.")
    c1,c2=st.columns(2)
    with c1:
        panel("120 V Prototype Findings","<p>Both PV placements produced identical measured results: load resistor <b>15 Ω</b>, load current <b>7.836 A</b>, load power <b>921.04 W</b>. Tx Line losses under 0.15 W total. PV branch impedance so large relative to circuit that placement has negligible effect.</p>")
    with c2:
        panel("30 V Prototype Findings","<p>PV location had a measurable effect. Load-bus PV: load resistor <b>10 Ω</b>, current <b>2.492 A</b>, power <b>62.10 W</b>. Midline-bus PV: same resistor, current <b>2.536 A</b>, power <b>64.31 W</b>. Lower-voltage circuit is more sensitive to PV connection point — consistent with simulation findings.</p>")
    section_heading("Component Cost Summary","Bill of materials for the hardware prototype.")
    st.dataframe(pd.DataFrame({
        "Component":["TX Line 1 Resistor","TX Line 1 Inductor","TX Line 2 Resistor","PV Inductor","PV Resistor","Load Resistor","Dual Throw Switch","Measuring Device","Power Supply"],
        "Value":["1 Ω, 100W","1 mH, 5A","0.5 Ω, 100W","10 mH, 5A","5 Ω, 100W","20 Ω, 100W","125 VAC, 20A","NA","30VAC, 150VA"],
        "Qty":[1,1,1,1,1,1,1,3,1],
        "Cost ($)":[7.50,2.29,5.15,65.21,6.65,4.75,5.20,"NA",60.00],
    }),use_container_width=True,hide_index=True)
    section_heading("Full Prototype Data Table","All measured values from the spreadsheet.")
    t1,t2=st.columns(2)
    with t1:
        st.markdown("### 120 V Prototype")
        st.dataframe(proto[proto["scenario"]=="120 V"],use_container_width=True,hide_index=True)
    with t2:
        st.markdown("### 30 V Prototype")
        st.dataframe(proto[proto["scenario"]=="30 V"],use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FILES
# ═══════════════════════════════════════════════════════════════════════════
def page_excel_data():
    section_heading("Project Files","All simulation data, scripts, and notebooks used in this project. Click any file to download.")
    panel("About These Files","<p>Every data file, Python script, and notebook used in the capstone project. Excel files contain raw PSSE simulation outputs. Python scripts contain the PSSE automation code.</p>")

    def _render_file(filename, title, desc, icon, mime, accent=None):
        full_path = p(filename)
        if not os.path.exists(full_path): return
        with open(full_path,"rb") as f_obj:
            b64=base64.b64encode(f_obj.read()).decode()
        _accent=accent or C["purple"]
        st.markdown(f"""
        <div class="file-link-card" style="margin-bottom:0.55rem;padding:0.9rem 1.2rem;border-left:3px solid {_accent};">
            <span style="font-size:1.5rem;flex-shrink:0;">{icon}</span>
            <div style="flex:1;">
                <a href="data:{mime};base64,{b64}" download="{filename}" target="_blank" style="font-size:1rem;color:{_accent};">{title}</a>
                <div style="margin-top:0.12rem;"><code style="font-size:0.72rem;background:#f4f0fb;padding:1px 6px;border-radius:4px;color:{_accent};">{filename}</code></div>
                <div class="file-link-desc" style="margin-top:0.35rem;line-height:1.5;">{desc}</div>
            </div>
        </div>""",unsafe_allow_html=True)

    XLSX_MIME="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    PY_MIME="text/x-python"
    PDF_MIME="application/pdf"

    st.markdown(f'<h3 style="border-left:4px solid #b86ce0;padding-left:10px;">📊 Excel Data Files</h3>',unsafe_allow_html=True)
    for fname,title,desc in [
        ("AllResults.xlsx","All Results — Pivot Tables","Complete pivot table analysis of all 5,184 Dx feeder simulation cases."),
        ("TrainingData.xlsx","AI Training Data","Consolidated Dx feeder simulation results (5,184 rows). Used to train the AI surrogate model."),
        ("ConstantZLoad (Consolidated data).xlsx","Constant-Z Load Study","Dx feeder PSSE outputs for constant impedance load."),
        ("ConstantILoad (Consolidated Data).xlsx","Constant-I Load Study","Dx feeder PSSE outputs for constant current load."),
        ("ZIPLoad.xlsx","ZIP Load Study","Dx feeder PSSE outputs for ZIP load model."),
        ("IEEE14busresults.xlsx","IEEE 14-Bus System Results","168 IEEE 14-bus simulation cases."),
        ("Final Cost Savings Analysis.xlsx","Cost Savings Analysis","Hourly CVR energy savings ($) and peak demand value."),
        ("Capstone Prototype Data(Sheet1).xlsx","Hardware Prototype Measurements","Measured branch currents and power for 120 V and 30 V prototypes."),
        ("ZIPLoad(Analysis).xlsx","ZIP Load Analysis","Summary analysis of ZIP load study results."),
    ]:
        _render_file(fname,title,desc,"📊",XLSX_MIME,accent=C["purple"])

    st.markdown(f'<h3 style="border-left:4px solid #7678ed;padding-left:10px;">🐍 Python Scripts</h3>',unsafe_allow_html=True)
    for fname,title,desc in [
        ("app.py","Dashboard App","Main Streamlit dashboard application (this file)."),
        ("cvr_forecasting_optimized.py","Optimized AI Module","Standalone optimized CVR forecasting module with physics-based PF interpolation."),
        ("DxFeederCases.py","Dx Feeder Cases","Python automation script for running Dx feeder PSSE load flow cases."),
        ("IEEE14buscases.py","IEEE 14-Bus Cases","Python automation script for running IEEE 14-bus PSSE simulations."),
        ("findLoadRange.py","Find Load Range","Script for determining load range and scaling parameters."),
        ("hourly_data.py","Hourly Data","Script for processing IESO hourly demand data."),
    ]:
        _render_file(fname,title,desc,"🐍",PY_MIME,accent=C["blue"])

    st.markdown(f'<h3 style="border-left:4px solid #ffa600;padding-left:10px;">📄 Reports</h3>',unsafe_allow_html=True)
    for fname,title,desc in [
        ("Design-Validation-Test-Plan-Report-Group4.pdf","Design Validation & Test Plan Report","Full design validation and test plan report for the capstone project."),
        ("Midterm Progress Report Final - Group 4.pdf","Midterm Progress Report","Midterm progress report for ECE 4416 Group 4."),
    ]:
        _render_file(fname,title,desc,"📄",PDF_MIME,accent=C["gold"])

    st.markdown(f'<h3 style="border-left:4px solid {C["pink"]};padding-left:10px;">🖼 Images & Media</h3>',unsafe_allow_html=True)
    for fname,title,desc in [
        ("Dx_Feeder_Image.png","Dx Feeder Network Diagram","Modified Dx distribution feeder network diagram."),
        ("IEEE14_Image.png","IEEE 14-Bus System Diagram","IEEE 14-bus standard test network."),
        ("solar-energy-2026-01-21-12-26-38-utc.mp4","Solar Farm Hero Video","Background video used in the dashboard header."),
    ]:
        full_path=p(fname)
        if not os.path.exists(full_path): continue
        with open(full_path,"rb") as f_obj:
            b64=base64.b64encode(f_obj.read()).decode()
        ext=fname.rsplit(".",1)[-1].lower()
        mime="video/mp4" if ext=="mp4" else ("image/png" if ext=="png" else "image/jpeg")
        _accent=C["pink"]
        st.markdown(f"""
        <div class="file-link-card" style="margin-bottom:0.55rem;padding:0.9rem 1.2rem;border-left:3px solid {_accent};">
            <span style="font-size:1.5rem;flex-shrink:0;">🖼</span>
            <div style="flex:1;">
                <a href="data:{mime};base64,{b64}" download="{fname}" target="_blank" style="font-size:1rem;color:{_accent};">{title}</a>
                <div style="margin-top:0.12rem;"><code style="font-size:0.72rem;background:#f4f0fb;padding:1px 6px;border-radius:4px;color:{_accent};">{fname}</code></div>
                <div class="file-link-desc" style="margin-top:0.35rem;line-height:1.5;">{desc}</div>
            </div>
        </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DESIGN THOUGHT PROCESS
# ═══════════════════════════════════════════════════════════════════════════
def page_design():
    section_heading("Design Thought Process",
        "Detailed rationale for every decision made in the Dx feeder and IEEE 14-bus studies.")

    HOURS=list(range(1,25))
    LT_Z=[5.40,5.83,5.87,5.87,5.82,5.31,5.10,4.57,4.47,4.27,4.02,3.95,3.91,3.93,3.97,3.95,3.88,3.84,3.86,3.90,4.00,4.56,5.13,5.30]
    LT_I=[2.79,3.02,3.04,3.04,2.81,2.75,2.64,2.36,2.13,2.09,2.05,2.01,2.02,2.04,2.05,2.04,2.00,1.98,1.99,2.01,2.06,2.30,2.65,2.74]
    LT_COMM=[2.74,2.96,2.98,2.98,2.95,2.69,2.58,2.31,2.20,2.07,2.02,1.97,1.97,1.98,2.00,1.99,1.95,1.93,1.94,1.96,2.01,2.30,2.67,2.74]
    LT_RES=[2.87,3.10,3.12,3.12,3.09,2.82,2.71,2.43,2.37,2.26,2.13,2.09,2.09,2.09,2.10,2.09,2.05,2.04,2.05,2.07,2.12,2.42,2.79,2.88]
    PV_B3=[2.33,2.48,2.48,2.48,2.44,2.33,2.33,2.04,1.97,1.90,1.83,1.76,1.74,1.75,1.81,1.85,1.89,1.91,1.91,1.91,1.91,2.06,2.35,2.46]
    PV_B4=[3.39,3.95,3.95,3.95,3.83,3.39,3.39,2.97,2.79,2.56,2.34,2.31,2.32,2.34,2.37,2.39,2.42,2.43,2.43,2.43,2.43,2.94,3.39,3.55]
    PV_B5=[4.64,4.76,4.83,4.83,4.73,4.47,4.06,3.73,3.63,3.56,3.50,3.44,3.43,3.44,3.42,3.31,3.10,3.01,3.03,3.12,3.30,3.67,4.18,4.39]
    PF_90=[2.70,3.27,3.30,3.30,3.15,2.62,2.43,1.61,1.57,1.56,1.54,1.53,1.52,1.52,1.50,1.44,1.35,1.31,1.32,1.36,1.44,1.59,1.98,2.25]
    PF_95=[3.61,3.86,3.89,3.89,3.81,3.55,3.40,3.28,3.11,2.98,2.74,2.67,2.64,2.65,2.68,2.64,2.58,2.55,2.56,2.59,2.65,3.27,3.66,3.79]
    PF_98=[4.03,4.05,4.06,4.06,4.05,4.01,3.95,3.86,3.70,3.47,3.38,3.32,3.33,3.35,3.42,3.48,3.48,3.48,3.49,3.52,3.56,3.82,4.07,4.17]
    SUN_C=[3.45,3.73,3.75,3.75,3.67,3.39,3.26,2.93,2.83,2.76,2.68,2.66,2.63,2.62,2.60,2.55,2.48,2.45,2.46,2.49,2.55,2.89,3.33,3.51]
    SUN_M=[3.45,3.73,3.75,3.75,3.67,3.39,3.26,2.93,2.82,2.73,2.64,2.61,2.60,2.59,2.59,2.55,2.48,2.45,2.46,2.49,2.55,2.89,3.33,3.51]
    SUN_V=[3.45,3.73,3.75,3.75,3.67,3.39,3.25,2.90,2.73,2.53,2.35,2.24,2.26,2.31,2.41,2.45,2.44,2.44,2.46,2.49,2.55,2.89,3.33,3.51]
    IESO_PCT=[75.47,73.20,72.01,71.82,73.25,77.21,83.31,88.27,90.17,90.96,91.71,92.45,92.58,92.51,92.80,94.69,98.03,100.0,99.69,98.02,95.25,90.38,84.31,78.95]
    IESO_AVG_MW=[13265.24,12866.76,12657.11,12623.87,12874.23,13570.89,14643.22,15515.62,15849.33,15987.43,16119.22,16249.04,16273.37,16260.37,16310.68,16643.63,17230.55,17576.71,17522.97,17228.52,16741.53,15885.76,14819.50,13876.85]

    def _lc2(title,traces,y_title="% Reduction",target_line=2.0,height=300):
        f=go.Figure()
        palette=[C["purple"],C["blue"],C["orange"],C["gold"],C["pink"],C["good"]]
        for i,(name,vals) in enumerate(traces):
            f.add_trace(go.Scatter(x=HOURS,y=vals,name=name,mode="lines",line=dict(color=palette[i%len(palette)],width=2.5),showlegend=True))
        if target_line:
            f.add_hline(y=target_line,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_position="bottom right",annotation_font_size=10)
        lay=base_layout(title,height=height)
        lay["margin"]=dict(l=20,r=20,t=60,b=40)
        lay["showlegend"]=True
        lay["legend"]=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(255,255,255,0.85)",bordercolor=C["border"],borderwidth=1)
        f.update_layout(**lay)
        f.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,3)))
        f.update_yaxes(title=y_title)
        return f

    tab_dx,tab_ieee,tab_other=st.tabs(["Dx Distribution Feeder","IEEE 14-Bus System","CVR Study Design Decisions"])

    with tab_dx:
        section_heading("Dx Feeder — Design Decisions","")
        panel("Why a Distribution Feeder?","<p>Distribution feeders connect high-voltage transmission lines to homes and businesses, typically operating at 27.6 kV serving loads up to ~10 MW. CVR is most applicable at this level because: (1) voltage is close to end-users, (2) utilities already have voltage-regulating equipment, and (3) PV solar farms are increasingly connected at this level. The feeder was based on an existing lab network from ECE 4464 — a deliberate choice for a realistic, validated network.</p>")
        panel("Why Was the Transformer Moved?","<p>In the original ECE 4464 lab network, the step-down transformer sat between Buses 1 and 2. The load bus (Bus 5) started at 0.92 pu with a peak load of 10 MW — already below the 0.97 pu CVR target voltage, so no headroom existed for CVR. Moving the transformer to between Buses 2 and 3 raised Bus 5 voltage to ~0.995 pu under peak load, creating the headroom needed to pull voltage down to 0.97 pu without violating the 0.95 pu minimum.</p>")
        render_image("img_tx_moved","Original (top) and Modified (bottom) Dx Feeder Network Used in Studies",max_width="92%")
        panel("Why 0.97 pu as the Target CVR Voltage?","<p>ANSI C84.1 defines 0.95 pu as the absolute minimum and 1.05 pu as the maximum. We chose <b>0.97 pu</b> because: it is low enough to achieve meaningful demand reduction; it leaves a 0.02 pu safety margin above the 0.95 pu minimum; and it is achievable across all 5,184 tested cases without any voltage violations.</p>")
        panel("Why Three PV Bus Locations (Buses 3, 4, 5)?","<p>Bus 5 is the load bus, Bus 4 is one step upstream, Bus 3 is two steps upstream. Placing the PV farm at each location tests the effect of electrical distance between voltage support and load. Results confirmed <b>Bus 5 (load bus) PV</b> achieves the highest % reduction because reactive power is injected directly where voltage needs to be controlled.</p>")
        g1,g2=st.columns(2)
        with g1:
            show_chart(_lc2("% CVR Reduction by PV Bus Location",[("Bus 3",PV_B3),("Bus 4",PV_B4),("Bus 5",PV_B5)]))
            analysis_box("Bus 5 (at load) consistently outperforms Bus 3 (near substation). Reactive power losses in line impedance reduce effectiveness at more distant buses.")
        with g2:
            avgs=[round(sum(PV_B3)/24,2),round(sum(PV_B4)/24,2),round(sum(PV_B5)/24,2)]
            fb=go.Figure()
            fb.add_trace(go.Bar(x=["Bus 3","Bus 4","Bus 5"],y=avgs,marker_color=[C["purple"],C["blue"],C["orange"]],text=[f"{v:.2f}%" for v in avgs],textposition="outside",textfont=dict(size=11,color=C["text"])))
            fb.add_hline(y=2.0,line_dash="dot",line_color=C["warn"],annotation_text="2% target",annotation_font_size=10)
            lay_bus=base_layout("Average Daily CVR Reduction by PV Bus",height=300)
            lay_bus["yaxis"]={"title":"Avg % Reduction","range":[0,max(avgs)*1.35]}
            fb.update_layout(**lay_bus); show_chart(fb)
            analysis_box(f"Bus 5: {avgs[2]}% avg · Bus 4: {avgs[1]}% avg · Bus 3: {avgs[0]}% avg.")
        panel("Why Three Power Factors (0.90, 0.95, 0.98)?","<p>PF determines how much reactive power the load draws. Lower PF → less reactive capacity for voltage control. <b>PF 0.90</b> = hardest condition; <b>PF 0.95</b> = IESO standard minimum (most realistic); <b>PF 0.98</b> = high PF with capacitor banks.</p>")
        g5,g6=st.columns(2)
        with g5:
            show_chart(_lc2("% CVR Reduction by Power Factor",[("PF 0.90",PF_90),("PF 0.95",PF_95),("PF 0.98",PF_98)]))
            analysis_box("Higher PF = more reactive headroom = higher CVR effectiveness. PF 0.98 nearly doubles the reduction vs PF 0.90 at peak hours.")
        with g6:
            pf_avgs=[round(sum(PF_90)/24,2),round(sum(PF_95)/24,2),round(sum(PF_98)/24,2)]
            fp=go.Figure()
            fp.add_trace(go.Bar(x=["PF 0.90","PF 0.95","PF 0.98"],y=pf_avgs,marker_color=[C["purple"],C["blue"],C["orange"]],text=[f"{v:.2f}%" for v in pf_avgs],textposition="outside",textfont=dict(size=11,color=C["deep"])))
            fp.add_hline(y=2.0,line_dash="dot",line_color=C["warn"])
            lay_pf=base_layout("Average CVR Reduction by Power Factor",height=340)
            lay_pf["yaxis"]={"title":"Avg % Reduction","range":[0,max(pf_avgs)*1.35]}
            fp.update_layout(**lay_pf); show_chart(fp)
            analysis_box(f"PF 0.90: {pf_avgs[0]}% · PF 0.95: {pf_avgs[1]}% · PF 0.98: {pf_avgs[2]}%.")

    with tab_ieee:
        section_heading("IEEE 14-Bus — Design Decisions","")
        panel("Why the IEEE 14-Bus System?","<p>The IEEE 14-bus system is the smallest standardized test network that captures the complexity of a real transmission/sub-transmission power system. Unlike the single-feeder Dx study, the IEEE 14-bus tests CVR in a <em>meshed</em> network where voltage at one bus affects all others — validating that the CVR strategy extends beyond simple radial feeders.</p>")
        render_image("img_ieee","IEEE 14-Bus System Network","90%")
        panel("Why Were Buses 4, 9, and 14 Chosen?","<p>Selected based on <b>load size</b> (three largest non-industrial loads) and <b>load type</b> (industrial loads at buses 2 and 3 are constant power — CVR has minimal effect). Bus 14 is most electrically distant from generators, Bus 9 intermediate, Bus 4 relatively close.</p>")
        panel("Why 3 PV Farm Combinations?","<p><b>1. One large farm at Bus 4</b> — Tests whether concentrating reactive power at one bus is sufficient. Bus 4 benefits greatly (~4.5%) but Bus 14 drops below 2%.<br><b>2. Medium at Bus 4 + two small at Bus 9 and 14</b> — All buses exceed 2%; Bus 14 achieves ~3.5%. Best overall configuration.<br><b>3. Three equal small farms</b> — Bus 14 leads (~3.2%) but Bus 4 drops to ~1.1% — equal sizing is suboptimal.</p>")

    with tab_other:
        section_heading("CVR Study Design Decisions","")
        panel("How Are Hourly Loads Calculated?","<p>Hourly demand data from the IESO was analyzed to compute the average load at each hour as a percentage of the peak value. Studies run on the Dx Feeder and IEEE 14-bus system used these percentages to scale the loads appropriately for different times of day.</p>")
        g_a,g_b=st.columns(2)
        with g_a:
            fi=go.Figure()
            fi.add_trace(go.Scatter(x=HOURS,y=IESO_PCT,mode="lines+markers",line=dict(color=C["purple"],width=3),marker=dict(size=5),fill="tozeroy",fillcolor="rgba(184,108,224,0.08)"))
            fi.add_hline(y=100,line_dash="dot",line_color=C["warn"],annotation_text="100% peak (hr 18)")
            fi.update_layout(**base_layout("Average Hourly Demand in 2024 (% of Peak)",height=320))
            fi.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2))); fi.update_yaxes(title="% of Peak Demand")
            show_chart(fi)
        with g_b:
            fi2=go.Figure()
            fi2.add_trace(go.Bar(x=HOURS,y=IESO_AVG_MW,marker_color=[C["purple"] if v==max(IESO_AVG_MW) else C["blue"] for v in IESO_AVG_MW],opacity=0.85))
            fi2.update_layout(**base_layout("Average Ontario Demand by Hour in 2024 (MW)",height=320))
            fi2.update_xaxes(title="Hour of Day",tickvals=list(range(1,25,2))); fi2.update_yaxes(title="Average MW")
            show_chart(fi2)
        panel("How Is % Reduction Calculated?","""
        <p>The load reduction percentage is defined as:</p>
        <div style="text-align:center;padding:0.8rem;font-size:1.1rem;font-weight:600;background:#f8f4fe;border-radius:8px;margin:0.5rem 0 0.8rem 0;">
            % Reduction = (MW<sub>after CVR</sub> − MW<sub>before CVR</sub>) / MW<sub>before CVR</sub>
        </div>
        <p>Where "MW" refers to the load at <b>Bus 5 in the Dx feeder</b> and <b>buses 4, 9, and 14 in the IEEE 14-bus system</b>.</p>
        """)
        panel("Why Three Sun Ratings (Very Sunny, Moderately Sunny, Cloudy)?","<p>The sun rating determines how much active power the PV farm produces and how much reactive power support it can provide. IEEE Standard 2800-2022 specifies that inverters must maintain PF ≥ 0.95 at all active power levels. The maximum reactive power available is Q<sub>max</sub> = √(S² − P²), which decreases as P increases. <b>Counterintuitive result:</b> CVR is most effective on cloudy days because the inverter has the most reactive capacity available when producing little or no real power.</p>")
        _SUN_HRS=list(range(1,25))
        _SUNNY_P=[0.000,0.000,0.000,0.000,0.000,0.006,0.063,0.237,0.448,0.617,0.734,0.786,0.771,0.746,0.663,0.532,0.354,0.167,0.050,0.008,0.000,0.000,0.000,0.000]
        _SUNNY_Q=[1.000,1.000,1.000,1.000,1.000,1.000,0.998,0.972,0.894,0.787,0.679,0.619,0.637,0.666,0.749,0.847,0.935,0.986,0.999,1.000,1.000,1.000,1.000,1.000]
        _MOD_P=[0.000,0.000,0.000,0.000,0.000,0.000,0.002,0.044,0.150,0.250,0.323,0.357,0.322,0.303,0.209,0.103,0.030,0.006,0.000,0.000,0.000,0.000,0.000,0.000]
        _MOD_Q=[1.000,1.000,1.000,1.000,1.000,1.000,1.000,0.999,0.989,0.968,0.946,0.934,0.947,0.953,0.978,0.995,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000]
        _CLOUDY_P=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.007,0.022,0.052,0.087,0.116,0.116,0.101,0.070,0.045,0.028,0.007,0.000,0.000,0.000,0.000,0.000,0.000]
        _CLOUDY_Q=[1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,0.999,0.996,0.993,0.993,0.995,0.998,0.999,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000]
        def _pq_chart(title,p_vals,q_vals,p_color,q_color,title_color):
            f=go.Figure()
            f.add_trace(go.Scatter(x=_SUN_HRS,y=p_vals,name="P Output",mode="lines",line=dict(color=p_color,width=2.5),showlegend=True))
            f.add_trace(go.Scatter(x=_SUN_HRS,y=q_vals,name="Q Availability",mode="lines",line=dict(color=q_color,width=2.5),showlegend=True))
            lay=base_layout("",height=260)
            lay["title"]=dict(text=f"P and Q Curves of Solar Farm on<br><b style='color:{title_color};'>{title}</b> in p.u.",x=0.5,xanchor="center",font=dict(size=13,color=C["deep"]))
            lay["margin"]=dict(l=10,r=10,t=70,b=40)
            lay["legend"]=dict(orientation="h",y=-0.18,x=0.5,xanchor="center",font=dict(size=10))
            f.update_layout(**lay)
            f.update_xaxes(title="Hour",tickvals=[1,6,11,16,21],range=[1,24])
            f.update_yaxes(title="p.u.",range=[0,1.25])
            return f
        sc1,sc2,sc3=st.columns(3)
        with sc1: show_chart(_pq_chart("Sunny Day",_SUNNY_P,_SUNNY_Q,C["orange"],C["gold"],C["gold"]))
        with sc2: show_chart(_pq_chart("Moderately Sunny Day",_MOD_P,_MOD_Q,C["blue"],C["purple"],C["blue"]))
        with sc3: show_chart(_pq_chart("Cloudy Day",_CLOUDY_P,_CLOUDY_Q,C["pink"],C["purple"],C["purple"]))
        panel("Why the 2% Reduction Requirement?","<p>The 2% demand reduction threshold was set based on published CVR factor literature and Ontario utility practice. A CVR factor of approximately 0.5–0.8 is typical for mixed residential/commercial loads, meaning a 3% voltage reduction produces roughly 1.5–2.4% demand reduction. Our target of 0.97 pu represents a 3% drop from 1.00 pu, which should reliably produce ≥2% demand reduction for the load types modelled.</p>")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR & MAIN
# ═══════════════════════════════════════════════════════════════════════════
def sidebar_menu() -> str:
    st.sidebar.markdown(f"""
    <div style="padding:0.4rem 0 0.8rem 0;">
        <div style="font-size:0.75rem;color:{C["purple"]};font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">
            ECE 4416 · Group 4
        </div>
        <div style="font-size:1.25rem;font-weight:800;color:{C["deep"]};margin-top:0.2rem;">CVR Dashboard</div>
        <div style="font-size:0.85rem;color:{C["muted"]};margin-top:0.15rem;">Western University</div>
    </div>""", unsafe_allow_html=True)
    return st.sidebar.radio("Menu", [
        "Problem Statement",
        "Dx Feeder Results",
        "IEEE 14-Bus Results",
        "Design Thought Process",
        "Forecasting Model",
        "Prototype",
        "Files",
    ], index=0)

# ── Main ──────────────────────────────────────────────────────
try:
    constz_raw, zip_df, zip_analysis, ieee, cost_dx, cost_full, consti_raw = load_data()
    constz = prepare_constz(constz_raw)
except Exception as e:
    st.error(f"Failed to load files: {e}"); st.stop()

selected_page = sidebar_menu()

if selected_page == "Problem Statement":
    page_about()
elif selected_page == "Dx Feeder Results":
    page_dx_results(constz_raw, constz, cost_dx, cost_full)
elif selected_page == "IEEE 14-Bus Results":
    page_ieee_results(ieee)
elif selected_page == "Forecasting Model":
    page_ai(constz_raw, consti_raw, zip_df)
elif selected_page == "Prototype":
    page_prototype()
elif selected_page == "Files":
    page_excel_data()
elif selected_page == "Design Thought Process":
    page_design()