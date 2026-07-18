"""
Anvil — Forge a Model
A single-file Streamlit prototype: upload a CSV, pick a target column,
AutoML trains 5 algorithms, picks a winner, and lets you test predictions live.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

# ONNX support for "bring your own model" import is optional — if the
# packages aren't installed, the rest of the app (training, prediction,
# deploy) still works fine; only the Import Model page degrades to a
# friendly message instead of crashing the whole app.
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

MAX_ONNX_BYTES = 50 * 1024 * 1024  # 50MB

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anvil — Forge a Model",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Design tokens (forge / foundry aesthetic)
#   iron-black   #16140F  background
#   forge-char   #221D17  panels
#   ash-grey     #9C9488  secondary text / borders
#   ember        #E8580C  primary accent (molten steel)
#   spark-yellow #FFC145  secondary accent (hot spark)
#   steel-white  #F3EEE4  primary text
# ----------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --iron-black: #16140F;
    --forge-char: #221D17;
    --forge-char-2: #2B241C;
    --ash-grey: #9C9488;
    --ember: #E8580C;
    --ember-dim: #B8460A;
    --spark: #FFC145;
    --steel-white: #F3EEE4;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, var(--iron-black) 0%, #100E0A 100%);
    color: var(--steel-white);
}

/* Kill default Streamlit chrome that fights the theme.
   No sidebar in this layout, so no collapse arrow to worry about. */
#MainMenu, footer, header {visibility: hidden;}

.block-container {padding-top: 1.5rem; max-width: 1200px;}

h1, h2, h3, h4 {
    font-family: 'Oswald', sans-serif !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--steel-white) !important;
}

h1 {
    border-bottom: 3px solid var(--ember);
    padding-bottom: 0.4rem;
    display: inline-block;
}

p, li, span, label, .stMarkdown {
    color: var(--steel-white);
}

/* Station dashboard cards */
.station-card {
    background: var(--forge-char);
    border: 1px solid #3A3126;
    border-top: 3px solid var(--ember);
    padding: 18px 18px 6px 18px;
    height: 100%;
    transition: border-color 0.15s ease;
}
div[data-testid="column"]:has(.station-card):hover .station-card {
    border-top-color: var(--spark);
}

/* Top status strip */
.forge-topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid #3A3126;
    padding-bottom: 12px;
    margin-bottom: 6px;
}

/* Buttons: struck-metal look */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(180deg, var(--ember) 0%, var(--ember-dim) 100%);
    color: var(--steel-white);
    border: none;
    border-radius: 0px;
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    box-shadow: 0 0 0 1px #000 inset;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    box-shadow: 0 0 14px var(--ember);
    transform: translateY(-1px);
}

/* Cards */
div[data-testid="stMetric"] {
    background: var(--forge-char);
    border: 1px solid #3A3126;
    border-left: 3px solid var(--ember);
    padding: 14px 16px;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--spark) !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--ash-grey) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem !important;
}

/* Dataframes / tables */
.stDataFrame, .stTable {
    border: 1px solid #3A3126 !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    background: var(--forge-char-2);
    border: 1px dashed var(--ash-grey);
    border-radius: 0px;
}

/* Selects / inputs */
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
    background: var(--forge-char-2) !important;
    color: var(--steel-white) !important;
    border-radius: 0px !important;
    border: 1px solid #3A3126 !important;
}

/* Progress bar as a molten pour */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--ember-dim), var(--ember), var(--spark));
}

hr {border-color: #3A3126;}

code {
    background: var(--forge-char-2) !important;
    color: var(--spark) !important;
}

.eyebrow {
    color: var(--ember);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-size: 0.78rem;
}

.forge-caption {
    color: var(--ash-grey);
    font-size: 0.92rem;
}

/* ---- Training overlay: two hammers pounding while the forge trains ---- */
.training-overlay {
    position: fixed;
    inset: 0;
    background: rgba(16, 14, 10, 0.93);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.training-rig {
    position: relative;
    width: 260px;
    height: 150px;
}
.ov-anvil {
    position: absolute;
    bottom: 18px;
    left: 50%;
    transform: translateX(-50%);
    width: 110px;
    height: 30px;
    background: #3A3126;
    clip-path: polygon(8% 100%, 0% 40%, 20% 0%, 80% 0%, 100% 40%, 92% 100%);
    box-shadow: 0 0 24px rgba(0, 0, 0, 0.6);
}
.ov-hammer {
    position: absolute;
    top: 0;
    width: 20px;
    height: 88px;
    transform-origin: top center;
}
.ov-hammer-left {
    left: 46px;
    animation: ov-swing-left 1.1s ease-in-out infinite;
}
.ov-hammer-right {
    right: 46px;
    animation: ov-swing-right 1.1s ease-in-out infinite;
    animation-delay: 0.55s;
}
.ov-hammer-handle {
    width: 8px;
    height: 68px;
    margin: 0 auto;
    background: #6B5A44;
    border-radius: 3px;
}
.ov-hammer-head {
    width: 34px;
    height: 18px;
    margin: -4px auto 0;
    background: linear-gradient(180deg, var(--ash-grey), #514A40);
    border-radius: 3px;
    box-shadow: 0 0 0 1px #000 inset;
}
@keyframes ov-swing-left {
    0%, 100% { transform: rotate(-40deg); }
    45%      { transform: rotate(8deg); }
    55%      { transform: rotate(8deg); }
}
@keyframes ov-swing-right {
    0%, 100% { transform: rotate(40deg); }
    45%      { transform: rotate(-8deg); }
    55%      { transform: rotate(-8deg); }
}
.ov-impact {
    position: absolute;
    bottom: 44px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--spark);
    opacity: 0;
}
.ov-impact-left  { left: 88px;  animation: ov-impact-flash 1.1s ease-in-out infinite; }
.ov-impact-right { right: 88px; animation: ov-impact-flash 1.1s ease-in-out infinite; animation-delay: 0.55s; }
@keyframes ov-impact-flash {
    0%, 40%  { opacity: 0; transform: scale(0.4); }
    50%      { opacity: 0.9; transform: scale(1.6); box-shadow: 0 0 20px 6px rgba(255, 193, 69, 0.6); }
    60%, 100% { opacity: 0; transform: scale(0.4); }
}
.overlay-caption {
    margin-top: 24px;
    color: var(--steel-white);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em;
    font-size: 0.9rem;
    text-transform: uppercase;
    text-align: center;
}

/* ---- Station hub hero: drifting molten waves behind the heading ---- */
.hub-hero {
    position: relative;
    height: 110px;
    overflow: hidden;
    margin-bottom: 12px;
    border-bottom: 1px solid #3A3126;
    display: flex;
    align-items: center;
}
.hub-hero .wave-svg {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 200%;
    height: 100%;
}
.hub-hero .wave-back  { fill: var(--spark);      opacity: 0.05; animation: hub-wave-scroll 26s linear infinite; }
.hub-hero .wave-mid   { fill: var(--ember-dim);  opacity: 0.08; animation: hub-wave-scroll 18s linear infinite reverse; }
.hub-hero .wave-front { fill: var(--ember);      opacity: 0.10; animation: hub-wave-scroll 12s linear infinite; }
@keyframes hub-wave-scroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}
.hub-hero-label {
    position: relative;
    z-index: 1;
    padding-left: 4px;
}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def training_overlay_html(caption):
    return f"""
    <div class="training-overlay">
        <div class="training-rig">
            <div class="ov-hammer ov-hammer-left">
                <div class="ov-hammer-handle"></div>
                <div class="ov-hammer-head"></div>
            </div>
            <div class="ov-hammer ov-hammer-right">
                <div class="ov-hammer-handle"></div>
                <div class="ov-hammer-head"></div>
            </div>
            <div class="ov-anvil"></div>
            <div class="ov-impact ov-impact-left"></div>
            <div class="ov-impact ov-impact-right"></div>
        </div>
        <p class="overlay-caption">{caption}</p>
    </div>
    """


# ----------------------------------------------------------------------------
# Splash: the unveiling — five hammer strikes forge the word ANVIL
# ----------------------------------------------------------------------------

def render_splash():
    splash_html = """
    <style>
    body {margin:0;}
    .forge-stage {
        position: relative;
        height: 380px;
        width: 100%;
        background: radial-gradient(ellipse at 50% 65%, #1F1A13 0%, #100E0A 70%);
        overflow: hidden;
        font-family: 'Oswald', sans-serif;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .letters {
        display: flex;
        gap: 0.35em;
        z-index: 3;
    }
    .letters span {
        font-size: 5.5rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #F3EEE4;
        opacity: 0;
        display: inline-block;
        text-shadow: 0 0 24px rgba(232, 88, 12, 0.0);
        transform: translateY(-14px) scale(1.3);
        filter: blur(6px);
        animation: strike-in 0.5s ease-out forwards;
    }
    .letters span:nth-child(1) { animation-delay: 0.15s; }
    .letters span:nth-child(2) { animation-delay: 0.75s; }
    .letters span:nth-child(3) { animation-delay: 1.35s; }
    .letters span:nth-child(4) { animation-delay: 1.95s; }
    .letters span:nth-child(5) { animation-delay: 2.55s; }

    @keyframes strike-in {
        0%   { opacity: 0; transform: translateY(-14px) scale(1.3); filter: blur(6px); text-shadow: 0 0 0 rgba(232,88,12,0);}
        35%  { opacity: 1; transform: translateY(2px) scale(0.96); filter: blur(0px); text-shadow: 0 0 34px rgba(255,193,69,0.9);}
        70%  { transform: translateY(0px) scale(1.03); text-shadow: 0 0 14px rgba(232,88,12,0.55);}
        100% { opacity: 1; transform: translateY(0px) scale(1); text-shadow: 0 0 6px rgba(232,88,12,0.35);}
    }

    .anvil-base {
        position: absolute;
        bottom: 62px;
        left: 50%;
        transform: translateX(-50%);
        width: 130px;
        height: 34px;
        background: #3A3126;
        clip-path: polygon(8% 100%, 0% 40%, 20% 0%, 80% 0%, 100% 40%, 92% 100%);
        z-index: 1;
        box-shadow: 0 0 20px rgba(0,0,0,0.6);
    }

    .flash {
        position: absolute;
        bottom: 96px;
        left: 50%;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #FFC145;
        transform: translate(-50%, 50%) scale(0);
        opacity: 0;
        z-index: 2;
        animation: flash-pulse 0.5s ease-out forwards;
    }
    .flash:nth-of-type(1) { animation-delay: 0.15s; }
    .flash:nth-of-type(2) { animation-delay: 0.75s; }
    .flash:nth-of-type(3) { animation-delay: 1.35s; }
    .flash:nth-of-type(4) { animation-delay: 1.95s; }
    .flash:nth-of-type(5) { animation-delay: 2.55s; }

    @keyframes flash-pulse {
        0%   { transform: translate(-50%, 50%) scale(0.2); opacity: 0.95; box-shadow: 0 0 0 rgba(255,193,69,0);}
        60%  { transform: translate(-50%, 50%) scale(2.6); opacity: 0.5; box-shadow: 0 0 40px 10px rgba(255,193,69,0.5);}
        100% { transform: translate(-50%, 50%) scale(3.4); opacity: 0; box-shadow: 0 0 0 rgba(255,193,69,0);}
    }

    .spark {
        position: absolute;
        bottom: 96px;
        left: 50%;
        width: 4px;
        height: 4px;
        background: #FFC145;
        z-index: 2;
        opacity: 0;
        animation-name: spark-fly;
        animation-duration: 0.65s;
        animation-timing-function: cubic-bezier(.2,.8,.4,1);
        animation-fill-mode: forwards;
    }
    @keyframes spark-fly {
        0%   { opacity: 1; transform: translate(-50%, 0) scale(1); }
        100% { opacity: 0; transform: translate(calc(-50% + var(--tx)), var(--ty)) scale(0.3); }
    }

    .tagline {
        position: absolute;
        bottom: 26px;
        left: 50%;
        transform: translateX(-50%);
        color: #9C9488;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        opacity: 0;
        animation: fade-in 0.8s ease-out forwards;
        animation-delay: 3.1s;
        white-space: nowrap;
    }
    @keyframes fade-in { to { opacity: 1; } }
    </style>

    <div class="forge-stage">
        <div class="anvil-base"></div>
        <div class="letters">
            <span>A</span><span>N</span><span>V</span><span>I</span><span>L</span>
        </div>
        <div class="flash"></div><div class="flash"></div><div class="flash"></div>
        <div class="flash"></div><div class="flash"></div>
        SPARKS_PLACEHOLDER
        <div class="tagline">Where raw data becomes a trained model</div>
    </div>
    """

    # Generate spark bursts (5 strikes x 6 sparks each) with pseudo-random trajectories.
    rng = np.random.default_rng(7)
    sparks = []
    for strike_i in range(5):
        delay = 0.15 + strike_i * 0.6
        for _ in range(6):
            angle = rng.uniform(0, 3.14159)
            dist = rng.uniform(40, 110)
            tx = np.cos(angle) * dist * rng.choice([-1, 1])
            ty = -abs(np.sin(angle) * dist) - rng.uniform(10, 40)
            sparks.append(
                f'<div class="spark" style="--tx:{tx:.0f}px; --ty:{ty:.0f}px; '
                f'animation-delay:{delay:.2f}s;"></div>'
            )
    splash_html = splash_html.replace("SPARKS_PLACEHOLDER", "".join(sparks))

    components.html(splash_html, height=400)


# ----------------------------------------------------------------------------
# ML helpers
# ----------------------------------------------------------------------------

def detect_task_type(y: pd.Series) -> str:
    # is_numeric_dtype correctly returns False for object, category, and the
    # newer pandas "string" / "string[pyarrow]" dtypes alike — checking dtype
    # against `object` directly missed those newer string dtypes and let a
    # text column like "Up"/"Down" fall through to regression.
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= max(10, int(0.02 * len(y))):
        return "classification"
    return "regression"


def get_candidate_models(task_type: str):
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True, random_state=42),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Support Vector Machine": SVR(),
    }


def build_pipeline(estimator, numeric_cols, categorical_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def score_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def score_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
    }


def primary_metric(task_type):
    return "accuracy" if task_type == "classification" else "r2"


def train_all_models(X, y, task_type, numeric_cols, categorical_cols, progress_cb=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None,
    )
    candidates = get_candidate_models(task_type)
    results = {}
    for i, (name, estimator) in enumerate(candidates.items()):
        if progress_cb:
            progress_cb(i, len(candidates), name)
        pipe = build_pipeline(estimator, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = (
            score_classification(y_test, preds)
            if task_type == "classification"
            else score_regression(y_test, preds)
        )
        try:
            imp = permutation_importance(
                pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
            )
            importances = pd.Series(imp.importances_mean, index=X.columns).sort_values(
                ascending=False
            )
        except Exception:
            importances = pd.Series(dtype=float)

        results[name] = {"pipeline": pipe, "metrics": metrics, "importances": importances}
    if progress_cb:
        progress_cb(len(candidates), len(candidates), "Done")
    return results, X_test, y_test


# ----------------------------------------------------------------------------
# ONNX import helpers — "bring your own model"
#
# This prototype only ever unpickles models it just trained itself in this
# same process, which is safe. It deliberately does NOT accept uploaded
# .pkl files: unpickling is arbitrary code execution (`__reduce__`), so a
# pickle a user hands you is a program, not a data file. ONNX is a
# protobuf-described computation graph — the runtime only evaluates the
# numeric ops declared in it, so importing one can't run arbitrary Python.
# That's why "bring your own model" here is ONNX-only.
# ----------------------------------------------------------------------------

class OnnxImportError(Exception):
    pass


def validate_onnx_bytes(file_bytes: bytes):
    if len(file_bytes) > MAX_ONNX_BYTES:
        raise OnnxImportError(f"File is too large (max {MAX_ONNX_BYTES // (1024*1024)}MB).")
    try:
        model = onnx.load_model_from_string(file_bytes)
    except Exception as e:
        raise OnnxImportError(f"Not a valid ONNX file: {e}")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        raise OnnxImportError(f"ONNX model failed validation: {e}")
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            raise OnnxImportError(
                "This model references external data files, which isn't allowed for "
                "imported models — re-export with weights embedded in one .onnx file."
            )
    if len(model.graph.input) == 0:
        raise OnnxImportError("Model graph has no inputs.")
    return model


def load_onnx_session(file_bytes: bytes):
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    return ort.InferenceSession(file_bytes, sess_options=so, providers=["CPUExecutionProvider"])


def _onnx_np_dtype(elem_type: int):
    return {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.STRING: np.str_,
    }.get(elem_type, np.float32)


def _onnx_type_name(ort_type: str) -> str:
    mapping = {
        "tensor(float)": "FLOAT", "tensor(double)": "DOUBLE",
        "tensor(int64)": "INT64", "tensor(int32)": "INT32", "tensor(string)": "STRING",
    }
    return mapping.get(ort_type, "FLOAT")


def _onnx_coerce(val, dtype):
    if val is None or val == "":
        return "" if dtype == np.str_ else 0
    if dtype == np.str_:
        return str(val)
    try:
        return dtype(val)
    except (TypeError, ValueError):
        return 0


def _onnx_scalar(arr):
    val = np.asarray(arr).reshape(-1)[0]
    return val.item() if hasattr(val, "item") else val


def _onnx_proba_dict(arr, class_names):
    try:
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            return {str(k): float(v) for k, v in arr[0].items()}
        vec = np.asarray(arr).reshape(-1)
        if class_names and len(class_names) == len(vec):
            return {c: float(p) for c, p in zip(class_names, vec)}
        return {str(i): float(p) for i, p in enumerate(vec)}
    except Exception:
        return None


def onnx_predict_single(session, feature_columns, class_names, task_type, input_dict):
    inputs = session.get_inputs()
    feed = {}
    if len(inputs) == 1 and len(inputs[0].shape) in (1, 2):
        inp = inputs[0]
        dtype = _onnx_np_dtype(getattr(onnx.TensorProto, _onnx_type_name(inp.type)))
        row = [_onnx_coerce(input_dict.get(col), dtype) for col in feature_columns]
        feed[inp.name] = np.array([row], dtype=dtype)
    else:
        for inp in inputs:
            if inp.name not in feature_columns:
                raise OnnxImportError(f"Model expects an input named '{inp.name}' not in your feature columns.")
            dtype = _onnx_np_dtype(getattr(onnx.TensorProto, _onnx_type_name(inp.type)))
            feed[inp.name] = np.array([[_onnx_coerce(input_dict.get(inp.name), dtype)]], dtype=dtype)

    outputs = session.run(None, feed)
    output_names = [o.name for o in session.get_outputs()]
    pred, proba = None, None
    for name, val in zip(output_names, outputs):
        if "label" in name.lower() or (pred is None and "prob" not in name.lower()):
            pred = _onnx_scalar(val)
        elif "prob" in name.lower() or proba is None:
            proba = _onnx_proba_dict(val, class_names)
    if pred is None:
        pred = _onnx_scalar(outputs[0])
    return pred, (proba if task_type == "classification" else None)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

def page_upload_train():
    st.markdown('<div class="eyebrow">Step 1</div>', unsafe_allow_html=True)
    st.markdown("# Upload &amp; Forge")
    st.markdown(
        '<p class="forge-caption">Drop in a CSV, choose what you want to predict, '
        "and Anvil trains five algorithms to find the strongest one.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Dataset (CSV)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df = df

    if st.session_state.get("df") is None:
        st.info("Upload a CSV to begin.")
        return

    df = st.session_state.df
    st.markdown("### Preview")
    st.dataframe(df.head(8), use_container_width=True)
    st.caption(f"{df.shape[0]:,} rows \u00b7 {df.shape[1]} columns")

    st.markdown("### Choose a target column")
    target = st.selectbox("What should Anvil learn to predict?", options=df.columns, index=len(df.columns) - 1)

    if st.button("Start training"):
        clean = df.dropna(subset=[target]).copy()
        y_raw = clean[target]
        X = clean.drop(columns=[target])

        task_type = detect_task_type(y_raw)

        label_encoder = None
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index)
        else:
            y = y_raw

        numeric_cols = list(X.select_dtypes(include=np.number).columns)
        categorical_cols = list(X.select_dtypes(exclude=np.number).columns)

        # Fill simple gaps so training doesn't choke on a prototype dataset.
        for c in numeric_cols:
            X[c] = X[c].fillna(X[c].median())
        for c in categorical_cols:
            X[c] = X[c].fillna("missing")

        overlay = st.empty()
        overlay.markdown(training_overlay_html("Warming the forge\u2026"), unsafe_allow_html=True)

        def progress_cb(i, total, name):
            overlay.markdown(
                training_overlay_html(f"Striking model {min(i + 1, total)} of {total}: {name}"),
                unsafe_allow_html=True,
            )
            time.sleep(0.15)

        results, X_test, y_test = train_all_models(
            X, y, task_type, numeric_cols, categorical_cols, progress_cb=progress_cb
        )
        overlay.empty()

        metric = primary_metric(task_type)
        best_name = max(results, key=lambda n: results[n]["metrics"][metric])

        st.session_state.task_type = task_type
        st.session_state.target = target
        st.session_state.feature_cols = list(X.columns)
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols
        st.session_state.X_train_sample = X
        st.session_state.results = results
        st.session_state.best_name = best_name
        st.session_state.label_encoder = label_encoder
        st.session_state.trained = True

        st.success(f"Best model: **{best_name}**  \u2014  head to the Leaderboard tab.")


def page_import_model():
    st.markdown('<div class="eyebrow">Bring your own model</div>', unsafe_allow_html=True)
    st.markdown("# Import Model")
    st.markdown(
        '<p class="forge-caption">Already trained something outside Anvil? Import it as an '
        "<b>ONNX</b> file rather than a pickle. Anvil only ever evaluates the numeric ops in an "
        "ONNX graph, so importing one can't run arbitrary code the way loading an uploaded "
        "pickle/joblib file could \u2014 a pickle is a program, not a data format. If your model is "
        "currently a pickle, convert it first (e.g. <code>skl2onnx.to_onnx(...)</code> for "
        "scikit-learn) and upload the resulting <code>.onnx</code> file.</p>",
        unsafe_allow_html=True,
    )

    if not ONNX_AVAILABLE:
        st.warning(
            "The `onnx` and `onnxruntime` packages aren't installed in this environment. "
            "Run `pip install onnx onnxruntime` and restart the app to enable model import."
        )
        return

    uploaded = st.file_uploader("Model file (.onnx)", type=["onnx"])
    model_name = st.text_input("Model name", value="Imported model")
    task_type = st.selectbox("Task type", options=["classification", "regression"])
    feature_cols_raw = st.text_input(
        "Feature columns, in the exact order your model expects (comma-separated)",
        placeholder="age, income, region",
    )
    class_names_raw = ""
    if task_type == "classification":
        class_names_raw = st.text_input(
            "Class names, in the order your model outputs them",
            placeholder="rejected, approved",
        )

    if st.button("Import ONNX model"):
        if uploaded is None:
            st.error("Choose an .onnx file first.")
            return
        feature_columns = [c.strip() for c in feature_cols_raw.split(",") if c.strip()]
        class_names = [c.strip() for c in class_names_raw.split(",") if c.strip()]
        if not feature_columns:
            st.error("List the feature columns your model expects, comma-separated.")
            return
        if task_type == "classification" and not class_names:
            st.error("List the class names your model outputs, comma-separated.")
            return

        file_bytes = uploaded.getvalue()
        try:
            validate_onnx_bytes(file_bytes)
            session = load_onnx_session(file_bytes)
            sample = {c: 0 for c in feature_columns}
            onnx_predict_single(session, feature_columns, class_names, task_type, sample)
        except OnnxImportError as e:
            st.error(f"Couldn't import that model: {e}")
            return
        except Exception as e:
            st.error(f"Model loaded but a test prediction failed \u2014 check feature columns/order: {str(e)[:300]}")
            return

        st.session_state.imported_model = {
            "name": model_name or "Imported model",
            "session": session,
            "feature_columns": feature_columns,
            "class_names": class_names,
            "task_type": task_type,
            "file_bytes": file_bytes,
        }
        st.success(f"Imported '{model_name}' \u2014 head to Test a Prediction or Deploy.")

    if st.session_state.get("imported_model"):
        m = st.session_state.imported_model
        st.markdown("---")
        st.markdown(f"**Currently imported:** {m['name']} ({m['task_type']})")
        st.caption(f"Feature columns: {', '.join(m['feature_columns'])}")
        if st.button("Remove imported model"):
            del st.session_state.imported_model
            st.rerun()


def page_leaderboard():
    st.markdown('<div class="eyebrow">Step 2</div>', unsafe_allow_html=True)
    st.markdown("# Leaderboard")

    if not st.session_state.get("trained"):
        st.info("Train a model first on the Upload & Forge tab.")
        return

    task_type = st.session_state.task_type
    results = st.session_state.results
    metric = primary_metric(task_type)
    best_name = st.session_state.best_name

    rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update({k.upper(): round(v, 4) for k, v in r["metrics"].items()})
        rows.append(row)
    board = pd.DataFrame(rows).sort_values(metric.upper(), ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", [f"#{i+1}" for i in range(len(board))])

    st.dataframe(board, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    best_metrics = results[best_name]["metrics"]
    with c1:
        st.metric("Champion", best_name)
    with c2:
        st.metric(metric.upper(), f"{best_metrics[metric]:.3f}")
    with c3:
        other = [k for k in best_metrics if k != metric][0]
        st.metric(other.upper(), f"{best_metrics[other]:.3f}")

    st.markdown("### Score by model")
    chart_df = board[["Model", metric.upper()]].set_index("Model")
    st.bar_chart(chart_df, color="#E8580C")

    st.markdown(f"### What {best_name} pays attention to")
    imp = results[best_name]["importances"]
    if len(imp):
        st.bar_chart(imp.head(10), color="#FFC145")
    else:
        st.caption("Feature importance unavailable for this model/data combination.")


def page_predict():
    st.markdown('<div class="eyebrow">Step 3</div>', unsafe_allow_html=True)
    st.markdown("# Test a Prediction")

    trained = st.session_state.get("trained")
    imported = st.session_state.get("imported_model")

    if not trained and not imported:
        st.info("Train a model on the Upload & Forge tab, or import one on the Import Model tab, first.")
        return

    options = []
    if trained:
        options += list(st.session_state.results.keys())
    if imported:
        options += [f"{imported['name']} (imported ONNX)"]
    default_idx = options.index(st.session_state.best_name) if trained else 0
    model_choice = st.selectbox("Model", options=options, index=default_idx)

    is_imported_choice = imported and model_choice == f"{imported['name']} (imported ONNX)"

    if is_imported_choice:
        st.markdown("### Enter feature values")
        input_row = {}
        cols = st.columns(2)
        for i, c in enumerate(imported["feature_columns"]):
            with cols[i % 2]:
                input_row[c] = st.text_input(c, value="0")

        if st.button("Predict"):
            try:
                pred, proba = onnx_predict_single(
                    imported["session"], imported["feature_columns"], imported["class_names"],
                    imported["task_type"], input_row,
                )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)[:300]}")
                return
            if imported["task_type"] == "classification":
                st.success(f"Predicted: `{pred}`")
                if proba:
                    proba_df = pd.DataFrame(
                        {"class": list(proba.keys()), "probability": list(proba.values())}
                    ).sort_values("probability", ascending=False)
                    st.bar_chart(proba_df.set_index("class"), color="#E8580C")
            else:
                st.success(f"Predicted: `{pred:.4f}`" if isinstance(pred, float) else f"Predicted: `{pred}`")
        return

    results = st.session_state.results
    pipe = results[model_choice]["pipeline"]
    X_ref = st.session_state.X_train_sample
    numeric_cols = st.session_state.numeric_cols
    categorical_cols = st.session_state.categorical_cols

    st.markdown("### Enter feature values")
    input_row = {}
    cols = st.columns(2)
    for i, c in enumerate(numeric_cols):
        with cols[i % 2]:
            default_val = float(X_ref[c].median())
            input_row[c] = st.number_input(c, value=default_val)
    for i, c in enumerate(categorical_cols):
        with cols[(i + len(numeric_cols)) % 2]:
            options = sorted(X_ref[c].astype(str).unique().tolist())
            input_row[c] = st.selectbox(c, options=options)

    if st.button("Predict"):
        row_df = pd.DataFrame([input_row])[st.session_state.feature_cols]
        pred = pipe.predict(row_df)[0]

        if st.session_state.task_type == "classification":
            le = st.session_state.label_encoder
            label = le.inverse_transform([int(pred)])[0] if le is not None else pred
            st.success(f"Predicted **{st.session_state.target}**: `{label}`")
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(row_df)[0]
                classes = le.classes_ if le is not None else pipe.classes_
                proba_df = pd.DataFrame({"class": classes, "probability": proba}).sort_values(
                    "probability", ascending=False
                )
                st.bar_chart(proba_df.set_index("class"), color="#E8580C")
        else:
            st.success(f"Predicted **{st.session_state.target}**: `{pred:.4f}`")


def page_deploy():
    st.markdown('<div class="eyebrow">Step 4</div>', unsafe_allow_html=True)
    st.markdown("# Deploy")
    st.markdown(
        '<p class="forge-caption">This prototype trains and tests in one session. '
        "For a real deployment, export the model and serve it behind an API.</p>",
        unsafe_allow_html=True,
    )

    trained = st.session_state.get("trained")
    imported = st.session_state.get("imported_model")

    if not trained and not imported:
        st.info("Train a model on the Upload & Forge tab, or import one on the Import Model tab, first.")
        return

    if imported:
        st.markdown("### Imported ONNX model")
        st.download_button(
            f"Download {imported['name']} (.onnx)",
            data=imported["file_bytes"],
            file_name=f"{imported['name'].replace(' ', '_')}.onnx",
            mime="application/octet-stream",
        )
        st.caption(
            "This is the exact file you uploaded — it runs with onnxruntime (or any "
            "ONNX-compatible runtime) in any language, no scikit-learn required."
        )
        if trained:
            st.markdown("---")

    if not trained:
        return

    best_name = st.session_state.best_name
    pipe = st.session_state.results[best_name]["pipeline"]

    st.markdown("### Trained model")
    buf = io.BytesIO()
    pickle.dump(pipe, buf)
    st.download_button(
        "Download trained model (.pkl)",
        data=buf.getvalue(),
        file_name="anvil_model.pkl",
        mime="application/octet-stream",
    )

    st.markdown("### How you'd call this model")
    st.code(
        f"""import pickle
import pandas as pd

model = pickle.load(open("anvil_model.pkl", "rb"))

new_data = pd.DataFrame([{{
{chr(10).join(f'    "{c}": ...,' for c in st.session_state.feature_cols)}
}}])

prediction = model.predict(new_data)
print(prediction)
""",
        language="python",
    )

    st.markdown("### What a production API endpoint would look like")
    st.code(
        f"""POST /api/v1/predict/{best_name.lower().replace(' ', '-')}
Authorization: Bearer <api-key>
Content-Type: application/json

{{
{chr(10).join(f'  "{c}": ...,' for c in st.session_state.feature_cols)}
}}

\u2192 200 OK
{{ "prediction": ... }}""",
        language="text",
    )
    st.caption(
        "In Anvil's full Flask version, this endpoint is real and gated by a "
        "per-team API key. In this prototype it's illustrative only."
    )


def page_image_classification():
    st.markdown('<div class="eyebrow">Coming Soon</div>', unsafe_allow_html=True)
    st.markdown("# Image Classification")
    st.markdown(
        '<p class="forge-caption">Deep learning support is on the anvil \u2014 not '
        "ready yet, but here's what's planned.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: var(--forge-char); border: 1px solid #3A3126;
                    border-left: 3px solid var(--spark); padding: 24px 28px; margin-top: 12px;">
            <p style="color: var(--spark); font-family: 'IBM Plex Mono', monospace;
                      letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.8rem;
                      margin-bottom: 10px;">In progress</p>
            <p style="color: var(--steel-white); font-size: 1.05rem; margin-bottom: 6px;">
                Upload a zip of labeled image folders and fine-tune a pretrained model
                on your own classes.
            </p>
            <ul class="forge-caption" style="margin-top: 14px; line-height: 1.9;">
                <li>Transfer learning on a pretrained image backbone (no training from scratch)</li>
                <li>Live training curve while it fine-tunes</li>
                <li>Test the model on a new image right in the browser</li>
                <li>Download the trained weights when it's done</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("This tab is a placeholder for now \u2014 the tabular pages are fully working.")


# ----------------------------------------------------------------------------
# Onboarding tour — shown once per session, right after the splash
# ----------------------------------------------------------------------------

TOUR_STEPS = [
    {
        "eyebrow": "Welcome",
        "title": "Anvil forges a model from your data",
        "body": "Upload a spreadsheet, tell it what to predict, and Anvil trains "
                "several algorithms to find the one that fits your data best. "
                "This quick tour walks through each step \u2014 takes about 20 seconds.",
    },
    {
        "eyebrow": "Step 1",
        "title": "Upload & Forge",
        "body": "Drop in a CSV and pick the column you want to predict. Anvil "
                "figures out whether that's a category (classification) or a "
                "number (regression) on its own, then trains 5 models on it.",
    },
    {
        "eyebrow": "Optional",
        "title": "Already have a trained model?",
        "body": "You don't have to train from scratch here \u2014 the Import Model tab "
                "lets you upload a model you trained elsewhere and try it out on this "
                "project. It takes an ONNX file rather than a pickle, so bring an "
                "exported .onnx (e.g. via skl2onnx) along with the feature columns it "
                "expects, and you can test predictions with it right alongside anything "
                "trained in Anvil.",
    },
    {
        "eyebrow": "Step 2",
        "title": "Leaderboard",
        "body": "See all 5 models ranked side by side, with the winner highlighted "
                "and a chart of which columns in your data mattered most to it.",
    },
    {
        "eyebrow": "Step 3",
        "title": "Test a Prediction",
        "body": "A form built automatically from your dataset's own columns \u2014 "
                "fill it in and get a live prediction from any of the trained models "
                "(or an imported one, if you added one).",
    },
    {
        "eyebrow": "Step 4",
        "title": "Deploy",
        "body": "Download the winning model, plus a ready-made code snippet "
                "showing exactly how to load it and call it elsewhere.",
    },
]


def render_onboarding_tour():
    step_i = st.session_state.get("tour_step", 0)
    step = TOUR_STEPS[step_i]
    total = len(TOUR_STEPS)

    dots = "".join(
        f'<span style="display:inline-block; width:8px; height:8px; margin:0 4px; '
        f'background:{"var(--ember)" if i == step_i else "#3A3126"};"></span>'
        for i in range(total)
    )

    st.markdown(
        f"""
        <div style="max-width: 620px; margin: 40px auto 0; background: var(--forge-char);
                    border: 1px solid #3A3126; border-top: 3px solid var(--ember);
                    padding: 36px 40px;">
            <p class="eyebrow" style="margin-bottom: 4px;">{step['eyebrow']}</p>
            <h2 style="margin-top: 0; margin-bottom: 16px; font-size: 1.6rem;">{step['title']}</h2>
            <p style="color: var(--steel-white); font-size: 1.02rem; line-height: 1.65;">
                {step['body']}
            </p>
            <div style="text-align:center; margin-top: 22px;">{dots}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, c1, c2, c3, _ = st.columns([2, 1, 1, 1, 2])
    with c1:
        if step_i > 0 and st.button("\u2190 Back", use_container_width=True):
            st.session_state.tour_step = step_i - 1
            st.rerun()
    with c2:
        if st.button("Skip tour", use_container_width=True):
            st.session_state.tour_done = True
            st.rerun()
    with c3:
        label = "Start Forging \u2192" if step_i == total - 1 else "Next \u2192"
        if st.button(label, use_container_width=True):
            if step_i == total - 1:
                st.session_state.tour_done = True
            else:
                st.session_state.tour_step = step_i + 1
            st.rerun()


# ----------------------------------------------------------------------------
# App shell — station dashboard
# ----------------------------------------------------------------------------

STATIONS = [
    {
        "key": "upload",
        "eyebrow": "Step 1",
        "title": "Upload & Forge",
        "desc": "Drop in a CSV, pick a target column, and AutoML trains five algorithms for you.",
        "render": lambda: page_upload_train(),
    },
    {
        "key": "import",
        "eyebrow": "Optional",
        "title": "Import Model",
        "desc": "Already trained something elsewhere? Bring your own ONNX model into the forge.",
        "render": lambda: page_import_model(),
    },
    {
        "key": "leaderboard",
        "eyebrow": "Step 2",
        "title": "Leaderboard",
        "desc": "See all five models ranked side by side, with the winner highlighted.",
        "render": lambda: page_leaderboard(),
    },
    {
        "key": "predict",
        "eyebrow": "Step 3",
        "title": "Test a Prediction",
        "desc": "A form built automatically from your dataset's own columns — try it live.",
        "render": lambda: page_predict(),
    },
    {
        "key": "deploy",
        "eyebrow": "Step 4",
        "title": "Deploy",
        "desc": "Download the winning model plus a ready-made snippet to put it to work.",
        "render": lambda: page_deploy(),
    },
    {
        "key": "image",
        "eyebrow": "Coming Soon",
        "title": "Image Classification",
        "desc": "Deep learning support is on the anvil — not ready yet, but here's the plan.",
        "render": lambda: page_image_classification(),
    },
]


def render_topbar():
    brand_col, status_col, btn1_col, btn2_col = st.columns([3, 3, 1, 1])
    with brand_col:
        st.markdown(
            '<h1 style="font-size: 1.3rem; border: none; margin-bottom: 0; padding-bottom: 0;">ANVIL</h1>'
            '<p class="forge-caption" style="margin-top: -6px;">Forge a model from raw data.</p>',
            unsafe_allow_html=True,
        )
    with status_col:
        bits = []
        if st.session_state.get("trained"):
            bits.append(f'<span class="forge-caption">Best</span> <strong>{st.session_state.best_name}</strong>')
        if st.session_state.get("imported_model"):
            bits.append(
                f'<span class="forge-caption">Imported</span> '
                f'<strong>{st.session_state.imported_model["name"]}</strong> (ONNX)'
            )
        if bits:
            st.markdown(
                '<div style="text-align: right; padding-top: 10px;">' + " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(bits) + "</div>",
                unsafe_allow_html=True,
            )
    with btn1_col:
        if st.button("Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k != "splash_done":
                    del st.session_state[k]
            st.rerun()
    with btn2_col:
        if st.button("Tour", use_container_width=True):
            st.session_state.tour_step = 0
            st.session_state.tour_done = False
            st.rerun()
    st.markdown('<hr style="margin-top: 4px;">', unsafe_allow_html=True)


def render_station_hub():
    wave_path = (
        "M0,64 C100,90 200,20 300,50 C400,80 500,20 600,50 C700,80 800,20 900,50 "
        "C1000,80 1100,40 1200,64 C1300,90 1400,20 1500,50 C1600,80 1700,20 1800,50 "
        "C1900,80 2000,20 2100,50 C2200,80 2300,40 2400,64 L2400,120 L0,120 Z"
    )
    st.markdown(
        f"""
        <div class="hub-hero">
            <svg class="wave-svg wave-back" viewBox="0 0 2400 120" preserveAspectRatio="none">
                <path d="{wave_path}"></path>
            </svg>
            <svg class="wave-svg wave-mid" viewBox="0 0 2400 120" preserveAspectRatio="none">
                <path d="{wave_path}"></path>
            </svg>
            <svg class="wave-svg wave-front" viewBox="0 0 2400 120" preserveAspectRatio="none">
                <path d="{wave_path}"></path>
            </svg>
            <div class="hub-hero-label">
                <p class="eyebrow" style="margin-bottom: 4px;">Home</p>
                <h3 style="margin: 0;">Choose a station</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = [STATIONS[0:3], STATIONS[3:6]]
    for row in rows:
        cols = st.columns(3)
        for col, station in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="station-card">
                        <p class="eyebrow" style="margin-bottom: 6px;">{station['eyebrow']}</p>
                        <h3 style="margin: 0 0 8px 0; font-size: 1.05rem;">{station['title']}</h3>
                        <p class="forge-caption" style="min-height: 54px;">{station['desc']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Enter \u2192", key=f"enter_{station['key']}", use_container_width=True):
                    st.session_state.current_station = station["key"]
                    st.rerun()


def render_station_page(key):
    if st.button("\u2190 Back to stations", key="back_to_hub"):
        st.session_state.current_station = None
        st.rerun()
    station = next(s for s in STATIONS if s["key"] == key)
    station["render"]()


def main():
    inject_css()

    if not st.session_state.get("splash_done"):
        render_splash()
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            if st.button("Enter the Forge \u2192", use_container_width=True):
                st.session_state.splash_done = True
                st.rerun()
        return

    if not st.session_state.get("tour_done"):
        render_onboarding_tour()
        return

    render_topbar()

    if st.session_state.get("current_station") is None:
        render_station_hub()
    else:
        render_station_page(st.session_state["current_station"])


if __name__ == "__main__":
    main()
