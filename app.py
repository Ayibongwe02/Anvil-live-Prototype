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

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anvil — Forge a Model",
    page_icon="\u2692\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
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

/* Kill default Streamlit chrome that fights the theme */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}

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

/* Sidebar = the tool rack */
section[data-testid="stSidebar"] {
    background: var(--forge-char);
    border-right: 1px solid #3A3126;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--spark) !important;
    font-size: 0.95rem;
    letter-spacing: 0.12em;
}

/* Radio nav styled as riveted tabs */
div[role="radiogroup"] label {
    background: var(--forge-char-2);
    border: 1px solid #3A3126;
    border-radius: 0px;
    padding: 10px 14px;
    margin-bottom: 6px;
    width: 100%;
    transition: all 0.15s ease;
}
div[role="radiogroup"] label:hover {
    border-color: var(--ember);
    background: #322A20;
}
div[role="radiogroup"] label[data-checked="true"] {
    border-left: 3px solid var(--ember);
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
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


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
    if y.dtype == object or str(y.dtype).startswith("category"):
        return "classification"
    if y.nunique() <= max(10, int(0.02 * len(y))) and y.dtype in (np.int64, np.int32):
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

    if st.button("\u2692\ufe0f  Start training"):
        clean = df.dropna(subset=[target]).copy()
        y_raw = clean[target]
        X = clean.drop(columns=[target])

        task_type = detect_task_type(y_raw)

        label_encoder = None
        if task_type == "classification" and y_raw.dtype == object:
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

        progress_bar = st.progress(0, text="Warming the forge\u2026")
        status = st.empty()

        def progress_cb(i, total, name):
            pct = int((i / total) * 100)
            progress_bar.progress(pct, text=f"Training {name}\u2026")
            status.markdown(
                f'<span class="forge-caption">Striking model {min(i+1, total)} of {total}: '
                f"<b>{name}</b></span>",
                unsafe_allow_html=True,
            )
            time.sleep(0.15)

        results, X_test, y_test = train_all_models(
            X, y, task_type, numeric_cols, categorical_cols, progress_cb=progress_cb
        )
        progress_bar.progress(100, text="Quenched \u2014 training complete")

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

    medal = {0: "\U0001F947", 1: "\U0001F948", 2: "\U0001F949"}
    board["Rank"] = [f"{medal.get(i, '')} {board.loc[i, 'Rank']}".strip() for i in range(len(board))]

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

    if not st.session_state.get("trained"):
        st.info("Train a model first on the Upload & Forge tab.")
        return

    results = st.session_state.results
    names = list(results.keys())
    default_idx = names.index(st.session_state.best_name)
    model_choice = st.selectbox("Model", options=names, index=default_idx)

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

    if st.button("\u26A1 Predict"):
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

    if not st.session_state.get("trained"):
        st.info("Train a model first on the Upload & Forge tab.")
        return

    best_name = st.session_state.best_name
    pipe = st.session_state.results[best_name]["pipeline"]

    buf = io.BytesIO()
    pickle.dump(pipe, buf)
    st.download_button(
        "\u2B07\ufe0f  Download trained model (.pkl)",
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


# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------

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

    with st.sidebar:
        st.markdown("## \            ANVIL")
        st.markdown('<p class="forge-caption">Forge a model from raw data.</p>', unsafe_allow_html=True)
        st.markdown("### Navigate")
        page = st.radio(
            "Navigate",
            ["Upload & Forge", "Leaderboard", "Test a Prediction", "Deploy"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        if st.session_state.get("trained"):
            st.markdown('<p class="forge-caption">Current best</p>', unsafe_allow_html=True)
            st.markdown(f"**{st.session_state.best_name}**")
        if st.button("Reset session"):
            for k in list(st.session_state.keys()):
                if k != "splash_done":
                    del st.session_state[k]
            st.rerun()

    if page == "Upload & Forge":
        page_upload_train()
    elif page == "Leaderboard":
        page_leaderboard()
    elif page == "Test a Prediction":
        page_predict()
    elif page == "Deploy":
        page_deploy()


if __name__ == "__main__":
    main()
