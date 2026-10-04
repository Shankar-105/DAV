import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Orbital Engine", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#172033; --muted:#65748b; --line:#dbe3ed; --blue:#2563eb; --coral:#e76f51; }
.stApp { background:#f7f9fc; color:var(--ink); }
[data-testid="stSidebar"] { background:#111827; }
[data-testid="stSidebar"] * { color:#f8fafc; }
.hero h1 { color:var(--ink); font-size:2.35rem; letter-spacing:-.04em; margin:0; }
.hero p { color:var(--muted); margin:.35rem 0 1.4rem; }
.metric-label { color:var(--muted); font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.metric-value { color:var(--ink); font-size:1.55rem; font-weight:800; margin-top:.25rem; }
.brand { font-size:1.1rem; font-weight:800; letter-spacing:.05em; margin:.4rem 0 1.6rem; }
.auth-copy { text-align:center; padding:1.2rem 0 .5rem; }
.auth-copy h1 { color:var(--ink); letter-spacing:-.04em; }
</style>
""", unsafe_allow_html=True)

AGENCIES = ["Private Space Enterprise", "National Space Agency", "International Consortium"]
FEATURES = ["agency_type", "payload_mass_kg", "orbit_altitude_km", "budget_millions_usd", "fuel_efficiency_score", "risk_mitigation_index", "testing_hours_logged", "crewed_status", "window_alignment_pct"]
TARGET = "final_mission_cost_millions"
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in ["agency_type", "crewed_status"]]
CATEGORICAL_FEATURES = ["agency_type", "crewed_status"]


def hash_password(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metric_card(label, value, accent="var(--blue)"):
    with st.container(border=True):
        st.markdown(f'<div class="metric-label">{label}</div><div class="metric-value" style="color:{accent}">{value}</div>', unsafe_allow_html=True)


def section_heading(title, description=""):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def generate_dataset(seed=20261004):
    rng = np.random.default_rng(seed)
    n = 10_000
    agency = rng.choice(AGENCIES, n, p=[.43, .35, .22])
    payload = np.clip(rng.lognormal(np.log(7000), .7, n), 500, 25000)
    orbit = rng.uniform(200, 36000, n)
    budget = np.clip(rng.gamma(4.2, 55, n), 10, 450)
    fuel = np.clip(rng.normal(72, 17, n), 0, 100)
    risk = np.clip(rng.beta(7, 2.5, n), 0, 1)
    testing = np.clip(rng.gamma(4.8, 430, n), 50, 5000).astype(int)
    crewed = rng.choice(["Yes", "No"], n, p=[.24, .76])
    window = np.clip(rng.normal(76, 18, n), 0, 100)
    agency_cost = np.select([agency == AGENCIES[0], agency == AGENCIES[1]], [5, 10], 14)
    cost = np.clip(budget * .82 + payload * .0022 + orbit * .00038 + testing * .0055 + np.where(crewed == "Yes", 62, 0) + (100 - fuel) * .16 + (1 - risk) * 34 + (100 - window) * .11 + agency_cost + rng.normal(0, 14, n), 10, 700)
    mission_score = fuel * .35 + risk * 100 * .30 + window * .20 + np.clip(testing / 50, 0, 100) * .15 + rng.normal(0, 6, n)
    status = np.select([mission_score >= 72, mission_score >= 48], ["Success", "Partial Failure"], default="Critical Failure")
    data = pd.DataFrame({
        "mission_id": [f"MS_{i:05d}" for i in range(1, n + 1)],
        "agency_type": agency,
        "payload_mass_kg": payload.round(2),
        "orbit_altitude_km": orbit.round(2),
        "budget_millions_usd": budget.round(2),
        "fuel_efficiency_score": fuel.round(2),
        "risk_mitigation_index": risk.round(4),
        "testing_hours_logged": testing,
        "crewed_status": crewed,
        "window_alignment_pct": window.round(2),
        "final_mission_cost_millions": cost.round(2),
        "primary_status_rating": status,
    })
    data.loc[rng.choice(n, int(n * .03), replace=False), "fuel_efficiency_score"] = np.nan
    data.loc[rng.choice(n, int(n * .02), replace=False), "window_alignment_pct"] = np.nan
    return data


@st.cache_data(show_spinner=False)
def clean_dataset(raw):
    cleaned = raw.copy()
    missing_before = cleaned.isna().sum()
    medians = {}
    for column in NUMERIC_FEATURES + [TARGET]:
        medians[column] = float(cleaned[column].median())
        cleaned[column] = cleaned[column].fillna(medians[column])
    duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned, missing_before, medians, duplicates


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])


def train_models(data):
    x_train, x_test, y_train, y_test = train_test_split(data[FEATURES], data[TARGET], test_size=.2, random_state=42)
    models = {
        "Linear Regression": Pipeline([("preprocessor", make_preprocessor()), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("preprocessor", make_preprocessor()), ("model", RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42, n_jobs=-1))]),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, prediction), "RMSE": np.sqrt(mean_squared_error(y_test, prediction)), "R² Score": r2_score(y_test, prediction)})
    return models, pd.DataFrame(rows)


def auth_gate():
    with st.container(border=True):
        st.markdown('<div class="auth-copy"><h1>🚀 ORBITAL ENGINE</h1><p>Global Spaceflight Mission Success & Cost Analytics</p></div>', unsafe_allow_html=True)
        mode = st.radio("Access mode", ["Login", "Create account"], horizontal=True)
        email = st.text_input("Email address", placeholder="analyst@domain.edu")
        password = st.text_input("Password", type="password")
        if mode == "Create account":
            confirmation = st.text_input("Confirm password", type="password")
            if st.button("Create account", use_container_width=True):
                key = email.strip().lower()
                if not key or not password or not confirmation:
                    st.error("Complete all fields before continuing.")
                elif password != confirmation:
                    st.error("Passwords do not match.")
                elif len(password) < 8:
                    st.error("Password must contain at least eight characters.")
                elif key in st.session_state.users:
                    st.error("That account already exists.")
                else:
                    st.session_state.users[key] = hash_password(password)
                    st.success("Account created. Switch to Login to continue.")
        elif st.button("Login", type="primary", use_container_width=True):
            key = email.strip().lower()
            if not key or not password:
                st.error("Email address and password are required.")
            elif st.session_state.users.get(key) != hash_password(password):
                st.error("The email address or password is incorrect.")
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = key
                st.rerun()
        st.caption("Demo account: demo@spaceflight.local / Spaceflight123!")


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "users" not in st.session_state:
    st.session_state.users = {"demo@spaceflight.local": hash_password("Spaceflight123!")}
if "models" not in st.session_state:
    st.session_state.models = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "history" not in st.session_state:
    st.session_state.history = []

raw_data = generate_dataset()
data, missing_before, medians, duplicates_removed = clean_dataset(raw_data)

if not st.session_state.authenticated:
    auth_gate()
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand">🚀 ORBITAL ENGINE</div>', unsafe_allow_html=True)
    st.caption(f"👤 Session: Active ({st.session_state.current_user})")
    if st.button("Clear Session / Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.models = None
        st.session_state.metrics = None
        st.rerun()
    st.divider()
    st.caption("WORKSPACE")
    page = st.radio("Navigation", ["📊 Operations Overview", "🧼 Data Cleaning Pipeline", "📈 Interactive Discovery", "🤖 Model Training Hub", "🔮 Live Mission Evaluator"], label_visibility="collapsed")
    st.divider()
    st.caption(f"{len(raw_data):,} mission records online")
    st.caption(f"Last activity {datetime.now().strftime('%H:%M:%S')}")

st.session_state.history.append({"page": page, "at": datetime.now().isoformat(timespec="seconds")})
st.session_state.history = st.session_state.history[-20:]

if page == "📊 Operations Overview":
    section_heading("Operations Overview", "A live command layer for the global mission portfolio.")
    with st.container(border=True):
        columns = st.columns(4)
        values = [f"{len(raw_data):,}", f"${data.budget_millions_usd.mean():,.1f}M", f"{data.fuel_efficiency_score.mean():.1f}/100", f"{(data.primary_status_rating == 'Success').mean() * 100:.1f}%"]
        labels = ["Total database volume", "Average budget", "Mean efficiency score", "Global success ratio"]
        accents = ["#2563eb", "#0f766e", "#e76f51", "#7c3aed"]
        for column, label, value, accent in zip(columns, labels, values, accents):
            with column:
                metric_card(label, value, accent)
    st.info("Synthetic operations dataset online: exactly 10,000 generated mission records with controlled missing-value conditions.")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            histogram = px.histogram(data, x=TARGET, nbins=42, marginal="rug", title="Final mission cost distribution", color_discrete_sequence=["#2563eb"])
            histogram.update_traces(opacity=.84)
            histogram.update_layout(template="plotly_white", height=410, bargap=.04)
            st.plotly_chart(histogram, use_container_width=True)
        with right:
            statuses = data.primary_status_rating.value_counts().rename_axis("Status").reset_index(name="Missions")
            donut = px.pie(statuses, names="Status", values="Missions", hole=.64, title="Mission status mix", color="Status", color_discrete_map={"Success": "#0f766e", "Partial Failure": "#e76f51", "Critical Failure": "#263449"})
            donut.update_layout(template="plotly_white", height=410)
            st.plotly_chart(donut, use_container_width=True)
    with st.container(border=True):
        st.subheader("Data review dock")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True, height=360)

elif page == "🧼 Data Cleaning Pipeline":
    section_heading("Data Cleaning Pipeline", "Transparent quality controls for every mission parameter.")
    with st.container(border=True):
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Missing parameter inventory")
            audit = missing_before[missing_before > 0].rename("Missing rows").reset_index().rename(columns={"index": "Feature"})
            for row in audit.itertuples(index=False):
                percentage = row[1] / len(raw_data)
                st.write(f"**{row[0]}** · {row[1]:,} rows · {percentage:.1%}")
                st.progress(float(percentage))
            st.caption("All other fields arrived complete in the generated source array.")
        with right:
            st.subheader("Validation log")
            st.success("✔ Imputed missing parameters via median method")
            st.success(f"✔ Validated {len(data):,} rows after cleaning")
            st.success(f"✔ Removed {duplicates_removed:,} duplicate records")
            st.success("✔ Preserved sequential mission identifiers")
            st.metric("Rows affected", f"{int(missing_before.sum()):,} / {len(raw_data):,}")
    with st.container(border=True):
        st.subheader("Applied median values")
        imputed = pd.DataFrame({"Feature": list(medians), "Median used": list(medians.values())})
        st.dataframe(imputed[imputed.Feature.isin(audit.Feature)], use_container_width=True, hide_index=True)

elif page == "📈 Interactive Discovery":
    section_heading("Interactive Discovery", "Find the relationships shaping mission cost and outcome.")
    with st.container(border=True):
        scatter = px.scatter(data.sample(3000, random_state=42), x="budget_millions_usd", y=TARGET, color="agency_type", hover_data=["mission_id", "primary_status_rating"], title="Budget versus final mission cost", color_discrete_sequence=["#2563eb", "#e76f51", "#0f766e"])
        scatter.update_layout(template="plotly_white", height=470)
        st.plotly_chart(scatter, use_container_width=True)
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            box = px.box(data, x="primary_status_rating", y="budget_millions_usd", color="primary_status_rating", title="Budget distribution by status", color_discrete_map={"Success": "#0f766e", "Partial Failure": "#e76f51", "Critical Failure": "#263449"})
            box.update_layout(template="plotly_white", showlegend=False, height=500)
            st.plotly_chart(box, use_container_width=True)
        with right:
            correlation = px.imshow(data.select_dtypes(include=np.number).corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric correlation matrix", aspect="auto")
            correlation.update_layout(template="plotly_white", height=500)
            st.plotly_chart(correlation, use_container_width=True)

elif page == "🤖 Model Training Hub":
    section_heading("Model Training Hub", "Run the engine core against a reproducible 80/20 validation split.")
    with st.container(border=True):
        st.write("Two production-style regression pipelines estimate final mission cost from operational parameters.")
        if st.button("▶ Run Engine Core Diagnostics", type="primary", use_container_width=True):
            with st.spinner("Encoding mission features and running diagnostics..."):
                time.sleep(.6)
                st.session_state.models, st.session_state.metrics = train_models(data)
            st.success("Engine diagnostics complete.")
    if st.session_state.metrics is not None:
        with st.container(border=True):
            metrics = st.session_state.metrics
            display = metrics.copy()
            for column in ["MAE", "RMSE", "R² Score"]:
                display[column] = display[column].map(lambda value: f"{value:.4f}")
            st.subheader("Validation scorecard")
            st.dataframe(display, use_container_width=True, hide_index=True)
            chart_data = metrics.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R² Score"], var_name="Metric", value_name="Score")
            chart = px.bar(chart_data, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f", title="Model metric comparison", color_discrete_sequence=["#2563eb", "#e76f51"])
            chart.update_layout(template="plotly_white", height=430)
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Run diagnostics to populate the model scorecard.")

else:
    section_heading("Live Mission Evaluator", "Stress-test a proposed mission against both trained estimators.")
    if st.session_state.models is None:
        with st.spinner("Preparing evaluator models..."):
            st.session_state.models, st.session_state.metrics = train_models(data)
    with st.container(border=True):
        left, right = st.columns([.95, 1.4])
        with left:
            st.subheader("Mission controls")
            agency = st.selectbox("Agency type", AGENCIES)
            crewed = st.selectbox("Crewed status", ["No", "Yes"])
            payload = st.slider("Payload mass (kg)", 500, 25000, 8500, 100)
            orbit = st.slider("Orbit altitude (km)", 200, 36000, 12000, 100)
            budget = st.number_input("Budget (millions USD)", 10.0, 450.0, 120.0, 5.0)
            fuel = st.slider("Fuel efficiency score", 0.0, 100.0, 72.0, .5)
            risk = st.slider("Risk mitigation index", 0.0, 1.0, .72, .01)
            testing = st.number_input("Testing hours logged", 50, 5000, 1800, 50)
            window = st.slider("Window alignment percentage", 0.0, 100.0, 76.0, .5)
            evaluate = st.button("Evaluate mission estimate", type="primary", use_container_width=True)
        with right:
            st.subheader("Prediction output")
            if evaluate:
                candidate = pd.DataFrame([{"agency_type": agency, "payload_mass_kg": payload, "orbit_altitude_km": orbit, "budget_millions_usd": budget, "fuel_efficiency_score": fuel, "risk_mitigation_index": risk, "testing_hours_logged": testing, "crewed_status": crewed, "window_alignment_pct": window}])
                linear = st.session_state.models["Linear Regression"].predict(candidate)[0]
                forest = st.session_state.models["Random Forest"].predict(candidate)[0]
                expected = (linear + forest) / 2
                cards = st.columns(3)
                for card, label, value, accent in zip(cards, ["Linear Regression", "Random Forest", "Expected cost"], [linear, forest, expected], ["#2563eb", "#e76f51", "#0f766e"]):
                    with card:
                        metric_card(label, f"${value:,.2f}M", accent)
                recommendations = []
                if fuel < 60: recommendations.append("Raise propulsion efficiency testing before review.")
                if risk < .60: recommendations.append("Increase redundancy and mitigation controls.")
                if window < 60: recommendations.append("Improve launch-window planning readiness.")
                if testing < 1000: recommendations.append("Add integrated system test hours.")
                if budget < expected * .65: recommendations.append("Revisit the budget against modeled complexity.")
                with st.container(border=True):
                    if recommendations:
                        st.warning("Operational recommendations")
                        for recommendation in recommendations:
                            st.write(f"• {recommendation}")
                    else:
                        st.success("Readiness profile is balanced across the submitted controls.")
            else:
                st.info("Tune the controls, then evaluate the mission to populate this output panel.")
