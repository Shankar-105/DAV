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

st.set_page_config(page_title="Spaceflight Mission Analytics", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.stApp { background: #f4f7fb; }
[data-testid="stSidebar"] { background: #101b30; }
[data-testid="stSidebar"] * { color: #f4f7fb; }
.title { color:#10233f; font-size:2.35rem; font-weight:800; }
.subtitle { color:#50627a; margin-bottom:1.3rem; }
.card { background:white; border:1px solid #dce5ef; border-radius:12px; padding:1rem; box-shadow:0 3px 12px #142a4b10; }
.label { color:#60738b; font-size:.85rem; font-weight:600; }
.value { color:#10233f; font-size:1.5rem; font-weight:800; margin-top:.25rem; }
.auth { max-width:560px; margin:4rem auto; background:white; border:1px solid #dce5ef; border-radius:16px; padding:2rem; box-shadow:0 8px 30px #142a4b17; }
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "users" not in st.session_state:
    st.session_state.users = {"demo@spaceflight.local": hashlib.sha256(b"Spaceflight123!").hexdigest()}
if "models" not in st.session_state:
    st.session_state.models = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "history" not in st.session_state:
    st.session_state.history = []


def password_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metric_card(label, value):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)


@st.cache_data
def generate_data(seed=2026):
    rng = np.random.default_rng(seed)
    n = 1200
    agency = rng.choice(
        ["Private Space Enterprise", "National Space Agency", "International Consortium"],
        n, p=[.42, .36, .22]
    )
    payload = rng.uniform(500, 25000, n)
    orbit = rng.uniform(200, 36000, n)
    budget = rng.uniform(10, 450, n)
    fuel = rng.uniform(0, 100, n)
    risk = rng.uniform(0, 1, n)
    testing = rng.integers(50, 5001, n)
    crewed = rng.choice(["Yes", "No"], n, p=[.28, .72])
    window = rng.uniform(0, 100, n)
    agency_bonus = np.select([agency == "Private Space Enterprise", agency == "National Space Agency"], [5, 8], 11)
    cost = np.clip(
        budget * .78 + payload * .0023 + orbit * .00042 + testing * .006
        + np.where(crewed == "Yes", 58, 0) + (100 - fuel) * .16
        + (1 - risk) * 31 + (100 - window) * .08 + agency_bonus + rng.normal(0, 16, n), 10, 650
    )
    score = fuel * .34 + risk * 100 * .31 + window * .20 + np.clip(testing / 50, 0, 100) * .15 + rng.normal(0, 7, n)
    status = np.select([score >= 72, score >= 48], ["Success", "Partial Failure"], default="Critical Failure")
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
    data.loc[rng.choice(n, int(n * .035), replace=False), "fuel_efficiency_score"] = np.nan
    data.loc[rng.choice(n, int(n * .04), replace=False), "window_alignment_pct"] = np.nan
    return data


@st.cache_data
def clean_data(data):
    result = data.copy()
    before = result.isna().sum()
    numeric = ["payload_mass_kg", "orbit_altitude_km", "budget_millions_usd", "fuel_efficiency_score", "risk_mitigation_index", "testing_hours_logged", "window_alignment_pct", "final_mission_cost_millions"]
    medians = {}
    for column in numeric:
        medians[column] = float(result[column].median())
        result[column] = result[column].fillna(medians[column])
    duplicates = int(result.duplicated().sum())
    result = result.drop_duplicates().reset_index(drop=True)
    return result, before, medians, duplicates


raw = generate_data()
data, missing_before, medians, duplicates = clean_data(raw)
FEATURES = ["agency_type", "payload_mass_kg", "orbit_altitude_km", "budget_millions_usd", "fuel_efficiency_score", "risk_mitigation_index", "testing_hours_logged", "crewed_status", "window_alignment_pct"]
TARGET = "final_mission_cost_millions"
NUMERIC = [x for x in FEATURES if x not in ["agency_type", "crewed_status"]]
CATEGORICAL = ["agency_type", "crewed_status"]


def preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def train():
    x_train, x_test, y_train, y_test = train_test_split(data[FEATURES], data[TARGET], test_size=.2, random_state=42)
    models = {
        "Linear Regression": Pipeline([("preprocessor", preprocessor()), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("preprocessor", preprocessor()), ("model", RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42, n_jobs=-1))]),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, prediction), "RMSE": np.sqrt(mean_squared_error(y_test, prediction)), "R²": r2_score(y_test, prediction)})
    return models, pd.DataFrame(rows)


def auth_gate():
    st.markdown('<div class="auth"><div class="title">Mission Analytics Console</div><div class="subtitle">Global Spaceflight Mission Success & Cost Analytics</div>', unsafe_allow_html=True)
    mode = st.radio("Access mode", ["Login", "Create account"], horizontal=True)
    email = st.text_input("Email address", placeholder="analyst@example.com")
    password = st.text_input("Password", type="password")
    if mode == "Create account":
        confirmation = st.text_input("Confirm password", type="password")
        if st.button("Create account", use_container_width=True):
            key = email.strip().lower()
            if not key or not password or not confirmation:
                st.error("Complete all account fields before continuing.")
            elif password != confirmation:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password must contain at least eight characters.")
            elif key in st.session_state.users:
                st.error("An account with this email already exists.")
            else:
                st.session_state.users[key] = password_hash(password)
                st.success("Account created. Switch to Login to continue.")
    else:
        if st.button("Login", use_container_width=True):
            key = email.strip().lower()
            if not key or not password:
                st.error("Email address and password are required.")
            elif st.session_state.users.get(key) != password_hash(password):
                st.error("The email address or password is incorrect.")
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = key
                st.rerun()
    st.caption("Demo account: demo@spaceflight.local / Spaceflight123!")
    st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.authenticated:
    auth_gate()
    st.stop()

with st.sidebar:
    st.markdown("## 🚀 Mission Analytics")
    st.caption(f"Logged in as: {st.session_state.current_user}")
    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()
    st.divider()
    page = st.radio("Navigation", ["Overview", "Dataset & Cleaning", "Exploratory Analysis", "Model Comparison", "Predict Performance"])
    st.divider()
    st.caption(f"Records loaded: {len(raw):,}")
    st.caption(f"Last activity: {datetime.now().strftime('%H:%M:%S')}")

st.session_state.history.append({"page": page, "timestamp": datetime.now().isoformat(timespec="seconds")})
st.session_state.history = st.session_state.history[-20:]

if page == "Overview":
    st.markdown('<div class="title">Overview</div><div class="subtitle">Mission portfolio health and cost intelligence.</div>', unsafe_allow_html=True)
    project_score = (data.fuel_efficiency_score * .35 + data.window_alignment_pct * .25 + data.risk_mitigation_index * 100 * .25 + np.clip(data.testing_hours_logged / 50, 0, 100) * .15).mean()
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Total records", "Global mean cost", "Average project score", "Average risk rating"], [f"{len(raw):,}", f"${data[TARGET].mean():,.1f}M", f"{project_score:.1f}/100", f"{data.risk_mitigation_index.mean():.3f}"]):
        with col:
            metric_card(label, value)
    st.info("This is a locally generated synthetic dataset for spaceflight mission analytics. No external assets or database are required.")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(data, x=TARGET, nbins=32, title="Final mission cost distribution", color_discrete_sequence=["#1c7ed6"])
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = data.primary_status_rating.value_counts().rename_axis("Status").reset_index(name="Missions")
        fig = px.pie(counts, names="Status", values="Missions", title="Mission status breakdown", color="Status", color_discrete_map={"Success": "#2f9e44", "Partial Failure": "#f08c00", "Critical Failure": "#c92a2a"})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top 10 loaded records")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

elif page == "Dataset & Cleaning":
    st.markdown('<div class="title">Dataset & Cleaning</div><div class="subtitle">Data quality controls and preprocessing audit.</div>', unsafe_allow_html=True)
    st.write("The pipeline generates 1,200 records, injects missing values into fuel efficiency and launch-window alignment, imputes numeric gaps with medians, and removes duplicates.")
    cols = st.columns(3)
    for col, label, value in zip(cols, ["Rows before cleaning", "Missing cells detected", "Duplicates removed"], [f"{len(raw):,}", f"{int(raw.isna().sum().sum()):,}", f"{duplicates:,}"]):
        with col:
            metric_card(label, value)
    audit = missing_before.rename("Missing values").reset_index().rename(columns={"index": "Feature"})
    audit["Status"] = np.where(audit["Missing values"] > 0, "Imputed with median", "No action required")
    st.subheader("Missing-value audit")
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.success(f"Cleaning completed: {len(data):,} usable rows remain and numeric gaps have been handled.")
    st.dataframe(data.head(25), use_container_width=True, hide_index=True)

elif page == "Exploratory Analysis":
    st.markdown('<div class="title">Exploratory Analysis</div><div class="subtitle">Explore cost drivers, outcomes, and feature relationships.</div>', unsafe_allow_html=True)
    fig = px.scatter(data, x="budget_millions_usd", y=TARGET, color="primary_status_rating", size="testing_hours_logged", hover_data=["mission_id", "agency_type"], title="Budget and testing effort versus final mission cost")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        fig = px.box(data, x="primary_status_rating", y=TARGET, color="primary_status_rating", title="Final cost by mission outcome")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(data, x="agency_type", y=TARGET, color="agency_type", title="Final cost by agency type")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    fig = px.imshow(data.select_dtypes(include=np.number).corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric feature correlation matrix", aspect="auto")
    fig.update_layout(template="plotly_white", height=700)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Comparison":
    st.markdown('<div class="title">Model Comparison</div><div class="subtitle">Compare regression models for mission cost estimation.</div>', unsafe_allow_html=True)
    if st.button("Train and compare models", type="primary", use_container_width=True):
        with st.spinner("Training Linear Regression..."):
            time.sleep(.8)
        with st.spinner("Training Random Forest Regression..."):
            st.session_state.models, st.session_state.metrics = train()
            time.sleep(.8)
        st.success("Both models were trained and evaluated successfully.")
    if st.session_state.metrics is None:
        st.warning("Run the training workflow to display model metrics.")
    else:
        metrics = st.session_state.metrics
        display = metrics.copy()
        for column in ["MAE", "RMSE", "R²"]:
            display[column] = display[column].map(lambda x: f"{x:.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        chart_data = metrics.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R²"], var_name="Metric", value_name="Score")
        fig = px.bar(chart_data, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f", title="Model validation metrics")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Recommended model by RMSE: {metrics.sort_values('RMSE').iloc[0]['Model']}")

else:
    st.markdown('<div class="title">Predict Performance</div><div class="subtitle">Estimate the final cost of a proposed mission.</div>', unsafe_allow_html=True)
    if st.session_state.models is None:
        st.session_state.models, st.session_state.metrics = train()
    left, right = st.columns(2)
    with left:
        agency = st.selectbox("Agency type", ["Private Space Enterprise", "National Space Agency", "International Consortium"])
        crewed = st.selectbox("Crewed status", ["No", "Yes"])
        payload = st.number_input("Payload mass (kg)", 500.0, 25000.0, 8500.0, 100.0)
        orbit = st.slider("Orbit altitude (km)", 200, 36000, 12000, 100)
        budget = st.number_input("Budget (millions USD)", 10.0, 450.0, 120.0, 5.0)
    with right:
        fuel = st.slider("Fuel efficiency score", 0.0, 100.0, 72.0, .5)
        risk = st.slider("Risk mitigation index", 0.0, 1.0, .72, .01)
        testing = st.number_input("Testing hours logged", 50, 5000, 1800, 50)
        window = st.slider("Window alignment percentage", 0.0, 100.0, 76.0, .5)
    if st.button("Evaluate mission estimate", type="primary", use_container_width=True):
        candidate = pd.DataFrame([{"agency_type": agency, "payload_mass_kg": payload, "orbit_altitude_km": orbit, "budget_millions_usd": budget, "fuel_efficiency_score": fuel, "risk_mitigation_index": risk, "testing_hours_logged": testing, "crewed_status": crewed, "window_alignment_pct": window}])
        linear = st.session_state.models["Linear Regression"].predict(candidate)[0]
        forest = st.session_state.models["Random Forest"].predict(candidate)[0]
        expected = (linear + forest) / 2
        cols = st.columns(3)
        for col, label, value in zip(cols, ["Linear Regression estimate", "Random Forest estimate", "Expected mission cost"], [linear, forest, expected]):
            with col:
                metric_card(label, f"${value:,.2f}M")
        recommendations = []
        if fuel < 60: recommendations.append("Increase propulsion and fuel-system efficiency testing.")
        if risk < .60: recommendations.append("Strengthen redundancy reviews and risk mitigation controls.")
        if window < 60: recommendations.append("Improve launch-window planning and weather readiness.")
        if testing < 1000: recommendations.append("Increase integrated system testing before launch approval.")
        if budget < expected * .65: recommendations.append("Review whether the proposed budget covers mission complexity.")
        if recommendations:
            st.warning("Recommended improvements:")
            for item in recommendations:
                st.write(f"- {item}")
        else:
            st.success("The submitted configuration shows strong readiness indicators.")
        st.caption("Predictions use the locally generated synthetic dataset and are analytical estimates, not operational guarantees.")
