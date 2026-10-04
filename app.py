
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Orbital Engine", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07111f;--panel:#0d1a2b;--line:rgba(148,163,184,.16);--text:#f5f7fb;--muted:#8ea0b8;--cyan:#55d6ff;--violet:#8b7cff;--green:#5ee6a8;--orange:#ffad66}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif}.stApp{background:radial-gradient(circle at 88% 5%,rgba(85,214,255,.10),transparent 25%),radial-gradient(circle at 10% 20%,rgba(139,124,255,.08),transparent 24%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(7,17,31,.72)}[data-testid="stSidebar"]{background:#091625;border-right:1px solid var(--line)}[data-testid="stSidebar"] *{color:#dbe6f3}
.block-container{max-width:1480px;padding-top:2.2rem;padding-bottom:3rem}h1,h2,h3,.brand,.eyebrow{font-family:"Space Grotesk",sans-serif}h1{letter-spacing:-.045em;font-size:2.55rem!important;margin-bottom:.2rem}h2{letter-spacing:-.025em}
.topbar{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:1.8rem}.eyebrow{color:var(--cyan);font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.55rem}.subtitle{color:var(--muted);font-size:1rem;max-width:760px}
.status-pill{border:1px solid rgba(94,230,168,.25);background:rgba(94,230,168,.07);color:var(--green);padding:.48rem .75rem;border-radius:999px;font-size:.78rem;font-weight:600;white-space:nowrap}
.brand{font-size:1.15rem;font-weight:700;letter-spacing:-.02em;padding:.4rem 0 1.7rem}.brand-mark{color:var(--cyan);margin-right:.35rem}.side-label{color:#6f829b;font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;margin:.9rem 0 .55rem}
.kpi{background:linear-gradient(145deg,rgba(16,34,56,.95),rgba(11,27,45,.95));border:1px solid var(--line);border-radius:16px;padding:1rem 1.05rem;min-height:112px;box-shadow:0 14px 40px rgba(0,0,0,.16)}.kpi-label{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}.kpi-value{color:var(--text);font-family:"Space Grotesk",sans-serif;font-size:1.65rem;font-weight:700;margin-top:.35rem}.kpi-note{color:#71859f;font-size:.74rem;margin-top:.25rem}
.section{color:#cbd7e6;font-family:"Space Grotesk",sans-serif;font-size:1rem;font-weight:600;margin:1.55rem 0 .65rem}.insight{background:linear-gradient(135deg,rgba(85,214,255,.10),rgba(139,124,255,.08));border:1px solid rgba(85,214,255,.16);border-radius:16px;padding:1rem 1.1rem;color:#dce8f6;line-height:1.5;margin:.8rem 0 1.1rem}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}div[data-testid="stButton"] button{border-radius:10px;border:1px solid rgba(85,214,255,.24);background:linear-gradient(135deg,#12324a,#162c4d);color:#f5f7fb;font-weight:600}
div[data-testid="stButton"] button:hover{border-color:rgba(85,214,255,.55);color:white}div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#0d1a2b;border-color:var(--line)}.footer{color:#60738d;font-size:.72rem;text-align:center;padding-top:2rem}
</style>
""", unsafe_allow_html=True)

AGENCIES=["Private Space Enterprise","National Space Agency","International Consortium"]
FEATURES=["agency_type","payload_mass_kg","orbit_altitude_km","budget_millions_usd","fuel_efficiency_score","risk_mitigation_index","testing_hours_logged","crewed_status","window_alignment_pct"]
TARGET="final_mission_cost_millions"
NUMERIC_FEATURES=[f for f in FEATURES if f not in ["agency_type","crewed_status"]]
CATEGORICAL_FEATURES=["agency_type","crewed_status"]

@st.cache_data(show_spinner=False)
def generate_dataset(seed=20261004):
    rng=np.random.default_rng(seed); n=10000
    agency=rng.choice(AGENCIES,n,p=[.43,.35,.22]); payload=np.clip(rng.lognormal(np.log(7000),.7,n),500,25000); orbit=rng.uniform(200,36000,n)
    budget=np.clip(rng.gamma(4.2,55,n),10,450); fuel=np.clip(rng.normal(72,17,n),0,100); risk=np.clip(rng.beta(7,2.5,n),0,1)
    testing=np.clip(rng.gamma(4.8,430,n),50,5000).astype(int); crewed=rng.choice(["Yes","No"],n,p=[.24,.76]); window=np.clip(rng.normal(76,18,n),0,100)
    agency_cost=np.select([agency==AGENCIES[0],agency==AGENCIES[1]],[5,10],14)
    cost=np.clip(budget*.82+payload*.0022+orbit*.00038+testing*.0055+np.where(crewed=="Yes",62,0)+(100-fuel)*.16+(1-risk)*34+(100-window)*.11+agency_cost+rng.normal(0,14,n),10,700)
    score=fuel*.35+risk*100*.30+window*.20+np.clip(testing/50,0,100)*.15+rng.normal(0,6,n)
    status=np.select([score>=72,score>=48],["Success","Partial Failure"],default="Critical Failure")
    data=pd.DataFrame({"mission_id":[f"MS_{i:05d}" for i in range(1,n+1)],"agency_type":agency,"payload_mass_kg":payload.round(2),"orbit_altitude_km":orbit.round(2),"budget_millions_usd":budget.round(2),"fuel_efficiency_score":fuel.round(2),"risk_mitigation_index":risk.round(4),"testing_hours_logged":testing,"crewed_status":crewed,"window_alignment_pct":window.round(2),"final_mission_cost_millions":cost.round(2),"primary_status_rating":status})
    data.loc[rng.choice(n,int(n*.03),replace=False),"fuel_efficiency_score"]=np.nan
    data.loc[rng.choice(n,int(n*.02),replace=False),"window_alignment_pct"]=np.nan
    return data

@st.cache_data(show_spinner=False)
def clean_dataset(raw):
    cleaned=raw.copy(); missing=cleaned.isna().sum()
    for column in NUMERIC_FEATURES+[TARGET]: cleaned[column]=cleaned[column].fillna(cleaned[column].median())
    duplicates=int(cleaned.duplicated().sum()); cleaned=cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned,missing,duplicates

def make_preprocessor():
    return ColumnTransformer([
        ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUMERIC_FEATURES),
        ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL_FEATURES),
    ])

@st.cache_resource(show_spinner=False)
def train_models(data):
    x_train,x_test,y_train,y_test=train_test_split(data[FEATURES],data[TARGET],test_size=.2,random_state=42)
    models={
        "Linear Regression":Pipeline([("preprocessor",make_preprocessor()),("model",LinearRegression())]),
        "Random Forest":Pipeline([("preprocessor",make_preprocessor()),("model",RandomForestRegressor(n_estimators=120,max_depth=6,random_state=42,n_jobs=-1))])
    }
    rows=[]
    for name,model in models.items():
        model.fit(x_train,y_train); prediction=model.predict(x_test)
        rows.append({"Model":name,"MAE":mean_absolute_error(y_test,prediction),"RMSE":np.sqrt(mean_squared_error(y_test,prediction)),"R² Score":r2_score(y_test,prediction)})
    return models,pd.DataFrame(rows)

def chart_layout(fig,height=380):
    fig.update_layout(template="plotly_dark",height=height,margin=dict(l=10,r=10,t=48,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans",color="#dbe6f3"),title_font=dict(family="Space Grotesk",size=16,color="#f5f7fb"),legend=dict(bgcolor="rgba(0,0,0,0)"),hoverlabel=dict(bgcolor="#102238",font_color="#f5f7fb"))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)",zeroline=False); fig.update_yaxes(gridcolor="rgba(148,163,184,.10)",zeroline=False)
    return fig

def kpi(label,value,note):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

def page_header(title,description,tag="MISSION ANALYTICS"):
    st.markdown(f'<div class="topbar"><div><div class="eyebrow">{tag}</div><h1>{title}</h1><div class="subtitle">{description}</div></div><div class="status-pill">● Dataset online · 10,000 records</div></div>',unsafe_allow_html=True)


raw_data=generate_dataset()
data,missing_before,duplicates_removed=clean_dataset(raw_data)
models,metrics=train_models(data)

missing_summary=pd.DataFrame({
    "Feature": missing_before.index,
    "Missing Before": missing_before.values,
})
missing_summary["Missing After"]=data[missing_summary["Feature"]].isna().sum().values
missing_summary["Cleaning Result"]="Cleaned / no missing values"

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">◈</span> ORBITAL ENGINE</div>',unsafe_allow_html=True)
    st.markdown('<div class="side-label">Project flow</div>',unsafe_allow_html=True)
    page=st.radio(
        "Project flow",
        ["1 · Actual Dataset","2 · Data Cleaning","3 · Exploratory Analysis","4 · Model Comparison","5 · Predict Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="side-label">Mini project</div>',unsafe_allow_html=True)
    st.caption("Space-mission cost analytics")
    st.caption("10,000 records · 12 columns")
    st.caption("No login required")

if page=="1 · Actual Dataset":
    page_header(
        "1. Actual dataset",
        "Start here. This is the raw dataset generated for the space-mission domain, before any cleaning or preprocessing.",
        "STEP 01 · RAW DATA"
    )
    c1,c2,c3,c4=st.columns(4)
    with c1:kpi("Records",f"{len(raw_data):,}","raw rows")
    with c2:kpi("Columns",f"{raw_data.shape[1]}","features + target + status")
    with c3:kpi("Missing cells",f"{int(raw_data.isna().sum().sum()):,}","present before cleaning")
    with c4:kpi("Duplicate rows",f"{int(raw_data.duplicated().sum()):,}","checked before cleaning")
    st.markdown('<div class="insight"><b>What is this data?</b><br>A synthetic space-mission dataset representing agencies, payload, orbit altitude, budget, fuel efficiency, risk mitigation, testing effort, crewed missions, launch-window alignment, final mission cost, and mission outcome.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Raw dataset preview</div>',unsafe_allow_html=True)
    st.dataframe(raw_data.head(100),use_container_width=True,hide_index=True,height=470)
    st.markdown('<div class="section">Column guide</div>',unsafe_allow_html=True)
    guide=pd.DataFrame([
        ["agency_type","Mission organization / agency","Category"],
        ["payload_mass_kg","Payload carried by the mission","Numeric"],
        ["orbit_altitude_km","Target orbit altitude","Numeric"],
        ["budget_millions_usd","Planned mission budget","Numeric"],
        ["fuel_efficiency_score","Propulsion/fuel efficiency score","Numeric"],
        ["risk_mitigation_index","Risk mitigation level from 0 to 1","Numeric"],
        ["testing_hours_logged","Recorded testing effort","Numeric"],
        ["crewed_status","Whether the mission is crewed","Category"],
        ["window_alignment_pct","Launch-window alignment percentage","Numeric"],
        ["final_mission_cost_millions","Final mission cost — prediction target","Target"],
        ["primary_status_rating","Mission outcome rating","Category"],
    ],columns=["Column","Meaning","Type"])
    st.dataframe(guide,use_container_width=True,hide_index=True)

elif page=="2 · Data Cleaning":
    page_header(
        "2. Data cleaning & preprocessing",
        "See exactly what was missing, what was changed, and how the cleaned dataset is prepared for analysis and machine learning.",
        "STEP 02 · CLEANING"
    )
    c1,c2,c3=st.columns(3)
    with c1:kpi("Rows before",f"{len(raw_data):,}","raw dataset")
    with c2:kpi("Missing cells fixed",f"{int(missing_before.sum()):,}","median imputation")
    with c3:kpi("Duplicate rows removed",f"{duplicates_removed:,}","duplicate check")
    st.markdown('<div class="section">What was cleaned?</div>',unsafe_allow_html=True)
    cleaning_cards=[
        ("01","Missing fuel efficiency","300 values were missing. They were filled with the median fuel-efficiency score of the available records."),
        ("02","Missing window alignment","200 values were missing. They were filled with the median window-alignment percentage."),
        ("03","Duplicate records","The complete rows were checked for duplicates. None were found, so no valid record was removed."),
        ("04","Model preprocessing","Numeric inputs are median-imputed and standardized; categorical inputs are filled with the most frequent value and one-hot encoded inside the model pipeline."),
    ]
    for number,title,body in cleaning_cards:
        st.markdown(f'<div class="insight"><b>{number} · {title}</b><br>{body}</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Before vs after</div>',unsafe_allow_html=True)
    shown=missing_summary[missing_summary["Missing Before"]>0].copy()
    st.dataframe(shown,use_container_width=True,hide_index=True)
    st.markdown('<div class="section">Cleaned dataset preview</div>',unsafe_allow_html=True)
    st.dataframe(data.head(100),use_container_width=True,hide_index=True,height=430)
    st.success("Cleaning complete: the analytical dataset has no missing values and is ready for EDA and model training.")

elif page=="3 · Exploratory Analysis":
    page_header(
        "3. Exploratory analysis",
        "Now that the data is clean, we look for distributions, relationships, and patterns before comparing models.",
        "STEP 03 · EDA"
    )
    c1,c2,c3,c4=st.columns(4)
    with c1:kpi("Clean records",f"{len(data):,}","ready for analysis")
    with c2:kpi("Avg. mission cost",f"USD {data[TARGET].mean():,.1f}M","final modeled cost")
    with c3:kpi("Success rate",f"{(data.primary_status_rating=='Success').mean()*100:.1f}%","mission outcome")
    with c4:kpi("Avg. budget",f"USD {data.budget_millions_usd.mean():,.1f}M","planned budget")
    left,right=st.columns([1.35,.9])
    with left:
        fig=px.histogram(data,x=TARGET,nbins=42,title="Final mission cost distribution",color_discrete_sequence=["#55d6ff"])
        fig.update_traces(opacity=.85)
        st.plotly_chart(chart_layout(fig,390),use_container_width=True)
    with right:
        counts=data.primary_status_rating.value_counts().rename_axis("Status").reset_index(name="Missions")
        fig=px.pie(counts,names="Status",values="Missions",hole=.68,title="Mission outcome mix",color="Status",color_discrete_map={"Success":"#5ee6a8","Partial Failure":"#ffad66","Critical Failure":"#ff647c"})
        st.plotly_chart(chart_layout(fig,390),use_container_width=True)
    left,right=st.columns(2)
    with left:
        fig=px.scatter(data.sample(min(2500,len(data)),random_state=42),x="budget_millions_usd",y=TARGET,color="agency_type",hover_data=["mission_id","primary_status_rating"],title="Budget vs. final mission cost",color_discrete_sequence=["#55d6ff","#8b7cff","#5ee6a8"])
        st.plotly_chart(chart_layout(fig,400),use_container_width=True)
    with right:
        agency_cost=data.groupby("agency_type",as_index=False)[TARGET].mean().sort_values(TARGET)
        fig=px.bar(agency_cost,x=TARGET,y="agency_type",orientation="h",title="Average cost by agency",color=TARGET,color_continuous_scale=["#203b58","#55d6ff"])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(chart_layout(fig,400),use_container_width=True)
    corr=data.select_dtypes(include=np.number).corr().round(2)
    fig=px.imshow(corr,text_auto=True,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Numeric correlation matrix",aspect="auto")
    st.plotly_chart(chart_layout(fig,560),use_container_width=True)

elif page=="4 · Model Comparison":
    page_header(
        "4. Model comparison",
        "Both models see the same cleaned data, the same 80/20 train-test split, and the same preprocessing pipeline. That makes the comparison fair.",
        "STEP 04 · MODELS"
    )
    best=metrics.loc[metrics["R² Score"].idxmax(),"Model"]
    c1,c2,c3,c4=st.columns(4)
    with c1:kpi("Best R²",f"{metrics['R² Score'].max():.3f}",best)
    with c2:kpi("Best MAE",f"{metrics['MAE'].min():.2f}","lower is better")
    with c3:kpi("Best RMSE",f"{metrics['RMSE'].min():.2f}","lower is better")
    with c4:kpi("Validation split","80 / 20","train / test")
    st.markdown('<div class="section">Validation scorecard</div>',unsafe_allow_html=True)
    display=metrics.copy()
    display["MAE"]=display["MAE"].map(lambda x:f"{x:.3f}")
    display["RMSE"]=display["RMSE"].map(lambda x:f"{x:.3f}")
    display["R² Score"]=display["R² Score"].map(lambda x:f"{x:.3f}")
    st.dataframe(display,use_container_width=True,hide_index=True)
    melted=metrics.melt(id_vars="Model",value_vars=["MAE","RMSE","R² Score"],var_name="Metric",value_name="Score")
    fig=px.bar(melted,x="Metric",y="Score",color="Model",barmode="group",text_auto=".3f",title="Model metric comparison",color_discrete_sequence=["#55d6ff","#8b7cff"])
    st.plotly_chart(chart_layout(fig,420),use_container_width=True)
    if best=="Linear Regression":
        st.markdown('<div class="insight"><b>Best model: Linear Regression.</b><br>It has the highest R² and the lowest MAE and RMSE. The dataset target was created mainly from additive relationships, so a linear model fits the underlying pattern well.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight"><b>Best model: Random Forest.</b><br>It gives the stronger validation results across the selected metrics.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">How to judge the models</div>',unsafe_allow_html=True)
    st.markdown('<div class="insight"><b>MAE</b> tells us the average absolute prediction error. <b>RMSE</b> also measures error but penalizes larger mistakes more strongly. <b>R²</b> tells us how much of the variation in final mission cost is explained by the model. For MAE/RMSE, lower is better; for R², higher is better.</div>',unsafe_allow_html=True)

else:
    page_header(
        "5. Predict performance",
        "Enter a new mission profile and compare the final-cost estimates from both trained models.",
        "STEP 05 · PREDICTION"
    )
    left,right=st.columns([.85,1.15],gap="large")
    with left:
        st.markdown('<div class="section">New mission inputs</div>',unsafe_allow_html=True)
        agency=st.selectbox("Agency type",AGENCIES)
        crewed=st.selectbox("Crewed status",["No","Yes"])
        payload=st.slider("Payload mass (kg)",500,25000,8500,100)
        orbit=st.slider("Orbit altitude (km)",200,36000,12000,100)
        budget=st.number_input("Budget (millions USD)",10.0,450.0,120.0,5.0)
        fuel=st.slider("Fuel efficiency score",0.0,100.0,72.0,.5)
        risk=st.slider("Risk mitigation index",0.0,1.0,.72,.01)
        testing=st.number_input("Testing hours logged",50,5000,1800,50)
        window=st.slider("Window alignment (%)",0.0,100.0,76.0,.5)
        evaluate=st.button("Predict mission cost",type="primary",use_container_width=True)
    with right:
        st.markdown('<div class="section">Prediction output</div>',unsafe_allow_html=True)
        candidate=pd.DataFrame([{"agency_type":agency,"payload_mass_kg":payload,"orbit_altitude_km":orbit,"budget_millions_usd":budget,"fuel_efficiency_score":fuel,"risk_mitigation_index":risk,"testing_hours_logged":testing,"crewed_status":crewed,"window_alignment_pct":window}])
        linear=models["Linear Regression"].predict(candidate)[0]
        forest=models["Random Forest"].predict(candidate)[0]
        expected=(linear+forest)/2
        if evaluate:
            a,b,c=st.columns(3)
            with a:kpi("Linear Regression",f"USD {linear:,.2f}M","estimated final cost")
            with b:kpi("Random Forest",f"USD {forest:,.2f}M","estimated final cost")
            with c:kpi("Average estimate",f"USD {expected:,.2f}M","reference estimate")
            gauge=go.Figure(go.Indicator(mode="gauge+number",value=expected,number={"prefix":"USD ","suffix":"M","font":{"size":34}},title={"text":"Expected final mission cost"},gauge={"axis":{"range":[0,700]},"bar":{"color":"#55d6ff"},"bgcolor":"#102238","borderwidth":0,"steps":[{"range":[0,200],"color":"#102a35"},{"range":[200,450],"color":"#243047"},{"range":[450,700],"color":"#3a2734"}]}))
            st.plotly_chart(chart_layout(gauge,310),use_container_width=True)
            recommendations=[]
            if fuel<60:recommendations.append("Raise propulsion efficiency testing.")
            if risk<.60:recommendations.append("Increase risk mitigation and redundancy.")
            if window<60:recommendations.append("Improve launch-window planning.")
            if testing<1000:recommendations.append("Add more integrated system test hours.")
            if budget<expected*.65:recommendations.append("Revisit the budget against the modeled cost.")
            if recommendations:
                st.warning("A few things stand out:")
                for item in recommendations:st.write(f"• {item}")
            else:
                st.success("The submitted mission profile looks balanced across the main controls.")
        else:
            st.markdown('<div class="insight"><b>Ready when you are.</b><br>Set the mission controls and run the prediction. Both models use the same cleaned dataset from the earlier project steps.</div>',unsafe_allow_html=True)

st.markdown('<div class="footer">Orbital Engine · space-mission data analytics lab project · no authentication required</div>',unsafe_allow_html=True)
