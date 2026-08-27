import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sqlite3
import shap
import numpy as np

from pathlib import Path
from PIL import Image
from datetime import datetime


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = BASE_DIR / "Results" / "risk_results_with_alerts.csv"

COST_MODEL_PATH = BASE_DIR / "Models" / "cost_overrun_model.pkl"
TIME_MODEL_PATH = BASE_DIR / "Models" / "time_overrun_model.pkl"

LOGO_PATH = BASE_DIR / "src" / "Insight_Logo.png"

DATABASE_DIR = BASE_DIR / "Database"
DATABASE_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "insight.db"


# =========================================================
# LOAD LOGO
# =========================================================

logo = Image.open(LOGO_PATH)


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Insight",
    page_icon=logo,
    layout="wide"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    cost_model = joblib.load(COST_MODEL_PATH)
    time_model = joblib.load(TIME_MODEL_PATH)

    return cost_model, time_model


cost_model, time_model = load_models()


# =========================================================
# LOAD HISTORICAL / DEMO RESULTS
# =========================================================

@st.cache_data
def load_results():

    return pd.read_csv(RESULTS_PATH)


data = load_results()


# =========================================================
# DATABASE
# =========================================================

def initialize_database():

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_snapshots (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            project_id TEXT,
            project_name TEXT,

            reporting_month TEXT,

            sector TEXT,
            ministry TEXT,
            state TEXT,
            implementing_agency_type TEXT,

            original_cost_crore REAL,
            cumulative_expenditure_crore REAL,

            planned_duration_months INTEGER,
            elapsed_duration_months INTEGER,

            schedule_consumption_pct REAL,

            expected_physical_progress_pct REAL,
            physical_progress_pct REAL,
            financial_progress_pct REAL,

            progress_gap_pct REAL,
            financial_physical_gap_pct REAL,

            milestones_total INTEGER,
            milestones_expected_to_date INTEGER,
            milestones_delayed INTEGER,
            milestone_delay_rate_pct REAL,

            land_acquisition_delay INTEGER,
            clearance_delay INTEGER,
            contractor_delay INTEGER,
            funding_issue INTEGER,

            expenditure_ratio_pct REAL,

            cost_risk_probability REAL,
            time_risk_probability REAL,

            overall_risk_score REAL,
            risk_level TEXT,
            early_warning TEXT,

            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# =========================================================
# MODEL FEATURES
# MUST MATCH THE FEATURES USED DURING TRAINING
# =========================================================

features = [

    "original_cost_crore",
    "cumulative_expenditure_crore",

    "planned_duration_months",
    "elapsed_duration_months",
    "schedule_consumption_pct",

    "expected_physical_progress_pct",
    "physical_progress_pct",
    "financial_progress_pct",

    "progress_gap_pct",
    "financial_physical_gap_pct",

    "milestones_total",
    "milestones_expected_to_date",
    "milestones_delayed",
    "milestone_delay_rate_pct",

    "land_acquisition_delay",
    "clearance_delay",
    "contractor_delay",
    "funding_issue",

    "expenditure_ratio_pct",

    "sector",
    "ministry",
    "state",
    "implementing_agency_type"
]


# =========================================================
# RISK FUNCTIONS
# =========================================================

def get_risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


def get_warning(score):

    if score >= 85:
        return "CRITICAL"

    elif score >= 70:
        return "HIGH ALERT"

    elif score >= 40:
        return "WATCH"

    else:
        return "NORMAL"


# =========================================================
# SAVE PROJECT TO DATABASE
# =========================================================

def save_project(record):

    conn = sqlite3.connect(DATABASE_PATH)

    record.to_sql(
        "project_snapshots",
        conn,
        if_exists="append",
        index=False
    )

    conn.close()


# =========================================================
# LOAD STORED PROJECTS
# =========================================================

def load_stored_projects():

    conn = sqlite3.connect(DATABASE_PATH)

    stored = pd.read_sql_query(
        """
        SELECT *
        FROM project_snapshots
        ORDER BY reporting_month
        """,
        conn
    )

    conn.close()

    return stored


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

def explain_prediction(model, project_data, top_n=5):

    # Extract preprocessing pipeline and XGBoost classifier
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform input using the exact training preprocessing
    transformed_data = preprocessor.transform(project_data)

    # Names after scaling / one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # SHAP model
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(transformed_data)

    # Handle SHAP versions returning different structures
    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            project_shap = shap_values[1][0]
        else:
            project_shap = shap_values[0][0]

    else:

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            project_shap = shap_values[0, :, 1]
        else:
            project_shap = shap_values[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": project_shap
    })

    explanation["importance"] = (
        explanation["shap_value"].abs()
    )

    explanation = (
        explanation
        .sort_values(
            "importance",
            ascending=False
        )
        .head(top_n)
    )

    return explanation


# =========================================================
# CLEAN SHAP FEATURE NAMES
# =========================================================

def clean_feature_name(name):

    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    name = name.replace("_", " ")

    return name.title()


# =========================================================
# PREPARE COMBINED PORTFOLIO DATA
# =========================================================

def get_combined_portfolio():

    stored_data = load_stored_projects()

    dashboard_columns = [
        "project_id",
        "project_name",
        "reporting_month",
        "sector",
        "cost_risk_probability",
        "time_risk_probability",
        "overall_risk_score",
        "risk_level",
        "early_warning"
    ]

    original_dashboard = data[
        dashboard_columns
    ].copy()

    if not stored_data.empty:

        stored_dashboard = stored_data[
            dashboard_columns
        ].copy()

        combined = pd.concat(
            [
                original_dashboard,
                stored_dashboard
            ],
            ignore_index=True
        )

    else:

        combined = original_dashboard

    # Convert dates properly
    combined["reporting_month_sort"] = pd.to_datetime(
        combined["reporting_month"],
        errors="coerce"
    )

    return combined, stored_data


# =========================================================
# HEADER
# =========================================================

header_col1, header_col2 = st.columns([1, 7])

with header_col1:

    st.image(
        logo,
        width=100
    )

with header_col2:

    st.title("Insight")

    st.caption(
        "Predictive Analytics and Early Warning System "
        "for Infrastructure Project Monitoring"
    )


st.divider()


# =========================================================
# NAVIGATION
# =========================================================

dashboard_tab, add_tab, stored_tab = st.tabs(
    [
        "📊 Portfolio Dashboard",
        "➕ Analyse New Project",
        "🗃️ Stored Projects"
    ]
)


# =========================================================
# GET CURRENT PORTFOLIO
# =========================================================

combined_data, stored_data = get_combined_portfolio()


# =========================================================
# TAB 1 - PORTFOLIO DASHBOARD
# =========================================================

with dashboard_tab:

    # Latest snapshot for every project
    latest = (
        combined_data
        .sort_values("reporting_month_sort")
        .groupby("project_id")
        .tail(1)
    )


    # -----------------------------------------------------
    # PORTFOLIO METRICS
    # -----------------------------------------------------

    total_projects = latest[
        "project_id"
    ].nunique()


    high_risk = (
        latest["risk_level"] == "HIGH"
    ).sum()


    critical_alerts = (
        latest["early_warning"] == "CRITICAL"
    ).sum()


    average_risk = (
        latest["overall_risk_score"].mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Projects",
        total_projects
    )


    col2.metric(
        "High Risk Projects",
        high_risk
    )


    col3.metric(
        "Critical Alerts",
        critical_alerts
    )


    col4.metric(
        "Average Risk",
        f"{average_risk:.1f}%"
    )


    st.divider()


    # -----------------------------------------------------
    # PROJECTS REQUIRING ATTENTION
    # -----------------------------------------------------

    st.subheader(
        "🚨 Projects Requiring Attention"
    )


    attention_projects = latest[
        latest["risk_level"].isin(
            ["HIGH", "MEDIUM"]
        )
    ].sort_values(
        "overall_risk_score",
        ascending=False
    )


    st.dataframe(

        attention_projects[
            [
                "project_id",
                "project_name",
                "sector",
                "cost_risk_probability",
                "time_risk_probability",
                "overall_risk_score",
                "risk_level",
                "early_warning"
            ]
        ],

        use_container_width=True
    )


    st.divider()


    # -----------------------------------------------------
    # RISK CHARTS
    # -----------------------------------------------------

    col_left, col_right = st.columns(2)


    # -----------------------------------------------------
    # RISK DISTRIBUTION
    # -----------------------------------------------------

    with col_left:

        st.subheader(
            "Portfolio Risk Distribution"
        )


        risk_counts = (
            latest["risk_level"]
            .value_counts()
            .reset_index()
        )


        risk_counts.columns = [
            "Risk Level",
            "Projects"
        ]


        fig = px.bar(

            risk_counts,

            x="Risk Level",
            y="Projects",

            title="Projects by Risk Level"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SECTOR RISK
    # -----------------------------------------------------

    with col_right:

        st.subheader(
            "Risk by Sector"
        )


        sector_risk = (

            latest
            .groupby("sector")[
                "overall_risk_score"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()

        )


        fig_sector = px.bar(

            sector_risk,

            x="sector",
            y="overall_risk_score",

            title="Average Risk Score by Sector"
        )


        st.plotly_chart(
            fig_sector,
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------
    # PROJECT INTELLIGENCE
    # -----------------------------------------------------

    st.subheader(
        "🔎 Project Intelligence"
    )


    selected_project = st.selectbox(

        "Select a project",

        sorted(
            latest["project_id"].unique()
        )
    )


    # All historical snapshots,
    # including manually submitted projects
    project_history = (

        combined_data[
            combined_data["project_id"]
            == selected_project
        ]

        .sort_values(
            "reporting_month_sort"
        )

    )


    if not project_history.empty:

        project_latest = (
            project_history.iloc[-1]
        )


        p1, p2, p3 = st.columns(3)


        p1.metric(

            "Cost Overrun Risk",

            f"{project_latest['cost_risk_probability']:.1f}%"

        )


        p2.metric(

            "Time Overrun Risk",

            f"{project_latest['time_risk_probability']:.1f}%"

        )


        p3.metric(

            "Overall Risk",

            f"{project_latest['overall_risk_score']:.1f}%"

        )


        # Only show risk trend if multiple snapshots exist
        if len(project_history) > 1:

            fig_trend = px.line(

                project_history,

                x="reporting_month_sort",
                y="overall_risk_score",

                markers=True,

                title="Risk Trend Over Time"
            )


            st.plotly_chart(
                fig_trend,
                use_container_width=True
            )


        warning = (
            project_latest[
                "early_warning"
            ]
        )


        if warning == "CRITICAL":

            st.error(
                "🚨 CRITICAL: Immediate intervention recommended."
            )


        elif warning == "HIGH ALERT":

            st.error(
                "⚠️ HIGH ALERT: Project requires urgent review."
            )


        elif warning == "RISING RISK":

            st.warning(
                "📈 RISING RISK: Risk has increased significantly."
            )


        elif warning == "WATCH":

            st.warning(
                "👀 WATCH: Continue monitoring this project."
            )


        else:

            st.success(
                "✅ Project currently within normal risk range."
            )


# =========================================================
# TAB 2 - ADD / ANALYSE NEW PROJECT
# =========================================================

with add_tab:

    st.header(
        "➕ Analyse New Project"
    )


    st.caption(
        "Enter the latest project monitoring data. "
        "Insight will generate cost, time and overall risk predictions."
    )


    # =====================================================
    # SHOW LAST SUBMITTED PREDICTION AFTER REFRESH
    # =====================================================

    if "last_prediction" in st.session_state:

        prediction = (
            st.session_state[
                "last_prediction"
            ]
        )


        st.success(
            f"Project {prediction['project_id']} analysed successfully."
        )


        result1, result2, result3 = st.columns(3)


        result1.metric(

            "Cost Overrun Risk",

            f"{prediction['cost_risk']:.1f}%"

        )


        result2.metric(

            "Time Overrun Risk",

            f"{prediction['time_risk']:.1f}%"

        )


        result3.metric(

            "Overall Risk",

            f"{prediction['overall_risk']:.1f}%"

        )


        risk_level = prediction[
            "risk_level"
        ]


        if risk_level == "HIGH":

            st.error(
                f"Risk Classification: {risk_level}"
            )


        elif risk_level == "MEDIUM":

            st.warning(
                f"Risk Classification: {risk_level}"
            )


        else:

            st.success(
                f"Risk Classification: {risk_level}"
            )


        # =================================================
        # SHAP EXPLAINABILITY
        # =================================================

        st.divider()


        st.subheader(
            "🔍 Risk Explainability"
        )


        st.caption(
            "The overall score is calculated using "
            "50% Cost Overrun Risk + 50% Time Overrun Risk. "
            "The factors below had the strongest influence "
            "on the AI predictions."
        )


        explanation_col1, explanation_col2 = (
            st.columns(2)
        )


        # -------------------------------------------------
        # COST EXPLANATION
        # -------------------------------------------------

        with explanation_col1:

            st.markdown(
                "### 💰 Cost Overrun Risk Drivers"
            )


            cost_explanation = prediction[
                "cost_explanation"
            ]


            for item in cost_explanation:

                feature = (
                    clean_feature_name(
                        item["feature"]
                    )
                )


                if item["shap_value"] > 0:

                    st.markdown(
                        f"🔺 **{feature}** — "
                        "increases predicted cost risk"
                    )


                else:

                    st.markdown(
                        f"🔻 **{feature}** — "
                        "reduces predicted cost risk"
                    )


        # -------------------------------------------------
        # TIME EXPLANATION
        # -------------------------------------------------

        with explanation_col2:

            st.markdown(
                "### ⏱️ Time Overrun Risk Drivers"
            )


            time_explanation = prediction[
                "time_explanation"
            ]


            for item in time_explanation:

                feature = (
                    clean_feature_name(
                        item["feature"]
                    )
                )


                if item["shap_value"] > 0:

                    st.markdown(
                        f"🔺 **{feature}** — "
                        "increases predicted time risk"
                    )


                else:

                    st.markdown(
                        f"🔻 **{feature}** — "
                        "reduces predicted time risk"
                    )


        st.info(
            "Prediction stored in the Insight database "
            "and included in the Portfolio Dashboard."
        )


        if st.button(
            "Clear Previous Prediction"
        ):

            del st.session_state[
                "last_prediction"
            ]

            st.rerun()


        st.divider()


    # =====================================================
    # NEW PROJECT FORM
    # =====================================================

    with st.form(
        "new_project_form"
    ):


        st.subheader(
            "Project Information"
        )


        c1, c2 = st.columns(2)


        with c1:

            project_id = st.text_input(
                "Project ID"
            )


            project_name = st.text_input(
                "Project Name"
            )


            reporting_month = st.date_input(
                "Reporting Date"
            )


        with c2:

            sector = st.selectbox(

                "Sector",

                [
                    "Transport & Logistics",
                    "Energy",
                    "Water & Sanitation",
                    "Communication",
                    "Social Infrastructure",
                    "Coal",
                    "Steel",
                    "Mining"
                ]

            )


            ministry = st.text_input(
                "Ministry"
            )


            state = st.text_input(
                "State"
            )


            agency = st.selectbox(

                "Implementing Agency Type",

                [
                    "Central Government Department",
                    "PSU",
                    "Railway Agency",
                    "State-linked Implementing Agency",
                    "Infrastructure Authority"
                ]

            )


        # =================================================
        # COST & SCHEDULE
        # =================================================

        st.subheader(
            "Cost & Schedule"
        )


        c1, c2 = st.columns(2)


        with c1:

            original_cost = st.number_input(

                "Original Cost (₹ Crore)",

                min_value=0.0,

                value=500.0
            )


            expenditure = st.number_input(

                "Cumulative Expenditure (₹ Crore)",

                min_value=0.0,

                value=200.0
            )


            planned_duration = st.number_input(

                "Planned Duration (Months)",

                min_value=1,

                value=48
            )


        with c2:

            elapsed_duration = st.number_input(

                "Elapsed Duration (Months)",

                min_value=0,

                value=24
            )


            expected_progress = st.slider(

                "Expected Physical Progress %",

                0.0,
                100.0,
                50.0
            )


            physical_progress = st.slider(

                "Actual Physical Progress %",

                0.0,
                100.0,
                40.0
            )


            financial_progress = st.slider(

                "Financial Progress %",

                0.0,
                100.0,
                45.0
            )


        # =================================================
        # MILESTONES
        # =================================================

        st.subheader(
            "Milestones"
        )


        m1, m2, m3 = st.columns(3)


        with m1:

            milestones_total = st.number_input(

                "Total Milestones",

                min_value=1,

                value=20
            )


        with m2:

            milestones_expected = st.number_input(

                "Milestones Expected To Date",

                min_value=0,

                value=10
            )


        with m3:

            milestones_delayed = st.number_input(

                "Delayed Milestones",

                min_value=0,

                value=2
            )


        # =================================================
        # IMPLEMENTATION ISSUES
        # =================================================

        st.subheader(
            "Implementation Issues"
        )


        i1, i2, i3, i4 = (
            st.columns(4)
        )


        with i1:

            land_delay = st.checkbox(
                "Land Acquisition Delay"
            )


        with i2:

            clearance_delay = st.checkbox(
                "Clearance Delay"
            )


        with i3:

            contractor_delay = st.checkbox(
                "Contractor Delay"
            )


        with i4:

            funding_issue = st.checkbox(
                "Funding Issue"
            )


        submitted = st.form_submit_button(
            "🤖 Analyse Project"
        )


    # =====================================================
    # RUN MODEL
    # =====================================================

    if submitted:


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if not project_id or not project_name:

            st.error(
                "Project ID and Project Name are required."
            )


        elif elapsed_duration > planned_duration * 3:

            st.error(
                "Elapsed duration appears unusually high. "
                "Please verify the entered value."
            )


        elif milestones_delayed > milestones_total:

            st.error(
                "Delayed milestones cannot exceed total milestones."
            )


        elif milestones_expected > milestones_total:

            st.error(
                "Expected milestones cannot exceed total milestones."
            )


        else:


            # =================================================
            # FEATURE ENGINEERING
            # =================================================

            schedule_consumption = (

                elapsed_duration /
                planned_duration

            ) * 100


            progress_gap = (

                expected_progress -
                physical_progress

            )


            financial_physical_gap = (

                financial_progress -
                physical_progress

            )


            expenditure_ratio = (

                expenditure /
                original_cost * 100

                if original_cost > 0

                else 0

            )


            milestone_delay_rate = (

                milestones_delayed /
                max(
                    milestones_expected,
                    1
                )

            ) * 100


            # =================================================
            # CREATE MODEL INPUT
            # =================================================

            new_project = pd.DataFrame(
                [
                    {

                        "original_cost_crore":
                            original_cost,

                        "cumulative_expenditure_crore":
                            expenditure,

                        "planned_duration_months":
                            planned_duration,

                        "elapsed_duration_months":
                            elapsed_duration,

                        "schedule_consumption_pct":
                            schedule_consumption,

                        "expected_physical_progress_pct":
                            expected_progress,

                        "physical_progress_pct":
                            physical_progress,

                        "financial_progress_pct":
                            financial_progress,

                        "progress_gap_pct":
                            progress_gap,

                        "financial_physical_gap_pct":
                            financial_physical_gap,

                        "milestones_total":
                            milestones_total,

                        "milestones_expected_to_date":
                            milestones_expected,

                        "milestones_delayed":
                            milestones_delayed,

                        "milestone_delay_rate_pct":
                            milestone_delay_rate,

                        "land_acquisition_delay":
                            int(
                                land_delay
                            ),

                        "clearance_delay":
                            int(
                                clearance_delay
                            ),

                        "contractor_delay":
                            int(
                                contractor_delay
                            ),

                        "funding_issue":
                            int(
                                funding_issue
                            ),

                        "expenditure_ratio_pct":
                            expenditure_ratio,

                        "sector":
                            sector,

                        "ministry":
                            ministry,

                        "state":
                            state,

                        "implementing_agency_type":
                            agency
                    }
                ]
            )


            # Correct feature order
            new_project = (
                new_project[
                    features
                ]
            )


            # =================================================
            # AI PREDICTIONS
            # =================================================

            cost_probability = (

                cost_model
                .predict_proba(
                    new_project
                )[0][1]

            ) * 100


            time_probability = (

                time_model
                .predict_proba(
                    new_project
                )[0][1]

            ) * 100


            overall_risk = (

                0.5 * cost_probability
                +
                0.5 * time_probability

            )


            risk_level = (
                get_risk_level(
                    overall_risk
                )
            )


            warning = (
                get_warning(
                    overall_risk
                )
            )


            # =================================================
            # SHAP EXPLANATION
            # =================================================

            cost_explanation_df = (
                explain_prediction(
                    cost_model,
                    new_project,
                    top_n=5
                )
            )


            time_explanation_df = (
                explain_prediction(
                    time_model,
                    new_project,
                    top_n=5
                )
            )


            # Convert DataFrames to records so they are easy
            # to store temporarily in Streamlit session state
            cost_explanation_records = (

                cost_explanation_df[
                    [
                        "feature",
                        "shap_value"
                    ]
                ]
                .to_dict(
                    orient="records"
                )

            )


            time_explanation_records = (

                time_explanation_df[
                    [
                        "feature",
                        "shap_value"
                    ]
                ]
                .to_dict(
                    orient="records"
                )

            )


            # =================================================
            # DATABASE RECORD
            # =================================================

            database_record = pd.DataFrame(
                [
                    {

                        "project_id":
                            project_id,

                        "project_name":
                            project_name,

                        "reporting_month":
                            reporting_month.strftime(
                                "%Y-%m-%d"
                            ),

                        "sector":
                            sector,

                        "ministry":
                            ministry,

                        "state":
                            state,

                        "implementing_agency_type":
                            agency,

                        "original_cost_crore":
                            original_cost,

                        "cumulative_expenditure_crore":
                            expenditure,

                        "planned_duration_months":
                            planned_duration,

                        "elapsed_duration_months":
                            elapsed_duration,

                        "schedule_consumption_pct":
                            schedule_consumption,

                        "expected_physical_progress_pct":
                            expected_progress,

                        "physical_progress_pct":
                            physical_progress,

                        "financial_progress_pct":
                            financial_progress,

                        "progress_gap_pct":
                            progress_gap,

                        "financial_physical_gap_pct":
                            financial_physical_gap,

                        "milestones_total":
                            milestones_total,

                        "milestones_expected_to_date":
                            milestones_expected,

                        "milestones_delayed":
                            milestones_delayed,

                        "milestone_delay_rate_pct":
                            milestone_delay_rate,

                        "land_acquisition_delay":
                            int(
                                land_delay
                            ),

                        "clearance_delay":
                            int(
                                clearance_delay
                            ),

                        "contractor_delay":
                            int(
                                contractor_delay
                            ),

                        "funding_issue":
                            int(
                                funding_issue
                            ),

                        "expenditure_ratio_pct":
                            expenditure_ratio,

                        "cost_risk_probability":
                            cost_probability,

                        "time_risk_probability":
                            time_probability,

                        "overall_risk_score":
                            overall_risk,

                        "risk_level":
                            risk_level,

                        "early_warning":
                            warning,

                        "created_at":
                            datetime.now().isoformat()
                    }
                ]
            )


            # =================================================
            # SAVE TO SQLITE
            # =================================================

            save_project(
                database_record
            )


            # =================================================
            # SAVE RESULT IN SESSION
            # =================================================

            st.session_state[
                "last_prediction"
            ] = {

                "project_id":
                    project_id,

                "project_name":
                    project_name,

                "cost_risk":
                    cost_probability,

                "time_risk":
                    time_probability,

                "overall_risk":
                    overall_risk,

                "risk_level":
                    risk_level,

                "warning":
                    warning,

                "cost_explanation":
                    cost_explanation_records,

                "time_explanation":
                    time_explanation_records
            }


            # =================================================
            # REFRESH ENTIRE APP
            #
            # This causes the Portfolio Dashboard to immediately
            # reload the SQLite data and therefore increment the
            # project total.
            # =================================================

            st.rerun()


# =========================================================
# TAB 3 - STORED PROJECTS
# =========================================================

with stored_tab:


    st.header(
        "🗃️ Stored Project Predictions"
    )


    stored_projects = (
        load_stored_projects()
    )


    if stored_projects.empty:


        st.info(
            "No manually submitted projects have been stored yet."
        )


    else:


        # -----------------------------------------------------
        # STORED PROJECT TABLE
        # -----------------------------------------------------

        st.dataframe(

            stored_projects[
                [
                    "project_id",
                    "project_name",
                    "reporting_month",
                    "sector",
                    "cost_risk_probability",
                    "time_risk_probability",
                    "overall_risk_score",
                    "risk_level",
                    "early_warning",
                    "created_at"
                ]
            ],

            use_container_width=True
        )


        # -----------------------------------------------------
        # SELECT PROJECT
        # -----------------------------------------------------

        selected_stored_project = (
            st.selectbox(

                "View stored project",

                stored_projects[
                    "project_id"
                ].unique(),

                key="stored_project_selector"
            )
        )


        selected_history = (

            stored_projects[
                stored_projects[
                    "project_id"
                ]
                == selected_stored_project
            ]

            .copy()

        )


        selected_history[
            "reporting_month_sort"
        ] = pd.to_datetime(

            selected_history[
                "reporting_month"
            ],

            errors="coerce"

        )


        selected_history = (
            selected_history
            .sort_values(
                "reporting_month_sort"
            )
        )


        # -----------------------------------------------------
        # LATEST STORED PREDICTION
        # -----------------------------------------------------

        stored_latest = (
            selected_history.iloc[-1]
        )


        s1, s2, s3 = (
            st.columns(3)
        )


        s1.metric(

            "Cost Overrun Risk",

            f"{stored_latest['cost_risk_probability']:.1f}%"

        )


        s2.metric(

            "Time Overrun Risk",

            f"{stored_latest['time_risk_probability']:.1f}%"

        )


        s3.metric(

            "Overall Risk",

            f"{stored_latest['overall_risk_score']:.1f}%"

        )


        # -----------------------------------------------------
        # RISK CLASSIFICATION
        # -----------------------------------------------------

        stored_risk_level = stored_latest["risk_level"]

        if stored_risk_level == "HIGH":

            st.error(
                f"Risk Classification: {stored_risk_level}"
            )

        elif stored_risk_level == "MEDIUM":

            st.warning(
                f"Risk Classification: {stored_risk_level}"
            )

        else:

            st.success(
                f"Risk Classification: {stored_risk_level}"
            )


        # -----------------------------------------------------
        # RISK EXPLAINABILITY
        # -----------------------------------------------------

        st.divider()

        st.subheader(
            "🔍 Risk Analysis"
        )

        st.caption(
            "The factors below had the strongest influence on "
            "the AI prediction for this stored project."
        )


        # Rebuild model input from the stored project snapshot
        stored_project_input = pd.DataFrame(
            [
                {
                    "original_cost_crore":
                        stored_latest["original_cost_crore"],

                    "cumulative_expenditure_crore":
                        stored_latest["cumulative_expenditure_crore"],

                    "planned_duration_months":
                        stored_latest["planned_duration_months"],

                    "elapsed_duration_months":
                        stored_latest["elapsed_duration_months"],

                    "schedule_consumption_pct":
                        stored_latest["schedule_consumption_pct"],

                    "expected_physical_progress_pct":
                        stored_latest["expected_physical_progress_pct"],

                    "physical_progress_pct":
                        stored_latest["physical_progress_pct"],

                    "financial_progress_pct":
                        stored_latest["financial_progress_pct"],

                    "progress_gap_pct":
                        stored_latest["progress_gap_pct"],

                    "financial_physical_gap_pct":
                        stored_latest["financial_physical_gap_pct"],

                    "milestones_total":
                        stored_latest["milestones_total"],

                    "milestones_expected_to_date":
                        stored_latest["milestones_expected_to_date"],

                    "milestones_delayed":
                        stored_latest["milestones_delayed"],

                    "milestone_delay_rate_pct":
                        stored_latest["milestone_delay_rate_pct"],

                    "land_acquisition_delay":
                        stored_latest["land_acquisition_delay"],

                    "clearance_delay":
                        stored_latest["clearance_delay"],

                    "contractor_delay":
                        stored_latest["contractor_delay"],

                    "funding_issue":
                        stored_latest["funding_issue"],

                    "expenditure_ratio_pct":
                        stored_latest["expenditure_ratio_pct"],

                    "sector":
                        stored_latest["sector"],

                    "ministry":
                        stored_latest["ministry"],

                    "state":
                        stored_latest["state"],

                    "implementing_agency_type":
                        stored_latest["implementing_agency_type"]
                }
            ]
        )

        stored_project_input = stored_project_input[features]


        # Generate SHAP explanations for the stored project
        stored_cost_explanation = explain_prediction(
            cost_model,
            stored_project_input,
            top_n=5
        )

        stored_time_explanation = explain_prediction(
            time_model,
            stored_project_input,
            top_n=5
        )


        explain_col1, explain_col2 = st.columns(2)


        with explain_col1:

            st.markdown(
                "### 💰 Cost Overrun Risk Drivers"
            )

            for _, row in stored_cost_explanation.iterrows():

                feature = clean_feature_name(
                    row["feature"]
                )

                if row["shap_value"] > 0:

                    st.markdown(
                        f"🔺 **{feature}** — increases predicted cost risk"
                    )

                else:

                    st.markdown(
                        f"🔻 **{feature}** — reduces predicted cost risk"
                    )


        with explain_col2:

            st.markdown(
                "### ⏱️ Time Overrun Risk Drivers"
            )

            for _, row in stored_time_explanation.iterrows():

                feature = clean_feature_name(
                    row["feature"]
                )

                if row["shap_value"] > 0:

                    st.markdown(
                        f"🔺 **{feature}** — increases predicted time risk"
                    )

                else:

                    st.markdown(
                        f"🔻 **{feature}** — reduces predicted time risk"
                    )


        # -----------------------------------------------------
        # RISK HISTORY
        # -----------------------------------------------------

        if len(
            selected_history
        ) > 1:


            fig_stored = px.line(

                selected_history,

                x="reporting_month_sort",
                y="overall_risk_score",

                markers=True,

                title="Stored Project Risk History"
            )


            st.plotly_chart(

                fig_stored,

                use_container_width=True
            )