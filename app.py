import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Charging Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATA_FILE = os.path.join(
    DATA_DIR,
    "feature_engineered_data.csv"
)

REG_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_regression_model.pkl"
)

CLS_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_classification_model.pkl"
)

THRESHOLD_FILE = os.path.join(
    MODEL_DIR,
    "demand_threshold.pkl"
)

REG_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "regression_model_comparison.csv"
)

CLS_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "classification_model_comparison.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .status-badge-high {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 15px;
        border: 1px solid #fca5a5;
    }
    .status-badge-normal {
        display: inline-block;
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 15px;
        border: 1px solid #93c5fd;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA & MODELS
# ============================================================

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    backup_file = os.path.join(DATA_DIR, "cleaned_charging_data.csv")
    if os.path.exists(backup_file):
        return pd.read_csv(backup_file)
    return pd.DataFrame()


@st.cache_data
def load_regression_results():
    if os.path.exists(REG_RESULTS_FILE):
        return pd.read_csv(REG_RESULTS_FILE)
    return pd.DataFrame()


@st.cache_data
def load_classification_results():
    if os.path.exists(CLS_RESULTS_FILE):
        return pd.read_csv(CLS_RESULTS_FILE)
    return pd.DataFrame()


@st.cache_resource
def load_regression_model():
    if os.path.exists(REG_MODEL_FILE):
        return joblib.load(REG_MODEL_FILE)
    return None


@st.cache_resource
def load_classification_model():
    if os.path.exists(CLS_MODEL_FILE):
        return joblib.load(CLS_MODEL_FILE)
    return None


@st.cache_resource
def load_threshold():
    if os.path.exists(THRESHOLD_FILE):
        try:
            return float(joblib.load(THRESHOLD_FILE))
        except Exception:
            return 14.115
    return 14.115


def get_feature_importances(pipeline):
    """Safely extracts feature importances from a fitted Pipeline with ColumnTransformer."""
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        model = pipeline.named_steps.get("model")
        if preprocessor is None or model is None:
            return None
        if not hasattr(model, "feature_importances_"):
            return None
        
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [
            f.replace("num__", "").replace("cat__", "") for f in feature_names
        ]
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df_imp
    except Exception:
        return None


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

df = load_data()
regression_results = load_regression_results()
classification_results = load_classification_results()
regression_model = load_regression_model()
classification_model = load_classification_model()
demand_threshold = load_threshold()

if df.empty:
    st.error(f"Dataset not found at `{DATA_FILE}`. Please check the data directory.")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## EV Charging Analytics")
st.sidebar.caption("Intelligent Demand Forecasting & Pattern Analysis")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Dashboard",
        "Regression Models",
        "Classification Models",
        "Demand Prediction",
        "Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(f"**Total Sessions:** `{len(df):,}`")
if "Demand" in df.columns:
    st.sidebar.markdown(f"**Avg Demand:** `{df['Demand'].mean():.2f} kWh`")
    st.sidebar.markdown(f"**Demand Threshold:** `{demand_threshold:.2f} kWh`")

st.sidebar.divider()
st.sidebar.caption("EV Charging Intelligence Platform © 2026")


# ============================================================
# 1. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown('<div class="main-title">EV Charging Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Overview of charging infrastructure metrics, demand distribution, and session dynamics</div>', unsafe_allow_html=True)

    # Key KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric("Total Sessions", f"{len(df):,}")
    with kpi2:
        avg_d = df["Demand"].mean() if "Demand" in df.columns else 0
        st.metric("Average Demand", f"{avg_d:.2f} kWh")
    with kpi3:
        med_d = df["Demand"].median() if "Demand" in df.columns else 0
        st.metric("Median Demand", f"{med_d:.2f} kWh")
    with kpi4:
        max_d = df["Demand"].max() if "Demand" in df.columns else 0
        st.metric("Max Demand", f"{max_d:.2f} kWh")
    with kpi5:
        high_pct = (df["Demand"] >= demand_threshold).mean() * 100 if "Demand" in df.columns else 0
        st.metric("High Demand Ratio", f"{high_pct:.1f}%")

    st.divider()

    # Visualizations Row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Demand Distribution")
        fig_hist = px.histogram(
            df,
            x="Demand",
            nbins=50,
            marginal="box",
            title="Distribution of Charging Demand (kWh)",
            color_discrete_sequence=["#3b82f6"]
        )
        if demand_threshold:
            fig_hist.add_vline(
                x=demand_threshold,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Threshold: {demand_threshold:.2f} kWh",
                annotation_position="top right"
            )
        fig_hist.update_layout(xaxis_title="Demand (kWh)", yaxis_title="Session Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Sessions by Time Period")
        if "TimePeriod" in df.columns:
            period_df = df["TimePeriod"].value_counts().reset_index()
            period_df.columns = ["TimePeriod", "Sessions"]
            fig_pie = px.pie(
                period_df,
                names="TimePeriod",
                values="Sessions",
                title="Charging Session Volume by Time Period",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Visualizations Row 2
    col_row2_left, col_row2_right = st.columns(2)

    with col_row2_left:
        st.subheader("Average Demand by Location Type")
        if "Location" in df.columns and "Demand" in df.columns:
            loc_df = df.groupby("Location")["Demand"].agg(["mean", "count"]).reset_index()
            loc_df.columns = ["Location", "Average Demand", "Sessions"]
            loc_df = loc_df.sort_values(by="Average Demand", ascending=False)
            fig_loc = px.bar(
                loc_df,
                x="Average Demand",
                y="Location",
                orientation="h",
                color="Average Demand",
                color_continuous_scale="Viridis",
                title="Average Demand Across Location Types",
                text_auto=".2f"
            )
            fig_loc.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_loc, use_container_width=True)

    with col_row2_right:
        st.subheader("Hourly Demand Pattern")
        if "StartHour" in df.columns and "Demand" in df.columns:
            hour_df = df.groupby("StartHour")["Demand"].mean().reset_index()
            fig_hour = px.line(
                hour_df,
                x="StartHour",
                y="Demand",
                markers=True,
                title="Average Charging Demand by Hour of the Day (0-23)",
                color_discrete_sequence=["#10b981"]
            )
            fig_hour.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2), xaxis_title="Hour of Day", yaxis_title="Avg Demand (kWh)")
            st.plotly_chart(fig_hour, use_container_width=True)


# ============================================================
# 2. REGRESSION MODELS
# ============================================================

elif page == "Regression Models":

    st.markdown('<div class="main-title">Regression Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Continuous charging demand (kWh) prediction benchmark across multiple algorithms</div>', unsafe_allow_html=True)

    if regression_results.empty:
        st.warning("No regression results file found at `outputs/regression_model_comparison.csv`. Please run the regression notebook first.")
        st.stop()

    # Standardize R2 column
    r2_col = "R2" if "R2" in regression_results.columns else ("R²" if "R²" in regression_results.columns else None)

    # Sort results
    reg_sorted = regression_results.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    best_reg = reg_sorted.iloc[0]

    # Best Model Banner
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Best Model", str(best_reg["Model"]))
    with c2:
        st.metric("RMSE (Lower is Better)", f"{best_reg['RMSE']:.4f}")
    with c3:
        st.metric("MAE", f"{best_reg['MAE']:.4f}")
    with c4:
        if r2_col:
            st.metric("R² Score", f"{best_reg[r2_col]:.4f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Performance Comparison", "Feature Importance", "Detailed Metrics Table"])

    with tab1:
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_rmse = px.bar(
                reg_sorted,
                x="RMSE",
                y="Model",
                orientation="h",
                color="RMSE",
                color_continuous_scale="Reds_r",
                title="Root Mean Squared Error (RMSE) Comparison",
                text_auto=".3f"
            )
            fig_rmse.update_layout(yaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col_fig2:
            if r2_col:
                fig_r2 = px.bar(
                    reg_sorted.sort_values(by=r2_col, ascending=False),
                    x=r2_col,
                    y="Model",
                    orientation="h",
                    color=r2_col,
                    color_continuous_scale="Blues",
                    title="R² Score (Variance Explained) Comparison",
                    text_auto=".3f"
                )
                fig_r2.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_r2, use_container_width=True)

    with tab2:
        st.subheader("Top Influential Features for Demand Prediction")
        if regression_model:
            df_reg_imp = get_feature_importances(regression_model)
            if df_reg_imp is not None:
                fig_imp = px.bar(
                    df_reg_imp.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    title=f"Top 15 Feature Importances ({best_reg['Model']})",
                    text_auto=".0f"
                )
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance is available for tree-based models (LightGBM, XGBoost, Random Forest).")
        else:
            st.info("Trained regression model not loaded.")

    with tab3:
        st.subheader("Complete Regression Benchmark Table")
        st.dataframe(reg_sorted, use_container_width=True, hide_index=True)


# ============================================================
# 3. CLASSIFICATION MODELS
# ============================================================

elif page == "Classification Models":

    st.markdown('<div class="main-title">High vs Normal Demand Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predicting whether a charging session will be High Demand or Normal Demand</div>', unsafe_allow_html=True)

    if classification_results.empty:
        st.warning("No classification results file found at `outputs/classification_model_comparison.csv`. Please run the classification notebook first.")
        st.stop()

    f1_col = "F1 Score" if "F1 Score" in classification_results.columns else ("F1" if "F1" in classification_results.columns else "Accuracy")
    cls_sorted = classification_results.sort_values(by=f1_col, ascending=False).reset_index(drop=True)
    best_cls = cls_sorted.iloc[0]

    # Best Model Banner
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Best Model", str(best_cls["Model"]))
    with c2:
        st.metric("Accuracy", f"{best_cls['Accuracy']:.4f}")
    with c3:
        st.metric("Precision", f"{best_cls['Precision']:.4f}")
    with c4:
        st.metric("Recall", f"{best_cls['Recall']:.4f}")
    with c5:
        st.metric("F1 Score", f"{best_cls[f1_col]:.4f}")

    st.info(f"**Classification Split Rule:** High Demand (Class 1) = Demand ≥ {demand_threshold:.2f} kWh | Normal Demand (Class 0) = Demand < {demand_threshold:.2f} kWh")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Performance Comparison", "Feature Importance", "Detailed Metrics Table"])

    with tab1:
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_f1 = px.bar(
                cls_sorted.sort_values(by=f1_col, ascending=True),
                x=f1_col,
                y="Model",
                orientation="h",
                color=f1_col,
                color_continuous_scale="Greens",
                title="F1 Score Comparison (Higher is Better)",
                text_auto=".3f"
            )
            fig_f1.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_f1, use_container_width=True)

        with col_fig2:
            auc_col = "ROC AUC" if "ROC AUC" in cls_sorted.columns else ("ROC-AUC" if "ROC-AUC" in cls_sorted.columns else None)
            if auc_col:
                fig_auc = px.bar(
                    cls_sorted.sort_values(by=auc_col, ascending=True),
                    x=auc_col,
                    y="Model",
                    orientation="h",
                    color=auc_col,
                    color_continuous_scale="Purples",
                    title="ROC-AUC Score Comparison",
                    text_auto=".3f"
                )
                fig_auc.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_auc, use_container_width=True)

    with tab2:
        st.subheader("Top Influential Factors for High-Demand Classification")
        if classification_model:
            df_cls_imp = get_feature_importances(classification_model)
            if df_cls_imp is not None:
                fig_imp = px.bar(
                    df_cls_imp.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Teal",
                    title=f"Top 15 Feature Importances ({best_cls['Model']})",
                    text_auto=".0f"
                )
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance is available for tree-based models.")
        else:
            st.info("Trained classification model not loaded.")

    with tab3:
        st.subheader("Complete Classification Benchmark Table")
        st.dataframe(cls_sorted, use_container_width=True, hide_index=True)


# ============================================================
# 4. DEMAND PREDICTION
# ============================================================

elif page == "Demand Prediction":

    st.markdown('<div class="main-title">Interactive Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Configure charging session attributes to predict continuous energy demand (kWh) and demand classification in real-time</div>', unsafe_allow_html=True)

    st.divider()

    # User input form
    col_input1, col_input2, col_input3 = st.columns(3)

    # 1. Charger & Location
    with col_input1:
        st.markdown("#### Hardware & Location")
        
        # Charger Company
        company_options = sorted(df["ChargerCompany"].dropna().unique().tolist()) if "ChargerCompany" in df.columns else [0, 1]
        charger_company = st.selectbox(
            "Charger Company",
            company_options,
            format_func=lambda x: f"Company Network {x}"
        )

        # Charger Type
        type_options = sorted(df["ChargerType"].dropna().unique().tolist()) if "ChargerType" in df.columns else [0, 1]
        charger_type = st.selectbox(
            "Charger Type",
            type_options,
            format_func=lambda x: f"Type {x} (Fast DC Charger)" if x == 1 else f"Type {x} (Standard AC Charger)"
        )

        # Location
        loc_options = sorted(df["Location"].dropna().astype(str).unique().tolist()) if "Location" in df.columns else ["apartment", "company", "hotel"]
        location = st.selectbox("Installation Location", loc_options)

    # 2. Timing
    with col_input2:
        st.markdown("#### Time & Schedule")

        start_hour = st.slider("Start Hour of Day (24-hr)", min_value=0, max_value=23, value=14, help="Hour when charging starts (0 = 12 AM, 23 = 11 PM)")
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day_name = st.selectbox("Day of Week", day_names, index=2)
        start_day_of_week = day_names.index(selected_day_name)
        
        # Auto-detect weekend
        is_weekend_val = 1 if start_day_of_week in [5, 6] else 0

        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        selected_month_name = st.selectbox("Month of Year", month_names, index=5)
        start_month = month_names.index(selected_month_name) + 1

    # 3. Calendar & Context
    with col_input3:
        st.markdown("#### Context & Period")

        years = sorted(df["StartYear"].dropna().astype(int).unique().tolist()) if "StartYear" in df.columns else [2022, 2023]
        start_year = st.selectbox("Year", years, index=0)

        is_weekend = st.selectbox(
            "Is Weekend?",
            [0, 1],
            index=is_weekend_val,
            format_func=lambda x: "Yes (Weekend)" if x == 1 else "No (Weekday)"
        )

        # Infer TimePeriod dynamically with fallback
        if start_hour < 6:
            default_tp = "Night"
        elif start_hour < 12:
            default_tp = "Morning"
        elif start_hour < 18:
            default_tp = "Afternoon"
        else:
            default_tp = "Evening"

        period_options = sorted(df["TimePeriod"].dropna().astype(str).unique().tolist()) if "TimePeriod" in df.columns else ["Morning", "Afternoon", "Evening", "Night"]
        default_tp_index = period_options.index(default_tp) if default_tp in period_options else 0

        time_period = st.selectbox(
            "Time Period",
            period_options,
            index=default_tp_index
        )

    # Build input DataFrame
    input_data = pd.DataFrame([{
        "ChargerCompany": charger_company,
        "Location": location,
        "ChargerType": charger_type,
        "StartHour": int(start_hour),
        "StartDayOfWeek": int(start_day_of_week),
        "StartMonth": int(start_month),
        "StartYear": int(start_year),
        "IsWeekend": int(is_weekend),
        "TimePeriod": time_period
    }])

    st.divider()

    # Predict Button
    predict_btn = st.button("Predict Demand and kWh", type="primary", use_container_width=True)

    if predict_btn:
        if regression_model is None or classification_model is None:
            st.error("Model files are missing. Please ensure models are trained and saved.")
        else:
            try:
                # 1. Regression Prediction
                pred_demand = float(regression_model.predict(input_data)[0])

                # 2. Classification Prediction & Probability
                pred_class = int(classification_model.predict(input_data)[0])
                prob_high = float(classification_model.predict_proba(input_data)[0][1]) if hasattr(classification_model, "predict_proba") else (1.0 if pred_class == 1 else 0.0)

                st.markdown("### Prediction Results")

                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Predicted Energy Demand</div>
                            <div class="metric-val">{pred_demand:.2f} <span style="font-size:16px;color:#64748b;">kWh</span></div>
                            <div style="margin-top:8px;font-size:13px;color:#64748b;">
                                Baseline Avg: {df['Demand'].mean():.2f} kWh ({'+' if pred_demand >= df['Demand'].mean() else ''}{pred_demand - df['Demand'].mean():.2f} kWh)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_col2:
                    badge = '<span class="status-badge-high">HIGH DEMAND</span>' if pred_class == 1 else '<span class="status-badge-normal">NORMAL DEMAND</span>'
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Predicted Demand Category</div>
                            <div style="margin-top:10px;">{badge}</div>
                            <div style="margin-top:10px;font-size:13px;color:#64748b;">
                                Threshold: {demand_threshold:.2f} kWh
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_col3:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">High Demand Probability</div>
                            <div class="metric-val">{prob_high * 100:.1f}%</div>
                            <div style="margin-top:8px;font-size:13px;color:#64748b;">
                                Model Confidence: {max(prob_high, 1 - prob_high) * 100:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Gauge chart for probability
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_high * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "High Demand Probability Meter", "font": {"size": 16}},
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444" if prob_high >= 0.5 else "#3b82f6"},
                        "steps": [
                            {"range": [0, 50], "color": "#eff6ff"},
                            {"range": [50, 100], "color": "#fef2f2"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.divider()

                # 3. 24-Hour Continuous Regression Demand Forecast Profile
                st.subheader("24-Hour Energy Demand Forecast Profile")
                st.caption("Hourly regression projection simulating energy demand throughout the day under the chosen configuration.")

                hours_range = list(range(24))
                sim_time_periods = [
                    "Night" if h < 6 else ("Morning" if h < 12 else ("Afternoon" if h < 18 else "Evening"))
                    for h in hours_range
                ]
                sim_24h_df = pd.DataFrame({
                    "ChargerCompany": [charger_company] * 24,
                    "Location": [location] * 24,
                    "ChargerType": [charger_type] * 24,
                    "StartHour": hours_range,
                    "StartDayOfWeek": [int(start_day_of_week)] * 24,
                    "StartMonth": [int(start_month)] * 24,
                    "StartYear": [int(start_year)] * 24,
                    "IsWeekend": [int(is_weekend)] * 24,
                    "TimePeriod": sim_time_periods
                })

                hourly_pred = regression_model.predict(sim_24h_df)

                # Metrics for 24h forecast
                peak_idx = int(np.argmax(hourly_pred))
                min_idx = int(np.argmin(hourly_pred))

                fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
                with fc_col1:
                    st.metric("Peak Demand Hour", f"{peak_idx}:00", f"{hourly_pred[peak_idx]:.2f} kWh")
                with fc_col2:
                    st.metric("Lowest Demand Hour", f"{min_idx}:00", f"{hourly_pred[min_idx]:.2f} kWh")
                with fc_col3:
                    st.metric("24-Hr Average Demand", f"{np.mean(hourly_pred):.2f} kWh")
                with fc_col4:
                    cur_diff = pred_demand - np.mean(hourly_pred)
                    st.metric("Selected vs Daily Avg", f"{pred_demand:.2f} kWh", f"{'+' if cur_diff >= 0 else ''}{cur_diff:.2f} kWh")

                # Build 24h Forecast Curve
                fig_forecast = go.Figure()

                # Add historical benchmark line if available
                if "StartHour" in df.columns and "Demand" in df.columns:
                    hist_avg = df.groupby("StartHour")["Demand"].mean().reindex(hours_range).values
                    fig_forecast.add_trace(go.Scatter(
                        x=hours_range,
                        y=hist_avg,
                        mode="lines",
                        name="Historical Dataset Avg",
                        line=dict(color="#94a3b8", width=2, dash="dot"),
                        hovertemplate="Historical Avg: %{y:.2f} kWh<extra></extra>"
                    ))

                # Add predicted 24h regression curve
                fig_forecast.add_trace(go.Scatter(
                    x=hours_range,
                    y=hourly_pred,
                    mode="lines+markers",
                    name="Forecasted Demand (Regression)",
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=6, color="#2563eb"),
                    fill="tozeroy",
                    fillcolor="rgba(37, 99, 235, 0.08)",
                    hovertemplate="Hour %{x}:00<br>Forecast: %{y:.2f} kWh<extra></extra>"
                ))

                # Add marker for selected hour
                fig_forecast.add_trace(go.Scatter(
                    x=[int(start_hour)],
                    y=[pred_demand],
                    mode="markers+text",
                    name=f"Selected Hour ({start_hour}:00)",
                    marker=dict(size=14, color="#ef4444", symbol="star"),
                    text=[f"  {pred_demand:.2f} kWh"],
                    textposition="top right",
                    textfont=dict(color="#ef4444", size=12, family="sans-serif"),
                    hovertemplate=f"Selected Hour: {start_hour}:00<br>Demand: {pred_demand:.2f} kWh<extra></extra>"
                ))

                # Add demand threshold line
                if demand_threshold:
                    fig_forecast.add_hline(
                        y=demand_threshold,
                        line_dash="dash",
                        line_color="#dc2626",
                        annotation_text=f"High Demand Cutoff ({demand_threshold:.2f} kWh)",
                        annotation_position="bottom right"
                    )

                fig_forecast.update_layout(
                    title="24-Hour Continuous Demand Forecast Curve",
                    xaxis=dict(
                        tickmode="linear",
                        tick0=0,
                        dtick=2,
                        title="Hour of Day (0:00 - 23:00)"
                    ),
                    yaxis=dict(title="Energy Demand (kWh)"),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=420,
                    margin=dict(l=20, r=20, t=60, b=30)
                )

                st.plotly_chart(fig_forecast, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)


# ============================================================
# 5. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown('<div class="main-title">EV Charging Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Filter, inspect, and export clean feature-engineered charging session records</div>', unsafe_allow_html=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
    with c2:
        st.metric("Total Features", len(df.columns))
    with c3:
        st.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    with c4:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / (1024*1024):.2f} MB")

    st.divider()

    st.markdown("#### Filter Records")
    fcol1, fcol2, fcol3 = st.columns(3)

    filtered_df = df.copy()

    with fcol1:
        if "Location" in df.columns:
            all_locs = sorted(df["Location"].dropna().unique())
            selected_locs = st.multiselect("Location", all_locs)
            if selected_locs:
                filtered_df = filtered_df[filtered_df["Location"].isin(selected_locs)]

    with fcol2:
        if "TimePeriod" in df.columns:
            all_periods = sorted(df["TimePeriod"].dropna().unique())
            selected_periods = st.multiselect("Time Period", all_periods)
            if selected_periods:
                filtered_df = filtered_df[filtered_df["TimePeriod"].isin(selected_periods)]

    with fcol3:
        if "ChargerType" in df.columns:
            all_types = sorted(df["ChargerType"].dropna().unique())
            selected_types = st.multiselect("Charger Type", all_types, format_func=lambda x: f"Type {x}")
            if selected_types:
                filtered_df = filtered_df[filtered_df["ChargerType"].isin(selected_types)]

    # Demand range slider
    if "Demand" in df.columns and not df["Demand"].dropna().empty:
        min_d = float(df["Demand"].dropna().min())
        max_d = float(df["Demand"].dropna().max())
        if min_d < max_d:
            demand_filter = st.slider(
                "Filter by Demand Range (kWh)",
                min_value=min_d,
                max_value=max_d,
                value=(min_d, max_d)
            )
            filtered_df = filtered_df[filtered_df["Demand"].between(demand_filter[0], demand_filter[1])]

    st.markdown(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records ({len(filtered_df)/len(df)*100:.1f}%)")

    tab_table, tab_stats = st.tabs(["Data Table", "Summary Statistics"])

    with tab_table:
        st.dataframe(filtered_df, use_container_width=True, height=450)
        
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered CSV",
            data=csv_bytes,
            file_name="filtered_ev_charging_data.csv",
            mime="text/csv",
            type="primary"
        )

    with tab_stats:
        st.dataframe(filtered_df.describe(), use_container_width=True)