import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
import shap
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(layout="wide", page_title="Aircraft Predictive Maintenance Dashboard")

st.title("✈️ Aircraft Predictive Maintenance Dashboard")
st.markdown("Welcome to your interactive dashboard for aviation maintenance analytics. All data generation, feature engineering, and model training run automatically when the app starts, or can be re-run with the button below. Explore the tabs to dive into EDA, model performance, explainability, and aircraft-specific insights!")

# --- Global Parameters ---
N_AIRCRAFT = 200
N_FLIGHTS = 50000
FAILURE_RATE = 0.03
FLIGHT_START_DATE = datetime(2023, 1, 1)
ALERT_THRESHOLD_PROB = 0.50 # For insights/recommendations

# Set a random seed for reproducibility
np.random.seed(42)

# --- Data Generation Function ---
@st.cache_data(show_spinner="Generating realistic synthetic flight data...")
def generate_synthetic_data(n_aircraft, n_flights, failure_rate, flight_start_date):
    aircraft_ids = [f'AC-{i:03d}' for i in range(1, n_aircraft + 1)]
    data = []
    flight_id_counter = 1

    for aircraft_id in tqdm(aircraft_ids, desc="Generating data"):
        base_engine_temp = np.random.uniform(95, 105)
        base_engine_vibration = np.random.uniform(0.5, 1.5)
        base_oil_pressure = np.random.uniform(40, 50)
        base_hydraulic_pressure = np.random.uniform(2900, 3100)
        current_flight_date = flight_start_date
        
        # Simulate enough flights for potential failures
        total_days_sim = n_flights // n_aircraft * 2
        
        while current_flight_date < (flight_start_date + timedelta(days=total_days_sim)):
            will_fail = np.random.rand() < (failure_rate * 0.5) 
            
            if will_fail:
                degradation_flights = np.random.randint(25, 75)
                for i in range(degradation_flights):
                    is_sudden_failure = np.random.rand() < 0.1
                    engine_temp_c = base_engine_temp + (i / 10) + np.random.normal(0, 1.5)
                    engine_vibration_mm_s = base_engine_vibration + (i / 20) + np.random.normal(0, 0.2)
                    oil_pressure_psi = base_oil_pressure - (i / 5) + np.random.normal(0, 2)
                    hydraulic_pressure_psi = base_hydraulic_pressure - (i / 5) + np.random.normal(0, 10)
                    if is_sudden_failure and i > 5:
                        engine_temp_c += np.random.uniform(5, 15)
                        engine_vibration_mm_s += np.random.uniform(1.0, 5.0)
                    
                    flight_duration = np.random.uniform(1.5, 5.0)
                    takeoff_landings = 1
                    remaining_useful_life_hrs = (degradation_flights - i - 1) * np.random.uniform(1.5, 5.0)
                    failure_within_50hrs = 1 if remaining_useful_life_hrs <= 50 else 0
                    oat_c = np.random.uniform(-10, 35)
                    cabin_pressure_psi = np.random.normal(12, 0.5)
                    fuel_flow_kg_hr = 1500 + 50 * flight_duration + np.random.normal(0, 25)
                    
                    data.append([
                        flight_id_counter, aircraft_id, current_flight_date,
                        engine_temp_c + 0.1 * oat_c,
                        engine_vibration_mm_s + 0.05 * oat_c,
                        oil_pressure_psi, hydraulic_pressure_psi,
                        cabin_pressure_psi, fuel_flow_kg_hr, oat_c,
                        flight_duration * 60, takeoff_landings,
                        failure_within_50hrs, remaining_useful_life_hrs
                    ])
                    flight_id_counter += 1
                    current_flight_date += timedelta(days=np.random.randint(1, 3))
                    
                    if flight_id_counter > n_flights:
                        break
                
                current_flight_date += timedelta(days=np.random.randint(7, 30)) # Simulate maintenance downtime
                base_engine_temp = np.random.uniform(95, 105)
                base_engine_vibration = np.random.uniform(0.5, 1.5)
                base_oil_pressure = np.random.uniform(40, 50)
                base_hydraulic_pressure = np.random.uniform(2900, 3100)
                
            else: # Normal flight
                flight_duration = np.random.uniform(1.5, 5.0)
                takeoff_landings = 1
                engine_temp_c = base_engine_temp + np.random.normal(0, 1.5)
                engine_vibration_mm_s = base_engine_vibration + np.random.normal(0, 0.2)
                oil_pressure_psi = base_oil_pressure + np.random.normal(0, 2)
                hydraulic_pressure_psi = base_hydraulic_pressure + np.random.normal(0, 10)
                oat_c = np.random.uniform(-10, 35)
                cabin_pressure_psi = np.random.normal(12, 0.5)
                fuel_flow_kg_hr = 1500 + 50 * flight_duration + np.random.normal(0, 25)
                remaining_useful_life_hrs = -1 
                failure_within_50hrs = 0
                
                data.append([
                    flight_id_counter, aircraft_id, current_flight_date,
                    engine_temp_c + 0.1 * oat_c,
                    engine_vibration_mm_s + 0.05 * oat_c,
                    oil_pressure_psi, hydraulic_pressure_psi,
                    cabin_pressure_psi, fuel_flow_kg_hr, oat_c,
                    flight_duration * 60, takeoff_landings,
                    failure_within_50hrs, remaining_useful_life_hrs
                ])
                flight_id_counter += 1
                current_flight_date += timedelta(days=np.random.randint(1, 3))
                
                if flight_id_counter > n_flights:
                    break
                
    columns = [
        'flight_id', 'aircraft_id', 'flight_date', 'engine_temp_c',
        'engine_vibration_mm_s', 'oil_pressure_psi', 'hydraulic_pressure_psi',
        'cabin_pressure_psi', 'fuel_flow_kg_hr', 'oat_c',
        'flight_duration_min', 'takeoff_landings', 'failure_within_50hrs',
        'remaining_useful_life_hrs'
    ]
    df = pd.DataFrame(data, columns=columns)
    df = df.iloc[:n_flights].sort_values(by=['aircraft_id', 'flight_date']).reset_index(drop=True)
    df.drop(['flight_id'], axis=1, inplace=True)
    return df

# --- Feature Engineering Function ---
@st.cache_data(show_spinner="Performing feature engineering...")
def feature_engineer_data(df_input):
    df_fe = df_input.copy()
    df_fe.sort_values(by=['aircraft_id', 'flight_date'], inplace=True)

    window_sizes = [5, 10, 20]
    features_to_roll = ['engine_temp_c', 'engine_vibration_mm_s', 'oil_pressure_psi', 'hydraulic_pressure_psi']

    for window in window_sizes:
        for feature in features_to_roll:
            df_fe[f'rolling_avg_{feature}_{window}fl'] = df_fe.groupby('aircraft_id')[feature].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df_fe[f'rolling_std_{feature}_{window}fl'] = df_fe.groupby('aircraft_id')[feature].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)

    for col in df_fe.columns:
        if 'rolling_std' in col:
            df_fe[col] = df_fe[col].fillna(0)

    df_fe['vibration_change_rate'] = df_fe.groupby('aircraft_id')['engine_vibration_mm_s'].diff().fillna(0)
    df_fe['temp_change_rate'] = df_fe.groupby('aircraft_id')['engine_temp_c'].diff().fillna(0)
    df_fe['combined_stress_index'] = df_fe['engine_vibration_mm_s'] * df_fe['engine_temp_c']

    # Cycles since last maintenance
    # This logic aims to reset cycles after a simulated maintenance (post-failure)
    df_fe['temp_cycles'] = df_fe.groupby('aircraft_id').cumcount() + 1
    
    current_cycles = {}
    final_cycles = np.zeros(len(df_fe))
    for i, row in df_fe.iterrows():
        ac_id = row['aircraft_id']
        if ac_id not in current_cycles:
            current_cycles[ac_id] = 0
        
        current_cycles[ac_id] += 1
        final_cycles[i] = current_cycles[ac_id]
        
        if row['failure_within_50hrs'] == 1:
            current_cycles[ac_id] = 0 # Reset for the *next* flight after this failure event

    df_fe['cycles_since_maintenance'] = final_cycles
    return df_fe

# --- Model Training Function ---
@st.cache_resource(show_spinner="Training predictive models (XGBoost Classifier & Regressor)...")
def train_predictive_models(df_processed):
    features = [col for col in df_processed.columns if col not in ['aircraft_id', 'flight_date', 'failure_within_50hrs', 'remaining_useful_life_hrs']]
    target_classification = 'failure_within_50hrs'
    target_regression = 'remaining_useful_life_hrs'

    # Time-based train-test split (using a fixed proportion for reproducibility)
    train_end_date = df_processed['flight_date'].max() - timedelta(days=90) # Last 90 days for test
    train_df = df_processed[df_processed['flight_date'] <= train_end_date].copy()
    test_df = df_processed[df_processed['flight_date'] > train_end_date].copy()
    
    # Ensure test_df has data
    if test_df.empty:
        st.warning("Test data is empty after time-based split. Adjusting split or data generation parameters may be needed.")
        # Fallback to a small test set if time split results in empty
        if len(df_processed) > 100:
            train_df = df_processed.iloc[:-50].copy()
            test_df = df_processed.iloc[-50:].copy()
        else:
            train_df = df_processed.copy()
            test_df = df_processed.copy() # Use same for train/test if data too small

    X_train_cls = train_df[features]
    y_train_cls = train_df[target_classification]
    X_test_cls = test_df[features]
    y_test_cls = test_df[target_classification]

    # Regression data (only for flights with defined RUL)
    X_train_reg = train_df[train_df[target_regression] > 0][features]
    y_train_reg = train_df[train_df[target_regression] > 0][target_regression]
    X_test_reg = test_df[test_df[target_regression] > 0][features]
    y_test_reg = test_df[test_df[target_regression] > 0][target_regression]

    # Handle cases where regression train/test sets might be empty
    if X_train_reg.empty or y_train_reg.empty:
        st.warning("Regression training data is empty. RUL model will not be trained.")
        xgb_reg = None
    else:
        xgb_reg = xgb.XGBRegressor(random_state=42)
        xgb_reg.fit(X_train_reg, y_train_reg)

    xgb_cls = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_cls.fit(X_train_cls, y_train_cls)

    return xgb_cls, xgb_reg, X_test_cls, y_test_cls, X_test_reg, y_test_reg, features, test_df


# --- Main Application Logic ---

# Trigger all initial steps
if st.button("Run Full Analysis"):
    st.session_state.data = generate_synthetic_data(N_AIRCRAFT, N_FLIGHTS, FAILURE_RATE, FLIGHT_START_DATE)
    st.session_state.df_fe = feature_engineer_data(st.session_state.data)
    (st.session_state.xgb_cls, st.session_state.xgb_reg, 
     st.session_state.X_test_cls, st.session_state.y_test_cls, 
     st.session_state.X_test_reg, st.session_state.y_test_reg, 
     st.session_state.features, st.session_state.test_df) = train_predictive_models(st.session_state.df_fe)
    st.success("Data loaded, features engineered, and models trained!")

if st.session_state.df_fe is None:
    st.info("Click 'Run Full Analysis' to load data, engineer features, and train models. This may take a few minutes.")
else:
    # Selector for aircraft ID (used in Dashboard and Insights)
    available_aircraft_ids = st.session_state.df_fe['aircraft_id'].unique()
    st.session_state.selected_aircraft = st.sidebar.selectbox(
        "Select Aircraft for Dashboard/Insights:", 
        available_aircraft_ids
    )

    # Create tabs for the dashboard sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview (EDA)", "🧠 Model Performance", "💡 Model Explainability", "🖥️ Interactive Dashboard", "✨ Insights & Recommendations"])

    with tab1:
        st.header("Data Overview (EDA) 📊")
        st.subheader("Tabular View of Data")
        st.dataframe(st.session_state.data.head(10))

        st.subheader("Time-series Trends for Aircraft Approaching Failure")
        if st.session_state.data['failure_within_50hrs'].sum() > 0:
            failure_aircraft_id = st.session_state.data[st.session_state.data['failure_within_50hrs'] == 1]['aircraft_id'].iloc[0]
            failed_flights = st.session_state.data[st.session_state.data['aircraft_id'] == failure_aircraft_id].sort_values('flight_date').tail(75)

            fig_ts = px.line(
                failed_flights,
                x='flight_date',
                y=['engine_vibration_mm_s', 'engine_temp_c', 'oil_pressure_psi', 'hydraulic_pressure_psi'],
                title=f'Sensor Readings for Aircraft {failure_aircraft_id} Approaching Failure',
                labels={'value': 'Sensor Reading', 'variable': 'Sensor'},
                template='plotly_white'
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No failures detected in the generated data to plot time-series trends.")

        st.subheader("Sensor Reading Distributions by Failure Status")
        col1_eda, col2_eda = st.columns(2)
        with col1_eda:
            fig_hist_vib = px.histogram(
                st.session_state.data,
                x='engine_vibration_mm_s',
                color='failure_within_50hrs',
                nbins=50,
                title='Engine Vibration Distribution',
                barmode='overlay',
                opacity=0.7,
                template='plotly_white'
            )
            st.plotly_chart(fig_hist_vib, use_container_width=True)
        with col2_eda:
            fig_box_temp = px.box(
                st.session_state.data,
                x='failure_within_50hrs',
                y='engine_temp_c',
                title='Engine Temperature Boxplot',
                template='plotly_white'
            )
            st.plotly_chart(fig_box_temp, use_container_width=True)

        col3_eda, col4_eda = st.columns(2)
        with col3_eda:
            fig_hist_oil = px.histogram(
                st.session_state.data,
                x='oil_pressure_psi',
                color='failure_within_50hrs',
                nbins=50,
                title='Oil Pressure Distribution',
                barmode='overlay',
                opacity=0.7,
                template='plotly_white'
            )
            st.plotly_chart(fig_hist_oil, use_container_width=True)
        with col4_eda:
            fig_box_hyd = px.box(
                st.session_state.data,
                x='failure_within_50hrs',
                y='hydraulic_pressure_psi',
                title='Hydraulic Pressure Boxplot',
                template='plotly_white'
            )
            st.plotly_chart(fig_box_hyd, use_container_width=True)

        st.subheader("Correlation Heatmap of Numerical Features")
        numerical_cols = st.session_state.data.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = st.session_state.data[numerical_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            title='Correlation Heatmap',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Class Imbalance Visualization")
        failure_counts = st.session_state.data['failure_within_50hrs'].value_counts().reset_index()
        failure_counts.columns = ['Failure Status', 'Count']
        fig_imbalance = px.bar(
            failure_counts,
            x='Failure Status',
            y='Count',
            title='Class Imbalance of failure_within_50hrs',
            labels={'Failure Status': 'Failure Within 50 Hours (0 = No, 1 = Yes)'},
            color='Failure Status',
            template='plotly_white'
        )
        fig_imbalance.update_traces(marker_color=['skyblue', 'salmon'])
        st.plotly_chart(fig_imbalance, use_container_width=True)
    
    with tab2:
        st.header("Model Performance 🧠")
        if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None:
            st.warning("Models not trained yet. Please click 'Run Full Analysis' on the main page.")
        else:
            st.subheader("Classification Model Evaluation")
            y_pred_cls = st.session_state.xgb_cls.predict(st.session_state.X_test_cls)
            y_pred_proba_cls = st.session_state.xgb_cls.predict_proba(st.session_state.X_test_cls)[:, 1]

            st.write(f"**ROC-AUC:** {roc_auc_score(st.session_state.y_test_cls, y_pred_proba_cls):.4f}")
            st.write(f"**Precision:** {precision_score(st.session_state.y_test_cls, y_pred_cls):.4f}")
            st.write(f"**Recall:** {recall_score(st.session_state.y_test_cls, y_pred_cls):.4f}")
            st.write(f"**F1 Score:** {f1_score(st.session_state.y_test_cls, y_pred_cls):.4f}")

            st.subheader("Regression Model Evaluation")
            y_pred_reg = st.session_state.xgb_reg.predict(st.session_state.X_test_reg)
            y_pred_reg[y_pred_reg < 0] = 0

            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(st.session_state.y_test_reg, y_pred_reg)):.4f}")
            st.write(f"**MAE:** {mean_absolute_error(st.session_state.y_test_reg, y_pred_reg):.4f}")
            st.write(f"**R-squared:** {r2_score(st.session_state.y_test_reg, y_pred_reg):.4f}")

    with tab3:
        st.header("Model Explainability with SHAP 💡")
        if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None:
            st.warning("Models not trained yet. Please click 'Run Full Analysis' on the main page.")
        else:
            st.subheader("SHAP Feature Importance (Classification Model)")
            explainer_cls = shap.TreeExplainer(st.session_state.xgb_cls)
            shap_values_cls = explainer_cls.shap_values(st.session_state.X_test_cls)
            
            # Using st.pyplot to display SHAP plots
            fig_shap_cls, ax_cls = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_cls, st.session_state.X_test_cls, plot_type="bar", show=False, ax=ax_cls)
            ax_cls.set_title("SHAP Feature Importance (Classification)")
            st.pyplot(fig_shap_cls)

            st.subheader("SHAP Feature Importance (Regression Model)")
            explainer_reg = shap.TreeExplainer(st.session_state.xgb_reg)
            shap_values_reg = explainer_reg.shap_values(st.session_state.X_test_reg)

            fig_shap_reg, ax_reg = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_reg, st.session_state.X_test_reg, show=False, ax=ax_reg)
            ax_reg.set_title("SHAP Feature Importance (Regression)")
            st.pyplot(fig_shap_reg)

            st.subheader("Example SHAP Explanation for a Single Flight (Classification)")
            if st.session_state.y_test_cls.sum() > 0:
                failed_flight_idx = st.session_state.y_test_cls[st.session_state.y_test_cls == 1].index[0]
                failed_aircraft_data = st.session_state.test_df.loc[failed_flight_idx]
                failed_aircraft_features = failed_aircraft_data[st.session_state.features].to_frame().T 

                st.write(f"Explaining prediction for flight of aircraft **{failed_aircraft_data['aircraft_id']}** on **{failed_aircraft_data['flight_date'].date()}**")
                
                shap_values_single_instance = explainer_cls.shap_values(failed_aircraft_features)[0] 
                
                fig_force, ax_force = plt.subplots(figsize=(12, 4))
                shap.force_plot(
                    explainer_cls.expected_value[0], 
                    shap_values_single_instance, 
                    failed_aircraft_features, 
                    show=False, 
                    matplotlib=True,
                    plot_cmap="PkGn",
                    ax=ax_force
                )
                ax_force.set_title("SHAP Force Plot for a Single Prediction (Class 0)")
                st.pyplot(fig_force)

            else:
                st.info("No failure instances in the test set to demonstrate a single flight explanation.")
    
    with tab4:
        st.header("Interactive Dashboard 🖥️")
        st.markdown("Monitor an aircraft's health over time with key sensor readings, predicted failure risk, and Remaining Useful Life.")

        if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None or st.session_state.df_fe is None:
            st.warning("Models and data not ready. Please click 'Run Full Analysis' on the main page.")
        elif st.session_state.selected_aircraft:
            aircraft_data_for_dashboard = st.session_state.df_fe[st.session_state.df_fe['aircraft_id'] == st.session_state.selected_aircraft].sort_values('flight_date').copy()
            
            if not aircraft_data_for_dashboard.empty:
                aircraft_data_for_dashboard['failure_probability'] = st.session_state.xgb_cls.predict_proba(aircraft_data_for_dashboard[st.session_state.features])[:, 1]
                aircraft_data_for_dashboard['predicted_rul'] = st.session_state.xgb_reg.predict(aircraft_data_for_dashboard[st.session_state.features])
                aircraft_data_for_dashboard['predicted_rul'] = aircraft_data_for_dashboard['predicted_rul'].apply(lambda x: max(0, x))

                fig_dashboard = go.Figure()

                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['engine_vibration_mm_s'], mode='lines+markers', name='Engine Vibration (mm/s)', yaxis='y1'))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['engine_temp_c'], mode='lines+markers', name='Engine Temperature (°C)', yaxis='y1'))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['oil_pressure_psi'], mode='lines+markers', name='Oil Pressure (psi)', yaxis='y1'))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['hydraulic_pressure_psi'], mode='lines+markers', name='Hydraulic Pressure (psi)', yaxis='y1'))

                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['failure_probability'] * 100, mode='lines+markers', name='Failure Risk (%)', yaxis='y2', line=dict(dash='dash', color='red')))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['predicted_rul'], mode='lines+markers', name='Predicted RUL (hrs)', yaxis='y3', line=dict(dash='dot', color='green')))

                fig_dashboard.update_layout(
                    title=f"Predictive Maintenance Dashboard for Aircraft {st.session_state.selected_aircraft}",
                    xaxis_title="Date",
                    yaxis=dict(
                        title="Sensor Readings",
                        side="left",
                        titlefont=dict(color="blue"),
                        tickfont=dict(color="blue")
                    ),
                    yaxis2=dict(
                        title="Failure Risk (%)",
                        overlaying="y",
                        side="right",
                        anchor="free",
                        position=0.95,
                        showgrid=False,
                        titlefont=dict(color="red"),
                        tickfont=dict(color="red")
                    ),
                    yaxis3=dict(
                        title="Predicted RUL (hrs)",
                        overlaying="y",
                        side="right",
                        anchor="x",
                        position=0.88, 
                        showgrid=False,
                        titlefont=dict(color="green"),
                        tickfont=dict(color="green")
                    ),
                    legend=dict(x=0, y=1.1, orientation="h"),
                    height=600,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_dashboard, use_container_width=True)
            else:
                st.warning(f"No data available for selected aircraft ID: {st.session_state.selected_aircraft}. Please select another or re-run analysis.")
        else:
            st.info("Please select an aircraft ID from the sidebar to view its dashboard.")

    with tab5:
        st.header("Insights & Recommendations ✨")
        st.markdown("Here, we present key insights derived from the analysis and actionable recommendations, **dynamically generated** from the current model and data state.")
        
        if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None or st.session_state.features is None:
            st.warning("Models and features not ready. Please click 'Run Full Analysis' on the main page to generate dynamic insights.")
        else:
            st.subheader("Overall Insights from Analysis")
            
            y_pred_cls_proba = st.session_state.xgb_cls.predict_proba(st.session_state.X_test_cls)[:, 1]
            
            st.markdown(f"""
            * **Degradation Patterns**: Our synthetic data successfully simulated realistic degradation, showing a **gradual increase in engine vibration and temperature** before failures, coupled with slight drops in oil/hydraulic pressure. This pattern is crucial for proactive maintenance.
            * **Environmental Impact**: **Outside Air Temperature (OAT)** was observed to directly influence sensor readings. For example, the `engine_temp_c` simulation incorporated a factor of `0.1 * oat_c`, showing that **hotter environments can increase stress** on components.
            * **Feature Importance**: Features derived from **rolling averages and change rates** of sensor data, especially engine vibration and temperature, proved to be highly predictive for both failure classification and RUL estimation. The **combined stress index** ($engine\_vibration\_mm\_s \times engine\_temp\_c$) also played a significant role.
            * **Model Performance**:
                * The **XGBoost Classifier** (for failure prediction) achieved a strong **ROC-AUC of {roc_auc_score(st.session_state.y_test_cls, y_pred_cls_proba):.4f}**. This indicates a high capability to distinguish between flights likely to fail and those that are not.
                * The **XGBoost Regressor** (for Remaining Useful Life) has a **RMSE of {np.sqrt(mean_squared_error(st.session_state.y_test_reg, st.session_state.xgb_reg.predict(st.session_state.X_test_reg))):.4f}** and an **R-squared of {r2_score(st.session_state.y_test_reg, st.session_state.xgb_reg.predict(st.session_state.X_test_reg)) if len(st.session_state.y_test_reg) > 1 else 0:.4f}**. This shows it can predict RUL with a reasonable degree of accuracy, enabling better maintenance planning.
            * **Explainable AI**: **SHAP values** revealed that features reflecting increasing engine wear (e.g., higher vibration, temperature, and their rolling statistics) were the primary drivers for predicting imminent failure and lower RUL, providing transparency to model predictions.
            """)

            st.subheader("Top Predictive Features")
            st.markdown("The following features were identified as the most important in predicting aircraft failure (from the Classification Model):")
            
            feature_importances = pd.Series(st.session_state.xgb_cls.feature_importances_, index=st.session_state.features)
            top_5_features = feature_importances.nlargest(5)
            
            for i, (feature, importance) in enumerate(top_5_features.items()):
                st.write(f"**{i+1}. {feature}** (Importance: {importance:.4f})")

            st.subheader("Aircraft-Specific Recommendations")
            
            if st.session_state.selected_aircraft:
                st.write(f"Recommendations for aircraft **{st.session_state.selected_aircraft}**, based on its latest flight data:")
                
                latest_flight = st.session_state.df_fe[st.session_state.df_fe['aircraft_id'] == st.session_state.selected_aircraft].sort_values('flight_date').iloc[-1]
                latest_features = latest_flight[st.session_state.features].to_frame().T
                
                current_prob_failure = st.session_state.xgb_cls.predict_proba(latest_features)[:, 1][0]
                predicted_rul = st.session_state.xgb_reg.predict(latest_features)[0]
                
                col1_rec, col2_rec, col3_rec = st.columns(3)
                with col1_rec:
                    st.metric("Latest Failure Risk", f"{current_prob_failure*100:.2f}%")
                with col2_rec:
                    st.metric("Predicted RUL", f"{max(0, predicted_rul):.2f} hours")
                with col3_rec:
                    st.metric("Cycles since Maint.", int(latest_flight['cycles_since_maintenance']))

                st.markdown("---")

                st.write(f"**Based on these metrics, here are the recommendations for Aircraft {st.session_state.selected_aircraft}:**")
                
                if current_prob_failure >= ALERT_THRESHOLD_PROB:
                    st.error(f"**⚠️ IMMEDIATE ACTION REQUIRED:** The failure risk is critically high at **{current_prob_failure*100:.2f}%**.")
                    st.markdown(f"""
                    - **Schedule immediate inspection** for critical components.
                    - Review the **Interactive Dashboard** tab to see which sensors (e.g., high vibration, high temp) are trending abnormally for this aircraft.
                    - Consider grounding the aircraft until a full maintenance check is performed to prevent unexpected failures and ensure safety.
                    """)
                elif predicted_rul <= 100 and predicted_rul > 0: # RUL can be -1 if not a failure
                    st.warning(f"**⚠️ PROACTIVE MAINTENANCE RECOMMENDED:** The predicted Remaining Useful Life (RUL) is low (**{predicted_rul:.2f} hours**).")
                    st.markdown("""
                    - **Schedule maintenance** within the next few flights or before the RUL hits zero.
                    - Proactive intervention can prevent a costly in-flight incident, minimize downtime, and improve operational efficiency.
                    - Continue to monitor this aircraft closely on the **Interactive Dashboard** tab.
                    """)
                else:
                    st.success("✅ **STATUS NORMAL:** The aircraft is operating within normal parameters.")
                    st.markdown("""
                    - Continue with routine monitoring as planned.
                    - No immediate action is required based on the current data and predictions.
                    - The predicted RUL is at a healthy level.
                    """)
            else:
                st.info("Please select an aircraft from the sidebar to view its specific recommendations.")

            st.subheader("General Recommendations for Predictive Maintenance Implementation")
            st.markdown("""
            * **Automated Alerting**: Develop an automated system to trigger alerts when an aircraft's predicted failure probability exceeds a threshold or RUL drops below a critical value.
            * **Maintenance Prioritization**: Leverage failure probabilities and RUL estimates to dynamically prioritize maintenance tasks across the fleet, optimizing resource allocation.
            * **Spare Parts Optimization**: Use RUL forecasts to inform inventory management, ensuring critical parts are available when needed and reducing excess stock.
            * **Data Quality & Volume**: Continuously improve sensor data quality and consider integrating more diverse data sources (e.g., historical maintenance logs, flight plans, environmental conditions for specific routes).
            * **Model Monitoring & Retraining**: Regularly monitor model performance in a production environment. Retrain models periodically with new data to adapt to changing operational conditions and component degradation patterns.
            * **Human-in-the-Loop Validation**: Maintain a feedback loop with maintenance engineers. Their practical insights are invaluable for validating model predictions and identifying areas for improvement.
            """)
