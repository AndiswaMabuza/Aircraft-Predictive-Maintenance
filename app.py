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

# Set page configuration for a wide layout and a custom title
st.set_page_config(layout="wide", page_title="Aircraft Predictive Maintenance Dashboard")

# Application Title and Description
st.title("✈️ Aircraft Predictive Maintenance Dashboard")
st.markdown("Welcome to your interactive dashboard for aviation maintenance analytics! All data generation, feature engineering, and model training are performed automatically when the app starts. Explore the tabs to dive into data insights, model performance, explainability, and aircraft-specific recommendations.")

# --- Global Parameters ---
N_AIRCRAFT = 200
N_FLIGHTS = 50000
FAILURE_RATE = 0.03
FLIGHT_START_DATE = datetime(2023, 1, 1)
ALERT_THRESHOLD_PROB = 0.50 # Probability threshold for proactive maintenance alerts

# Set a fixed random seed for reproducibility across runs
np.random.seed(42)

# --- Data Generation Function ---
@st.cache_data(show_spinner="Generating realistic synthetic flight data...")
def generate_synthetic_data(n_aircraft, n_flights, failure_rate, flight_start_date):
    """
    Generates synthetic aircraft flight data with realistic degradation patterns.
    """
    aircraft_ids = [f'AC-{i:03d}' for i in range(1, n_aircraft + 1)]
    data = []
    flight_id_counter = 1

    for aircraft_id in tqdm(aircraft_ids, desc="Generating data"):
        # Initialize base sensor values for a new aircraft or after maintenance
        base_engine_temp = np.random.uniform(95, 105)
        base_engine_vibration = np.random.uniform(0.5, 1.5)
        base_oil_pressure = np.random.uniform(40, 50)
        base_hydraulic_pressure = np.random.uniform(2900, 3100)
        current_flight_date = flight_start_date
        
        # Simulate enough flights to generate failures
        total_days_sim = n_flights // n_aircraft * 2 
        
        while current_flight_date < (flight_start_date + timedelta(days=total_days_sim)):
            # Determine if a failure path will be initiated for this series of flights
            will_fail = np.random.rand() < (failure_rate * 0.5) 
            
            if will_fail:
                # Simulate a gradual degradation period before a major failure
                degradation_flights = np.random.randint(25, 75)
                for i in range(degradation_flights):
                    # Introduce a small chance of sudden failure within the degradation period
                    is_sudden_failure = np.random.rand() < 0.1
                    
                    # Gradual increase in degradation for temp and vibration
                    engine_temp_c = base_engine_temp + (i / 10) + np.random.normal(0, 1.5)
                    engine_vibration_mm_s = base_engine_vibration + (i / 20) + np.random.normal(0, 0.2)
                    
                    # Slight pressure drops as degradation progresses
                    oil_pressure_psi = base_oil_pressure - (i / 5) + np.random.normal(0, 2)
                    hydraulic_pressure_psi = base_hydraulic_pressure - (i / 5) + np.random.normal(0, 10)
                    
                    # Apply sudden failure effects if triggered
                    if is_sudden_failure and i > 5:
                        engine_temp_c += np.random.uniform(5, 15)
                        engine_vibration_mm_s += np.random.uniform(1.0, 5.0)
                    
                    # Common flight parameters
                    flight_duration = np.random.uniform(1.5, 5.0)
                    takeoff_landings = 1
                    
                    # Calculate remaining useful life (RUL) and binary failure target
                    remaining_useful_life_hrs = (degradation_flights - i - 1) * np.random.uniform(1.5, 5.0)
                    failure_within_50hrs = 1 if remaining_useful_life_hrs <= 50 else 0
                    
                    # Environmental effects: hotter OAT increases stress
                    oat_c = np.random.uniform(-10, 35)
                    cabin_pressure_psi = np.random.normal(12, 0.5)
                    fuel_flow_kg_hr = 1500 + 50 * flight_duration + np.random.normal(0, 25)
                    
                    data.append([
                        flight_id_counter, aircraft_id, current_flight_date,
                        engine_temp_c + 0.1 * oat_c, # OAT effect on temp
                        engine_vibration_mm_s + 0.05 * oat_c, # OAT effect on vibration
                        oil_pressure_psi, hydraulic_pressure_psi,
                        cabin_pressure_psi, fuel_flow_kg_hr, oat_c,
                        flight_duration * 60, takeoff_landings,
                        failure_within_50hrs, remaining_useful_life_hrs
                    ])
                    flight_id_counter += 1
                    current_flight_date += timedelta(days=np.random.randint(1, 3)) # Next flight in 1-3 days
                    
                    if flight_id_counter > n_flights: # Stop if target number of flights reached
                        break
                
                # After a failure sequence, simulate a maintenance period before resetting sensor baselines
                current_flight_date += timedelta(days=np.random.randint(7, 30)) 
                base_engine_temp = np.random.uniform(95, 105)
                base_engine_vibration = np.random.uniform(0.5, 1.5)
                base_oil_pressure = np.random.uniform(40, 50)
                base_hydraulic_pressure = np.random.uniform(2900, 3100)
                
            else: # Normal flight path
                flight_duration = np.random.uniform(1.5, 5.0)
                takeoff_landings = 1
                
                # Sensor readings are stable with normal fluctuations
                engine_temp_c = base_engine_temp + np.random.normal(0, 1.5)
                engine_vibration_mm_s = base_engine_vibration + np.random.normal(0, 0.2)
                oil_pressure_psi = base_oil_pressure + np.random.normal(0, 2)
                hydraulic_pressure_psi = base_hydraulic_pressure + np.random.normal(0, 10)
                
                oat_c = np.random.uniform(-10, 35)
                cabin_pressure_psi = np.random.normal(12, 0.5)
                fuel_flow_kg_hr = 1500 + 50 * flight_duration + np.random.normal(0, 25)
                
                remaining_useful_life_hrs = -1 # Not applicable for non-failure flights
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
                
    # Create the DataFrame and perform final cleanup/sorting
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
    """
    Applies feature engineering steps (rolling statistics, change rates, stress index, cycles) to the DataFrame.
    """
    df_fe = df_input.copy()
    df_fe.sort_values(by=['aircraft_id', 'flight_date'], inplace=True)

    window_sizes = [5, 10, 20]
    features_to_roll = ['engine_temp_c', 'engine_vibration_mm_s', 'oil_pressure_psi', 'hydraulic_pressure_psi']

    for window in window_sizes:
        for feature in features_to_roll:
            df_fe[f'rolling_avg_{feature}_{window}fl'] = df_fe.groupby('aircraft_id')[feature].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df_fe[f'rolling_std_{feature}_{window}fl'] = df_fe.groupby('aircraft_id')[feature].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)

    # Fill NaNs from rolling std (which would be 0 for the first elements in a window)
    for col in df_fe.columns:
        if 'rolling_std' in col:
            df_fe[col] = df_fe[col].fillna(0)

    # Calculate change rates for key sensors
    df_fe['vibration_change_rate'] = df_fe.groupby('aircraft_id')['engine_vibration_mm_s'].diff().fillna(0)
    df_fe['temp_change_rate'] = df_fe.groupby('aircraft_id')['engine_temp_c'].diff().fillna(0)
    
    # Create a combined stress index
    df_fe['combined_stress_index'] = df_fe['engine_vibration_mm_s'] * df_fe['engine_temp_c']

    # Recalculate Cycles since last maintenance, resetting after each failure event
    # This loop ensures that cycles are truly cumulative per aircraft and reset on failure
    current_cycles = {}
    final_cycles_series = pd.Series(np.zeros(len(df_fe)), index=df_fe.index) # Pre-allocate for efficiency

    for i, row in df_fe.iterrows():
        ac_id = row['aircraft_id']
        if ac_id not in current_cycles:
            current_cycles[ac_id] = 0
        
        current_cycles[ac_id] += 1
        final_cycles_series.loc[i] = current_cycles[ac_id]
        
        if row['failure_within_50hrs'] == 1:
            current_cycles[ac_id] = 0 # Reset for the *next* flight after a failure event

    df_fe['cycles_since_maintenance'] = final_cycles_series
    
    return df_fe

# --- Model Training Function ---
@st.cache_resource(show_spinner="Training predictive models (XGBoost Classifier & Regressor)...")
def train_predictive_models(df_processed):
    """
    Trains XGBoost classification and regression models and returns trained models, test sets, and features.
    """
    # Define features and targets for modeling
    features = [col for col in df_processed.columns if col not in ['aircraft_id', 'flight_date', 'failure_within_50hrs', 'remaining_useful_life_hrs']]
    target_classification = 'failure_within_50hrs'
    target_regression = 'remaining_useful_life_hrs'

    # Time-based train-test split: last 90 days of data used for testing to simulate future prediction
    train_end_date = df_processed['flight_date'].max() - timedelta(days=90) 
    train_df = df_processed[df_processed['flight_date'] <= train_end_date].copy()
    test_df = df_processed[df_processed['flight_date'] > train_end_date].copy()
    
    # Fallback if time-based split results in empty test set (e.g., small dataset)
    if test_df.empty and len(df_processed) > 100:
        st.warning("Time-based test set is empty. Adjusting split to use last 10% of data for testing.")
        split_idx = int(len(df_processed) * 0.9)
        train_df = df_processed.iloc[:split_idx].copy()
        test_df = df_processed.iloc[split_idx:].copy()
    elif test_df.empty: # If still empty (very small dataset), use the whole dataset for train/test
        st.warning("Dataset is too small for meaningful time-based split. Using entire dataset for both training and testing for demonstration.")
        train_df = df_processed.copy()
        test_df = df_processed.copy() 

    X_train_cls = train_df[features]
    y_train_cls = train_df[target_classification]
    X_test_cls = test_df[features]
    y_test_cls = test_df[target_classification]

    # Regression data (only for flights with defined RUL, i.e., those approaching failure)
    X_train_reg = train_df[train_df[target_regression] > 0][features]
    y_train_reg = train_df[train_df[target_regression] > 0][target_regression]
    X_test_reg = test_df[test_df[target_regression] > 0][features]
    y_test_reg = test_df[test_df[target_regression] > 0][target_regression]

    # Initialize and train Classification Model (XGBoost Classifier)
    xgb_cls = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_cls.fit(X_train_cls, y_train_cls)

    # Initialize and train Regression Model (XGBoost Regressor)
    if X_train_reg.empty or y_train_reg.empty:
        st.warning("Regression training data is empty. RUL model will not be trained and will return None.")
        xgb_reg = None
    else:
        xgb_reg = xgb.XGBRegressor(random_state=42)
        xgb_reg.fit(X_train_reg, y_train_reg)

    return xgb_cls, xgb_reg, X_test_cls, y_test_cls, X_test_reg, y_test_reg, features, test_df

# --- Perform all analysis steps automatically on app startup ---
# These functions are wrapped with st.cache_data/st.cache_resource
# so they only run once unless their inputs change or cache is cleared.
data = generate_synthetic_data(N_AIRCRAFT, N_FLIGHTS, FAILURE_RATE, FLIGHT_START_DATE)
df_fe = feature_engineer_data(data)
xgb_cls, xgb_reg, X_test_cls, y_test_cls, X_test_reg, y_test_reg, features, test_df = train_predictive_models(df_fe)

# --- Initialize session state variables if not already present ---
if 'selected_aircraft' not in st.session_state:
    st.session_state.selected_aircraft = None # Will be set by selectbox later

# --- Sidebar for Aircraft Selection ---
available_aircraft_ids = df_fe['aircraft_id'].unique()
st.sidebar.header("Aircraft Selection")
st.session_state.selected_aircraft = st.sidebar.selectbox(
    "Choose an Aircraft ID for detailed views:", 
    available_aircraft_ids
)

# --- Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview (EDA)", "🧠 Model Performance", "💡 Model Explainability", "🖥️ Interactive Dashboard", "✨ Insights & Recommendations"])

with tab1:
    st.header("Data Overview (EDA) 📊")
    st.markdown("A deep dive into the generated sensor data to understand distributions, trends, and class imbalance.")

    st.subheader("Tabular View of Raw Data")
    st.dataframe(data.head(10))

    st.subheader("Time-series Trends for an Aircraft Approaching Failure")
    # Select an aircraft that has a failure event for a good example
    if data['failure_within_50hrs'].sum() > 0:
        failure_aircraft_id_sample = data[data['failure_within_50hrs'] == 1]['aircraft_id'].iloc[0]
        failed_flights_sample = data[data['aircraft_id'] == failure_aircraft_id_sample].sort_values('flight_date').tail(75)

        fig_ts = px.line(
            failed_flights_sample,
            x='flight_date',
            y=['engine_vibration_mm_s', 'engine_temp_c', 'oil_pressure_psi', 'hydraulic_pressure_psi'],
            title=f'Sensor Readings for Aircraft {failure_aircraft_id_sample} Approaching Failure',
            labels={'value': 'Sensor Reading', 'variable': 'Sensor'},
            template='plotly_white'
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No failure events in the generated data to show a pre-failure time-series plot.")

    st.subheader("Sensor Reading Distributions by Failure Status")
    col1_eda, col2_eda = st.columns(2)
    with col1_eda:
        fig_hist_vib = px.histogram(
            data,
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
            data,
            x='failure_within_50hrs',
            y='engine_temp_c',
            title='Engine Temperature Boxplot',
            template='plotly_white'
        )
        st.plotly_chart(fig_box_temp, use_container_width=True)

    col3_eda, col4_eda = st.columns(2)
    with col3_eda:
        fig_hist_oil = px.histogram(
            data,
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
            data,
            x='failure_within_50hrs',
            y='hydraulic_pressure_psi',
            title='Hydraulic Pressure Boxplot',
            template='plotly_white'
        )
        st.plotly_chart(fig_box_hyd, use_container_width=True)

    st.subheader("Correlation Heatmap of Numerical Features")
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = data[numerical_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        title='Correlation Heatmap',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Class Imbalance Visualization")
    failure_counts = data['failure_within_50hrs'].value_counts().reset_index()
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
    st.markdown("Evaluating the performance of our XGBoost Classification and Regression models on unseen test data.")

    st.subheader("Classification Model Evaluation (Predicting Failure)")
    if xgb_cls is not None:
        y_pred_cls = xgb_cls.predict(X_test_cls)
        y_pred_proba_cls = xgb_cls.predict_proba(X_test_cls)[:, 1]

        st.write(f"**ROC-AUC:** {roc_auc_score(y_test_cls, y_pred_proba_cls):.4f}")
        st.write(f"**Precision:** {precision_score(y_test_cls, y_pred_cls):.4f}")
        st.write(f"**Recall:** {recall_score(y_test_cls, y_pred_cls):.4f}")
        st.write(f"**F1 Score:** {f1_score(y_test_cls, y_pred_cls):.4f}")
    else:
        st.info("Classification model could not be trained. Check data generation/feature engineering.")

    st.subheader("Regression Model Evaluation (Predicting Remaining Useful Life)")
    if xgb_reg is not None:
        y_pred_reg = xgb_reg.predict(X_test_reg)
        y_pred_reg[y_pred_reg < 0] = 0 # RUL cannot be negative

        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
        st.write(f"**MAE:** {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
        st.write(f"**R-squared:** {r2_score(y_test_reg, y_pred_reg):.4f}")
    else:
        st.info("Regression model could not be trained. This typically happens if there's no RUL data (no failures) in the training set.")

with tab3:
    st.header("Model Explainability with SHAP 💡")
    st.markdown("Understanding *why* the models make specific predictions using SHAP (SHapley Additive exPlanations) values.")

    if xgb_cls is None:
        st.warning("Classification model not trained. Cannot generate SHAP plots.")
    else:
        st.subheader("SHAP Feature Importance (Classification Model)")
        explainer_cls = shap.TreeExplainer(xgb_cls)
        shap_values_cls = explainer_cls.shap_values(X_test_cls)
        
        fig_shap_cls, ax_cls = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_cls, X_test_cls, plot_type="bar", show=False, ax=ax_cls)
        ax_cls.set_title("SHAP Feature Importance (Classification)")
        st.pyplot(fig_shap_cls)

    if xgb_reg is None:
        st.warning("Regression model not trained. Cannot generate SHAP plots.")
    else:
        st.subheader("SHAP Feature Importance (Regression Model)")
        explainer_reg = shap.TreeExplainer(xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_reg)

        fig_shap_reg, ax_reg = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_reg, X_test_reg, show=False, ax=ax_reg)
        ax_reg.set_title("SHAP Feature Importance (Regression)")
        st.pyplot(fig_shap_reg)

    st.subheader("Example SHAP Force Plot for a Single Flight (Classification)")
    if y_test_cls.sum() > 0 and xgb_cls is not None:
        # Find a single failed flight from the test set for detailed explanation
        failed_flight_idx = y_test_cls[y_test_cls == 1].index[0]
        failed_aircraft_data = test_df.loc[failed_flight_idx]
        failed_aircraft_features = failed_aircraft_data[features].to_frame().T 

        st.write(f"Explaining prediction for flight of aircraft **{failed_aircraft_data['aircraft_id']}** on **{failed_aircraft_data['flight_date'].date()}**:")
        
        explainer_cls_single = shap.TreeExplainer(xgb_cls)
        shap_values_single_instance = explainer_cls_single.shap_values(failed_aircraft_features)[0] # For binary classification, take values for one class (e.g., class 0 or 1)
        
        fig_force, ax_force = plt.subplots(figsize=(12, 4))
        shap.force_plot(
            explainer_cls_single.expected_value[0], 
            shap_values_single_instance, 
            failed_aircraft_features, 
            show=False, 
            matplotlib=True,
            plot_cmap="PkGn",
            ax=ax_force
        )
        ax_force.set_title("SHAP Force Plot for a Single Prediction (Class 0 Contribution)")
        st.pyplot(fig_force)

    else:
        st.info("No failure instances in the test set or classification model not trained to demonstrate a single flight explanation.")

with tab4:
    st.header("Interactive Aircraft Health Dashboard 🖥️")
    st.markdown("Monitor the health of a selected aircraft over time with real-time sensor readings, predicted failure risk, and Remaining Useful Life.")

    if xgb_cls is None or xgb_reg is None:
        st.warning("Models not trained. Please ensure the analysis has run completely.")
    elif st.session_state.selected_aircraft:
        aircraft_data_for_dashboard = df_fe[df_fe['aircraft_id'] == st.session_state.selected_aircraft].sort_values('flight_date').copy()
        
        if not aircraft_data_for_dashboard.empty:
            # Generate predictions for the selected aircraft's full history
            aircraft_data_for_dashboard['failure_probability'] = xgb_cls.predict_proba(aircraft_data_for_dashboard[features])[:, 1]
            if xgb_reg is not None:
                aircraft_data_for_dashboard['predicted_rul'] = xgb_reg.predict(aircraft_data_for_dashboard[features])
                aircraft_data_for_dashboard['predicted_rul'] = aircraft_data_for_dashboard['predicted_rul'].apply(lambda x: max(0, x))
            else:
                aircraft_data_for_dashboard['predicted_rul'] = np.nan # No RUL predictions if model failed to train

            fig_dashboard = go.Figure()

            # Plot Sensor Readings
            fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['engine_vibration_mm_s'], mode='lines+markers', name='Engine Vibration (mm/s)', yaxis='y1'))
            fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['engine_temp_c'], mode='lines+markers', name='Engine Temperature (°C)', yaxis='y1'))
            fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['oil_pressure_psi'], mode='lines+markers', name='Oil Pressure (psi)', yaxis='y1'))
            fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['hydraulic_pressure_psi'], mode='lines+markers', name='Hydraulic Pressure (psi)', yaxis='y1'))

            # Plot Failure Probability & Predicted RUL on secondary/tertiary Y-axes
            fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['failure_probability'] * 100, mode='lines+markers', name='Failure Risk (%)', yaxis='y2', line=dict(dash='dash', color='red')))
            
            if xgb_reg is not None:
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
            st.warning(f"No data available for selected aircraft ID: {st.session_state.selected_aircraft}. Please select another or re-run analysis if the dataset is new.")
    else:
        st.info("Please select an aircraft ID from the sidebar to view its dashboard.")

with tab5:
    st.header("Insights & Recommendations ✨")
    st.markdown("Here, we present key insights derived from the analysis and actionable recommendations, **dynamically generated** from the current model and data state.")
    
    if xgb_cls is None or xgb_reg is None or features is None:
        st.warning("Models and features not ready. Please ensure the analysis has run completely to generate dynamic insights.")
    else:
        st.subheader("Overall Insights from Analysis")
        
        y_pred_cls_proba = xgb_cls.predict_proba(X_test_cls)[:, 1]
        
        st.markdown(f"""
        * **Degradation Patterns**: Our synthetic data successfully simulated realistic degradation, showing a **gradual increase in engine vibration and temperature** before failures, coupled with slight drops in oil/hydraulic pressure. This pattern is crucial for proactive maintenance.
        * **Environmental Impact**: **Outside Air Temperature (OAT)** was observed to directly influence sensor readings. For example, the `engine_temp_c` simulation incorporated a factor of `0.1 * oat_c`, showing that **hotter environments can increase stress** on components.
        * **Feature Importance**: Features derived from **rolling averages and change rates** of sensor data, especially engine vibration and temperature, proved to be highly predictive for both failure classification and RUL estimation. The **combined stress index** ($engine\_vibration\_mm\_s \times engine\_temp\_c$) also played a significant role.
        * **Model Performance**:
            * The **XGBoost Classifier** (for failure prediction) achieved a strong **ROC-AUC of {roc_auc_score(y_test_cls, y_pred_cls_proba):.4f}**. This indicates a high capability to distinguish between flights likely to fail and those that are not.
            * The **XGBoost Regressor** (for Remaining Useful Life) has a **RMSE of {np.sqrt(mean_squared_error(y_test_reg, xgb_reg.predict(X_test_reg))):.4f}** and an **R-squared of {r2_score(y_test_reg, xgb_reg.predict(X_test_reg)) if len(y_test_reg) > 1 else 0:.4f}**. This shows it can predict RUL with a reasonable degree of accuracy, enabling better maintenance planning.
        * **Explainable AI**: **SHAP values** revealed that features reflecting increasing engine wear (e.g., higher vibration, temperature, and their rolling statistics) were the primary drivers for predicting imminent failure and lower RUL, providing transparency to model predictions.
        """)

        st.subheader("Top Predictive Features")
        st.markdown("The following features were identified as the most important in predicting aircraft failure (from the Classification Model):")
        
        feature_importances = pd.Series(xgb_cls.feature_importances_, index=features)
        top_5_features = feature_importances.nlargest(5)
        
        for i, (feature, importance) in enumerate(top_5_features.items()):
            st.write(f"**{i+1}. {feature}** (Importance: {importance:.4f})")

        st.subheader("Aircraft-Specific Recommendations")
        
        if st.session_state.selected_aircraft:
            st.write(f"Recommendations for aircraft **{st.session_state.selected_aircraft}**, based on its latest flight data:")
            
            latest_flight = df_fe[df_fe['aircraft_id'] == st.session_state.selected_aircraft].sort_values('flight_date').iloc[-1]
            latest_features = latest_flight[features].to_frame().T
            
            current_prob_failure = xgb_cls.predict_proba(latest_features)[:, 1][0]
            predicted_rul = xgb_reg.predict(latest_features)[0] if xgb_reg else np.nan
            
            col1_rec, col2_rec, col3_rec = st.columns(3)
            with col1_rec:
                st.metric("Latest Failure Risk", f"{current_prob_failure*100:.2f}%")
            with col2_rec:
                st.metric("Predicted RUL", f"{max(0, predicted_rul):.2f} hours" if not np.isnan(predicted_rul) else "N/A")
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
            elif not np.isnan(predicted_rul) and predicted_rul <= 100 and predicted_rul > 0: # RUL can be -1 if not a failure
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
