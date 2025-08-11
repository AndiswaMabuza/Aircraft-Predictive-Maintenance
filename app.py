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
st.set_page_config(layout="wide", page_title="Aircraft Predictive Maintenance")

st.title("✈️ Aircraft Predictive Maintenance Analytics")
st.markdown("This application demonstrates a comprehensive predictive maintenance solution for aircraft, covering synthetic data generation, EDA, feature engineering, model training, explainability, and interactive dashboards.")

# Set a random seed for reproducibility
np.random.seed(42)

# --- Parameters for Synthetic Data ---
N_AIRCRAFT = 200
N_FLIGHTS = 50000
FAILURE_RATE = 0.03
FLIGHT_START_DATE = datetime(2023, 1, 1)
ALERT_THRESHOLD_PROB = 0.50

@st.cache_data
def generate_synthetic_data(n_aircraft, n_flights, failure_rate, flight_start_date):
    """
    Generates synthetic aircraft flight data with degradation patterns.
    """
    st.info("Generating synthetic data... This might take a moment.")
    aircraft_ids = [f'AC-{i:03d}' for i in range(1, n_aircraft + 1)]

    data = []
    flight_id_counter = 1

    for aircraft_id in tqdm(aircraft_ids, desc="Generating data"):
        
        base_engine_temp = np.random.uniform(95, 105)
        base_engine_vibration = np.random.uniform(0.5, 1.5)
        base_oil_pressure = np.random.uniform(40, 50)
        base_hydraulic_pressure = np.random.uniform(2900, 3100)
        
        current_flight_date = flight_start_date
        
        while current_flight_date < (flight_start_date + timedelta(days=n_flights // n_aircraft * 2)):
            
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
                
                # After failure, reset to maintenance
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
    st.success("Synthetic data generation complete!")
    return df

@st.cache_data
def feature_engineer_data(df_input):
    """
    Applies feature engineering steps to the DataFrame.
    """
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

    # Cycles since last maintenance - re-calculate based on failure events
    df_fe['cycles_since_maintenance'] = df_fe.groupby('aircraft_id').cumcount() + 1
    
    # Identify actual failure points for reset
    failure_indices = df_fe[df_fe['failure_within_50hrs'] == 1].index
    
    for idx in failure_indices:
        ac_id = df_fe.loc[idx, 'aircraft_id']
        # Set cycles_since_maintenance to 1 for the failure event itself and subsequent flights
        # For simplicity, we'll reset for the current aircraft after this flight.
        # A more robust approach might involve finding the *next* maintenance based on date.
        # Here, we simulate a 'reset' of cycles immediately after a failure event for that aircraft's future flights.
        df_fe.loc[(df_fe['aircraft_id'] == ac_id) & (df_fe.index > idx), 'cycles_since_maintenance'] = (
            df_fe.loc[(df_fe['aircraft_id'] == ac_id) & (df_fe.index > idx)].groupby('aircraft_id').cumcount() + 1
        )
    
    # Ensure all cycles are correctly assigned within each aircraft group
    # This loop ensures that cycles are truly cumulative per aircraft and reset on failure
    current_cycles = {}
    for i, row in df_fe.iterrows():
        ac_id = row['aircraft_id']
        if ac_id not in current_cycles:
            current_cycles[ac_id] = 0
        
        current_cycles[ac_id] += 1
        df_fe.loc[i, 'cycles_since_maintenance'] = current_cycles[ac_id]
        
        if row['failure_within_50hrs'] == 1:
            current_cycles[ac_id] = 0 # Reset for the *next* flight after failure

    return df_fe

@st.cache_resource
def train_models(train_df, test_df, features, target_classification, target_regression):
    """
    Trains XGBoost classification and regression models.
    """
    st.info("Training models... This may take a while.")
    
    # Classification data
    X_train_cls = train_df[features]
    y_train_cls = train_df[target_classification]
    X_test_cls = test_df[features]
    y_test_cls = test_df[target_classification]

    # Regression data (only for flights with defined RUL)
    X_train_reg = train_df[train_df[target_regression] > 0][features]
    y_train_reg = train_df[train_df[target_regression] > 0][target_regression]
    X_test_reg = test_df[test_df[target_regression] > 0][features]
    y_test_reg = test_df[test_df[target_regression] > 0][target_regression]

    # Classification Model
    xgb_cls = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_cls.fit(X_train_cls, y_train_cls)

    # Regression Model
    xgb_reg = xgb.XGBRegressor(random_state=42)
    xgb_reg.fit(X_train_reg, y_train_reg)

    st.success("Models trained successfully!")
    return xgb_cls, xgb_reg, X_test_cls, y_test_cls, X_test_reg, y_test_reg


# --- Main Application Flow ---

if 'data' not in st.session_state:
    st.session_state.data = None
if 'df_fe' not in st.session_state:
    st.session_state.df_fe = None
if 'xgb_cls' not in st.session_state:
    st.session_state.xgb_cls = None
if 'xgb_reg' not in st.session_state:
    st.session_state.xgb_reg = None
if 'X_test_cls' not in st.session_state:
    st.session_state.X_test_cls = None
if 'y_test_cls' not in st.session_state:
    st.session_state.y_test_cls = None
if 'X_test_reg' not in st.session_state:
    st.session_state.X_test_reg = None
if 'y_test_reg' not in st.session_state:
    st.session_state.y_test_reg = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'selected_aircraft' not in st.session_state:
    st.session_state.selected_aircraft = None


# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["1. Data Generation", "2. EDA", "3. Feature Engineering", "4. Modeling", "5. Explainability", "6. Dashboard", "7. Insights & Recommendations"])


if page == "1. Data Generation":
    st.header("1. Synthetic Data Generation ⚙️")
    st.markdown("We'll generate a synthetic dataset mimicking aircraft sensor readings with realistic degradation patterns. This ensures the project is self-contained and reproducible.")
    
    if st.button("Generate Data"):
        st.session_state.data = generate_synthetic_data(N_AIRCRAFT, N_FLIGHTS, FAILURE_RATE, FLIGHT_START_DATE)
        st.write("First 5 rows of the generated data:")
        st.dataframe(st.session_state.data.head())
        st.write(f"Total flights: {len(st.session_state.data)}")
        st.write(f"Failure rate: {st.session_state.data['failure_within_50hrs'].mean() * 100:.2f}%")
        st.write("Data Info:")
        st.text(st.session_state.data.info())

elif page == "2. EDA":
    st.header("2. Exploratory Data Analysis (EDA) 📊")
    st.markdown("Exploring the generated data to understand sensor trends, distributions, and relationships between features and failure targets.")

    if st.session_state.data is None:
        st.warning("Please generate data first from the 'Data Generation' section.")
    else:
        st.subheader("Tabular View of Data")
        st.dataframe(st.session_state.data.head(10))

        st.subheader("Time-series Trends for Aircraft Approaching Failure")
        # Ensure there's at least one failure for plotting
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
        col1, col2 = st.columns(2)
        with col1:
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
        with col2:
            fig_box_temp = px.box(
                st.session_state.data,
                x='failure_within_50hrs',
                y='engine_temp_c',
                title='Engine Temperature Boxplot',
                template='plotly_white'
            )
            st.plotly_chart(fig_box_temp, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
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
        with col4:
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

elif page == "3. Feature Engineering":
    st.header("3. Feature Engineering 🧩")
    st.markdown("Creating new features that capture temporal and relational aspects of sensor data to enhance predictive power.")

    if st.session_state.data is None:
        st.warning("Please generate data first from the 'Data Generation' section.")
    else:
        if st.button("Perform Feature Engineering"):
            st.session_state.df_fe = feature_engineer_data(st.session_state.data)
            st.write("Engineered features added. Preview of new features:")
            st.dataframe(st.session_state.df_fe[['aircraft_id', 'flight_date', 'engine_vibration_mm_s', 'rolling_avg_engine_vibration_mm_s_5fl', 'vibration_change_rate', 'combined_stress_index', 'cycles_since_maintenance']].head())
            
            # Define features after feature engineering
            st.session_state.features = [col for col in st.session_state.df_fe.columns if col not in ['aircraft_id', 'flight_date', 'failure_within_50hrs', 'remaining_useful_life_hrs']]
            st.success("Feature engineering complete!")

elif page == "4. Modeling":
    st.header("4. Predictive Modeling 🧠")
    st.markdown("Building XGBoost models for binary classification (failure prediction) and regression (RUL prediction). Data is split by time to simulate real-world deployment.")

    if st.session_state.df_fe is None:
        st.warning("Please perform feature engineering first from the 'Feature Engineering' section.")
    else:
        if st.button("Train Models"):
            target_classification = 'failure_within_50hrs'
            target_regression = 'remaining_useful_life_hrs'
            
            # Time-based train-test split
            train_end_date = st.session_state.df_fe['flight_date'].max() - timedelta(days=90)
            train_df = st.session_state.df_fe[st.session_state.df_fe['flight_date'] <= train_end_date].copy()
            test_df = st.session_state.df_fe[st.session_state.df_fe['flight_date'] > train_end_date].copy()
            
            xgb_cls, xgb_reg, X_test_cls, y_test_cls, X_test_reg, y_test_reg = train_models(
                train_df, test_df, st.session_state.features, target_classification, target_regression
            )
            st.session_state.xgb_cls = xgb_cls
            st.session_state.xgb_reg = xgb_reg
            st.session_state.X_test_cls = X_test_cls
            st.session_state.y_test_cls = y_test_cls
            st.session_state.X_test_reg = X_test_reg
            st.session_state.y_test_reg = y_test_reg

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

elif page == "5. Explainability":
    st.header("5. Model Explainability with SHAP 💡")
    st.markdown("Understanding *why* a model makes a specific prediction is crucial. SHAP values help interpret the predictions.")

    if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None:
        st.warning("Please train models first from the 'Modeling' section.")
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
        # Find a flight that actually failed in the test set for a good example
        if st.session_state.y_test_cls.sum() > 0:
            failed_flight_idx = st.session_state.y_test_cls[st.session_state.y_test_cls == 1].index[0]
            failed_aircraft_data = st.session_state.df_fe.loc[failed_flight_idx]
            failed_aircraft_features = failed_aircraft_data[st.session_state.features].to_frame().T # SHAP expects 2D array

            st.write(f"Explaining prediction for flight of aircraft **{failed_aircraft_data['aircraft_id']}** on **{failed_aircraft_data['flight_date'].date()}**")
            
            # Re-calculate SHAP values for the specific instance
            shap_values_single_instance = explainer_cls.shap_values(failed_aircraft_features)[0] # For binary, [0] or [1] for class
            
            # Using force plot with Matplotlib backend for Streamlit compatibility
            fig_force, ax_force = plt.subplots(figsize=(12, 4))
            shap.force_plot(
                explainer_cls.expected_value[0], # Expected value for class 0 or 1, depending on problem
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


elif page == "6. Dashboard":
    st.header("6. Interactive Visualization Dashboard 🖥️")
    st.markdown("An interactive dashboard to monitor an aircraft's health, showing key sensor readings, failure risk, and RUL predictions over time.")

    if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None or st.session_state.df_fe is None:
        st.warning("Please train models and perform feature engineering first.")
    else:
        # Get aircraft IDs for selection
        available_aircraft_ids = st.session_state.df_fe['aircraft_id'].unique()
        selected_aircraft_id = st.selectbox("Select an Aircraft ID:", available_aircraft_ids)
        st.session_state.selected_aircraft = selected_aircraft_id

        if selected_aircraft_id:
            # Filter data for the selected aircraft
            aircraft_data_for_dashboard = st.session_state.df_fe[st.session_state.df_fe['aircraft_id'] == selected_aircraft_id].sort_values('flight_date').copy()
            
            if not aircraft_data_for_dashboard.empty:
                # Add predictions to the DataFrame for the selected aircraft
                aircraft_data_for_dashboard['failure_probability'] = st.session_state.xgb_cls.predict_proba(aircraft_data_for_dashboard[st.session_state.features])[:, 1]
                aircraft_data_for_dashboard['predicted_rul'] = st.session_state.xgb_reg.predict(aircraft_data_for_dashboard[st.session_state.features])
                aircraft_data_for_dashboard['predicted_rul'] = aircraft_data_for_dashboard['predicted_rul'].apply(lambda x: max(0, x))

                fig_dashboard = go.Figure()

                # Plot 1: Sensor Readings
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['engine_vibration_mm_s'], mode='lines+markers', name='Engine Vibration (mm/s)', yaxis='y1'))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['engine_temp_c'], mode='lines+markers', name='Engine Temperature (°C)', yaxis='y1'))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['oil_pressure_psi'], mode='lines+markers', name='Oil Pressure (psi)', yaxis='y1'))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['hydraulic_pressure_psi'], mode='lines+markers', name='Hydraulic Pressure (psi)', yaxis='y1'))


                # Plot 2: Failure Probability & RUL
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['failure_probability'] * 100, mode='lines+markers', name='Failure Risk (%)', yaxis='y2', line=dict(dash='dash', color='red')))
                fig_dashboard.add_trace(go.Scatter(x=aircraft_data_for_dashboard['flight_date'], y=aircraft_data_for_dashboard['predicted_rul'], mode='lines+markers', name='Predicted RUL (hrs)', yaxis='y3', line=dict(dash='dot', color='green')))

                fig_dashboard.update_layout(
                    title=f"Predictive Maintenance Dashboard for Aircraft {selected_aircraft_id}",
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
                        position=0.88, # Adjust position to avoid overlap
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
                st.warning(f"No data available for aircraft ID: {selected_aircraft_id}")

elif page == "7. Insights & Recommendations":
    st.header("7. Insights & Recommendations ✨")
    st.markdown("Here, we present key insights derived from the analysis and actionable recommendations, **dynamically generated** from the current model and data state.")
    
    if st.session_state.xgb_cls is None or st.session_state.xgb_reg is None or st.session_state.features is None:
        st.warning("Please run through the 'Modeling' and 'Feature Engineering' sections first to generate dynamic insights.")
    else:
        st.subheader("Model Performance Summary")
        
        # Calculate performance metrics dynamically from the test set
        y_pred_cls_proba = st.session_state.xgb_cls.predict_proba(st.session_state.X_test_cls)[:, 1]
        y_pred_cls = (y_pred_cls_proba > ALERT_THRESHOLD_PROB).astype(int)
        
        st.markdown(f"""
        Our predictive models, trained on a time-based split of the data, show strong performance:
        * **Classification Model (Failure Prediction)**: Achieved a **ROC-AUC of {roc_auc_score(st.session_state.y_test_cls, y_pred_cls_proba):.4f}**. This indicates a high capability to distinguish between flights likely to fail and those that are not.
        * **Regression Model (RUL Prediction)**: The model for Remaining Useful Life has a **RMSE of {np.sqrt(mean_squared_error(st.session_state.y_test_reg, st.session_state.xgb_reg.predict(st.session_state.X_test_reg))):.4f}** and an **R-squared of {r2_score(st.session_state.y_test_reg, st.session_state.xgb_reg.predict(st.session_state.X_test_reg)):.4f}**. This shows it can predict RUL with a reasonable degree of accuracy.
        """)

        st.subheader("Top Predictive Features")
        st.markdown("The following features were identified as the most important in predicting aircraft failure and RUL:")
        
        # Get feature importance dynamically from the models
        feature_importances = pd.Series(st.session_state.xgb_cls.feature_importances_, index=st.session_state.features)
        top_5_features = feature_importances.nlargest(5)
        
        for i, (feature, importance) in enumerate(top_5_features.items()):
            st.write(f"**{i+1}. {feature}** (Importance: {importance:.4f})")

        st.subheader("Aircraft-Specific Recommendations")
        
        if st.session_state.selected_aircraft:
            st.write(f"Recommendations for aircraft **{st.session_state.selected_aircraft}**, based on its latest flight data:")
            
            # Get the latest data for the selected aircraft
            latest_flight = st.session_state.df_fe[st.session_state.df_fe['aircraft_id'] == st.session_state.selected_aircraft].sort_values('flight_date').iloc[-1]
            latest_features = latest_flight[st.session_state.features].to_frame().T
            
            # Make dynamic predictions for the latest flight
            current_prob_failure = st.session_state.xgb_cls.predict_proba(latest_features)[:, 1][0]
            predicted_rul = st.session_state.xgb_reg.predict(latest_features)[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest Failure Risk", f"{current_prob_failure*100:.2f}%")
            with col2:
                st.metric("Predicted RUL", f"{max(0, predicted_rul):.2f} hours")
            with col3:
                st.metric("Cycles since Maint.", int(latest_flight['cycles_since_maintenance']))

            st.markdown("---")

            st.write(f"**Based on these metrics, here are the recommendations for Aircraft {st.session_state.selected_aircraft}:**")
            
            if current_prob_failure >= ALERT_THRESHOLD_PROB:
                st.error(f"**⚠️ IMMEDIATE ACTION REQUIRED:** The failure risk is critically high at **{current_prob_failure*100:.2f}%**.")
                st.markdown(f"""
                - **Schedule immediate inspection** for critical components.
                - Review the dashboard to see which sensors (e.g., high vibration, high temp) are trending abnormally.
                - Consider grounding the aircraft until a full maintenance check is performed.
                """)
            elif predicted_rul <= 100:
                st.warning(f"**⚠️ PROACTIVE MAINTENANCE RECOMMENDED:** The predicted Remaining Useful Life (RUL) is low (**{predicted_rul:.2f} hours**).")
                st.markdown("""
                - **Schedule maintenance** within the next few flights.
                - Do not wait for a full failure to occur. A proactive intervention can prevent a costly in-flight incident and minimize downtime.
                - Monitor this aircraft closely on the dashboard.
                """)
            else:
                st.success("✅ **STATUS NORMAL:** The aircraft is operating within normal parameters.")
                st.markdown("""
                - Continue with routine monitoring as planned.
                - No immediate action is required based on the current data.
                - The predicted RUL is at a healthy level.
                """)
        else:
            st.info("Please select an aircraft from the 'Dashboard' page to view specific insights and recommendations.")
