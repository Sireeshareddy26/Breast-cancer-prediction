import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model components
# Ensure these files are in the same directory as your app.py when deploying
model = joblib.load('adaboost_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('feature_names.joblib') # Load feature names to ensure correct order

st.set_page_config(page_title="Breast Cancer Prediction App", layout="centered")

st.title("🩺 Breast Cancer Prediction")
st.write("Enter the patient's details to predict the likelihood of breast cancer.")

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    # Initialize with all feature names + 'Predicted Status' + 'Probability of Cancer'
    st.session_state.prediction_history = pd.DataFrame(columns=feature_names + ['Predicted Status', 'Probability of Cancer'])

# --- Input Form ---
with st.form("prediction_form"):
    st.header("Patient Information")

    # Group features for better UI organization
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics & Core Metrics")
        # Check if 'serian no' is in feature_names and exclude it from user input
        if 'serian no' not in feature_names:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
        else:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, format="%.2f")
        glucose = st.number_input("Glucose", min_value=50.0, max_value=300.0, value=100.0, format="%.2f")
        insulin = st.number_input("Insulin", min_value=0.5, max_value=100.0, value=10.0, format="%.2f")
        homa = st.number_input("HOMA", min_value=0.1, max_value=30.0, value=2.0, format="%.2f")
        leptin = st.number_input("Leptin", min_value=1.0, max_value=100.0, value=20.0, format="%.2f")

    with col2:
        st.subheader("Adipokines & Inflammation")
        adiponectin = st.number_input("Adiponectin", min_value=1.0, max_value=50.0, value=15.0, format="%.2f")
        resistin = st.number_input("Resistin", min_value=1.0, max_value=60.0, value=10.0, format="%.2f")
        mcp1 = st.number_input("MCP.1", min_value=10.0, max_value=2000.0, value=300.0, format="%.2f")

    with col3:
        st.subheader("Tumor & Status Details")
        tumor_size_cm = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=50.0, value=2.0, format="%.2f")
        node_invasion = st.number_input("Node Invasion (0-3)", min_value=0, max_value=3, value=0)
        estrogen_status = st.number_input("Estrogen Status (0/1)", min_value=0, max_value=1, value=0)
        prg_status = st.number_input("PRG Status (0/1)", min_value=0, max_value=1, value=0)
        her2_status = st.number_input("HER2 Status (0/1)", min_value=0, max_value=1, value=0)
        histology = st.number_input("Histology (e.g., 1, 2, 3)", min_value=1, max_value=3, value=1)
        malignant_status = st.number_input("Malignant Status (0/1)", min_value=0, max_value=1, value=0)


    submit_button = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submit_button:
    # Create input DataFrame
    input_data_dict = {
        'age': age, 'bmi': bmi, 'glucose': glucose, 'insulin': insulin, 'homa': homa,
        'leptin': leptin, 'adiponectin': adiponectin, 'resistin': resistin, 'mcp.1': mcp1,
        'tumor size (cm)': tumor_size_cm, 'node invasion': node_invasion,
        'estrogen status': estrogen_status, 'prg status': prg_status,
        'her2 status': her2_status, 'histology': histology,
        'malignant status': malignant_status
    }

    # Add 'serian no' if it's part of the features but not in input, fill with 0 or a default value
    if 'serian no' in feature_names and 'serian no' not in input_data_dict:
        input_data_dict['serian no'] = 0  # Default value, adjust if needed

    input_df = pd.DataFrame([input_data_dict])

    # Add interaction term if it was part of the original training features
    if 'tumor_node_interaction' in feature_names:
        input_df['tumor_node_interaction'] = input_df['tumor size (cm)'] * input_df['node invasion']

    # Ensure the order of columns matches the training data by reindexing
    # This is crucial for correct predictions. Any feature in 'feature_names' not provided by user will be filled with 0.
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction_proba = model.predict_proba(scaled_input)[:, 1][0]
    prediction = model.predict(scaled_input)[0]

    predicted_status = label_encoder.inverse_transform([prediction])[0]

    st.subheader("Prediction Results:")
    # Assuming '1' means 'Present' for cancer based on previous LabelEncoder behavior
    if predicted_status == label_encoder.classes_[1]: 
        st.error(f"**Breast Cancer Status: {predicted_status}**")
        st.markdown(f"Probability of Cancer: **{prediction_proba:.2%}**")
    else:
        st.success(f"**Breast Cancer Status: {predicted_status}**")
        st.markdown(f"Probability of Cancer: **{prediction_proba:.2%}**")

    # Store prediction in history
    new_entry_dict = input_df.iloc[0].to_dict()
    new_entry_dict['Predicted Status'] = predicted_status
    new_entry_dict['Probability of Cancer'] = f"{prediction_proba:.2%}"
    
    # Create a DataFrame for the new entry to ensure correct concatenation
    new_entry_df = pd.DataFrame([new_entry_dict])
    
    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, new_entry_df],
        ignore_index=True
    )

st.subheader("Prediction History")
if not st.session_state.prediction_history.empty:
    st.dataframe(st.session_state.prediction_history)

    # Download button for history
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(st.session_state.prediction_history)
    st.download_button(
        label="Download Prediction History as CSV",
        data=csv,
        file_name="breast_cancer_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("No predictions made yet.")
