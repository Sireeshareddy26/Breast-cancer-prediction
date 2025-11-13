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
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        
        # Use radio buttons for binary 'Yes'/'No' inputs and map to 0/1
        menopause_status = st.radio("Menopause", ['No', 'Yes'], index=0, help="Is the patient post-menopausal?")
        menopause = 0 if menopause_status == 'Yes' else 1

    with col2:
        tumor_size_cm = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=50.0, value=2.0, format="%.2f")
        node_invasion = st.radio("invasion", ['No', 'Yes'], index=0, help="Is the cancer spread to lymph nodes ?")
        invasion = 1 if node_invasion == 'Yes' else 0

    with col3:
        metastasis_status = st.radio("Metastasis", ['No', 'Yes'], index=0, help="Is there evidence of metastasis?")
        metastasis = 1 if metastasis_status == 'Yes' else 0
        
        history_status = st.radio("History of Breast Cancer", ['No', 'Yes'], index=0, help="Does the patient have a history of breast cancer?")
        history = 1 if history_status == 'Yes' else 0

    submit_button = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submit_button:
    # Create a dictionary to hold user inputs
    input_data_dict = {
        'age': age,
        'menopause': menopause,
        'tumor size (cm)': tumor_size_cm,
        'node invasion': node_invasion,
        'metastasis': metastasis,
        'history': history
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data_dict])

    # Add 'serian no' if it's part of the features but not in input, fill with 0 or a default value
    # This assumes 'serial no' is not a user input feature and can be default-filled.
    if 'serian no' in feature_names:
        input_df['serian no'] = 0  # Default value, adjust if needed

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
