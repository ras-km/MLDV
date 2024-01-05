import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def get_user_input():
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
    blood_pressure_problems = st.sidebar.radio('Blood Pressure Problems', ['No', 'Yes'])
    any_transplants = st.sidebar.radio('Any Transplants', ['No', 'Yes'])
    any_chronic_diseases = st.sidebar.radio('Any Chronic Diseases', ['No', 'Yes'])
    known_allergies = st.sidebar.radio('Known Allergies', ['No', 'Yes'])
    history_of_cancer_in_family = st.sidebar.radio('History of Cancer in Family', ['No', 'Yes'])
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, 25.0)
    age_group = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'])
    major_surgeries = st.sidebar.selectbox('Number of Major Surgeries', [0, 1, 2, 3])

    # Encode categorical features manually
    diabetes_encoded = 1 if diabetes == 'Yes' else 0
    blood_pressure_problems_encoded = 1 if blood_pressure_problems == 'Yes' else 0
    any_transplants_encoded = 1 if any_transplants == 'Yes' else 0
    any_chronic_diseases_encoded = 1 if any_chronic_diseases == 'Yes' else 0
    known_allergies_encoded = 1 if known_allergies == 'Yes' else 0
    history_of_cancer_in_family_encoded = 1 if history_of_cancer_in_family == 'Yes' else 0

    # Create a DataFrame with the processed features
    user_features = pd.DataFrame({
        'Diabetes': diabetes_encoded,
        'Blood Pressure Problems': blood_pressure_problems_encoded,
        'Any Transplants': any_transplants_encoded,
        'Any Chronic Diseases': any_chronic_diseases_encoded,
        'Known Allergies': known_allergies_encoded,
        'History Of Cancer In Family': history_of_cancer_in_family_encoded,
        'BMI': bmi,
        'Age Group_31-40': 1 if age_group == '31-40' else 0,
        'Age Group_41-50': 1 if age_group == '41-50' else 0,
        'Age Group_51-60': 1 if age_group == '51-60' else 0,
        'Age Group_61-70': 1 if age_group == '61-70' else 0,
        'Major Surgeries_1': 1 if major_surgeries == 1 else 0,
        'Major Surgeries_2': 1 if major_surgeries == 2 else 0,
        'Major Surgeries_3': 1 if major_surgeries == 3 else 0,
    }, index=[0])

    return user_features

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Get user input
user_features = get_user_input()

# Preprocess input features (e.g., scale them)
try:
    # Manually encode categorical features
    user_features['Age Group_18-30'] = 0  # Add a dummy column for Age Group_18-30
    user_features['Major Surgeries_0'] = 0  # Add a dummy column for Major Surgeries_0

    # Prediction
    predicted_price = model.predict(scaler.transform(user_features.values))

    # Display result
    st.subheader('Prediction')
    st.write("The premium is estimated to be ${:,.2f}".format(predicted_price[0]))
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.write("User Features:")
    st.write(user_features)
