import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import pickle
import joblib

st.write("""
# Premium Prediction App
This app predicts the premium based on user input.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
    blood_pressure_problems = st.sidebar.radio('Blood Pressure Problems', ['No', 'Yes'])
    any_transplants = st.sidebar.radio('Any Transplants', ['No', 'Yes'])
    any_chronic_diseases = st.sidebar.radio('Any Chronic Diseases', ['No', 'Yes'])
    known_allergies = st.sidebar.radio('Known Allergies', ['No', 'Yes'])
    history_of_cancer_in_family = st.sidebar.radio('History of Cancer in Family', ['No', 'Yes'])
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, 25.0)
    age_group = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'])
    major_surgeries = st.sidebar.selectbox('Number of Major Surgeries', [0, 1, 2, 3])

    data = {
        'Diabetes': 1 if diabetes == 'Yes' else 0,
        'Blood Pressure Problems': 1 if blood_pressure_problems == 'Yes' else 0,
        'Any Transplants': 1 if any_transplants == 'Yes' else 0,
        'Any Chronic Diseases': 1 if any_chronic_diseases == 'Yes' else 0,
        'Known Allergies': 1 if known_allergies == 'Yes' else 0,
        'History Of Cancer In Family': 1 if history_of_cancer_in_family == 'Yes' else 0,
        'BMI': bmi,
        'Age Group': age_group,
        'Major Surgeries': major_surgeries
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_features = user_input_features()

# Load the model and scaler
ridge_model = joblib.load('ridge_model.joblib')
scaler = joblib.load('scaler.joblib')


# Convert 'Age Group' to one-hot encoding
user_features_encoded = pd.get_dummies(user_features, columns=['Age Group'], drop_first=True)

# Ensure the categorical columns are present and fill with zeros if not
expected_columns = ['Diabetes', 'Blood Pressure Problems', 'Any Transplants', 
                    'Any Chronic Diseases', 'Known Allergies', 'History Of Cancer In Family', 
                    'BMI', 'MajorSurgery_1','MajorSurgery_2','MajorSurgery_3', 'Age Group_31-40', 'Age Group_41-50', 
                    'Age Group_51-60', 'Age Group_61-70']
for col in expected_columns:
    if col not in user_features_encoded.columns:
        user_features_encoded[col] = 0

# Ensure all columns are present and in the correct order
user_features_encoded = user_features_encoded[expected_columns]


# Preprocess input features (e.g., scale them)
input_features_scaled = scaler.transform(user_features_encoded.values[:, :14])


# Make prediction using the loaded model
prediction = ridge_model.predict(input_features_scaled)

st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")

