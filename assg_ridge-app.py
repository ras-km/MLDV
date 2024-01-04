import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import pickle
import joblib


with st.container():
    st.title('Insurance Premium Prediction App')
    st.header('This app predicts the health insurance premium based on user information input')
    st.write('##')


st.image('medical_insurance4.jpeg')


st.sidebar.header('Please fill in these information for us to serve you better')

def user_input_features():
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'], help='Select either Yes or No')
    blood_pressure_problems = st.sidebar.radio('Blood Pressure Problems', ['No', 'Yes'], help='Select either Yes or No')
    any_transplants = st.sidebar.radio('Any Transplants', ['No', 'Yes'], help='Select either Yes or No')
    any_chronic_diseases = st.sidebar.radio('Any Chronic Diseases', ['No', 'Yes'], help='Select either Yes or No')
    known_allergies = st.sidebar.radio('Known Allergies', ['No', 'Yes'], help='Select either Yes or No')
    history_of_cancer_in_family = st.sidebar.radio('History of Cancer in Family', ['No', 'Yes'], help='Select either Yes or No')
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, 25.0, help='Use slider to select your BMI')
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
        'Age Group': f'Age Group_{age_group}',
        'Major Surgeries': f'MajorSurgery_{major_surgeries}'
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_features = user_input_features()

# Load the model and scaler
ridge_model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

# Convert 'Age Group' and 'Major Surgeries' to one-hot encoding manually
age_group_column = f'Age Group_{user_features["Age Group"].values[0]}'
major_surgeries_column = f'MajorSurgery_{user_features["Major Surgeries"].values[0]}'

# Create a DataFrame with one-hot encoded columns
user_features_encoded = pd.DataFrame(0, columns=['Age Group_31-40', 'Age Group_41-50', 'Age Group_51-60', 'Age Group_61-70',
                                                 'MajorSurgery_1', 'MajorSurgery_2', 'MajorSurgery_3'],
                                     index=user_features.index)

# Update the values in the DataFrame with the one-hot encoded features
user_features_encoded[age_group_column] = 1
user_features_encoded[major_surgeries_column] = 1

# Concatenate with the original DataFrame
user_features_encoded = pd.concat([user_features, user_features_encoded], axis=1)

# Ensure all columns are present and in the correct order
expected_columns = ['Diabetes', 'Blood Pressure Problems', 'Any Transplants', 
                    'Any Chronic Diseases', 'Known Allergies', 'History Of Cancer In Family', 
                    'BMI', 'MajorSurgery_1', 'MajorSurgery_2', 'MajorSurgery_3',
                    'Age Group_31-40', 'Age Group_41-50', 'Age Group_51-60', 'Age Group_61-70']

user_features_encoded = user_features_encoded.reindex(columns=expected_columns, fill_value=0)

# Preprocess input features (e.g., scale them)
input_features_scaled = scaler.transform(user_features_encoded.values)

# Make prediction using the loaded model
prediction = ridge_model.predict(input_features_scaled)

with st.container():
    st.write('Prediction')
    st.subheader(f"The predicted premium is: {prediction[0]}")
