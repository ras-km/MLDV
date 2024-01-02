#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Load the model
#with open('ridge_model.pkl', 'rb') as clf_file:
   # model = pickle.load(clf_file)
clf = joblib.load('ridge_model.joblib')    

# Load the scaler
#with open('scaler.pkl', 'rb') as scaler_file:
   # scaler = pickle.load(scaler_file)
scaler = joblib.load('scaler.joblib')

# Preprocess input features (e.g., scale them)
input_features_scaled = scaler.transform(user_features)

# Make prediction using the loaded model
prediction = ridge_model.predict(input_features_scaled)

st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")

