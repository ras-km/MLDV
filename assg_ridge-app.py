import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
# Assuming you have a dataset named 'premium_data.csv'
data = pd.read_csv('premium_data.csv')

# Preprocess your data and split into features (X) and target variable (y)

# Initialize and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit the Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_scaled, y)

# Save the trained model and scaler
joblib.dump(ridge_model, 'ridge_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

import streamlit as st
import pandas as pd
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

# Preprocess input features (e.g., scale them)
input_features_scaled = scaler.transform(user_features)

# Make prediction using the loaded model
prediction = ridge_model.predict(input_features_scaled)

st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")
