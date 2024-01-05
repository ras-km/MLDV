import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the model
    model = joblib.load('ridge_model.pkl')
    
    # Load the scaler
    scaler = joblib.load('scaler.pkl')

# Your prediction function
def predict_premium(features):

    # Preprocess input features (e.g., scale them)
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)

    return predictions
    
# Function to get user input
def get_user_input():
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
    blood_pressure_problems = st.sidebar.radio('Blood Pressure Problems', ['No', 'Yes'])
    any_transplants = st.sidebar.radio('Any Transplants', ['No', 'Yes'])
    any_chronic_diseases = st.sidebar.radio('Any Chronic Diseases', ['No', 'Yes'])
    known_allergies = st.sidebar.radio('Known Allergies', ['No', 'Yes'])
    history_of_cancer_in_family = st.sidebar.radio('History of Cancer in Family', ['No', 'Yes'])
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    age_group = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'])
    major_surgeries = st.sidebar.selectbox('Number of Major Surgeries', [0, 1, 2, 3])

    # Convert 'Age Group' to one-hot encoding
    age_group_encoded = pd.get_dummies(pd.Series([f'Age Group_{age_group}']), prefix='Age Group').iloc[:, 1:]
    
    # Convert 'Age Group' to one-hot encoding
    #user_features_encoded = pd.get_dummies(user_features, columns=['Age Group', 'Major Surgeries'], drop_first=True)
    

    # Create a DataFrame with the processed features
    user_features = pd.DataFrame({
        'Diabetes': 1 if diabetes == 'Yes' else 0,
        'Blood Pressure Problems': 1 if blood_pressure_problems == 'Yes' else 0,
        'Any Transplants': 1 if any_transplants == 'Yes' else 0,
        'Any Chronic Diseases': 1 if any_chronic_diseases == 'Yes' else 0,
        'Known Allergies': 1 if known_allergies == 'Yes' else 0,
        'History Of Cancer In Family': 1 if history_of_cancer_in_family == 'Yes' else 0,
        'BMI': bmi,
        'Age Group_31-40': 1 if age_group == '31-40' else 0,
        'Age Group_41-50': 1 if age_group == '41-50' else 0,
        'Age Group_51-60': 1 if age_group == '51-60' else 0,
        'Age Group_61-70': 1 if age_group == '61-70' else 0,
        'Major Surgeries_1': 1 if major_surgeries == 1 else 0,
        'Major Surgeries_2': 1 if major_surgeries == 2 else 0,
        'Major Surgeries_3': 1 if major_surgeries == 3 else 0,
    }, index=[0])

    # Concatenate the one-hot encoded 'Age Group' to the DataFrame
    user_features = pd.concat([user_features, age_group_encoded], axis=1)

    return user_features
    
def preprocess_user_input(diabetes, blood_pressure_problems, any_transplants,
                           any_chronic_diseases, known_allergies, history_of_cancer_in_family,
                           bmi, major_surgery_1, major_surgery_2, major_surgery_3,
                           age_31_40, age_41_50, age_51_60, age_61_70):
    # Convert categorical variables to numerical representation
    diabetes = 1 if diabetes == 'Yes' else 0
    blood_pressure_problems = 1 if blood_pressure_problems == 'Yes' else 0
    any_transplants = 1 if any_transplants == 'Yes' else 0
    any_chronic_diseases = 1 if any_chronic_diseases == 'Yes' else 0
    known_allergies = 1 if known_allergies == 'Yes' else 0
    history_of_cancer_in_family = 1 if history_of_cancer_in_family == 'Yes' else 0

    # Create a DataFrame with the processed features
    user_features = pd.DataFrame({
        'Diabetes': diabetes,
        'Blood Pressure Problems': blood_pressure_problems,
        'Any Transplants': any_transplants,
        'Any Chronic Diseases': any_chronic_diseases,
        'Known Allergies': known_allergies,
        'History Of Cancer In Family': history_of_cancer_in_family,
        'BMI': bmi,
        'Major Surgery 1': major_surgery_1,
        'Major Surgery 2': major_surgery_2,
        'Major Surgery 3': major_surgery_3,
        'Age Group 31-40': age_31_40,
        'Age Group 41-50': age_41_50,
        'Age Group 51-60': age_51_60,
        'Age Group 61-70': age_61_70
    }, index=[0])

    return user_features

# Convert 'Age Group' to one-hot encoding
user_features_encoded = pd.get_dummies(user_features, columns=['Age Group', 'Major Surgeries'], drop_first=True)
                               
# Ensure the categorical columns are present and fill with zeros if not
expected_columns = ['Diabetes', 'Blood Pressure Problems', 'Any Transplants', 
                    'Any Chronic Diseases', 'Known Allergies', 'History Of Cancer In Family', 
                    'BMI', 'MajorSurgery_1', 'MajorSurgery_2', 'MajorSurgery_3', 
                    'Age Group_31-40', 'Age Group_41-50', 'Age Group_51-60', 'Age Group_61-70']

user_features_encoded = user_features_encoded.reindex(columns=expected_columns, fill_value=0)
                               
# Sidebar for user input
st.sidebar.header('User Input Parameters')

try:
    # Get user input
    user_features = get_user_input()

    # Preprocess input features (e.g., scale them)
    input_features_scaled = scaler.transform(user_features.values)

    # Prediction
    predicted_price = predict_premium(input_features_scaled)

    # Display result
    st.subheader('Prediction')
    st.write("The premium is estimated to be ${:,.2f}".format(predicted_price))
except Exception as e:
    st.error(f"An error occurred: {e}")
