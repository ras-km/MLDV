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

    # Create a DataFrame with the processed features
    user_features = pd.DataFrame({
        'Diabetes': 1 if diabetes == 'Yes' else 0,
        'Blood Pressure Problems': 1 if blood_pressure_problems == 'Yes' else 0,
        'Any Transplants': 1 if any_transplants == 'Yes' else 0,
        'Any Chronic Diseases': 1 if any_chronic_diseases == 'Yes' else 0,
        'Known Allergies': 1 if known_allergies == 'Yes' else 0,
        'History Of Cancer In Family': 1 if history_of_cancer_in_family == 'Yes' else 0,
        'BMI': bmi,
        'Age Group': age_group,
        'Major Surgeries': f'Major Surgeries_{major_surgeries}'
    }, index=[0])

    return user_features

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Manually encode categorical features (Age Group and Major Surgeries)
age_group_encoded = pd.get_dummies(user_features['Age Group'], prefix='Age Group')
major_surgeries_encoded = pd.get_dummies(user_features['Major Surgeries'], prefix='Major Surgeries')

# Concatenate the one-hot encoded features to the DataFrame
user_features_encoded = pd.concat([user_features, age_group_encoded, major_surgeries_encoded], axis=1)

# Drop the original 'Age Group' and 'Major Surgeries' columns
user_features_encoded = user_features_encoded.drop(['Age Group', 'Major Surgeries'], axis=1)

# Ensure all columns are present and in the correct order
expected_columns = ['Diabetes', 'Blood Pressure Problems', 'Any Transplants', 
                    'Any Chronic Diseases', 'Known Allergies', 'History Of Cancer In Family', 
                    'BMI', 'Age Group_31-40', 'Age Group_41-50', 'Age Group_51-60', 'Age Group_61-70',
                    'Major Surgeries_1', 'Major Surgeries_2', 'Major Surgeries_3']

user_features_encoded = user_features_encoded.reindex(columns=expected_columns, fill_value=0)


# Ensure all columns are present and in the correct order
expected_columns = ['Diabetes', 'Blood Pressure Problems', 'Any Transplants', 
                    'Any Chronic Diseases', 'Known Allergies', 'History Of Cancer In Family', 
                    'BMI', 'Major Surgeries_1', 'Major Surgeries_2', 'Major Surgeries_3', 
                    'Age Group_31-40', 'Age Group_41-50', 'Age Group_51-60', 'Age Group_61-70']

user_features_encoded = user_features_encoded.reindex(columns=expected_columns, fill_value=0)

# Preprocess input features (e.g., scale them)
input_features_scaled = scaler.transform(user_features_encoded.values)

# Prediction
predicted_price = model.predict(input_features_scaled)

# Display result
st.subheader('Prediction')
st.write("The premium is estimated to be ${:,.2f}".format(predicted_price[0]))
