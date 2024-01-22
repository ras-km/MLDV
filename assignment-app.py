import streamlit as st
import pandas as pd
import numpy as np
import joblib

with st.container():
    st.title(':rainbow[Insurance Premium Prediction App]')
    st.write('This app predicts the health insurance premium based on user information input')
    st.write('##')

image_path = "images/medical_insurance3.jpeg"
image = st.image(image_path, use_column_width=True)

st.sidebar.header('Please fill in these information for an estimate of your premium')

# Load the model and scaler
model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def get_user_input():
    diabetes = st.sidebar.checkbox('Diabetes', value=False, help=':orange[Select if you have Diabetes]')
    blood_pressure_problems = st.sidebar.checkbox('Blood Pressure Problems', value=False, help=':orange[Select if you have Blood Pressure Problems]')
    any_transplants = st.sidebar.checkbox('Any Transplants', value=False, help=':orange[Select if you have any Transplants]')
    any_chronic_diseases = st.sidebar.checkbox('Any Chronic Diseases', value=False, help=':orange[Select if you have any Chronic Diseases]')
    known_allergies = st.sidebar.checkbox('Known Allergies', value=False, help=':orange[Select if you have Known Allergies]')
    history_of_cancer_in_family = st.sidebar.checkbox('History of Cancer in Family', value=False, help=':orange[Select if there is a History of Cancer in Family]')
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, 25.0, help=':orange[Use slider to select your BMI]')
    age_group = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'], help=':orange[Select your age group]')
    major_surgeries = st.sidebar.selectbox('Number of Major Surgeries', [0, 1, 2, 3], help=':orange[How many times have you had major surgeries before?]')

    # Create a DataFrame with the processed features
    user_features = pd.DataFrame({
        'Diabetes': 1 if diabetes else 0,
        'Blood Pressure Problems': 1 if blood_pressure_problems else 0,
        'Any Transplants': 1 if any_transplants else 0,
        'Any Chronic Diseases': 1 if any_chronic_diseases else 0,
        'Known Allergies': 1 if known_allergies else 0,
        'History Of Cancer In Family': 1 if history_of_cancer_in_family else 0,
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

# Get user input
user_features = get_user_input()

# Preprocess input features (e.g., scale them)
input_features_scaled = scaler.transform(user_features.values)

# Prediction
predicted_price = model.predict(input_features_scaled)

columns = st.columns(1)  # Use the number of columns you want

with columns[0]:
    st.subheader(f"The predicted premium is: :orange[${predicted_price[0]:,.2f}]")

with st.form("quotation_form"):
    st.write("---")
    st.write("Leave your name and email, and we will send you a :red[quotation]")
    st.write("##")

    name = st.text_input("Name", key="name", placeholder="Enter Name")
    email = st.text_input("Email", key="email", placeholder="someone@somewhere.com")

    submit_button = st.form_submit_button(":orange[Send quotation]:e-mail:")

# Process the form data after submission
if submit_button:
    # Perform actions with the collected data (name and email)
    st.write(f"Thank you, :orange[{name}]! A quotation has been sent to {email}")
