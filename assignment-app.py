import streamlit as st
import pandas as pd
import numpy as np
import joblib

with st.container():
    st.title('Insurance Premium Prediction App')
    st.write('This app predicts the health insurance premium based on user information input')
    st.write('##')

image_path = "images/medical_insurance3.jpeg"
image = st.image(image_path, caption="Your Image Caption", use_column_width=True)

st.sidebar.header('Please fill in these information for us to serve you better')

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
input_features_scaled = scaler.transform(user_features.values)

# Prediction
predicted_price = model.predict(input_features_scaled)

# Display result
#st.subheader('Prediction')
#st.write("The premium is estimated to be ${:,.2f}".format(predicted_price[0]))

columns = st.columns(2)  # Use the number of columns you want

with columns[0]:
    st.header('Prediction')

with columns[1]:
    st.subheader(f"The predicted premium is: ${predicted_price[0]:,.2f}")

with st.form("quotation_form"):
    st.write("---")
    st.write("Leave your name and email, and we will send you a quotation")
    st.write("##")

    name = st.text_input("Name", key="name")
    email = st.text_input("Email", key="email")

    submit_button = st.form_submit_button("Send quotation")

# Process the form data after submission
if submit_button:
    # Perform actions with the collected data (name and email)
    st.write(f"Name: {name}")
    st.write(f"Email: {email}")
#except Exception as e:
    #st.error(f"An error occurred: {e}")
