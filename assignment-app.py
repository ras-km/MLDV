import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title="Premium Prediction",  # String or None. Strings get appended with "• Streamlit". 
	page_icon="images/medical_insurance5.jpg",  # String, anything supported by st.image, or None.
)

with st.container(border=True):
    st.header(':rainbow[Insurance Premium Prediction]', divider='rainbow')
    st.write(':orange[This app predicts the health insurance premium based on user information input]')
    st.write('Please expand the sidebar on the top left arrow!')
    st.write('##')

image_path = "images/medical_insurance5.jpg"
image = st.image(image_path, use_column_width=True)

with st.container():
    st.header(':orange[Please fill in these information for an estimate of your premium]', divider='rainbow')

# Load the model and scaler
model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def get_user_input():
    diabetes = st.sidebar.checkbox(':orange[Diabetes]', value=False, help=':orange[Select if you have Diabetes]')
    blood_pressure_problems = st.sidebar.checkbox(':orange[Blood Pressure]', value=False, help=':orange[Select if you have Blood Pressure Problems]')
    any_transplants = st.sidebar.checkbox(':orange[Transplants]', value=False, help=':orange[Select if you have any Transplants]')
    any_chronic_diseases = st.sidebar.checkbox(':orange[Chronic Diseases]', value=False, help=':orange[Select if you have any Chronic Diseases]')
    known_allergies = st.sidebar.checkbox(':orange[Allergies]', value=False, help=':orange[Select if you have Known Allergies]')
    history_of_cancer_in_family = st.sidebar.checkbox(':orange[History of Cancer]', value=False, help=':orange[Select if there is a History of Cancer in Family]')
    bmi = st.sidebar.slider(':orange[BMI]', 0.0, 100.0, 25.0, help=':orange[Use slider to select your BMI]')
    age_group = st.sidebar.selectbox(':orange[Age Group]', ['18-30', '31-40', '41-50', '51-60', '61-70'], help=':orange[Select your age group]')
    major_surgeries = st.sidebar.selectbox(':orange[Major Surgeries]', [0, 1, 2, 3], help=':orange[How many times have you had major surgeries before?]')

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
    st.write("---")

with st.form("quotation_form"):
    st.subheader(f"The predicted premium is: :orange[${predicted_price[0]:,.2f}]", divider='rainbow')   
    st.write(":orange[Leave your name and email, and we will send you a quotation!]", divider='rainbow')
    st.write("##")

    name = st.text_input(":orange[Name]", key="name", placeholder="Enter Name")
    email = st.text_input(":orange[Email]", key="email", placeholder="someone@somewhere.com")

    submit_button = st.form_submit_button(":orange[Send quotation]:e-mail:")

# Process the form data after submission
if submit_button:
    # Perform actions with the collected data (name and email)
    progress_text = ":orange[Sending quotation....]"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.write(f"Thank you, :orange[{name}]! A quotation has been sent to {email}")
