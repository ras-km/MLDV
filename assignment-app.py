import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Set page configuration
st.set_page_config(
    layout="wide",  # Adjust the layout as needed
    initial_sidebar_state="collapsed",  # Expand the sidebar by default
    page_title="Premium Prediction",
    page_icon="images/medical_insurance5.jpg",
)

# Set the background image using CSS
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.pexels.com/photos/169789/pexels-photo-169789.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

input_style = """
<style>
div[data-baseweb="base-input"] > div:first-child > input {
    width: 100px;  # Set the desired width for text input
    height: 30px;  # Set the desired height for text input
    padding: 5px;  # Adjust padding if needed
}
</style>
"""

st.markdown(input_style, unsafe_allow_html=True)

# Main content
st.title('Insurance Premium Prediction')
st.write('This app predicts the health insurance premium based on user information input.')
st.write('Please expand the sidebar on the top left arrow!')
st.write('##')

# Sidebar
#st.sidebar.header('User Input')
st.sidebar.write('Fill in the following information for an estimate of your premium.')

# Load the model and scaler
model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def get_user_input():
    diabetes = st.sidebar.checkbox('Diabetes', value=False, help='Select if you have Diabetes')
    blood_pressure_problems = st.sidebar.checkbox('Blood Pressure', value=False, help='Select if you have Blood Pressure Problems')
    any_transplants = st.sidebar.checkbox('Transplants', value=False, help='Select if you have any Transplants')
    any_chronic_diseases = st.sidebar.checkbox('Chronic Diseases', value=False, help='Select if you have any Chronic Diseases')
    known_allergies = st.sidebar.checkbox('Allergies', value=False, help='Select if you have Known Allergies')
    history_of_cancer_in_family = st.sidebar.checkbox('History of Cancer', value=False, help='Select if there is a History of Cancer in Family') 
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0, help='Use slider to select your BMI')
    age_group = st.sidebar.selectbox('Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'], help='Select your age group')
    major_surgeries = st.sidebar.selectbox('Major Surgeries', [0, 1, 2, 3], help='How many times have you had major surgeries before?')

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

# Display result
st.subheader(f"The predicted premium is: ${predicted_price[0]:,.2f}")
st.write("Leave your name and email, and we will send you a quotation!")
st.write("##")

name = st.text_input("Name", key="name", placeholder="Enter Name")
email = st.text_input("Email", key="email", placeholder="someone@somewhere.com")

submit_button = st.button("Send quotation")

# Process the form data after submission
if submit_button:
    # Perform actions with the collected data (name and email)
    progress_text = "Sending quotation...."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.write(f"Thank you, {name}! A quotation has been sent to {email}")
