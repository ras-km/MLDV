import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Use a locally available dataset for testing
from sklearn.datasets import load_Medicalpremium
insurance = load_Medicalpremium()
X = Medicalpremium.data
y = Medicalpremium.target

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the RandomForestRegressor model
clf = RandomForestRegressor()

# Train the model
clf.fit(X_train, y_train)

# Define the user input function
def user_input_features():
    diabetes = st.sidebar.radio('Diabetes', 0, 1)
    blood_pressure_problems = st.sidebar.radio('Blood pressure problems', 0, 1)
    any_transplants = st.sidebar.radio('Any transplants', 0, 1)
    any_chronic_diseases = st.sidebar.radio('Any chronic diseases', 0, 1)
    known_allergies = st.sidebar.radio('Known allergies', 0, 1)
    history_of_cancer_in_family = st.sidebar.radio('History of cancer in family', 0, 1)
    bmi = st.sidebar.slider('BMI', 0, 100)
    major_surgery_1 = st.sidebar.radio('Major surgery 1', 0, 1)
    major_surgery_2 = st.sidebar.radio('Major surgery 2', 0, 1)
    major_surgery_3 = st.sidebar.radio('Major surgery 3', 0, 1)
    age_31_40 = st.sidebar.radio('Age 31-40', 0, 1)
    age_41_50 = st.sidebar.radio('Age 41-50', 0, 1)
    age_51_60 = st.sidebar.radio('Age 51-60', 0, 1)
    age_61_70 = st.sidebar.radio('Age 61-70', 0, 1)

    data = {'diabetes': diabetes,
            'blood_pressure_problems': blood_pressure_problems,
            'any_transplants': any_transplants,
            'any_chronic_diseases': any_chronic_diseases,
            'known_allergies': known_allergies,
            'history_of_cancer_in_family': history_of_cancer_in_family,
            'bmi': bmi,
            'major_surgery_1': major_surgery_1,
            'major_surgery_2': major_surgery_2,
            'major_surgery_3': major_surgery_3,
            'age_31_40': age_31_40,
            'age_41_50': age_41_50,
            'age_51_60': age_51_60,
            'age_61_70': age_61_70}

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(user_input)

# Make prediction
prediction = clf.predict(user_input)

# Display prediction
st.subheader('Prediction')
st.write(prediction)

