import streamlit as st
import pandas as pd
from sklearn import datasets
drom sklearn.ensemble impport RandomForestRegressor

st.write("""
This app predicts the premium for user after they input the required information""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    diabetes = st.sidebar.dropdown('Diabetes', 0, 1, o) 
    blood_pressure_problems = st.sidebar.radio('Blood pressure problems', 0, 1, o) 
    any_transplants = st.sidebar.radio('Any transplants', 0, 1, o)
    any_chronic_diseases = st.sidebar.radio('Any chronic diseases', 0, 1, o) 
    known_allergies = st.sidebar.radio('Known allergies', 0, 1, o) 
    history_of_cancer_in_family = st.sidebar.radio('History of cancer in family', 0, 1, o)
    bmi = st.sidebar.slider('BMI', 0, 1, o) 
    major_surgery_1 = st.sidebar.radio('Major surgery 1', 0, 1, o) 
    major_surgery_2 = st.sidebar.radio('Major surgery 2', 0, 1, o) 
    major_surgery_3 = st.sidebar.radio('Major surgery 3', 0, 1, o)
    age_31_40 = st.sidebar.radio('Age 31-40', 0, 1, o) 
    age_41_50 = st.sidebar.radio('Age 41-50', 0, 1, o) 
    age_51_60 = st.sidebar.radio('Age 51-60', 0, 1, o) 
    age_61_70 = st.sidebar.radio('Age 61-70', 0, 1, o)
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

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

premium = datasets.load_med_premium()
X = med_premium_feat.data
Y = med+premium.target

clf = RandomForestRegressor
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(med_premium.target_names)

st.subheader('Prediction')
st.write(med_premium.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

