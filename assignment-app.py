import streamlit as st

# Your prediction function
def predict_premium(features):
    # Load the model
    model = joblib.load('ridge_model.pkl')
    
    # Load the scaler
    scaler = joblib.load('scaler.pkl')

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
        'Age Group': f'Age Group_{age_group}',
        'Major Surgeries': f'MajorSurgery_{major_surgeries}'
    }

    features = pd.DataFrame(data, index=[0])
    return features

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

    # Create a dictionary with the processed features
    user_features = {
        'Diabetes': diabetes,
        'BloodPressureProblems': blood_pressure_problems,
        'AnyTransplants': any_transplants,
        'AnyChronicDiseases': any_chronic_diseases,
        'KnownAllergies': known_allergies,
        'HistoryOfCancerInFamily': history_of_cancer_in_family,
        'BMI': bmi,
        'MajorSurgery_1': major_surgery_1,
        'MajorSurgery_2': major_surgery_2,
        'MajorSurgery_3': major_surgery_3,
        'Age_31_40': age_31_40,
        'Age_41_50': age_41_50,
        'Age_51_60': age_51_60,
        'Age_61_70': age_61_70
    }

    return user_features

# Ensure the categorical columns are present and fill with zeros if not
expected_columns = ['Diabetes', 'Blood Pressure Problems', 'Any Transplants', 
                    'Any Chronic Diseases', 'Known Allergies', 'History Of Cancer In Family', 
                    'BMI', 'MajorSurgery_1', 'MajorSurgery_2', 'MajorSurgery_3', 
                    'Age Group_31-40', 'Age Group_41-50', 'Age Group_51-60', 'Age Group_61-70']
                               
# Sidebar for user input
st.sidebar.header('User Input Parameters')

try:
    # Ensure all columns are present and in the correct order
    user_features_encoded = user_features_encoded.reindex(columns=expected_columns, fill_value=0)

    # Preprocess input features (e.g., scale them)
    input_features_scaled = scaler.transform(user_features.values)


    # Prediction
    predicted_price = predict_premium(input_features_scaled)

    # Display result
    st.subheader('Prediction')
    st.write("The premium is estimated to be ${:,.2f}".format(predicted_price))
except Exception as e:
    st.error(f"An error occurred: {e}")
