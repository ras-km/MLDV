import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

st.write("""
This app predicts the premium for user after they input the required information""")

med_premium=pd.read_csv('Medicalpremium.csv')

target_variable = 'PremiumPrice'
med_premium_feat = med_premium.drop(columns=[target_variable])
# Extract features and target variable from med_premium
X_train = med_premium_feat
y_train = med_premium[target_variable]

# Extract features from premium_features (ensure it has the same columns as X_train)
X_test = med_premium[X_train.columns]

# Create arrays X to contain features and y for target only
X = med_premium_feat.values
y = med_premium['PremiumPrice'].values

# Split the data into training and testing sets
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.3, random_state=7)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=7)

# Train the Random Forest Regressor model on the training set
rf_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')
# Save the scaler if you used one during preprocessing
joblib.dump(scaler, 'scaler.pkl')

# Make predictions on the test set
y_test_pred = rf_model.predict(X_test_scaled)

def predict_premium(features):
    
    # Load the model
    loaded_model  = joblib.load('random_forest_model.pkl')
    
    # Load the scaler
    loaded_scaler = joblib.load('scaler.pkl')

    # Preprocess input features (e.g., scale them)
    features_scaled = loaded_scaler.transform(features)

    # Make predictions
    predictions = loaded_model.predict(features_scaled)

    return predictions

def estimate_insurance_premium_rf(model, features):
    # Assuming 'model' is a trained random forest model

    # Convert the input features to a 2D array
    input_features = [list(features.values())]

    # Make predictions using the random forest model
    premium_value = model.predict(input_features)

    # Assuming premium_value is the estimated insurance premium
    return premium_value[0]



st.sidebar.header('User Input Parameters')


def user_input_features():
    # Use radio buttons for yes/no selection
    Diabetes = st.sidebar.radio('Diabetes', [0, 1])
    BloodPressureProblems = st.sidebar.radio('Blood Pressure Problems', [0, 1])
    AnyTransplants = st.sidebar.radio('Any Transplants', [0, 1])
    AnyChronicDiseases = st.sidebar.radio('Any Chronic Diseases', [0, 1])
    KnownAllergies = st.sidebar.radio('Known Allergies', [0, 1])
    HistoryOfCancerInFamily = st.sidebar.radio('History of Cancer in Family', [0, 1])

    # Use sliders for continuous features
    BMI = st.sidebar.slider('BMI', 0.0, 100.0, 25.0)  # Assuming BMI range is 0.0 to 100.0
    
    # Use dropdown for age group selection
    AgeGroup = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'])

    # Use dropdown for major surgeries selection
    MajorSurgeries = st.sidebar.selectbox('Number of Major Surgeries', [0, 1, 2, 3])
    
    # Create a dictionary to hold user input
    features = {
        'Diabetes': Diabetes,
        'Blood Pressure Problems': BloodPressureProblems,
        'Any Transplants': AnyTransplants,
        'Any Chronic Diseases': AnyChronicDiseases,
        'Known Allergies': KnownAllergies,
        'History Of Cancer In Family': HistoryOfCancerInFamily,
        'BMI': BMI,
        'Age Group': AgeGroup,
        'Major Surgeries': MajorSurgeries
    }
    return features



# User input
features = user_input_features()

# Load the trained model
loaded_model = joblib.load('random_forest_model.pkl')

# Load the scaler
loaded_scaler = joblib.load('scaler.pkl')

# Preprocess input features (e.g., scale them)
features_array = [list(features.values())]
features_scaled = loaded_scaler.transform(features)

# Make prediction using the loaded model
prediction = loaded_model.predict(features_scaled)
# Display prediction
st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")

