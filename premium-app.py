import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset (replace with your actual dataset)
med_premium = pd.read_csv('Medicalpremium.csv')
target_variable = 'PremiumPrice'
med_premium_feat = med_premium.drop(columns=[target_variable])
# Extract features and target variable from med_premium
X_train = med_premium_feat
y_train = med_premium[target_variable]

# Extract features from premium_features (ensure it has the same columns as X_train)
X_test = med_premium_feat[X_train.columns]

# Print basic information about the loaded dataset
st.write("Loaded Dataset Information:")
st.write(med_premium.info())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=7)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create the RandomForestRegressor model
clf = RandomForestRegressor()

# Train the model
clf.fit(X_train_scaled, y_train)

# Save the model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Streamlit app
st.write("""
# Premium Prediction App
This app predicts the premium based on user input.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Use radio buttons for yes/no selection
    diabetes = st.sidebar.radio('Diabetes', [0, 1])
    blood_pressure_problems = st.sidebar.radio('Blood Pressure Problems', [0, 1])
    any_transplants = st.sidebar.radio('Any Transplants', [0, 1])
    any_chronic_diseases = st.sidebar.radio('Any Chronic Diseases', [0, 1])
    known_allergies = st.sidebar.radio('Known Allergies', [0, 1])
    history_of_cancer_in_family = st.sidebar.radio('History of Cancer in Family', [0, 1])

    # Use sliders for continuous features
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, 25.0)  # Assuming BMI range is 0.0 to 100.0
    
    # Use dropdown for age group selection
    age_group = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'])

    # Use dropdown for major surgeries selection
    major_surgeries = st.sidebar.selectbox('Number of Major Surgeries', [0, 1, 2, 3])
    
    # Create a dictionary to hold user input
    features = {
        'Diabetes': diabetes,
        'Blood Pressure Problems': blood_pressure_problems,
        'Any Transplants': any_transplants,
        'Any Chronic Diseases': any_chronic_diseases,
        'Known Allergies': known_allergies,
        'History Of Cancer In Family': history_of_cancer_in_family,
        'BMI': bmi,
        'Age Group': age_group,
        'Major Surgeries': major_surgeries
    }
    return features

# User input
features = user_input_features()

# Load the model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
# Preprocess input features (e.g., scale them)
input_features_array = [list(features.values())]
# Convert the input features to a NumPy array
input_features_array = np.array(input_features_array)

# Add print statements for debugging
print(f"input_features_array: {input_features_array}, type: {type(input_features_array)}")
# Reshape the input features array if needed
if len(input_features_array.shape) == 1:
    input_features_array = input_features_array.reshape(1, -1)
    
input_features_2d = scaler.transform(input_features_array)
print(f"input_features_2d: {input_features_2d}, type: {type(input_features_2d)}")
features_scaled = input_features_2d[0]  # Extract the scaled features from the 2D array
print(f"features_scaled: {features_scaled}, type: {type(features_scaled)}")

# Make prediction using the loaded model
input_features_2d = scaler.transform(input_features_array)
features_scaled = input_features_2d[0]  # Extract the scaled features from the 2D array

prediction = model.predict(features_scaled)
# Display prediction
st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")

# Optional: Display actual premium if available
# st.write(f"The actual premium is: {actual_premium}")

# Calculate and display mean absolute error (MAE)
# actual_premium = y_test.iloc[0]  # Replace with actual premium value
# mae = mean_absolute_error([actual_premium], [prediction[0]])
# st.write(f"Mean Absolute Error (MAE): {mae}")

