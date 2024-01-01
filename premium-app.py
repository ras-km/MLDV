import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load your dataset (replace with your actual dataset)
med_premium = pd.read_csv('Medicalpremium.csv')

# Assuming the target variable is named 'target', adjust as needed
X = med_premium.drop('PremiumPrice', axis=1)
y = med_premium['PremiumPrice']

# Print basic information about the loaded dataset
st.write("Loaded Dataset Information:")
st.write(df.info())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Print shapes of train and test sets
st.write("Shapes after train-test split:")
st.write("X_train shape:", X_train.shape)
st.write("X_test shape:", X_test.shape)
st.write("y_train shape:", y_train.shape)
st.write("y_test shape:", y_test.shape)

# Create the RandomForestRegressor model
clf = RandomForestRegressor()

# Train the model
clf.fit(X_train, y_train)

# Streamlit app
st.write("""
# Premium Prediction App
This app predicts the premium based on user input.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Use radio buttons for yes/no selection
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
    blood_pressure_problems = st.sidebar.radio('Blood Pressure Problems', ['No', 'Yes'])
    any_transplants = st.sidebar.radio('Any Transplants', ['No', 'Yes'])
    any_chronic_diseases = st.sidebar.radio('Any Chronic Diseases', ['No', 'Yes'])
    known_allergies = st.sidebar.radio('Known Allergies', ['No', 'Yes'])
    history_of_cancer_in_family = st.sidebar.radio('History of Cancer in Family', ['No', 'Yes'])
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, 25.0)  # Assuming BMI range is 0.0 to 100.0
    
    # Use dropdown for age group selection
    age_group = st.sidebar.selectbox('Select Age Group', ['18-30', '31-40', '41-50', '51-60', '61-70'])

    # Create a dictionary to hold user input
    features = {
        'Diabetes': 1 if diabetes == 'Yes' else 0,
        'Blood Pressure Problems': 1 if blood_pressure_problems == 'Yes' else 0,
        'Any Transplants': 1 if any_transplants == 'Yes' else 0,
        'Any Chronic Diseases': 1 if any_chronic_diseases == 'Yes' else 0,
        'Known Allergies': 1 if known_allergies == 'Yes' else 0,
        'History of Cancer in Family': 1 if history_of_cancer_in_family == 'Yes' else 0,
        'BMI': bmi,
        'Age Group': age_group
    }
    return features

# User input
features = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(features)

# Make prediction
prediction = clf.predict(pd.DataFrame(features, index=[0]))

# Display prediction
st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")

# Optional: Display actual premium if available
# st.write(f"The actual premium is: {actual_premium}")

# Calculate and display mean absolute error (MAE)
# actual_premium = y_test.iloc[0]  # Replace with actual premium value
# mae = mean_absolute_error([actual_premium], [prediction[0]])
# st.write(f"Mean Absolute Error (MAE): {mae}")

