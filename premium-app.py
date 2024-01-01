import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
st.write(med_premium .info())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=7)

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

# Display user input
st.subheader('User Input parameters')
st.write(features)

# Load the trained model
loaded_model = joblib.load(model_filename)

# Make prediction using the loaded model
prediction = loaded_model.predict(pd.DataFrame(features, index=[0]))

# Display prediction
st.subheader('Prediction')
st.write(f"The predicted premium is: {prediction[0]}")

# Optional: Display actual premium if available
# st.write(f"The actual premium is: {actual_premium}")

# Calculate and display mean absolute error (MAE)
# actual_premium = y_test.iloc[0]  # Replace with actual premium value
# mae = mean_absolute_error([actual_premium], [prediction[0]])
# st.write(f"Mean Absolute Error (MAE): {mae}")

