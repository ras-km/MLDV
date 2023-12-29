{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a56c0bc2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn import datasets\n",
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "043ea479",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.write(\"\"\"\n",
    "This app predicts the premium for user after they input the required information\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e63134e",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.sidebar.header('User Input Parameters')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efa265d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def user_input_features():\n",
    "    diabetes = st.sidebar.dropdown('Diabetes', 0, 1, o) \n",
    "    blood_pressure_problems = st.sidebar.radio('Blood pressure problems', 0, 1, o) \n",
    "    any_transplants = st.sidebar.radio('Any transplants', 0, 1, o)\n",
    "    any_chronic_diseases = st.sidebar.radio('Any chronic diseases', 0, 1, o) \n",
    "    known_allergies = st.sidebar.radio('Known allergies', 0, 1, o) \n",
    "    history_of_cancer_in_family = st.sidebar.radio('History of cancer in family', 0, 1, o)\n",
    "    bmi = st.sidebar.slider('BMI', 0, 1, o) \n",
    "    major_surgery_1 = st.sidebar.radio('Major surgery 1', 0, 1, o) \n",
    "    major_surgery_2 = st.sidebar.radio('Major surgery 2', 0, 1, o) \n",
    "    major_surgery_3 = st.sidebar.radio('Major surgery 3', 0, 1, o)\n",
    "    age_31_40 = st.sidebar.radio('Age 31-40', 0, 1, o) \n",
    "    age_41_50 = st.sidebar.radio('Age 41-50', 0, 1, o) \n",
    "    age_51_60 = st.sidebar.radio('Age 51-60', 0, 1, o) \n",
    "    age_61_70 = st.sidebar.radio('Age 61-70', 0, 1, o)\n",
    "    data = {'diabetes': diabetes,\n",
    "           'blood_pressure_problems': blood_pressure_problems,\n",
    "           'any_transplants': any_transplants,\n",
    "           'any_chronic_diseases': any_chronic_diseases,\n",
    "           'known_allergies': known_allergies,\n",
    "           'history_of_cancer_in_family': history_of_cancer_in_family,\n",
    "           'bmi': bmi,\n",
    "           'major_surgery_1': major_surgery_1,\n",
    "           'major_surgery_2': major_surgery_2,\n",
    "           'major_surgery_3': major_surgery_3,\n",
    "           'age_31_40': age_31_40,\n",
    "           'age_41_50': age_41_50,\n",
    "           'age_51_60': age_51_60,\n",
    "           'age_61_70': age_61_70}\n",
    "    features = pd.DataFrame(data, index=[0])\n",
    "    return features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44b09c10",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = user_input_features()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95b76449",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('User Input parameters')\n",
    "st.write(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beeec01f",
   "metadata": {},
   "outputs": [],
   "source": [
    "premium = datasets.load_med_premium()\n",
    "X = med_premium_feat.data\n",
    "Y = med+premium.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ffaffb85",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = RandomForestRegressor\n",
    "clf.fit(X,Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5dde7f37",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = clf.predict(df)\n",
    "prediction_proba = clf.predict_proba(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0146eba",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('Class labels and their corresponding index number')\n",
    "st.write(med_premium.target_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddde29dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('Prediction')\n",
    "st.write(med_premium.target_names[prediction])\n",
    "#st.write(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4afc50d",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('Prediction Probability')\n",
    "st.write(prediction_proba)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
