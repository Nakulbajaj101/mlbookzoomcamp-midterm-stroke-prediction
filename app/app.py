import streamlit as st
import requests


st.title("Brain Health Service - Is your patient at risk of having a stroke")



work_mapping = {
        "Manage Children": "Children",
        "Government Job": "Govt_Job",
        "Private Job": "Private",
        "Self Employed": "Self-employed",
        "Never Worked": "Never_worked"
}

placeholder=st.empty()
id = placeholder.number_input('Enter Patient ID:', value=0, max_value=99999)
gender = st.selectbox("Patient's Gender:", ('Male', 'Female'))
age = st.slider("Patient's Age:", min_value=18, max_value=120, step=1)
hypertension = st.selectbox("Does the patient have hyptertension?", ("Yes", "No"))
heart_disease = st.selectbox("Does the patient have heart disease?", ("Yes", "No"))
ever_married = st.selectbox("Is the patient married?", ("Yes", "No"))
work_type = st.selectbox("What type of work does the patient do?", ("Manage Children", "Government Job", "Private Job", "Self Employed"))
residence_type = st.selectbox("Which area type does the patient live in?", ("Urban", "Rural"))
avg_glucose_level = st.slider("Average glucose of the patient", min_value=40.0, max_value=300.0)
bmi = st.slider("BMI of the patient", min_value=20.0, max_value=100.0)
smoking_status = st.selectbox("What is the patient's smoking status>", ("Formerly Smoked", "Never Smoked", "Smokes", "unknown"))


url = "http://localhost:3000" #Update the url
endpoint = f"{url}/classify"
request = {
    "id" : id,
    "gender": gender,
    "age": int(age),
    "hypertension":  int(hypertension=="Yes"),
    "heart_disease": int(heart_disease=="Yes"),
    "ever_married": int(ever_married=="Yes"),
    "work_type": work_mapping[work_type],
    "residence_type": residence_type,
    "avg_glucose_level": float(avg_glucose_level),
    "bmi": float(bmi),
    "smoking_status": smoking_status.lower()
}

pred_value=""
if st.button('Predict Stroke Risk', disabled=id==0):
    prediction = requests.post(url=url, json=request).json()
    pred_value = st.write(f"Chance of stroke for your patient is {prediction['stroke_risk']}, and likelihood of stroke is {round(prediction['probability_of_stroke'],2)}") 
if st.button("Clear Form", disabled=pred_value==""):
    placeholder.number_input('Enter Patient ID:', value=0)