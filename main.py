# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
#import streamlit.components.v1 as components
#import shap
#import matplotlib.pyplot as plt
#import numpy as np
#import pyplot
#import matplotlib.pyplot as plt
#from PIL import Image

st.header("An AI Model for Predicting Postoperative In-Hospital Mortality in Geriatric Hip Fracture Patients")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Age = st.sidebar.selectbox("Age", ("60-69", "70-79", "80-89", "90-100", ">100"))
Sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
Fracture = st.sidebar.selectbox("Fracture type", ("Femoral neck fracture", "Intertrochanteric fracture"))
Operation = st.sidebar.selectbox("Operation", ("Hip joint replacement", "Internal fixation"))
Comorbidities = st.sidebar.selectbox("Number of comorbidities", ("0", "1", "2", "≧3"))
Coronarydisease = st.sidebar.selectbox("Coronary heart disease", ("No", "Yes"))
Cerebrovasculardisease = st.sidebar.selectbox("Cerebrovascular disease", ("No", "Yes"))
Heartfailure = st.sidebar.selectbox("Heart failure", ("No", "Yes"))
Renalfailure = st.sidebar.selectbox("Renal failure", ("No", "Yes"))
Nephroticsyndrome = st.sidebar.selectbox("Nephrotic syndrome", ("No", "Yes"))
Respiratorysystemdisease = st.sidebar.selectbox("Respiratory system disease", ("No", "Yes"))
Gastrointestinalbleeding = st.sidebar.selectbox("Gastrointestinal bleeding", ("No", "Yes"))
Gastrointestinalulcer = st.sidebar.selectbox("Gastrointestinal ulcer", ("No", "Yes"))
Liverfailure = st.sidebar.selectbox("Liver failure", ("No", "Yes"))
Cirrhosis = st.sidebar.selectbox("Cirrhosis", ("No", "Yes"))
Diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
Cancer = st.sidebar.selectbox("Cancer", ("No", "Yes"))


if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Age, Sex, Fracture, Operation, Comorbidities, Coronarydisease, Cerebrovasculardisease, Heartfailure, Renalfailure, Nephroticsyndrome, Respiratorysystemdisease, Gastrointestinalbleeding, Gastrointestinalulcer, Liverfailure, Cirrhosis, Diabetes, Cancer]],
                     columns=["Age", "Sex", "Fracture", "Operation", "Comorbidities", "Coronarydisease", "Cerebrovasculardisease", "Heartfailure", "Renalfailure", "Nephroticsyndrome", "Respiratorysystemdisease", "Gastrointestinalbleeding", "Gastrointestinalulcer", "Liverfailure", "Cirrhosis", "Diabetes", "Cancer"])
    x = x.replace(["60-69", "70-79", "80-89", "90-100", ">100"], [6, 7, 8, 9, 10])
    x = x.replace(["Male", "Female"], [1, 2])
    x = x.replace(["Femoral neck fracture", "Intertrochanteric fracture"], [1, 2])
    x = x.replace(["Hip joint replacement", "Internal fixation"], [1, 2])
    x = x.replace(["0", "1", "2", "≧3"], [0, 1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of death in hospital: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.550:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.550:
        st.markdown(f"Management Measures for Low-risk Population: Geriatric hip fracture patients identified as low-risk for postoperative in-hospital mortality also require comprehensive postoperative management. Standard care protocols such as pain management, wound care, and mobilization should be followed consistently. Early mobilization protocols and physical therapy should be integrated into their care plan to facilitate optimal recovery and functional outcomes. Adequate nutrition support, psychological support, and regular follow-up appointments are important for healing and well-being. Patient education and engagement in the care plan ensure a clear understanding of postoperative instructions and adherence to medication regimens, contributing to a smooth transition to a home setting.")
    else:
        st.markdown(f"Management Measures for High-risk Population: For geriatric hip fracture patients identified as high-risk for postoperative in-hospital mortality, a proactive and individualized management approach is essential. Close monitoring of vital signs and regular pain assessment should be implemented to promptly identify any signs of deterioration. Early intervention by a multidisciplinary team, comprehensive medication management, and optimization are crucial to address potential complications and minimize adverse drug events. Specialized rehabilitation programs tailored to their specific needs and comorbidities can optimize functional recovery. Postoperative infection prevention strategies and regular communication among healthcare providers ensure seamless coordination and continuity of care.")

st.subheader('Model information')
st.markdown('The AI prediction model, developed using the eXGBoosting Machine (eXGBM) algorithm, demonstrated outstanding performance in predicting postoperative in-hospital mortality in geriatric hip fracture patients. It exhibited the highest scores in various evaluation metrics, including accuracy, precision, specificity, F1 score, Brier score, and log loss. With an AUC of 0.908, the model showcased excellent discrimination ability. Additionally, the model showed favorable calibration, indicating its accuracy in estimating risk levels. The comprehensive scoring system ranked the eXGBM model as the top-performing model, further validating its predictive capability. This AI model is freely accessible for research purposes, providing a valuable tool for enhancing clinical decision-making in managing geriatric hip fracture patients’ in-hospital mortality risk.')