import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Diabetes Prediction App

This app predicts the **Diabetes** reason!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0.0, 8,5, 17.0)
    Glucose = st.sidebar.slider('Glucose', 0.0, 100.0, 200.0)
    BloodPressure = st.sidebar.slider('BloodPressure', 0.0, 75.0, 150.0)
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(diabetes.target_names)

st.subheader('Prediction')
st.write(diabetes.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
