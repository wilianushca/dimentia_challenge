import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_heartrate = data["le_heartrate"]
le_bloodoxygen = data["le_bloodoxygen"]
le_weight = data["le_weight"]
le_age = data["le_age"] 
le_smoking = data["le_smoking"]
le_gender = data["le_gender"]
le_physicalact = data["le_physicalact"]
le_depression = data["le_depression"] 
le_nutrition = data["le_nutrition"] 
le_sleep = data["le_sleep"] 

def show_predict_page():
    st.title("Software Developer Diabetes Prediction")

    st.write("""### We need some information to predict the risk""")

    smoking = (
        "Current Smoker",
        "Former Smoker",
        "Never Smoked",
    )

    gender = (
        "Male",
        "Female",
    )

    physicalact = (
        "Sedentary",
        "Moderate Activity",
        "Mild Activity",
    )

    depression = (
        "Yes",
        "No",
    )

    nutrition = (
        "Balanced Diet",
        "Mediterranean Diet",
        "Low-Carb Diet",
    )

    sleep = (
        "Good",
        "Poor",
    )


    smoke = st.selectbox("Smoking", smoking)
    gendr = st.selectbox("Gender", gender)
    activity = st.selectbox("Activeness", physicalact)
    depress = st.selectbox("Depression", depression)
    diet = st.selectbox("Diet", nutrition)
    sleephealth = st.selectbox("Sleep Health", sleep)
    
    hrtrate = st.slider("Heart Rate", 0, 150, 10)
    bloodoxy = st.slider("Blood Oxygen Level", 0, 150, 10)
    weight = st.slider("Weight", 0, 300, 50)
    ages = st.slider("Age", 0, 100, 1)

    ok = st.button("Calculate Risk")
    if ok:
        X = np.array([[hrtrate, bloodoxy, weight, ages, smoke, gendr, activity, depress, diet, sleephealth   ]])
        X.iloc[:, 0] = le_heartrate.transform(X.iloc[:, 0])
        X.iloc[:, 1] = le_bloodoxygen.transform(X.iloc[:, 1])
        X.iloc[:, 2] = le_weight.transform(X.iloc[:, 2])
        X.iloc[:, 3] = le_age.transform(X.iloc[:, 3])
        X.iloc[:, 4] = le_smoking.transform(X.iloc[:, 4])
        X.iloc[:, 5] = le_gender.transform(X.iloc[:, 5])
        X.iloc[:, 6] = le_physicalact.transform(X.iloc[:, 6])
        X.iloc[:, 7] = le_depression.transform(X.iloc[:, 7])
        X.iloc[:, 8] = le_nutrition.transform(X.iloc[:, 8])
        X.iloc[:, 9] = le_sleep.transform(X.iloc[:, 9])
        X = X.astype(float)

        risk = regressor.predict(X)
        st.subheader(f"The estimated salary is {risk[0]:.2f}")
