import streamlit as st
import pickle
import numpy as np

with open('lr.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival prediction")

feature1 = st.number_input("Enter Passanger id:")
feature2 = st.number_input("Enter Pclass:")
feature3= st.number_input("Enter Sex:")
feature4 = st.number_input("Enter Age:")
feature5 = st.number_input("Enter Sibsp:")
feature6 = st.number_input("Enter Parch:")
feature7 = st.number_input("Enter Ticket:")
feature8 = st.number_input("Enter Fare:")
feature9 = st.number_input("Enter Cabin:")
feature10 = st.number_input("Enter Embarked:")

if st.button("Predict"):
    prediction = model.predict(np.array([[feature1,feature2,feature3,feature4,feature5,feature6,
                                            feature7,feature8,feature9,feature10]]))

    result = "Survived" if prediction == 1 else "Did not Survive"

    st.markdown(f"### 🧠 Prediction: **{result}**")
    # st.write(f"Model confidence (probability of 'Pass'): **{prediction[0]:.2f}**")
