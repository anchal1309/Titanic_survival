import streamlit as st
import pickle
import numpy as np

with open('lr.pkl', 'rb') as f:
    model = pickle.load(f)

with open("le_sex.pkl", "rb") as f:
    le_sex = pickle.load(f)

decoded_sex = le_sex.inverse_transform([1])

with open("le_cabin.pkl", "rb") as f:
    le_cabin = pickle.load(f)

decoded_cabin = le_cabin.inverse_transform([1])

with open("le_embark.pkl", "rb") as f:
    le_embark = pickle.load(f)

decoded_embark = le_embark.inverse_transform([1])

with open("le_ticket.pkl", "rb") as f:
    le_ticket = pickle.load(f)

decoded_ticket = le_ticket.inverse_transform([1])

st.title("Titanic Survival prediction")

pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, step=1)
fare = st.number_input("Fare", value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical values using loaded LabelEncoders
sex_encoded = le_sex.transform([sex])[0]          # 'male' -> 1
embarked_encoded = le_embark.transform([embarked])[0]  # 'C' -> 2 (example)

# Build numeric input for the model
X_input = np.array([[pclass,               # already an int
                     sex_encoded,          # now an int
                     age,                  # float
                     sibsp,                # int
                     parch,                # int
                     fare,                 # float
                     embarked_encoded]],   # int
                   dtype=float)



if st.button("Predict Survival"):
    prediction = model.predict(X_input) 
    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to **Survive**.")
    else:
        st.error("😢 The passenger is likely to **Not Survive**.")

# feature1 = st.number_input("Enter Passanger id:")
# feature2 = st.number_input("Enter Pclass:")
# feature3= st.number_input("Enter Sex:")
# feature4 = st.number_input("Enter Age:")
# feature5 = st.number_input("Enter Sibsp:")
# feature6 = st.number_input("Enter Parch:")
# feature7 = st.number_input("Enter Ticket:")
# feature8 = st.number_input("Enter Fare:")
# feature9 = st.number_input("Enter Cabin:"
# feature10 = st.number_input("Enter Embarked:"))

# if st.button("Predict"):
#     prediction = model.predict(np.array([[feature1,feature2,feature3,feature4,feature5,feature6,
#                                             feature7,feature8,feature9,feature10]]))

#     result = "Survived" if prediction == 1 else "Did not Survive"

#     st.markdown(f"### 🧠 Prediction: **{result}**")
    # st.write(f"Model confidence (probability of 'Pass'): **{prediction[0]:.2f}**")
