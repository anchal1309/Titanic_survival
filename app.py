import streamlit as st
import pickle
import os
import numpy as np

for fname in ["lr.pkl","le_sex.pkl","le_cabin.pkl","le_ticket.pkl","le_embark.pkl"]:
    st.write(f"{fname}: {'✅ Found' if os.path.exists(fname) else '❌ Missing'}")

if not os.path.exists("lr.pkl"):
    st.error("❌ Model file 'lr.pkl' not found. The app cannot proceed.")
    st.stop()

with open('lr.pkl', 'rb') as f:
    model = pickle.load(f)

with open("le_sex.pkl", "rb") as f:
    le_sex = pickle.load(f)

with open("le_cabin.pkl", "rb") as f:
    le_cabin = pickle.load(f)

with open("le_embark.pkl", "rb") as f:
    le_embark = pickle.load(f)

with open("le_ticket.pkl", "rb") as f:
    le_ticket = pickle.load(f)

st.title("Titanic Survival Prediction")

# Inputs
Passenger_ID = st.number_input("Passenger ID", min_value=0, step=1)
pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, step=1)
ticket = st.selectbox("Ticket number",['330911', '363272', '240276', '315154', '3101298', '7538',
       '330972', '248738', '2657', 'A/4 48871', '349220', '694', '21228',
       '24065', 'W.E.P. 5734', 'SC/PARIS 2167', '233734', '2692',
       'STON/O2. 3101270', '2696', 'PC 17603', 'C 17368', 'PC 17598',
       'PC 17597', 'PC 17608', 'A/5. 3337', '113509', '2698', '113054',
       '2662', 'SC/AH 3085', 'C.A. 31029', 'C.A. 2315', 'W./C. 6607',
       '13236', '2682', '342712', '315087', '345768', '1601', '349256',
       '113778', 'SOTON/O.Q. 3101263', '237249', '11753',
       'STON/O 2. 3101291', 'PC 17594', '370374', '11813', 'C.A. 37671',
       '13695', 'SC/PARIS 2168', '29105', '19950', 'SC/A.3 2861',
       '382652', '349230', '348122', '386525', '349232', '237216',
       '347090', '334914', 'F.C.C. 13534', '330963', '113796', '2543',
       '382653', '349211', '3101297', 'PC 17562', '113503', '359306',
       '11770', '248744', '368702', '2678', 'PC 17483', '19924', '349238',
       '240261', '2660', '330844', 'A/4 31416', '364856', '29103',
       '347072', '345498', 'F.C. 12750', '376563', '13905', '350033',
       '19877', 'STON/O 2. 3101268', '347471', 'A./5. 3338', '11778',
       '228414', '365235', '347070', '2625', 'C 4001', '330920', '383162',
       '3410', '248734', '237734', '330968', 'PC 17531', '329944', '2680',
       '2681', 'PP 9549', '13050', 'SC/AH 29037', 'C.A. 33595', '367227',
       '392095', '368783', '371362', '350045', '367226', '211535',
       '342441', 'STON/OQ. 369943', '113780', '4133', '2621', '349226',
       '350409', '2656', '248659', 'SOTON/OQ 392083', 'CA 2144', '113781',
       '244358', '17475', '345763', '17463', 'SC/A4 23568', '113791',
       '250651', '11767', '349255', '3701', '350405', '347077',
       'S.O./P.P. 752', '347469', '110489', 'SOTON/O.Q. 3101315',
       '335432', '2650', '220844', '343271', '237393', '315153',
       'PC 17591', 'W./C. 6608', '17770', '7548', 'S.O./P.P. 251', '2670',
       '2673', '29750', 'C.A. 33112', '230136', 'PC 17756', '233478',
       '113773', '7935', 'PC 17558', '239059', 'S.O./P.P. 2', 'A/4 48873',
       'CA. 2343', '28221', '226875', '111163', 'A/5. 851', '235509',
       '28220', '347465', '16966', '347066', 'C.A. 31030', '65305',
       '36568', '347080', 'PC 17757', '26360', 'C.A. 34050', 'F.C. 12998',
       '9232', '28034', 'PC 17613', '349250', 'SOTON/O.Q. 3101308',
       'S.O.C. 14879', '347091', '113038', '330924', '36928', '32302',
       'SC/PARIS 2148', '342684', 'W./C. 14266', '350053', 'PC 17606',
       '2661', '350054', '370368', 'C.A. 6212', '242963', '220845',
       '113795', '3101266', '330971', 'PC 17599', '350416', '110813',
       '2679', '250650', 'PC 17761', '112377', '237789', '3470', '17464',
       '26707', 'C.A. 34651', 'SOTON/O2 3101284', '13508', '7266',
       '345775', 'C.A. 42795', 'AQ/4 3130', '363611', '28404', '345501',
       '345572', '350410', 'C.A. 34644', '349235', '112051', 'C.A. 49867',
       'A. 2. 39186', '315095', '368573', '370371', '2676', '236853',
       'SC 14888', '2926', 'CA 31352', 'W./C. 14260', '315085', '364859',
       '370129', 'A/5 21175', 'SOTON/O.Q. 3101314', '2655', 'A/5 1478',
       'PC 17607', '382650', '2652', '33638', '345771', '349202',
       'SC/Paris 2123', '113801', '347467', '347079', '237735', '315092',
       '383123', '112901', '392091', '12749', '350026', '315091', '2658',
       'LP 1588', '368364', 'PC 17760', 'AQ/3. 30631', 'PC 17569',
       '28004', '350408', '347075', '2654', '244368', '113790', '24160',
       'SOTON/O.Q. 3101309', 'PC 17585', '2003', '236854', 'PC 17580',
       '2684', '2653', '349229', '110469', '244360', '2675', '2622',
       'C.A. 15185', '350403', 'PC 17755', '348125', '237670', '2688',
       '248726', 'F.C.C. 13528', 'PC 17759', 'F.C.C. 13540', '113044',
       '11769', '1222', '368402', '349910', 'S.C./PARIS 2079', '315083',
       '11765', '2689', '3101295', '112378', 'SC/PARIS 2147', '28133',
       '112058', '248746', '315152', '29107', '680', '366713', '330910',
       '364498', '376566', 'SC/PARIS 2159', '349911', '244346', '364858',
       '349909', 'PC 17592', 'C.A. 2673', 'C.A. 30769', '371109', '13567',
       '347065', '21332', '28664', '113059', '17765', 'SC/PARIS 2166',
       '28666', '334915', '365237', '19928', '347086', 'A.5. 3236',
       'PC 17758', 'SOTON/O.Q. 3101262', '359309', '2668'])
fare = st.number_input("Fare", value=50.0)
cabin = st.selectbox("Cabin number",['B45', 'E31', 'B57', 'B59', 'B63', 'B66', 'B36', 'A21', 'C78', 'D34',
       'D19', 'A9', 'D15', 'C31', 'C23', 'C25', 'C27', 'F', 'G63', 'B61', 'C53',
       'D43', 'C130', 'C132', 'C101', 'C55 C57', 'B71', 'C46', 'C116',
       'F', 'A29', 'G6', 'C6', 'C28', 'C51', 'E46', 'C54', 'C97', 'D22',
       'B10', 'F4', 'E45', 'E52', 'D30', 'B58',' B60', 'E34', 'C62 C64',
       'A11', 'B11', 'C80', 'F33', 'C85', 'D37', 'C86', 'D21', 'C89',
       'F E46', 'A34', 'D', 'B26', 'C22 C26', 'B69', 'C32', 'B78',
       'F E57', 'F2', 'A18', 'C106', 'B51 B53 B55', 'D10 D12', 'E60',
       'E50', 'E39 E41', 'B52 B54 B56', 'C39', 'B24', 'D28', 'B41', 'C7',
       'D40', 'D38', 'C105'])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict Survival"):
    # Check if required string fields are not empty
    if not ticket or not cabin:
        st.warning("Please fill in both Ticket and Cabin fields.")
    else:
        try:
            # Encode categorical variables
            sex_encoded = le_sex.transform([sex])[0]
            embarked_encoded = le_embark.transform([embarked])[0]
            cabin_encoded = le_cabin.transform([cabin])[0]
            ticket_encoded = le_ticket.transform([ticket])[0]

            # Combine all inputs
            X_input = np.array([[int(Passenger_ID), int(pclass),
                                 sex_encoded, int(age), int(sibsp),
                                 int(parch), ticket_encoded, float(fare),
                                 cabin_encoded, embarked_encoded]])

            prediction = model.predict(X_input)

            if prediction[0] == 1:
                st.success("🎉 The passenger is likely to **Survive**.")
            else:
                st.error("😢 The passenger is likely to **Not Survive**.")

        except ValueError as e:
            st.error(f"Input error: {e}")
