"""
Streamlit App — Tourism Wellness Package Predictor
Loads the XGBoost pipeline from Hugging Face and serves purchase predictions.
"""

import os, joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_USER       = os.environ.get("HF_USERNAME", "your-hf-username")
MODEL_REPO_ID = f"{HF_USER}/tourism-package-model"

@st.cache_resource(show_spinner="Loading model from Hugging Face ...")
def load_model():
    path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="model_building/tourism_model.pkl",
        repo_type="model",
        token=HF_TOKEN or None,
    )
    return joblib.load(path)

model = load_model()

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Wellness Tourism Package — Purchase Predictor")
st.markdown("Enter customer details to predict purchase likelihood.")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Customer Profile")
    age               = st.slider("Age", 18, 80, 35)
    gender            = st.selectbox("Gender", ["Male", "Female"])
    marital_status    = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation        = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation       = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income    = st.number_input("Monthly Income", min_value=5000, max_value=100000, value=20000, step=1000)

with col2:
    st.subheader("Travel Preferences")
    city_tier                 = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact           = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched           = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    preferred_property_star   = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_trips           = st.slider("Number of Trips", 0, 20, 3)
    passport                  = st.radio("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car                   = st.radio("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col3:
    st.subheader("Pitch & Visit Details")
    number_of_person_visiting   = st.slider("Persons Visiting", 1, 10, 2)
    number_of_children_visiting = st.slider("Children (< 5 yrs)", 0, 5, 0)
    pitch_satisfaction_score    = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    number_of_followups         = st.slider("Number of Follow-ups", 0, 6, 2)
    duration_of_pitch           = st.slider("Duration of Pitch (mins)", 5, 60, 20)

st.divider()
if st.button("Predict Purchase Likelihood", use_container_width=True):
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting, "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status, "NumberOfTrips": number_of_trips,
        "Passport": passport, "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car, "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation, "MonthlyIncome": monthly_income,
    }])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Result")
    if pred == 1:
        st.success(f"✅ LIKELY to purchase — probability: {prob:.1%}")
    else:
        st.warning(f"❌ UNLIKELY to purchase — probability: {prob:.1%}")
    with st.expander("View Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))
