import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦")

# --- LOAD MODEL & SCALER ---
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_assets():
    try:
        model = joblib.load('bank_model_deploy.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files not found. Make sure 'bank_model_deploy.pkl' and 'scaler.pkl' are in the same folder.")
        return None, None

model, scaler = load_assets()

# --- TITLE & DESCRIPTION ---
st.title("🏦 Term Deposit Prediction System")
st.markdown("""
This AI tool predicts whether a bank client will subscribe to a Term Deposit.
Adjust the sliders below to simulate a customer profile.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Customer Profile")

# 1. Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", 18, 90, 35)
education_numeric = st.sidebar.selectbox(
    "Education Level", 
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Illiterate", "Basic 4y", "Basic 6y", "Basic 9y", "High School", "Professional", "University"][x],
    index=6
)

# 2. Campaign History
st.sidebar.subheader("Campaign History")
campaign = st.sidebar.slider("Number of Contacts (This Campaign)", 1, 20, 2)
pdays = st.sidebar.number_input("Days since last contact (999=Never)", -1, 999, 999)
previous = st.sidebar.slider("Previous Contacts", 0, 10, 0)

# 3. Economic Indicators (Macro)
st.sidebar.subheader("Economic Context")
emp_rate = st.sidebar.slider("Employment Var. Rate", -4.0, 2.0, -1.8)
cons_price = st.sidebar.slider("Consumer Price Index", 92.0, 95.0, 93.0)
cons_conf = st.sidebar.slider("Consumer Conf. Index", -55.0, -25.0, -40.0)
euribor = st.sidebar.slider("Euribor 3 Month Rate", 0.0, 6.0, 1.2)
employees = st.sidebar.number_input("Number of Employees", 4900, 5300, 5099)

# --- PREDICTION LOGIC ---
if st.button("Predict Subscription Probability", type="primary"):
    if model is not None:
        # 1. Organize Inputs exactly like training data
        input_data = pd.DataFrame({
            'age': [age],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'employment_variation_rate': [emp_rate],
            'consumer_price_index': [cons_price],
            'consumer_confidence_index': [cons_conf],
            'euribor_3month_rate': [euribor],
            'num_employees': [employees],
            'education_numeric': [education_numeric]
        })

        # 2. SCALE the data (using the saved scaler)
        # Note: Scaler expects these exact 10 columns in this order
        input_scaled = scaler.transform(input_data)
        
        # 3. FEATURE ENGINEERING (The "Smart" Part)
        # We need to recreate 'Econ_Pressure' using the SCALED values.
        # Column Indices: campaign is index 1, euribor is index 7
        scaled_campaign = input_scaled[:, 1]
        scaled_euribor = input_scaled[:, 7]
        econ_pressure = scaled_campaign * scaled_euribor
        
        # Add this new feature to the array
        # We assume the model expects 11 features (10 original + 1 interaction)
        final_input = np.column_stack((input_scaled, econ_pressure))

        # 4. PREDICT
        # Use our optimized threshold (0.35)
        probability = model.predict_proba(final_input)[0][1]
        threshold = 0.35
        
        # --- DISPLAY RESULTS ---
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Subscription Probability", f"{probability:.1%}")
            
        with col2:
            if probability >= threshold:
                st.success("Recommendation: **CALL CLIENT**")
                st.write("This customer has a **High Potential** to subscribe.")
            else:
                st.error("Recommendation: **DO NOT CALL**")
                st.write("Low conversion probability. Save resources.")
        
        # Visual Gauge
        st.progress(int(probability * 100))
        
        # Explainability (Why?)
        st.subheader("Why this prediction?")
        if econ_pressure[0] > 1.0: # High pressure
             st.info(f"💡 **Insight:** High 'Economic Pressure' detected (Euribor {euribor} x Calls {campaign}). Customer may be annoyed by persistence in this economy.")