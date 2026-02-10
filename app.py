import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Bank Marketing Assistant",
    page_icon="🎯",
    layout="centered"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('bank_model_deploy.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found. Please run your notebook to generate 'bank_model_deploy.pkl'.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model, scaler = load_assets()

# --- HEADER ---
st.title("🎯 Marketing Campaign Assistant")
st.markdown("""
Use this tool to identify **high-potential clients** for the Term Deposit campaign.
*Adjust the sliders below to match the client's profile.*
""")
st.markdown("---")

# --- INPUTS ---
st.subheader("👤 Client Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 35)

with col2:
    edu_options = {
        "University Degree": 6,
        "Professional Course": 5,
        "High School": 4,
        "Basic Education (9y)": 3,
        "Basic Education (6y)": 2,
        "Basic Education (4y)": 1,
        "Illiterate": 0
    }
    education_label = st.selectbox("Education Level", list(edu_options.keys()))
    education_numeric = edu_options[education_label]

st.subheader("📞 Engagement History")
col3, col4 = st.columns(2)

with col3:
    campaign = st.slider("Number of Calls (This Campaign)", 1, 10, 2)
    
with col4:
    previous = st.slider("Contacts in Previous Campaigns", 0, 10, 0)

is_new_client = st.checkbox("This is a New Client (Never contacted before)", value=True)
if is_new_client:
    pdays = 999
else:
    pdays = st.slider("Days since last contact", 0, 500, 30)

st.subheader("📈 Economic Context")
st.info("Values simulated based on current market data.")

col5, col6 = st.columns(2)
with col5:
    euribor = st.slider("Market Interest Rate (Euribor)", 0.0, 5.0, 1.2)
    emp_rate = st.slider("Economic Stability Index", -3.5, 1.5, -1.8)

with col6:
    cons_conf = st.slider("Consumer Confidence", -55.0, -25.0, -46.2)
    cons_idx = 93.5 
    num_emp = 5100  

# --- PREDICTION LOGIC ---
if st.button("Analyze Client Potential", type="primary", use_container_width=True):
    
    input_data = pd.DataFrame({
        'age': [age],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'employment_variation_rate': [emp_rate],
        'consumer_price_index': [cons_idx],
        'consumer_confidence_index': [cons_conf],
        'euribor_3month_rate': [euribor],
        'num_employees': [num_emp],
        'education_numeric': [education_numeric]
    })

    try:
        input_scaled = scaler.transform(input_data)
        
        # Manually create interaction term
        scaled_campaign = input_scaled[:, 1]
        scaled_euribor = input_scaled[:, 7]
        econ_pressure = scaled_campaign * scaled_euribor
        
        final_input = np.column_stack((input_scaled, econ_pressure))
        
        try:
            prob = model.predict_proba(final_input)[0][1]
        except ValueError:
            prob = model.predict_proba(input_scaled)[0][1]

        threshold = 0.35
        
        st.divider()
        st.progress(int(prob * 100))
        
        if prob >= threshold:
            st.success(f"✅ **HIGH PRIORITY LEAD (Score: {int(prob*100)})**")
            st.markdown("""
            **Recommendation:** Call this client immediately.
            * **Reasoning:** Strong match between client profile and economic climate.
            """)
            if econ_pressure[0] > 0.5:
                st.warning("⚠️ **Note:** Interest rates are high. Mention 'High Return Savings'.")
                
        else:
            st.error(f"❌ **Low Priority (Score: {int(prob*100)})**")
            st.markdown("""
            **Recommendation:** Do not call.
            * **Reasoning:** Conversion probability is too low.
            """)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")