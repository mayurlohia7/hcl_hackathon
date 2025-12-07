import streamlit as st
import pickle
import numpy as np

# Load model
with open("rf_return_risk_model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("📦 Return Risk Prediction App")

# User inputs
price = st.number_input("Price")
discount = st.number_input("Discount (%)")
tax_rate = st.number_input("Tax Rate")
stock_level = st.number_input("Stock Level")
shipping_cost = st.number_input("Shipping Cost")
popularity_index = st.number_input("Popularity Index")
product_id_freq = st.number_input("Product ID Frequency")
supplier_id_freq = st.number_input("Supplier ID Frequency")

# Prepare input
input_data = np.array([[  
    price, discount, tax_rate, stock_level, 
    shipping_cost, popularity_index,
    product_id_freq, supplier_id_freq
]])


if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Return Risk (Prob: {prob:.2f})")
    else:
        st.success(f"✅ Low Return Risk (Prob: {prob:.2f})")
