import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AI Delivery Risk System 🇮🇳",
    page_icon="🚚",
    layout="wide"
)

# ======================================
# LOAD ARTIFACTS
# ======================================
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_artifacts():
    # Only the pipeline needs to be loaded!
    model = joblib.load(MODEL_PATH)
    return model

model = load_artifacts()

# ======================================
# VISITOR COUNTER
# ======================================
def update_visitors():
    file = "visitors.txt"
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("0")

    with open(file, "r") as f:
        count = int(f.read())

    count += 1

    with open(file, "w") as f:
        f.write(str(count))

    return count

visits = update_visitors()

# ======================================
# HEADER
# ======================================
st.markdown(
    "<h1 style='text-align:center;color:#00F5D4;'>🚚 AI Delivery Risk Prediction System 🇮🇳</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<div style='background:#162447;padding:10px;border-radius:8px;text-align:center;color:white;'>👥 Total Visitors: <b>{visits}</b></div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ======================================
# MAIN UI
# ======================================
st.subheader("📦 Shipment Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    market = st.selectbox("Market", ["LATAM", "US", "Europe", "Asia"])
    ship_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
    order_region = st.selectbox("Order Region", ["West", "East", "Central", "South"])

with col2:
    customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
    quantity = st.slider("Order Quantity", 1, 10, 2)
    days = st.slider("Scheduled Shipment Days", 1, 10, 3)

with col3:
    price = st.slider("Product Price (Sales)", 10.0, 500.0, 100.0)
    discount = st.slider("Discount Rate", 0.0, 1.0, 0.1)

st.markdown("---")

# ======================================
# BUILD INPUT 
# ======================================
def build_input():
    # Because we used a pipeline in Data.py, we just need a standard DataFrame 
    # with the exact column names used during training.
    data = {
        "Days for shipment (scheduled)": [days],
        "Shipping Mode": [ship_mode],
        "Customer Segment": [customer_segment],
        "Market": [market],
        "Order Region": [order_region],
        "Order Item Quantity": [quantity],
        "Sales": [price],
        "Order Item Discount Rate": [discount]
    }
    return pd.DataFrame(data)

# ======================================
# PREDICTION
# ======================================
# Fixed Indentation block
if st.button("🔍 Predict Delivery Risk"):

    input_df = build_input()

    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    confidence = np.max(probs) * 100

    st.subheader("📊 Prediction Result")

    # Target is binary: 0 (On-Time) or 1 (Late)
    labels = ["On-Time Delivery", "Late Delivery Risk"]
    pred_label = labels[int(pred)]

    # Result display
    if pred == 0:
        st.success(f"✅ {pred_label} ({confidence:.0f}% confidence)")
    else:
        st.warning(f"⚠ {pred_label} ({confidence:.0f}% confidence)")

    # Probability chart
    prob_df = pd.DataFrame({"Probability": probs}, index=labels)
    st.bar_chart(prob_df)

    st.info(
        f"📌 Market: {market} | Mode: {ship_mode} | Region: {order_region} | Qty: {quantity} | Price: {price} | Discount: {discount} | Days: {days}"
    )

st.markdown("---")
st.markdown("<center>Built with ❤️ using Machine Learning</center>", unsafe_allow_html=True)