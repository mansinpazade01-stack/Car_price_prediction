import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# -------- STYLE / COLORS --------
st.markdown("""
<style>

/* Main Gradient Background */
.stApp {
    background: linear-gradient(135deg, #3F7C66, #8FD3A8);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F2F2F2;
}

/* Title Styling */
h1 {
    text-align: center;
    color: white;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: white;
    font-size:18px;
}

/* Result Card */
.result-box {
    background-color: #F4F4F4;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙ Technical Specs")

st.sidebar.metric("Model Accuracy (R²)", "0.80")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset Size: 2500 Cars")

st.sidebar.write("---")
st.sidebar.write("Developer")
st.sidebar.write("Mansi Pazade")
st.sidebar.write("AI & Data Science")

# ---------- MAIN TITLE ----------
st.markdown("<h1>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict the resale value of your car instantly</p>", unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# ---------- INPUT SECTION ----------
with col1:
    st.subheader("⚙ Enter Car Details")

    brand = st.selectbox(
        "Select Brand",
        ["Tesla", "BMW", "Audi", "Ford", "Toyota"]
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        max_value=500000,
        value=50000
    )

    car_age = st.number_input(
        "Car Age (years)",
        min_value=0,
        max_value=30,
        value=5
    )

    predict = st.button("🔮 Predict Price")

# ---------- PREDICTION ----------
with col2:
    st.subheader("💰 Prediction Result")

    if predict:

        input_data = np.array([[mileage, car_age]])

        prediction = model.predict(input_data)

        st.markdown(
            f"""
            <div class="result-box">
            💰 Estimated Car Price: ₹ {prediction[0]:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )