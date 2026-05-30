import streamlit as st
import pandas as pd
import pickle

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]

df = None

for enc in encodings:
    try:
        df = pd.read_csv(
            "data/laptop_price.csv",
            encoding=enc
        )
        break
    except:
        pass

# ==========================================================
# LOAD MODEL
# ==========================================================

model = pickle.load(
    open("models/stacking_regressor.pkl", "rb")
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <h1 style='text-align:center;color:#4F8BF9'>
    💻 Laptop Price Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center;color:gray'>
    Stacking Regression Model
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# INPUTS
# ==========================================================

st.subheader("📝 Laptop Specifications")

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "Company",
        sorted(df["Company"].unique())
    )

    product = st.selectbox(
        "Product",
        sorted(df["Product"].unique())
    )

    typename = st.selectbox(
        "Laptop Type",
        sorted(df["TypeName"].unique())
    )

    inches = st.selectbox(
        "Screen Size",
        sorted(df["Inches"].unique())
    )

    screen_resolution = st.selectbox(
        "Screen Resolution",
        sorted(df["ScreenResolution"].unique())
    )

    cpu = st.selectbox(
        "CPU",
        sorted(df["Cpu"].unique())
    )

with col2:

    ram = st.selectbox(
        "RAM",
        sorted(df["Ram"].unique())
    )

    memory = st.selectbox(
        "Memory",
        sorted(df["Memory"].unique())
    )

    gpu = st.selectbox(
        "GPU",
        sorted(df["Gpu"].unique())
    )

    opsys = st.selectbox(
        "Operating System",
        sorted(df["OpSys"].unique())
    )

    weight = st.selectbox(
        "Weight",
        sorted(df["Weight"].unique())
    )

# ==========================================================
# PREDICT
# ==========================================================

if st.button(
    "🔍 Predict Laptop Price",
    use_container_width=True
):

    input_df = pd.DataFrame({
        "Company": [company],
        "Product": [product],
        "TypeName": [typename],
        "Inches": [inches],
        "ScreenResolution": [screen_resolution],
        "Cpu": [cpu],
        "Ram": [ram],
        "Memory": [memory],
        "Gpu": [gpu],
        "OpSys": [opsys],
        "Weight": [weight]
    })

    prediction = model.predict(input_df)[0]

    st.divider()

    st.success(
        f"💰 Predicted Laptop Price: € {prediction:,.2f}"
    )

    # ======================================================
    # METRICS
    # ======================================================

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Estimated Price",
            f"€ {prediction:,.2f}"
        )

    with c2:

        if prediction < 700:
            category = "Budget"

        elif prediction < 1500:
            category = "Mid Range"

        else:
            category = "Premium"

        st.metric(
            "Category",
            category
        )

    st.divider()

    # ======================================================
    # PRICE METER
    # ======================================================

    st.subheader("💲 Price Indicator")

    progress = min(
        int((prediction / 4000) * 100),
        100
    )

    st.progress(progress)

    st.write(
        f"Price Position: {progress}%"
    )

    st.divider()

    # ======================================================
    # SUMMARY TABLE
    # ======================================================

    st.subheader("📋 Selected Configuration")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )