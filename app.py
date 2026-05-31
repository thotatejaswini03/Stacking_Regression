import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Stacking Regression Predictor",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

div[data-testid="stMetric"]{
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("📈 Stacking Regression Predictor")

st.markdown("""
### Insurance Charges Prediction using Stacking Regression
""")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================

for col in ["sex", "smoker", "region"]:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# TRAIN MODELS
# =====================================================

@st.cache_resource
def train_models():

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    gbr = GradientBoostingRegressor(
        random_state=42
    )

    rf.fit(X_train, y_train)
    gbr.fit(X_train, y_train)

    base_models = [
        ("rf", rf),
        ("gbr", gbr)
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression()
    )

    stacking_model.fit(X_train, y_train)

    return rf, gbr, stacking_model

rf_model, gbr_model, stacking_model = train_models()

# =====================================================
# PREDICTIONS
# =====================================================

rf_pred = rf_model.predict(X_test)
gbr_pred = gbr_model.predict(X_test)
stack_pred = stacking_model.predict(X_test)

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(y_test, stack_pred)
rmse = np.sqrt(mean_squared_error(y_test, stack_pred))
r2 = r2_score(y_test, stack_pred)

# =====================================================
# KPI SECTION
# =====================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("📁 Records", len(df))
m2.metric("📉 MAE", f"{mae:.2f}")
m3.metric("📊 RMSE", f"{rmse:.2f}")
m4.metric("🎯 R² Score", f"{r2:.4f}")

# =====================================================
# MAIN SECTION
# =====================================================

left, right = st.columns([1,1])

# =====================================================
# LEFT PANEL
# =====================================================

with left:

    st.subheader("📊 Performance Comparison")

    rf_r2 = r2_score(y_test, rf_pred)
    gbr_r2 = r2_score(y_test, gbr_pred)
    stack_r2 = r2_score(y_test, stack_pred)

    comparison_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Gradient Boosting",
            "Stacking Regressor"
        ],
        "R² Score": [
            rf_r2,
            gbr_r2,
            stack_r2
        ]
    })

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(2.2, 1.3))

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df["R² Score"]
    )

    ax.set_title(
        "R² Comparison",
        fontsize=7
    )

    ax.tick_params(
        axis="x",
        labelsize=4
    )

    ax.tick_params(
        axis="y",
        labelsize=4
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=4
        )

    plt.tight_layout()

    st.pyplot(fig)

# =====================================================
# RIGHT PANEL
# =====================================================

with right:

    st.subheader("💰 Predict Insurance Charges")

    with st.container(border=True):

        age = st.number_input(
            "Age",
            18,
            100,
            30
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            60.0,
            25.0
        )

        children = st.number_input(
            "Children",
            0,
            10,
            0
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

        if st.button(
            "Predict Charges",
            use_container_width=True
        ):

            sex_map = {
                "female": 0,
                "male": 1
            }

            smoker_map = {
                "no": 0,
                "yes": 1
            }

            region_map = {
                "northeast": 0,
                "northwest": 1,
                "southeast": 2,
                "southwest": 3
            }

            sample = pd.DataFrame([{
                "age": age,
                "sex": sex_map[sex],
                "bmi": bmi,
                "children": children,
                "smoker": smoker_map[smoker],
                "region": region_map[region]
            }])

            prediction = stacking_model.predict(sample)[0]

            st.success(
                f"Predicted Insurance Charges: ${prediction:,.2f}"
            )

# =====================================================
# MODEL EVALUATION
# =====================================================

with st.expander("📈 Model Evaluation"):

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.4f}")

    fig2, ax2 = plt.subplots(figsize=(2.5, 1.6))

    ax2.scatter(
        y_test,
        stack_pred,
        alpha=0.4,
        s=6
    )

    ax2.set_title(
        "Actual vs Predicted",
        fontsize=7
    )

    ax2.tick_params(
        axis='both',
        labelsize=5
    )

    st.pyplot(fig2)

# =====================================================
# DATASET OVERVIEW
# =====================================================

with st.expander("📂 Dataset Overview"):

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Missing Values: {df.isnull().sum().sum()}")

    st.dataframe(
        df.head(),
        height=150
    )