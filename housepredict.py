# HousePredict.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# Load dataset & train model
# ==========================
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏡 House Price Prediction Dashboard")

# --- Input form di sidebar ---
st.sidebar.header("📝 Input Data Rumah")
medinc = st.sidebar.number_input("Median Income (x10.000 USD)", 1.0, 15.0, 6.0, step=0.1)
houseage = st.sidebar.slider("House Age (tahun)", 1, 50, 20)
averooms = st.sidebar.slider("Average Rooms", 1, 10, 5, step=1)
avebedrms = st.sidebar.slider("Average Bedrooms", 1, 5, 1, step=1)
population = st.sidebar.number_input("Population", 100, 5000, 800, step=50)
aveoccup = st.sidebar.slider("Average Occupancy", 1.0, 6.0, 3.0, step=0.1)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0, step=0.1)
longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -118.0, step=0.1)

input_data = pd.DataFrame({
    "MedInc": [medinc],
    "HouseAge": [houseage],
    "AveRooms": [averooms],
    "AveBedrms": [avebedrms],
    "Population": [population],
    "AveOccup": [aveoccup],
    "Latitude": [latitude],
    "Longitude": [longitude],
})
prediksi = model.predict(input_data)[0]

# --- Highlight Prediksi ---
st.subheader("💰 Prediksi Harga Rumah")
st.metric(label="Estimated Price", value=f"${prediksi*100000:,.2f}")

# ==========================
# Bagian bawah: Evaluasi & Grafik
# ==========================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Model Evaluation")
    c1, c2 = st.columns(2)
    c1.metric("MSE", f"{mse:.3f}")
    c2.metric("R²", f"{r2:.3f}")

    # Prediksi vs Aktual
    st.subheader("📈 Prediksi vs Aktual")
    fig_scatter, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel("Harga Aktual (100k USD)")
    ax.set_ylabel("Harga Prediksi (100k USD)")
    st.pyplot(fig_scatter)

    # Tabel Harga Rumah per Wilayah
    st.subheader("📊 Harga Rumah per Wilayah (Top 10)")
    df_sample = df.copy()
    df_sample["Predicted"] = model.predict(X)
    df_group = df_sample.groupby("Latitude")[["MedHouseVal", "Predicted"]].mean().reset_index()
    st.dataframe(df_group.head(10).style.format("{:.2f}"), use_container_width=True)

with col2:
    st.subheader("⚖️ Feature Importance")
    importance = pd.Series(model.coef_, index=X.columns).sort_values()
    fig_imp, ax = plt.subplots(figsize=(5, 4))
    importance.plot(kind="barh", color="skyblue", ax=ax)
    st.pyplot(fig_imp)

st.caption("Dibuat dengan ❤️ menggunakan Streamlit & scikit-learn")
