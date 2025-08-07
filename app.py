# from flask import Flask, render_template, request
# from predict import load_model, predict_demand

# app = Flask(__name__)
# model = load_model()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return render_template('result.html', error="Model not loaded.")

#     try:
#         form_data = request.form.to_dict()
#         prediction, insights = predict_demand(model, form_data)
#         return render_template('result.html', prediction=prediction, insights=insights)
#     except Exception as e:
#         return render_template('result.html', error=f"Prediction failed: {e}")

# if __name__ == '__main__':
#     app.run(debug=True)
import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# Page config
st.set_page_config(
    page_title="Product Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hugging Face file URL (direct link to the raw pkl file)
model_url = "https://huggingface.co/sachin0910/product-demand-pipeline/blob/main/best_pipeline.pkl"
output_path = "models/best_pipeline.pkl"

st.write("📦 Checking model file...")

# If not already downloaded locally, download it from Hugging Face
if not os.path.exists(output_path):
    st.warning("⚠️ Model not found locally. Downloading from Hugging Face...")
    os.makedirs("models", exist_ok=True)
    try:
        urllib.request.urlretrieve(model_url, output_path)
        st.success("✅ Model downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(output_path)
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Main app
def main():
    st.title("📦 Product Demand Forecasting")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Promotional & Pricing")
        promocodeused = st.selectbox("Promocode Used", ["Yes", "No"])
        price = st.number_input("Price ($)", 0.0, value=100.0, step=0.01)
        discount_percent = st.slider("Discount Percent (%)", 0.0, 100.0, 10.0, 0.1)
        competitorprice = st.number_input("Competitor Price ($)", 0.0, value=110.0, step=0.01)
        adcampaign = st.selectbox("Ad Campaign", ["None", "Tv", "Online"])

        st.subheader("📅 Temporal Factors")
        isweekend = st.selectbox("Is Weekend", ["Yes", "No"])
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
        daytype = st.selectbox("Day Type", ["Weekday", "Holiday", "Weekend"])

    with col2:
        st.subheader("🌤️ Environmental")
        temp = st.slider("Temperature (°C)", -20.0, 50.0, 20.0, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 200.0, 0.0, 0.1)

        st.subheader("📦 Product Information")
        category = st.text_input("Category", "Electronics")
        brand = st.selectbox("Brand", ["BrandA", "BrandB", "BrandC", "BrandD"])
        material = st.text_input("Material", "Plastic")
        weight = st.number_input("Weight (kg)", 0.0, value=1.0, step=0.01)
        warranty = st.slider("Warranty (years)", 0.0, 10.0, 1.0, 0.1)
        productrating = st.slider("Product Rating", 1.0, 5.0, 4.0, 0.1)
        launchyear = st.number_input("Launch Year", 1990, 2024, 2023)
        stocklevel = st.number_input("Stock Level", 0, 100)
        supplierdelay = st.number_input("Supplier Delay (days)", 0, 30, 5)

    st.subheader("🛒 Product Metadata")
    productid = st.text_input("Product ID", "P0001")
    location = st.text_input("Location", "L01")
    date = st.date_input("Date")
    finalprice = st.number_input("Final Price ($)", 0.0, value=90.0, step=0.01)
    warehouse = st.selectbox("Warehouse", ["W1", "W2", "W3"])
    inventorytype = st.selectbox("Inventory Type", ["Fresh", "Returned", "Finished Goods", "Repaired"])

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
            input_data = {
                "productid": productid,
                "location": location,
                "date": pd.to_datetime(date),
                "promocodeused": promocodeused,
                "price": price,
                "discount_percent": discount_percent,
                "competitorprice": competitorprice,
                "adcampaign": adcampaign,
                "finalprice": finalprice,
                "isweekend": isweekend,
                "season": season,
                "daytype": daytype,
                "temp(c)": temp,
                "rainfall(mm)": rainfall,
                "category": category,
                "brand": brand,
                "material": material,
                "weight(kg)": weight,
                "warranty(years)": warranty,
                "productrating": productrating,
                "launchyear": launchyear,
                "stocklevel": stocklevel,
                "supplierdelay(days)": supplierdelay,
                "warehouse": warehouse,
                "inventorytype": inventorytype
            }

            try:
                # import warnings
                # warnings.filterwarnings("ignore")
                df = pd.DataFrame([input_data])

                with st.spinner("Generating prediction..."):
                    prediction = model.predict(df)

                st.success("✅ Prediction Generated Successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📦 Predicted Units Sold", f"{prediction[0]:.0f}", delta=f"{prediction[0] - stocklevel:.0f} vs Stock")
                with col2:
                    revenue = prediction[0] * price
                    st.metric("💰 Estimated Revenue", f"${revenue:,.2f}")
                with col3:
                    demand_level = "High" if prediction[0] > stocklevel else "Normal" if prediction[0] > stocklevel * 0.5 else "Low"
                    st.metric("📊 Demand Level", demand_level)

                st.markdown("---")
                st.subheader("📈 Business Insights")
                insights_col1, insights_col2 = st.columns(2)
                with insights_col1:
                    if prediction[0] > stocklevel:
                        st.warning(f"⚠️ **High Demand Alert**: Predicted demand ({prediction[0]:.0f}) exceeds stock ({stocklevel})")
                        st.info(f"💡 **Recommendation**: Increase stock by {prediction[0] - stocklevel:.0f} units")
                    else:
                        st.success("✅ **Stock Sufficient**: Stock meets demand")
                with insights_col2:
                    profit_margin = ((price - (price * discount_percent / 100)) / price) * 100
                    st.info(f"📊 **Profit Margin**: {profit_margin:.1f}%")
                    if competitorprice > price:
                        st.success("🎯 **Competitive Advantage**: Your price is lower than competitor")
                    else:
                        st.warning("⚡ **Price Pressure**: Competitor has lower price")

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.error("Please check your input or model files.")

# Sidebar
def show_sidebar():
    st.sidebar.title("ℹ️ About")
    st.sidebar.info(
        "This app uses machine learning to predict product demand based on pricing, promotions, seasonality, and more."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Model Features")
    for feature in [
        "✅ Promotional factors",
        "✅ Pricing strategies",
        "✅ Temporal patterns",
        "✅ Environmental conditions",
        "✅ Product specs",
        "✅ Stock management"
    ]:
        st.sidebar.markdown(feature)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Path**: `models/best_pipeline.pkl`")

if __name__ == "__main__":
    show_sidebar()
    main()

