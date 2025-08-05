import streamlit as st
import pandas as pd
import joblib
from src.feature_engineering import FeatureEngineer
from src.preprocessing import preprocess_data
import gdown
import os
#predict
# Page config
st.set_page_config(
    page_title="Product Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔗 Step 1: Provide the Google Drive file ID
file_id = "1cWrbb-nKeNt6naJ4PPqsGjUIqvpYlTA4"
output_path = "models/best_pipeline.pkl"

# ✅ Step 2: Create folder if not exists
os.makedirs("models", exist_ok=True)

# 📥 Step 3: Download from Google Drive if not already downloaded
if not os.path.exists(output_path):
    url = "https://drive.google.com/uc?id=1cWrbb-nKeNt6naJ4PPqsGjUIqvpYlTA4"
    gdown.download(url, output_path, quiet=False)

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model, preprocessor = joblib.load(output_path)
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

# Main app
def main():
    st.title("📦 Product Demand Forecasting")
    st.markdown("---")

    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Promotional & Pricing")
        promocodeused = st.selectbox("Promocode Used", options=["Yes", "No"], index=1)
        price = st.number_input("Price ($)", min_value=0.0, value=100.0, step=0.01)
        discount_percent = st.slider("Discount Percent (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        competitorprice = st.number_input("Competitor Price ($)", min_value=0.0, value=110.0, step=0.01)
        adcampaign = st.selectbox("Ad Campaign", options=["None", "Tv", "Online"], index=0)

        st.subheader("📅 Temporal Factors")
        isweekend = st.selectbox("Is Weekend", options=["Yes", "No"], index=1)
        season = st.selectbox("Season", options=["Winter", "Spring", "Summer", "Autumn"], index=0)
        daytype = st.selectbox("Day Type", options=["Weekday", "Holiday", "Weekend"], index=0)

    with col2:
        st.subheader("🌤️ Environmental")
        temp = st.slider("Temperature (°C)", -20.0, 50.0, 20.0, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 200.0, 0.0, 0.1)

        st.subheader("📦 Product Information")
        category = st.text_input("Category", value="Electronics")
        brand = st.selectbox("Brand", options=["BrandA", "BrandB", "BrandC", "BrandD"], index=0)
        material = st.text_input("Material", value="Plastic")
        weight = st.number_input("Weight (kg)", 0.0, value=1.0, step=0.01)
        warranty = st.slider("Warranty (years)", 0.0, 10.0, 1.0, 0.1)
        productrating = st.slider("Product Rating", 1.0, 5.0, 4.0, 0.1)
        launchyear = st.number_input("Launch Year", 1990, 2024, 2023)
        stocklevel = st.number_input("Stock Level", 0, 100, step=1)
        supplierdelay = st.number_input("Supplier Delay (days)", 0, 30, 5)

    st.subheader("🛒 Product Metadata")
    productid = st.text_input("Product ID", value="P0001")
    location = st.text_input("Location", value="L01")
    date = st.date_input("Date")
    finalprice = st.number_input("Final Price ($)", 0.0, value=90.0, step=0.01)
    warehouse = st.selectbox("Warehouse", options=["W1", "W2", "W3"], index=0)
    inventorytype = st.selectbox("Inventory Type", options=["Fresh", "Returned", "Finished Goods", "Repaired"], index=0)

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
                df = pd.DataFrame([input_data])

                # Feature Engineering
                fe = FeatureEngineer()
                df_fe = fe.transform(df)

                # Preprocessing using loaded preprocessor (DON'T fit again)
                df_processed, _ = preprocess_data(df_fe, preprocessor=preprocessor, fit=False)


                # Prediction
                with st.spinner("Generating prediction..."):
                    prediction = model.predict(df_processed)

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
