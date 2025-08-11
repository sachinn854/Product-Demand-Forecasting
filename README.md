APP LINK :-  https://product-demand-forecasting.onrender.com/



# 🛒 Product Demand Forecasting

A robust Machine Learning pipeline built to forecast product demand using advanced feature engineering, preprocessing, and model optimization techniques. The pipeline is designed to ensure reproducibility, modularity, and scalability.

---

## 📁 Project Structure

```bash
Product-Demand-Forecasting/
├── 1_DATA/                         # Raw data
├── 2_DATA CLEANING/               # Initial cleaning scripts/output
├── 3_CLEANED DATA/                # Cleaned datasets
├── 4_EDA/                         # Exploratory Data Analysis
├── 5_verification of data sets/   # Dataset verification
├── 6_Merge File/                  # Merging multiple datasets
├── 7_Feature Selection/           # Feature importance, removal
├── 8_Encoding/                    # Encoding categorical features
├── model/                         # Model outputs
├── models/                        # Saved ML models
├── src/
│   ├── feature_engineering.py     # Custom feature generation
│   ├── model_train.py             # Model training and evaluation
│   ├── pipeline.py                # End-to-end pipeline runner
│   ├── preprocessing.py           # Preprocessing logic (nulls, encoding)
│   └── __init__.py
├── test/                          # For future test cases
├── best_model.pkl                 # Serialized best model
├── main.py                        # Main entry point
├── All_command.txt                # Helpful CLI commands
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

```

## 💡 Problem Statement

The goal is to accurately **forecast the demand** for products based on features like:
- Product category and sub-category
- Warehouse location
- Time period
- Order priority and quantity
- Shipment mode and cost
- Discount and profit margins
- And many more real-world business features

---

## 🔎 Data Preprocessing & EDA

Before building the model:
- ✅ Missing values were handled
- ✅ Duplicates removed
- ✅ Outliers detected and treated using IQR/Z-score
- ✅ Columns with low or no variance were dropped
- ✅ Log transformation was applied to skewed distributions

### Basic EDA insights:
- Most products have low demand
- Certain categories/sub-categories show seasonal spikes
- Discounts affect demand patterns differently by region

---

## ⚙️ Machine Learning Pipeline

A complete pipeline was developed with modular components for **cleaning, preprocessing, feature engineering, training, and evaluation**.

### ✅ Steps Included:

1. **Feature Engineering**:
    - Temporal features from date columns
    - Interaction terms like `discount × quantity`
    - Encoding categorical features using target/one-hot encoding

2. **Preprocessing**:
    - Scaling using StandardScaler
    - Encoding using LabelEncoder / OneHotEncoder
    - Imputation (mean/median for numerical, mode for categorical)

3. **Model Training**:
    - Multiple models trained and evaluated:
      - `DecisionTreeRegressor`
      - `RandomForestRegressor`
      - `XGBoostRegressor`

4. **Evaluation**:
    - Models evaluated on **R² Score** using a validation set.
    - Best performing model saved using `joblib`.

---

## 📊 Results

| Model             | R² Score |
|------------------|----------|
| Decision Tree     | 0.8079   |
| Random Forest     | 0.9039   |

📌 **XGBoost** was selected as the best model based on its superior performance.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/sachinn854/Product-Demand-Forecasting.git
cd Product-Demand-Forecasting

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Run the pipeline
python main.py


# to run app 




📬 Contact
Made with ❤️ by Sachin Yadav