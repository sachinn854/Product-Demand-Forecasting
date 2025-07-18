# 🛒 Product Demand Forecasting

This project aims to **predict product demand** based on historical sales, pricing strategies, promotions, inventory data, and product characteristics using machine learning techniques.

---

## 📌 Project Objectives

- Predict demand (`units_sold`) category using classification
- Identify key drivers of demand such as price, discount, campaign, weather, etc.
- Apply advanced feature engineering and ML models to improve accuracy

---

## 🗂️ Dataset Overview

- Total Rows: **5,00,000+**
- Total Columns: **47**
- Source: Synthetic + structured retail data

### Key Features:

| Column Name            | Description                                 |
|------------------------|---------------------------------------------|
| `productid`            | Unique product ID                           |
| `location`             | Selling location                            |
| `date`                 | Date of sale                                |
| `units_sold`           | Target label (binned into demand category)  |
| `price`, `discount_percent` | Original price & discount offered   |
| `competitorprice`      | Competitor's product price                  |
| `adcampaign`           | Whether a campaign ran for the product      |
| `stocklevel`, `supplierdelay(days)` | Inventory & delay info       |
| `finalprice`           | Derived as: `price * (1 - discount%)`       |
| `temp(c)`, `rainfall(mm)` | Weather-related features                  |
| `productrating`        | Numeric product rating                      |
| `category`, `brand`, `material`, `warranty(years)` | Product info   |

---

## 🔧 Feature Engineering

We performed **extensive feature engineering** to enhance model learning:

- ✅ **Interaction Features**:  
  - `stock_delay_interaction = stocklevel * supplierdelay`
- ✅ **Binning**:  
  - `productrating` → `rating_category` (`low`, `medium`, `high`)
- ✅ **Log Transformations**:  
  - `log_price`, `log_competitorprice`, `log_supplierdelay`
- ✅ **Missing Value Handling**:
  - Filled `promocodeused` and other categorical nulls with `"None"`
  - Numeric columns filled using mean/mode strategies
- ✅ **Label Encoding & One-Hot Encoding** for categorical variables

---

## 📊 Exploratory Data Analysis (EDA)

- Visualized `units_sold` distribution across:
  - Days of week, seasons, weather types
- Heatmaps to find correlation between numeric variables
- Count plots for categorical features like `category`, `brand`, `warehouse`, etc.
- Found strong influence of:
  - **Discount %**, **Ad campaigns**, **Weekends**, **Temperature**

---

## 🤖 Models Trained & Accuracy

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 78.0%    |
| Random Forest       | 79.0%    |
| XGBoost (Tuned)     | **81.5%** ✅ |

> ✅ Final model selected: **XGBoost Classifier** with hyperparameter tuning

---

## 📈 Model Evaluation

- Used `accuracy_score`, `confusion_matrix`, and `classification_report`
- Train-Test split: **80-20**
- Cross-validation (`cv=3`) used during GridSearchCV
- Best model tuned with parameters:  
  - `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`

---

## 📂 Repository Structure

