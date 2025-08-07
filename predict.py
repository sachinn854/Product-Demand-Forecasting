import joblib
import pandas as pd
import os

def load_model():
    model_path = os.path.join("models", "best_pipeline.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Model load error: {e}")
        return None

def predict_demand(model, form_data):
    # Convert inputs
    form_data['date'] = pd.to_datetime(form_data['date'])
    form_data['price'] = float(form_data['price'])
    form_data['discount_percent'] = float(form_data['discount_percent'])
    form_data['competitorprice'] = float(form_data['competitorprice'])
    form_data['finalprice'] = float(form_data['finalprice'])
    form_data['temp(c)'] = float(form_data['temp'])
    form_data['rainfall(mm)'] = float(form_data['rainfall'])
    form_data['weight(kg)'] = float(form_data['weight'])
    form_data['warranty(years)'] = float(form_data['warranty'])
    form_data['productrating'] = float(form_data['productrating'])
    form_data['launchyear'] = int(form_data['launchyear'])
    form_data['stocklevel'] = int(form_data['stocklevel'])
    form_data['supplierdelay(days)'] = int(form_data['supplierdelay'])

    # Rename keys to match model columns
    input_dict = {
        'productid': form_data['productid'],
        'location': form_data['location'],
        'date': form_data['date'],
        'promocodeused': form_data['promocodeused'],
        'price': form_data['price'],
        'discount_percent': form_data['discount_percent'],
        'competitorprice': form_data['competitorprice'],
        'adcampaign': form_data['adcampaign'],
        'finalprice': form_data['finalprice'],
        'isweekend': form_data['isweekend'],
        'season': form_data['season'],
        'daytype': form_data['daytype'],
        'temp(c)': form_data['temp(c)'],
        'rainfall(mm)': form_data['rainfall(mm)'],
        'category': form_data['category'],
        'brand': form_data['brand'],
        'material': form_data['material'],
        'weight(kg)': form_data['weight(kg)'],
        'warranty(years)': form_data['warranty(years)'],
        'productrating': form_data['productrating'],
        'launchyear': form_data['launchyear'],
        'stocklevel': form_data['stocklevel'],
        'supplierdelay(days)': form_data['supplierdelay(days)'],
        'warehouse': form_data['warehouse'],
        'inventorytype': form_data['inventorytype']
    }

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]

    revenue = prediction * input_dict['price']
    demand_level = "High" if prediction > input_dict['stocklevel'] else "Normal" if prediction > 0.5 * input_dict['stocklevel'] else "Low"
    stock_diff = int(prediction - input_dict['stocklevel'])

    insights = {
        "revenue": revenue,
        "demand_level": demand_level,
        "stock_diff": stock_diff,
        "profit_margin": ((input_dict['price'] - (input_dict['price'] * input_dict['discount_percent'] / 100)) / input_dict['price']) * 100,
        "competitive_edge": input_dict['price'] < input_dict['competitorprice']
    }

    return prediction, insights
