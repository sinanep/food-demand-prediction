# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="XGBoost Sales Predictor", layout="wide")

st.title("🛒 Sales Prediction with XGBoost")

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    # Feature Engineering
    df['discount'] = df['base_price'] - df['checkout_price']
    df['discount_pct'] = df['discount'] / df['base_price']

    # Encode categorical features
    df['center_id'] = df['center_id'].astype('category').cat.codes
    df['meal_id'] = df['meal_id'].astype('category').cat.codes

    return df

# Load data
df = load_data()

# Display sample data
st.subheader("📄 Sample of Training Data")
st.dataframe(df.head())

# Define features and target
features = ['week', 'center_id', 'meal_id', 'checkout_price', 'base_price',
            'emailer_for_promotion', 'homepage_featured', 'discount', 'discount_pct']
target = 'num_orders'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.markdown(f"- **R² Score:** {r2:.4f}")
st.markdown(f"- **Mean Squared Error (MSE):** {mse:.2f}")
st.markdown(f"- **Mean Absolute Error (MAE):** {mae:.2f}")

# Plot actual vs predicted
st.subheader("📈 Actual vs Predicted Orders")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, ax=ax1)
ax1.set_xlabel("Actual Orders")
ax1.set_ylabel("Predicted Orders")
ax1.set_title("Actual vs Predicted Orders")
ax1.grid(True)
st.pyplot(fig1)

# Predict demand on full dataset
df['predicted_orders'] = model.predict(X)

# Identify highly demanded meals
top_meals = df.groupby('meal_id')['predicted_orders'].sum().sort_values(ascending=False).reset_index()

st.subheader("🍽️ Top 10 Predicted High-Demand Meals")
st.dataframe(top_meals.head(10))

# Barplot of top 10 meals
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_meals.head(10), x='predicted_orders', y='meal_id', palette='viridis', ax=ax2)
ax2.set_title("Top 10 Meals by Predicted Demand")
ax2.set_xlabel("Total Predicted Orders")
ax2.set_ylabel("Meal ID")
st.pyplot(fig2)

# -----------------------------
# 🔍 Custom Prediction Section
# -----------------------------
st.subheader("🔍 Predict Orders for Custom Inputs")

with st.form("prediction_form"):
    week = st.slider("Week", min_value=int(df['week'].min()), max_value=int(df['week'].max()), value=10)
    center_id = st.slider("Center ID", min_value=int(df['center_id'].min()), max_value=int(df['center_id'].max()), value=5)
    meal_id = st.slider("Meal ID", min_value=int(df['meal_id'].min()), max_value=int(df['meal_id'].max()), value=50)
    base_price = st.number_input("Base Price", min_value=50.0, max_value=1000.0, value=200.0)
    checkout_price = st.number_input("Checkout Price", min_value=50.0, max_value=1000.0, value=180.0)
    emailer_for_promotion = st.selectbox("Emailer Promotion", [0, 1])
    homepage_featured = st.selectbox("Homepage Featured", [0, 1])

    submit_button = st.form_submit_button("Predict")

    if submit_button:
        discount = base_price - checkout_price
        discount_pct = discount / base_price

        input_data = pd.DataFrame([{
            'week': week,
            'center_id': center_id,
            'meal_id': meal_id,
            'checkout_price': checkout_price,
            'base_price': base_price,
            'emailer_for_promotion': emailer_for_promotion,
            'homepage_featured': homepage_featured,
            'discount': discount,
            'discount_pct': discount_pct
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted number of orders: **{int(prediction)}**")
