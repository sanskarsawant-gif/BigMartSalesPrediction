
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import pickle

print("="*60)
print("BIG MART SALES PREDICTION - MODEL TRAINING")
print("="*60)

# Load the training data
print("\n[1/7] Loading training data...")
bigmart_data = pd.read_csv('Train.csv')
print(f"✓ Loaded {len(bigmart_data)} records")

# Handle missing values
print("\n[2/7] Handling missing values...")
# Fill Item_Weight with mean
bigmart_data['Item_Weight'].fillna(bigmart_data['Item_Weight'].mean(), inplace=True)

# Fill Outlet_Size with mode based on Outlet_Type
mode_of_outlet_size = bigmart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', 
                                                aggfunc=(lambda x: x.mode()[0]))
miss_values = bigmart_data['Outlet_Size'].isnull()
bigmart_data.loc[miss_values, 'Outlet_Size'] = bigmart_data.loc[miss_values, 'Outlet_Type'].apply(
    lambda x: mode_of_outlet_size[x])
print("✓ Missing values handled")

# Standardize Item_Fat_Content
print("\n[3/7] Standardizing categories...")
bigmart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}}, 
                     inplace=True)
print("✓ Categories standardized")

# Label Encoding
print("\n[4/7] Encoding categorical variables...")
encoder = LabelEncoder()

# Create encoders dictionary to save
encoders = {}

# Encode all categorical columns
categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
                       'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

for col in categorical_columns:
    bigmart_data[col] = encoder.fit_transform(bigmart_data[col])
    # Save the encoder for each column
    encoders[col] = encoder
    encoder = LabelEncoder()  # Create new encoder for next column

print("✓ All categorical variables encoded")

# Split features and target
print("\n[5/7] Splitting features and target...")
X = bigmart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = bigmart_data['Item_Outlet_Sales']
print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {Y.shape}")

# Train-test split
print("\n[6/7] Training XGBoost model...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Evaluate
train_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, train_prediction)

test_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_prediction)

print(f"✓ Training R² Score: {r2_train:.4f}")
print(f"✓ Testing R² Score: {r2_test:.4f}")

# Save the model and encoders
print("\n[7/7] Saving model and encoders...")
with open('bigmart_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

with open('bigmart_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Also save some statistics for the app
stats = {
    'item_weight_mean': float(bigmart_data['Item_Weight'].mean()),
    'r2_train': float(r2_train),
    'r2_test': float(r2_test)
}

with open('bigmart_stats.pkl', 'wb') as f:
    pickle.dump(stats, f)

print("✓ Model saved as 'bigmart_model.pkl'")
print("✓ Encoders saved as 'bigmart_encoders.pkl'")
print("✓ Stats saved as 'bigmart_stats.pkl'")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nYou can now run the Streamlit app:")
print("  streamlit run bigmart_sales_app.py")
