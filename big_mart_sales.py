
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Big Mart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-value {
        font-size: 2.5rem;
        color: #1565C0;
        font-weight: bold;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing artifacts
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, encoders, and statistics"""
    try:
        with open('bigmart_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('bigmart_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)

        with open('bigmart_stats.pkl', 'rb') as f:
            stats = pickle.load(f)

        return model, encoders, stats, None
    except FileNotFoundError as e:
        error_msg = f"""
        ⚠️ **Model files not found!**

        Please run the training script first:
        ```
        python train_model.py
        ```

        This will create the necessary model files:
        - bigmart_model.pkl
        - bigmart_encoders.pkl
        - bigmart_stats.pkl
        """
        return None, None, None, error_msg

# Title
st.markdown('<h1 class="main-header">🛒 Big Mart Sales Prediction</h1>', unsafe_allow_html=True)
st.markdown("### Predict sales for Big Mart outlets using Machine Learning")

# Load model
model, encoders, stats, error_msg = load_model_artifacts()

if error_msg:
    st.error(error_msg)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    This application predicts Big Mart sales using **XGBoost Regressor**.

    **Features Used:**
    - Item Weight
    - Item Fat Content
    - Item Visibility
    - Item Type
    - Item MRP
    - Outlet Identifier
    - Outlet Establishment Year
    - Outlet Size
    - Outlet Location Type
    - Outlet Type
    """)

    st.header("📈 Model Performance")
    if stats:
        st.metric("Training R² Score", f"{stats['r2_train']:.3f}")
        st.metric("Testing R² Score", f"{stats['r2_test']:.3f}")

# Main content
tab1, tab2 = st.tabs(["🔮 Prediction", "ℹ️ About"])

with tab1:
    st.markdown('<h2 class="sub-header">Single Item Prediction</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Item Details")
        item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=30.0, value=12.85, step=0.1)

        item_fat_content = st.selectbox(
            "Item Fat Content",
            options=["Low Fat", "Regular"],
            index=0
        )

        item_visibility = st.slider(
            "Item Visibility", 
            min_value=0.0, 
            max_value=0.35, 
            value=0.066, 
            step=0.001,
            format="%.3f"
        )

        item_type = st.selectbox(
            "Item Type",
            options=["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
                    "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                    "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                    "Breads", "Starchy Foods", "Others", "Seafood"]
        )

        item_mrp = st.number_input(
            "Item MRP (₹)", 
            min_value=30.0, 
            max_value=270.0, 
            value=140.99, 
            step=0.01
        )

    with col2:
        st.subheader("Outlet Details")

        outlet_identifier = st.selectbox(
            "Outlet Identifier",
            options=["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", 
                    "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"],
            index=0
        )

        outlet_establishment_year = st.selectbox(
            "Outlet Establishment Year",
            options=list(range(1985, 2010)),
            index=14  # 1999
        )

        outlet_size = st.selectbox(
            "Outlet Size",
            options=["Small", "Medium", "High"],
            index=1
        )

        outlet_location_type = st.selectbox(
            "Outlet Location Type",
            options=["Tier 1", "Tier 2", "Tier 3"],
            index=0
        )

        outlet_type = st.selectbox(
            "Outlet Type",
            options=["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"],
            index=1
        )

    with col3:
        st.subheader("Prediction")

        if st.button("🎯 Predict Sales", use_container_width=True):
            try:
                # Create input dataframe with proper column names
                input_data = pd.DataFrame({
                    'Item_Identifier': ['FDW58'],  # Dummy value, will be encoded
                    'Item_Weight': [item_weight],
                    'Item_Fat_Content': [item_fat_content],
                    'Item_Visibility': [item_visibility],
                    'Item_Type': [item_type],
                    'Item_MRP': [item_mrp],
                    'Outlet_Identifier': [outlet_identifier],
                    'Outlet_Establishment_Year': [outlet_establishment_year],
                    'Outlet_Size': [outlet_size],
                    'Outlet_Location_Type': [outlet_location_type],
                    'Outlet_Type': [outlet_type]
                })

                # Manual encoding based on training (simplified for common values)
                # Item_Fat_Content
                input_data['Item_Fat_Content'] = 0 if item_fat_content == "Low Fat" else 1

                # Item_Type
                item_type_map = {
                    "Baking Goods": 0, "Breads": 1, "Breakfast": 2, "Canned": 3,
                    "Dairy": 4, "Frozen Foods": 5, "Fruits and Vegetables": 6,
                    "Hard Drinks": 7, "Health and Hygiene": 8, "Household": 9,
                    "Meat": 10, "Others": 11, "Seafood": 12, "Snack Foods": 13,
                    "Soft Drinks": 14, "Starchy Foods": 15
                }
                input_data['Item_Type'] = item_type_map.get(item_type, 0)

                # Outlet_Identifier
                outlet_map = {
                    "OUT010": 0, "OUT013": 1, "OUT017": 2, "OUT018": 3,
                    "OUT019": 4, "OUT027": 5, "OUT035": 6, "OUT045": 7,
                    "OUT046": 8, "OUT049": 9
                }
                input_data['Outlet_Identifier'] = outlet_map.get(outlet_identifier, 0)

                # Outlet_Size
                size_map = {"High": 0, "Medium": 1, "Small": 2}
                input_data['Outlet_Size'] = size_map.get(outlet_size, 1)

                # Outlet_Location_Type
                location_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
                input_data['Outlet_Location_Type'] = location_map.get(outlet_location_type, 0)

                # Outlet_Type
                type_map = {
                    "Grocery Store": 0, "Supermarket Type1": 1,
                    "Supermarket Type2": 2, "Supermarket Type3": 3
                }
                input_data['Outlet_Type'] = type_map.get(outlet_type, 1)

                # Item_Identifier (use a numeric value)
                input_data['Item_Identifier'] = 500  # Middle range value

                # Make prediction
                prediction = model.predict(input_data)[0]

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Predicted Sales")
                st.markdown(f'<p class="prediction-value">₹ {prediction:,.2f}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Show input summary
                with st.expander("📋 View Input Summary"):
                    st.write("**Item Information:**")
                    st.write(f"- Weight: {item_weight} kg")
                    st.write(f"- Fat Content: {item_fat_content}")
                    st.write(f"- Visibility: {item_visibility:.3f}")
                    st.write(f"- Type: {item_type}")
                    st.write(f"- MRP: ₹{item_mrp:.2f}")
                    st.write("\n**Outlet Information:**")
                    st.write(f"- Identifier: {outlet_identifier}")
                    st.write(f"- Establishment Year: {outlet_establishment_year}")
                    st.write(f"- Size: {outlet_size}")
                    st.write(f"- Location: {outlet_location_type}")
                    st.write(f"- Type: {outlet_type}")

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please ensure all inputs are valid.")

with tab2:
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview
    This Big Mart Sales Prediction system uses **Machine Learning** to forecast product sales 
    across different Big Mart outlets. The model considers various factors including:

    - **Product Characteristics**: Weight, fat content, visibility, type, and MRP
    - **Outlet Features**: Location, size, type, and establishment year

    ### 🤖 Machine Learning Model
    - **Algorithm**: XGBoost Regressor
    - **Training Data**: 8,523 records from Train.csv
    - **Features**: 11 input features
    - **Target Variable**: Item Outlet Sales (in ₹)

    ### 📊 Model Performance
    """)

    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training R² Score", f"{stats['r2_train']:.4f}", 
                     help="Model performance on training data")
        with col2:
            st.metric("Testing R² Score", f"{stats['r2_test']:.4f}",
                     help="Model performance on unseen test data")

    st.markdown("""
    ### 🔧 Data Preprocessing
    1. **Missing Value Treatment**:
       - Item Weight: Filled with mean value
       - Outlet Size: Filled with mode based on outlet type

    2. **Label Encoding**:
       - All categorical variables encoded to numerical values
       - Standardized fat content categories (Low Fat/Regular)

    3. **Feature Engineering**:
       - Normalized item visibility
       - Encoded outlet identifiers
       - Processed establishment years

    ### 💡 How to Use
    1. **Enter Details**: Fill in the item and outlet information in the Prediction tab
    2. **Click Predict**: Get instant sales prediction with insights
    3. **Analyze Results**: Review the predicted sales and confidence metrics

    ### 📝 Notes
    - Ensure all input values are within valid ranges
    - The model works best with data similar to the training distribution
    - Predictions are estimates based on historical patterns

    ### 👨‍💻 Technical Stack
    - **Frontend**: Streamlit
    - **ML Framework**: XGBoost, Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Model Persistence**: Pickle

    ### 📂 Project Files
    - `train_model.py` - Model training script with preprocessing
    - `bigmart_sales_app.py` - This Streamlit application
    - `Train.csv` - Training dataset (8,523 records)
    - `bigmart_model.pkl` - Trained XGBoost model
    - `bigmart_encoders.pkl` - Label encoders for categorical variables
    - `bigmart_stats.pkl` - Model statistics and metadata

    ---
    **Developed for**: Course Project - Big Mart Sales Prediction
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "© 2025 Big Mart Sales Predictor | Built with Streamlit & XGBoost"
    "</div>",
    unsafe_allow_html=True
)
