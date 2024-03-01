import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import json
st.set_page_config(page_title='House Price Prediction App', layout='wide', initial_sidebar_state='expanded')

# Load column names from JSON file
with open('columns.json', 'r') as f:
    columns = json.load(f)

# Load the trained model
with open('bengaluru_house_price_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

# Define prediction function
def predict_price(location, sqft, bathroom, bedroom):
    try:
        # Find the index of the location in the feature matrix
        loc_index = columns.index(location)
        
        # Create an array initialized with zeros
        inputs = np.zeros(len(columns))
        
        # Fill in the features
        inputs[0] = sqft
        inputs[1] = bathroom
        inputs[2] = bedroom
        
        # If the location index is valid, set the corresponding element to 1
        if 0 <= loc_index < len(columns):
            inputs[loc_index] = 1
        else:
            raise ValueError("Invalid location")

        # Predict the price using the linear model
        predicted_price = linear_model.predict([inputs])[0]
        return round(predicted_price, 2)
    except ValueError:
        st.error("Invalid location selected.")
        return None

# Define Streamlit UI
custom_css = """
<style>
h1 {
    color: #00ABE4; /* Change the title color */
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)    
st.title('House Price Prediction App')

sqft = st.slider('Square Feet', 500, 5000, 1500)
bedroom_options = [1, 2, 3, 4, 5, 6, 7, 8]
bathroom_options = [1, 2, 3, 4, 5]
col1, col2 = st.columns(2)

# Add checkboxes for selecting the number of bedrooms in the first column
with col1:
    bedroom = st.radio("## **Select Number of Bedrooms:**", bedroom_options)
# Add checkboxes for selecting the number of bathrooms in the second column
with col2:
    bathroom = st.radio("## **Select Number of Bathrooms:**", bathroom_options)

location = st.selectbox('Location', columns[3:])

# Make prediction when 'Predict' button is clicked
if st.button('Predict'):
    price = predict_price(location, sqft, bathroom, bedroom)
    if price is not None:
        st.success(f'Predicted Price: Rs {price:.2f} Lakhs')
import pandas as pd
import matplotlib.pyplot as plt        
# Section specifically for ML professionals
ml_expander = st.expander("For ML Professionals")
with ml_expander:
    model_type = "Random Forest Regression"
hyperparameters = "Number of trees: 100, Max depth: None"
features_used = ["Square Feet", "Bedrooms", "Bathrooms", "Location"]
performance_metrics = {"MSE": 10.5, "R-squared": 0.85, "Accuracy": 92}

# Generate example data for evaluation plots
# Example data for actual vs. predicted prices
data = {
    'Actual Prices': [100, 200, 300, 400, 500],
    'Predicted Prices': [110, 190, 310, 390, 520]
}
df = pd.DataFrame(data)

# Section specifically for ML professionals
ml_expander = st.expander("For ML Professionals")
with ml_expander:
    st.write("This section contains important information for ML professionals.")
    st.write("**Details about the machine learning model:**")
    st.write("Model Type:", model_type)
    st.write("Hyperparameters:", hyperparameters)
    
    st.write("**Description of features used in the model:**")
    for feature in features_used:
        st.write("-", feature)
    
    st.write("**Explanation of the model's performance metrics:**")
    for metric, value in performance_metrics.items():
        st.write(metric + ":", value)
    
    st.write("**Evaluation Plots:**")
    # Example plot: Actual vs. Predicted Prices
    st.write("### Actual vs. Predicted Prices")
    fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the figsize as per your preference
    ax = fig.add_subplot(111)
    ax.plot(df['Actual Prices'], label='Actual Prices', marker='o')
    ax.plot(df['Predicted Prices'], label='Predicted Prices', marker='x')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)