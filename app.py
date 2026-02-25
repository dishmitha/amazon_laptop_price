import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# Title and description
st.title("💻 Laptop Price Predictor")
st.markdown("Predict the price of a laptop based on its specifications")

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('laptop_price_model.pkl')
    return model

model = load_model()

# Create two columns for layout
col1, col2 = st.columns(2)

# Input fields - Left column
with col1:
    st.subheader("Laptop Specifications")
    
    # Brand selection
    brand = st.selectbox(
        "Brand",
        ["Acer", "Apple", "Asus", "Dell", "HP", "Lenovo", "MSI"]
    )
    
    # Processor selection
    processor = st.selectbox(
        "Processor",
        ["Intel i3", "Intel i5", "Intel i7", "i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
    )
    
    # RAM selection
    ram = st.selectbox(
        "RAM (GB)",
        [4, 8, 16, 32]
    )
    
    # Storage selection
    storage = st.selectbox(
        "Storage (GB)",
        [128, 256, 512, 1024]
    )

# Input fields - Right column
with col2:
    st.subheader("Additional Details")
    
    # Operating System selection
    os = st.selectbox(
        "Operating System",
        ["Windows", "MacOS"]
    )
    
    # GPU selection
    gpu = st.selectbox(
        "GPU",
        ["Integrated", "AMD Radeon", "NVIDIA GTX 1650", "NVIDIA RTX 3050"]
    )
    
    # Rating selection
    rating = st.slider(
        "Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

# Processor encoding (matching the notebook)
processor_map = {
    'Intel i3': 1, 
    'Intel i5': 2, 
    'Intel i7': 3, 
    'i9': 4,
    'AMD Ryzen 3': 1, 
    'AMD Ryzen 5': 2, 
    'AMD Ryzen 7': 3
}

# GPU encoding (matching the notebook)
gpu_map = {
    'Integrated': 0,
    'AMD Radeon': 1,
    'NVIDIA GTX 1650': 2,
    'NVIDIA RTX 3050': 3
}

# Create feature dataframe
def preprocess_input(brand, processor, ram, storage, os, gpu, rating):
    # Create a dataframe with the input
    data = {
        'Processor': [processor_map[processor]],
        'RAM_GB': [ram],
        'Storage_GB': [storage],
        'Rating': [rating],
        'GPU_Level': [gpu_map[gpu]],
        'GPU': [gpu_map[gpu]],
        'Operating_System_macOS': [1 if os == 'MacOS' else 0]
    }
    
    df = pd.DataFrame(data)
    
    # Add brand one-hot encoding columns
    brands = ["Acer", "Apple", "Asus", "Dell", "HP", "Lenovo", "MSI"]
    for b in brands:
        df[f'Brand_{b}'] = 1 if brand == b else 0
    
    # Ensure columns are in the correct order (matching training data)
    # Reorder columns to match the model's expected input
    cols = ['Processor', 'RAM_GB', 'Storage_GB', 'GPU', 'Rating', 'GPU_Level']
    cols += [f'Brand_{b}' for b in brands]
    cols.append('Operating_System_macOS')
    
    # Make sure all columns exist
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[cols]

# Predict button
if st.button("Predict Price", type="primary"):
    try:
        # Preprocess the input
        input_df = preprocess_input(brand, processor, ram, storage, os, gpu, rating)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display result
        st.success(f"💰 Predicted Price: ₹{prediction:,.2f}")
        
        # Show input summary
        with st.expander("See input details"):
            st.write(f"**Brand:** {brand}")
            st.write(f"**Processor:** {processor}")
            st.write(f"**RAM:** {ram} GB")
            st.write(f"**Storage:** {storage} GB")
            st.write(f"**Operating System:** {os}")
            st.write(f"**GPU:** {gpu}")
            st.write(f"**Rating:** {rating}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add some styling
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>",
    unsafe_allow_html=True
)
st.markdown("This app uses a Linear Regression model trained on laptop data.")
st.markdown("</div>", unsafe_allow_html=True)
