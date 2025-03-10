import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Sample Data Generation
def generate_sample_data():
    np.random.seed(42)
    date_rng = pd.date_range(start="2024-01-01", periods=100, freq="D")
    stock_prices = np.cumsum(np.random.randn(100)) + 100  # Simulated stock prices
    df = pd.DataFrame({"Date": date_rng, "Close": stock_prices})
    return df

# Load Sample Data
data = generate_sample_data()

# Convert Date to string only for display in Streamlit
data_display = data.copy()
data_display["Date"] = data_display["Date"].astype(str)

# Streamlit UI
st.title("📈 Sample Time Series Graph")
st.write("This is a sample time series visualization using randomly generated stock price data.")

# Plot Time Series Graph
fig = px.line(data, x="Date", y="Close", title="Stock Price Over Time")
st.plotly_chart(fig)

# Display Data
st.subheader("Raw Data")
st.write(data_display.tail())  # Display last few rows
