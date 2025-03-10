# import streamlit as st
# from datetime import date
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go

# # Define start date for historical data
# START = "2015-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")

# # Streamlit Title
# st.title("📈 Stock Prediction App")

# # Dropdown for stock selection
# stocks = ("AAPL", "GOOG", "MSFT", "GME")
# selected_stock = st.selectbox("Select dataset for prediction", stocks)

# # Function to load data
# @st.cache_data
# def load_data(ticker):
#     data = yf.download(ticker, START, TODAY)
#     data.reset_index(inplace=True)  # Reset index to use Date as a column
#     return data

# # Load Data
# data_load_state = st.text("Loading data... Please Wait!")
# data = load_data(selected_stock)
# data_load_state.text("Loading complete!")

# # Ensure 'Date' column is in datetime format
# data["Date"] = pd.to_datetime(data["Date"])

# # Show raw data
# st.subheader("Raw Data (Latest)")
# st.write(data.tail())

# # Function to plot stock data
# def plot_stock_data():
#     # Ensure the Date column is sorted
#     data.sort_values(by="Date", inplace=True)

#     # Create plot
#     fig = go.Figure()

#     # Add Open and Close Prices as Line Charts
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Stock Open', line=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Stock Close', line=dict(color='red')))

#     # Update layout for better readability
#     fig.update_layout(
#         title="📊 Time Series Data",
#         xaxis_title="Date",
#         yaxis_title="Stock Price",
#         xaxis_rangeslider_visible=True,  # Enable zooming with range slider
#         xaxis=dict(
#             tickformat="%Y",  # Show years properly
#             tickangle=0
#         ),
#         template="plotly_white",  # Light background
#         legend=dict(x=0, y=1, traceorder="normal")  # Position legend properly
#     )

#     st.plotly_chart(fig)

# # Plot the stock data
# plot_stock_data()

# -----------------------------------------

import streamlit as st
from datetime import date

import yfinance as yf
# from  fbprophet import Prophet
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import pandas as pd
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction App")

stocks = ("AAPL","GOOG","MSFT","GME")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:",1,5)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker , START , TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data.... Please Wait!")
data = load_data(selected_stocks)
# st.write(f"Historical Data for {selected_stock}")
data_load_state.text("Loading data Done.. !")

st.subheader('Raw data')
# st.write(data.head())
st.write(data.tail()) 

def generate_sample_data():
    """Generates sample stock market data."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=400, freq="D")
    stock_open = np.cumsum(np.random.randn(400)) + 100  # Random walk
    stock_close = stock_open + np.random.randn(400) * 2  # Small variance
    
    data = pd.DataFrame({"Date": dates, "stock_open": stock_open, "stock_close": stock_close})
    return data

def create_time_series_chart(data):
    """Creates a Plotly time series chart with a range slider."""
    fig = go.Figure()

    # Add Stock Open line
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["stock_open"],
        mode="lines",
        name="Stock Open",
        line=dict(color="blue")
    ))

    # Add Stock Close line
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["stock_close"],
        mode="lines",
        name="Stock Close",
        line=dict(color="red")
    ))

    # Add range slider
    fig.update_layout(
        title="📈 Time Series Data",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis=dict(rangeslider=dict(visible=True)),
        template="plotly_white"
    )

    return fig

# Streamlit App
def main():
    st.title("📊 Stock Market Time Series Chart")
    
    data = generate_sample_data()  # Generate sample data
    fig = create_time_series_chart(data)  # Create chart
    
    st.plotly_chart(fig)  # Display chart

# Run the app
if __name__ == "__main__":
    main()
