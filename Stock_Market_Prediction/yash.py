import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px

# App title with emoji
st.title("📊 Stock Prediction App 📈")

# Sidebar for user input
st.sidebar.header("📌 User Input")

# Predefined list of stocks with names and tickers
STOCK_DATA = {
    "AAPL": "Apple Inc.",
    "GOOG": "Alphabet Inc. (Google)",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla Inc.",
    "AMZN": "Amazon.com Inc.",
    "GME": "GameStop Corp.",
    "ZOMATO.NS": "Zomato Ltd (India)",
    "RELIANCE.NS": "Reliance Industries (India)",
    "TATASTEEL.NS": "Tata Steel Ltd (India)",
    "HDFCBANK.NS": "HDFC Bank Ltd (India)",
}

# Define stock symbols for dropdown
stocks = list(STOCK_DATA.keys()) + ["Custom"]
selected_stock = st.sidebar.selectbox("🔍 Select a stock for prediction", stocks)

# Add custom stock input
if selected_stock == "Custom":
    custom_stock = st.sidebar.text_input("✏️ Enter a custom stock code (e.g., AAPL, GOOG, etc.):")
    
    # Suggest stocks based on user input
    if custom_stock:
        # Filter stocks that match the user's input (case-insensitive)
        suggestions = [f"{ticker} - {name}" for ticker, name in STOCK_DATA.items() if custom_stock.lower() in ticker.lower() or custom_stock.lower() in name.lower()]
        
        if suggestions:
            st.sidebar.write("### 🎯 Suggestions:")
            for suggestion in suggestions:
                st.sidebar.write(f"- {suggestion}")
        else:
            st.sidebar.warning("⚠️ No matching stocks found. Try a different keyword.")
        
        selected_stock = custom_stock
    else:
        st.sidebar.warning("⚠️ Please enter a stock code or select from the dropdown.")

# Define date range
START = st.sidebar.date_input("📅 Start date", date(2020, 1, 1))
TODAY = st.sidebar.date_input("📅 End date", date.today())

# Define prediction period
n_years = st.sidebar.slider("⏳ Years of prediction:", 1, 10)
period = n_years * 365

# Function to load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    # Flatten multi-level column names
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    return data

# Function to validate stock code
def validate_stock_code(ticker):
    try:
        data = yf.download(ticker, period="1d")  # Try fetching 1 day of data
        if data.empty:
            return False, "No data found for this stock code."
        return True, "Valid stock code."
    except Exception as e:
        return False, f"Error: {e}"

# Load data
if selected_stock:
    data_load_state = st.text("⏳ Loading data... Please Wait!")
    is_valid, validation_message = validate_stock_code(selected_stock)
    
    if not is_valid:
        data_load_state.text(f"❌ Error: {validation_message}")
        st.error(f"❌ Invalid stock code: {selected_stock}")
        
        # Suggest popular stock codes
        st.write("### 🎯 Suggested Stock Codes")
        st.write("Here are some popular stock codes you can try:")
        for ticker, name in STOCK_DATA.items():
            st.write(f"- **{ticker}**: {name}")
        
        st.stop()
    else:
        try:
            data = load_data(selected_stock)
            data_load_state.text("✅ Loading data... Done!")
        except Exception as e:
            data_load_state.text(f"❌ Error loading data: {e}")
            st.stop()
else:
    st.warning("⚠️ Please select or enter a stock code.")
    st.stop()

# Ensure Date column is in datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Prepare data for Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Ensure 'y' column is numeric and handle missing/invalid values
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert to numeric, set invalid values to NaN
df_train = df_train.dropna()  # Drop rows with NaN values

# Train Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe
future = m.make_future_dataframe(periods=period)

# Make predictions
forecast = m.predict(future)

# Sentiment Analysis: Compare last predicted value with current closing price
current_close = df_train['y'].iloc[-1]  # Current closing price
future_close = forecast['yhat'].iloc[-1]  # Predicted closing price

# Determine sentiment
if future_close > current_close:
    sentiment = "Positive"
    sentiment_color = "#77ff33"
else:
    sentiment = "Negative"
    sentiment_color = "red"

# Display sentiment in a styled box
st.markdown(
    f"""
    <div style="
        padding: 10px;
        border-radius: 10px;
        background-color: #303030;
        text-align: center;
        margin: 10px 0;
    ">
        <h2 style="color: {sentiment_color}; margin: 0;">Future Sentiment: {sentiment}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Debugging: Print fetched data
st.subheader("📄 Raw Data")
st.write("This section displays the raw stock data fetched from Yahoo Finance. It includes columns like Date, Open, High, Low, Close, Adj Close, and Volume.")
st.write(data.tail())  # Display last few rows
st.write(f"📏 Data shape: {data.shape}")  # Check data shape
st.write(data.columns)  # Check column names

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    
    # Add Closing Price trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        name='Closing Price',
        mode='lines',
        line=dict(color='blue')  # Blue color for closing price
    ))
    
    # Add Opening Price trace
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Open'],
        name='Opening Price',
        mode='lines',
        line=dict(color='red')  # Red color for opening price
    ))
    
    # Update layout
    fig.update_layout(
        title_text=f"{selected_stock} Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True
    )
    
    st.plotly_chart(fig)

st.subheader("📈 Line Chart of Stock Data")
st.write("This chart visualizes the historical stock prices (Opening and Closing) over time. The blue line represents the Closing Price, and the red line represents the Opening Price.")
plot_raw_data()

# Debugging: Print training data
st.subheader("📚 Training Data")
st.write("This section shows the data prepared for training the Prophet model. It includes two columns: 'ds' (date) and 'y' (closing price).")
st.write(df_train.tail())

# Debugging: Print cleaned training data
st.subheader("🧹 Cleaned Training Data")
st.write("This section displays the cleaned training data after removing any rows with missing or invalid values.")
st.write(df_train.tail())

# Display forecast data
st.subheader("🔮 Forecast Data")
st.write("This section shows the forecasted stock prices generated by the Prophet model. It includes columns like 'ds' (date), 'yhat' (predicted value), and uncertainty intervals.")
st.write(forecast.tail())

# Plot forecast
st.write("📊 Forecast Plot")
st.write("This chart visualizes the predicted stock prices over time. The black dots represent historical data, and the blue line represents the forecasted values.")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write("🧩 Forecast Components")
st.write("This section breaks down the forecast into its components: trend, yearly seasonality, and weekly seasonality. It helps in understanding the underlying patterns in the data.")
fig2 = m.plot_components(forecast)
st.write(fig2)