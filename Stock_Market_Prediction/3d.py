import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
import time

st.set_page_config(
    page_title="📊 Stock Prediction App"
)

st.title("📊 Stock Prediction App 📈")
st.sidebar.header("📌 User Input")

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

stocks = list(STOCK_DATA.keys()) + ["Custom"]
selected_stock = st.sidebar.selectbox("🔍 Select a stock for prediction", stocks)

if selected_stock == "Custom":
    custom_stock = st.sidebar.text_input("✏️ Enter a custom stock code (e.g., AAPL, GOOG, etc.):")

    if custom_stock:
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

START = st.sidebar.date_input("📅 Start date", date(2020, 1, 1))
TODAY = st.sidebar.date_input("📅 End date", date.today())
n_years = st.sidebar.slider("⏳ Years of prediction:", 1, 5)
period = n_years * 365

# ✅ Retry mechanism added
@st.cache_data
def load_data(ticker):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, START, TODAY, progress=False, threads=True)
            if data.empty:
                raise ValueError("⚠️ No data returned from Yahoo Finance.")
            data.reset_index(inplace=True)
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"⏳ Attempt {attempt+1} failed, retrying... Error: {e}")
                time.sleep(2)
            else:
                raise e

def validate_stock_code(ticker):
    try:
        data = yf.download(ticker, period="1d", progress=False, threads=True)
        if data.empty:
            return False, "⚠️ No data found for this stock code."
        return True, "✅ Valid stock code."
    except Exception as e:
        return False, f"⚠️ Error: {str(e)}"

if selected_stock:
    data_load_state = st.text("⏳ Loading data... Please Wait!")
    is_valid, validation_message = validate_stock_code(selected_stock)

    if not is_valid:
        data_load_state.text(f"❌ Error: {validation_message}")
        st.error(f"❌ Invalid stock code: {selected_stock}")
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
            if 'Too Many Requests' in str(e):
                st.error("❌ Yahoo Finance API se zyada request bhej di gayi hai. Thodi der ruk kar dubara try karo.")
            else:
                st.error(f"❌ Error loading data: {e}")
            st.stop()
else:
    st.warning("⚠️ Please select or enter a stock code.")
    st.stop()

data["Date"] = pd.to_datetime(data["Date"])
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

current_close = df_train['y'].iloc[-1]
future_close = forecast['yhat'].iloc[-1]

if future_close > current_close:
    sentiment = "Positive"
    sentiment_color = "#77ff33"
else:
    sentiment = "Negative"
    sentiment_color = "red"

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

st.subheader("📄 Raw Data")
st.write("This section displays the raw stock data fetched from Yahoo Finance.")
st.write(data.tail())
st.write(f"📏 Data shape: {data.shape}")
st.write(data.columns)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing Price', mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Opening Price', mode='lines', line=dict(color='red')))
    fig.update_layout(
        title_text=f"{selected_stock} Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)

st.subheader("📈 Line Chart of Stock Data")
plot_raw_data()

st.subheader("📚 Training Data")
st.write(df_train.tail())

st.subheader("🧹 Cleaned Training Data")
st.write(df_train.tail())

st.subheader("🔮 Forecast Data")
st.write(forecast.tail())

st.write("📊 Forecast Plot")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("🧩 Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
