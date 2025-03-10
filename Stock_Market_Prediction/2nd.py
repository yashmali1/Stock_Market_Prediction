# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Generate sample time-series data
# def generate_time_series_data(n=100):
#     date_rng = pd.date_range(start='2024-01-01', periods=n, freq='D')
#     data = np.cumsum(np.random.randn(n))  # Cumulative sum for a time series effect
#     df = pd.DataFrame({'Date': date_rng, 'Value': data})
#     return df

# # Streamlit app
# def main():
#     st.title("Time Series Data Viewer")
    
#     # Generate data
#     df = generate_time_series_data()
    
#     # Display DataFrame
#     st.write("### Sample Time Series Data:")
#     st.dataframe(df)
    
#     # Plot time series
#     st.write("### Time Series Chart:")
#     fig, ax = plt.subplots()
#     ax.plot(df['Date'], df['Value'], label='Time Series Data', color='b')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Value')
#     ax.set_title('Generated Time Series Data')
#     ax.legend()
#     st.pyplot(fig)

# if __name__ == "__main__":
#     main()


import yfinance as yf
import matplotlib.pyplot as plt

# ✅ Correct date format (YYYY-MM-DD)
startdt = "2021-06-10"
enddt = "2021-06-20"

# ✅ Fetch BTC data
ticker = "BTC-USD"
data = yf.download(ticker, start=startdt, end=enddt)

# ✅ Check if data is empty
if data.empty:
    print("⚠️ No data found! Check ticker or date range.")
else:
    # ✅ Plot closing price line chart
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Closing Price", color="blue", linestyle="-", marker="o")
    
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
