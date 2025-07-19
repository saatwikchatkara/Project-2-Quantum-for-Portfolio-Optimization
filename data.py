# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Vanguard ETF Ticker Symbols ---
# You can replace these with any other Vanguard (or other company) ETF/fund tickers
# that you are interested in for your portfolio.
vanguard_tickers = [
    "VOO",  # Vanguard S&P 500 ETF
    "VTI",  # Vanguard Total Stock Market ETF
    "BND",  # Vanguard Total Bond Market ETF
    "VXUS", # Vanguard Total International Stock ETF
    "VGT",  # Vanguard Information Technology ETF
]

# --- 2. Define the Date Range for Historical Data ---
# It's good practice to specify a start and end date for your data.
# For portfolio optimization, you typically need several years of data.
start_date = "2018-01-01"
end_date = "2023-12-31" # Or pd.Timestamp.now().strftime('%Y-%m-%d') for current date

print(f"Fetching historical data for Vanguard ETFs from {start_date} to {end_date}...\n")

# --- 3. Fetch Historical Data using yfinance ---
# With yfinance's default auto_adjust=True, the DataFrame returned
# directly contains the adjusted close prices with tickers as columns.
# We explicitly set auto_adjust=True for clarity, though it's the default.
data = yf.download(vanguard_tickers, start=start_date, end=end_date, auto_adjust=True)

# --- 4. Display the Raw Data (first few rows) ---
print("Raw Historical Adjusted Close Prices (Tickers are columns):")
print(data.head())
print("\n" + "="*70 + "\n")
print("Raw Historical Adjusted Close Prices (last few rows):")
print(data.tail())
print("\n" + "="*70 + "\n")

# --- 5. Check for Missing Values ---
# It's important to handle missing data before calculating returns/covariance.
print("Missing values per ticker:")
print(data.isnull().sum())
print("\n")

# --- 6. (Optional) Visualize the Price Trends ---
print("Plotting historical price trends...")
plt.figure(figsize=(12, 6))
data.plot(ax=plt.gca())
plt.title('Historical Adjusted Close Prices of Vanguard ETFs')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price ($)')
plt.grid(True)
plt.legend(title='Ticker')
plt.tight_layout()
plt.show()
print("\n" + "="*70 + "\n")

# --- Next Steps for Portfolio Optimization (Classical Pre-processing) ---
# For quantum portfolio optimization, you'll typically need:
# 1. Daily/Monthly Returns
# 2. Expected Returns (mean of historical returns)
# 3. Covariance Matrix of Returns

# Calculate daily returns (percentage change from previous day)
returns = data.pct_change().dropna() # dropna() removes the first row with NaN

print("Daily Returns (first few rows):")
print(returns.head())
print("\n" + "="*70 + "\n")

# Calculate Expected Returns (mean of daily returns)
expected_returns = returns.mean()
print("Expected Daily Returns:")
print(expected_returns)
print("\n" + "="*70 + "\n")

# Calculate Covariance Matrix
covariance_matrix = returns.cov()
print("Covariance Matrix of Daily Returns:")
