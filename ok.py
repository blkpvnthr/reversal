# Data Science 101 for Python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import yfinance as yf
from scipy.stats import norm, gamma, binom, poisson, bernoulli, expon, chi2, linregress
from scipy import stats
import requests
import configparser

warnings.filterwarnings("ignore")

# Load API Key from Config File
config = configparser.ConfigParser()
config.read('config.ini')

try:
    api_key = config.get('API', 'FRED_API_KEY')
except (configparser.NoSectionError, configparser.NoOptionError):
    raise ValueError("API key not found! Make sure 'config.ini' exists and contains the 'FRED_API_KEY' under [API] section.")

#plt.show()
#plt.pause(0.001)  # Allows non-blocking behavior\
#plt.ion()  # Turn on interactive

# Define stock symbol and date range
symbol = 'DIA'
market = 'SPY'
start = '2022-01-01'
end = '2025-01-01'

# Compare Stock to Market
#symbol = 'AAPL'

# Read data
#dataset1 = yf.download(symbol, start, end)
df2 = yf.download(market, start=start, end=end)
# Download stock data
df = yf.download(symbol, start=start, end=end)

# Check if data was downloaded successfully
if df.empty:
    print("No data was downloaded. Check the symbol and date range.")
    exit()

# 🔹 Fix MultiIndex issue by renaming columns properly
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(0)  # Drop "Ticker" level (DIA)

# 🔹 Manually rename columns to standard names
df.columns = ["Close", "High", "Low", "Open", "Volume"]

# Verify column names after fix
print("\nColumns after fixing:", df.columns.tolist())

# Pretty print DataFrame
print("\nStock Data (First 5 Rows):")
print(df.head())



# Fetch CPI Data from FRED (CPIAUCSL: Consumer Price Index for All Urban Consumers)
cpi_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={api_key}&file_type=json"
cpi_response = requests.get(cpi_url).json()

# Convert to DataFrame
cpi_data = pd.DataFrame(cpi_response['observations'])
cpi_data['value'] = cpi_data['value'].astype(float)
cpi_data['date'] = pd.to_datetime(cpi_data['date'])
cpi_data = cpi_data.set_index('date')

# Compute Inflation Rate (Monthly % Change)
cpi_data['inflation_rate'] = cpi_data['value'].pct_change()

# Define Realized Volatility Function
def realized_volatility(returns):
    return np.sqrt(np.sum(returns**2))

# Compute Realized Volatility
df_rv = (cpi_data.groupby(pd.Grouper(freq='M'))['inflation_rate'].apply(realized_volatility)
         .rename('Realized Volatility'))

# Annualize Realized Volatility
df_rv = df_rv * np.sqrt(12)

# Display CPI and Inflation Rate Data
plt.title("CPI and Inflation Rate Data")
plt.plot(cpi_data['value'], label="CPI Index", color='blue')
plt.plot(cpi_data['inflation_rate'], label="Inflation Rate", color='red')
plt.legend()



# Feature Engineering
df['Simple_Returns'] = df['Close'].pct_change()
df['Inflation_Rate'] = cpi_data['inflation_rate'].reindex(df.index, method='ffill')
df['Real_Return'] = ((df['Simple_Returns'] + 1) / (df['Inflation_Rate'] + 1)) - 1
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['Increase_Decrease'] = np.where(df['Volume'].diff() > 0, 1, -1)  # Volume Change Direction
df['Buy_Sell_on_Open'] = np.where(df['Open'].diff() > 0, 1, -1)  # Open Price Change
df['Buy_Sell'] = np.where(df['Close'].diff() > 0, 1, -1)  # Close Price Change
# Compute Stock Realized Volatility
df['Realized_Volatility'] = df['Simple_Returns'].rolling(window=21).apply(realized_volatility, raw=True)

# Drop NaN values caused by shifting
df.dropna(inplace=True)


# Merge Realized Volatility into DataFrame
df = df.join(df_rv, how='left')

# Plot CPI, Inflation Rate, and Realized Volatility
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

ax[0].plot(cpi_data.index, cpi_data['value'], label="CPI Index", color='blue')
ax[0].set_title("CPI Over Time")
ax[0].set_ylabel("CPI Index")
ax[0].legend()

ax[1].plot(df_rv.index, df_rv, label="Realized Volatility", color='red')
ax[1].set_title("Realized Volatility Over Time")
ax[1].set_ylabel("Realized Volatility")
ax[1].legend()

ax[2].plot(df.index, df['Real_Return'], label="Real Return", color='green')
ax[2].set_title("Real Return Over Time")
ax[2].set_ylabel("Real Return")
ax[2].legend()

plt.tight_layout()
plt.show()

# Show final DataFrame
print("\nFinal DataFrame (After Feature Engineering, First 5 Rows):")
print(df.head())

# Statistical Summary
print("\nMean Values in the Distribution")
print("-" * 35)
print(df.mean(numeric_only=True))  # Exclude non-numeric columns

print("\n***********************************")
print("Median Values in the Distribution")
print("-" * 35)
print(df.median(numeric_only=True))  # Exclude non-numeric columns

print("\nMode Values")
print("-" * 35)
print(df.mode(numeric_only=True).iloc[0])  # Show only first mode per column

# Measuring Standard Deviation
print("Measuring Standard Deviation")
print(df.std(numeric_only=True))

# Measuring Skewness
print("Measuring Skewness")
print(df.skew(numeric_only=True))

# Define Variables
mu = df['Simple_Returns'].mean()  # Mean Simple_Returns
sigma = df['Simple_Returns'].std()  # Volatility

# Plot Histogram with Normal Distribution Curve
plt.figure(figsize=(10, 6))
[n, bins, patches] = plt.hist(df['Simple_Returns'], bins=100, density=True, alpha=0.6, color='g')

# Plot CPI and Inflation Rate
fig, ax1 = plt.subplots(figsize=(12, 6))

# CPI Plot
ax1.set_xlabel("Date")
ax1.set_ylabel("CPI Index", color="tab:blue")
ax1.plot(cpi_data.index, cpi_data['value'], color="tab:blue", label="CPI Index")
ax1.tick_params(axis='y', labelcolor="tab:blue")

# Inflation Rate Plot
ax2 = ax1.twinx()
ax2.set_ylabel("Inflation Rate (%)", color="tab:red")
ax2.plot(cpi_data.index, cpi_data['inflation_rate'] * 100, color="tab:red", linestyle="dashed", label="Inflation Rate")
ax2.tick_params(axis='y', labelcolor="tab:red")

plt.tight_layout()
plt.show()

# Title and Legends
plt.title("CPI and Inflation Rate Over Time")
fig.tight_layout()
plt.show()

# Realized Volatility Plot
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Realized_Volatility'], label="Realized Volatility", color='red')
plt.title("Realized Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Realized Volatility")
plt.legend()
plt.show()

# Normal Distribution Curve
s = norm.pdf(bins, mu, sigma)
plt.plot(bins, s, color='y', lw=2)
plt.title("Stock Simple_Returns on Normal Distribution")
plt.xlabel("Simple_Returns")
plt.ylabel("Frequency")
plt.show()

# Plot Normal Distribution Comparisons
x_min, x_max = df['Simple_Returns'].min(), df['Simple_Returns'].max()
x = np.linspace(x_min, x_max, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, norm.pdf(x, mu, sigma), label="Stock Simple_Returns", color='black')
plt.plot(x, norm.pdf(x, -2, 1), color='red', lw=2, ls='-', alpha=0.5, label="Mean=-2, Std=1")
plt.plot(x, norm.pdf(x, 2, 1.2), color='blue', lw=2, ls='-', alpha=0.5, label="Mean=2, Std=1.2")
plt.plot(x, norm.pdf(x, 0, 0.8), color='green', lw=2, ls='-', alpha=0.5, label="Mean=0, Std=0.8")
plt.legend()
plt.title("Normal Distribution Comparisons")
plt.show()

# Fit Stock Simple_Returns to Normal Distribution
mu, std = norm.fit(df['Simple_Returns'])
plt.hist(df['Simple_Returns'], bins=25, density=True, alpha=0.6, color='g')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title(f"Fit results: mu = {mu:.2f},  std = {std:.2f}")
plt.show()

stock_ret = df['Close'].pct_change().dropna()
mkt_ret = df2['Close'].pct_change().dropna()

# Ensure both datasets have the same date range
common_dates = mkt_ret.index.intersection(df.index)
mkt_ret = mkt_ret.loc[common_dates]
stock_ret = df.loc[common_dates, "Simple_Returns"]

# Convert to NumPy arrays with the same shape
mkt_ret = mkt_ret.to_numpy().flatten()
stock_ret = stock_ret.to_numpy().flatten()

# Ensure sizes match before regression
print(f"Market Simple_Returns Shape: {mkt_ret.shape}, Stock Simple_Returns Shape: {stock_ret.shape}")

# Perform Regression Analysis
beta, alpha, r_value, p_value, std_err = linregress(mkt_ret, stock_ret)

print(f"Beta: {beta}, Alpha: {alpha}")
# Display Results
print(f"Beta: {beta:.4f}, Alpha: {alpha:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4f}")

# Hypothesis Testing
if p_value < 0.05:  # 5% significance level
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")

# Scatter Plot for Stock & Market Simple_Returns
plt.figure(figsize=(10, 6))
sns.regplot(x=mkt_ret, y=stock_ret, scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
plt.xlabel("Market Simple_Returns")
plt.ylabel("Stock Simple_Returns")
plt.title("Stock vs. Market Regression Analysis")
plt.show()

# Poisson Distribution
mu = df['Simple_Returns'].mean()
dist = poisson.rvs(mu=mu, loc=0, size=1000)
print(f"Mean: {df['Simple_Returns'].mean():.5f}")
print(f"SD: {df['Simple_Returns'].std(ddof=1):.5f}")

plt.hist(dist, bins=10, density=True, alpha=0.6, color='b')
plt.title("Poisson Distribution Curve")
plt.show()

# Bernoulli Distribution
countIncrease = df[df["Increase_Decrease"] == 1]["Increase_Decrease"].count()
countAll = df["Increase_Decrease"].count()
Increase_dist = bernoulli(countIncrease / countAll)

plt.figure(figsize=(6, 4))
plt.vlines(0, 0, Increase_dist.pmf(0), colors='r', lw=5, label="Probability of Decrease")
plt.vlines(1, 0, Increase_dist.pmf(1), colors='b', lw=5, label="Probability of Increase")
plt.legend()
plt.title("Bernoulli Distribution of Increase Variable")
plt.show()

# Exponential Distribution
x_m = df['Simple_Returns'].max()
x = np.linspace(0, x_m, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, expon.pdf(x, scale=sigma), 'r-', lw=2, alpha=0.6, label='Exponential PDF')
plt.xlabel("Simple_Returns")
plt.ylabel("Probability")
plt.legend()
plt.title("Exponential Distribution of Simple_Returns")
plt.show()

# Chi-Square Distribution
x = df['Simple_Returns']
fig, ax = plt.subplots(figsize=(10, 6))

linestyles = [':', '--', '-.', '-']
deg_of_freedom = [1, 4, 7, 6]
for df_, ls in zip(deg_of_freedom, linestyles):
    ax.plot(x, chi2.pdf(x, df_), linestyle=ls)

plt.xlim(0, 10)
plt.ylim(0, 0.4)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Chi-Square Distribution")
plt.legend()
plt.show()

# Scatter Plot of Stock Data
sns.regplot(x="Close", y="Open", data=df)
plt.show()
