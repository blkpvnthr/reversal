# CPI Inflation Calculation & Stock Market Analysis

This project fetches Consumer Price Index (CPI) data from the **FRED API**, processes it, and compares it to stock market data obtained from **Yahoo Finance**. The script also applies various data-filling methods and performs statistical analysis on stock returns.

<img src="stock.gif">

## 📦 Dependencies
Install the required Python packages:
```sh
pip install -r requirements.txt
```

## 🔑 Configuration
Set up your **API key** in a `config.ini` file:
```ini
[API]
FRED_API_KEY = your_fred_api_key_here
```

Or set it as an environment variable:
```sh
export FRED_API_KEY=your_fred_api_key_here
```

## 🛠 How to Run
Run the script with:
```sh
python ok.py
```

## 📊 Visualization Output
The script generates multiple plots:
1. **CPI Data with Different Fill Methods** (Original, Backward Fill, Forward Fill, Linear Interpolation)
2. **Realized Volatility of Inflation & Stock Data**
3. **Stock Returns Distribution**
4. **Stock vs. Market Regression Analysis**

## ⚠️ Error Handling
If you encounter errors:
1. Ensure your `FRED_API_KEY` is correct and active.
2. Check if the **FRED API** is online.
3. Verify that Yahoo Finance (`yfinance`) is returning data correctly.
4. Inspect missing values and ensure correct data preprocessing.

## 📜 License
This project is open-source and free to use.

---

📩 **Need Help?** Feel free to reach out or open an issue!

