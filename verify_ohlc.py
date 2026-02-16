from hybrid_model import fetch_stock_data
import pandas as pd

def test_fetch_ohlc():
    ticker = "AAPL"
    print(f"Testing fetch_stock_data for {ticker}...")
    df = fetch_stock_data(ticker)
    
    if df is not None:
        print("Data fetched successfully.")
        print("Columns:", df.columns.tolist())
        
        required_cols = ['ds', 'y', 'Open', 'High', 'Low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if not missing_cols:
            print("SUCCESS: All required OHLC columns are present.")
            print(df.head())
        else:
            print(f"FAILURE: Missing columns: {missing_cols}")
    else:
        print("FAILURE: Could not fetch data.")

if __name__ == "__main__":
    test_fetch_ohlc()
