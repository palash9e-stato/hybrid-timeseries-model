import os
import yfinance as yf
import pandas as pd
from prophet import Prophet
import google.generativeai as genai
import feedparser
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMNI_API_KEY_2")
TARGET_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BTC-USD', 'ETH-USD']
SENTIMENT_WEIGHT = 0.05  # 5% adjustment based on sentiment score

# Initialize Gemini
if not GEMINI_API_KEY:
    print("WARNING: GEMNI_API_KEY_2 not found in environment variables. Sentiment analysis will be skipped.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

def fetch_stock_data(ticker):
    """Fetches historical stock data using yfinance."""
    print(f"Fetching data for {ticker}...")
    try:
        # Fetch last 2 years of data
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty:
            print(f"No data found for {ticker}")
            return None
        
        # Prepare for Prophet: Reset index and rename columns
        data.reset_index(inplace=True)
        # Handle MultiIndex columns if present (common in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns]
            # Verify exact column names after flattening
            date_col = next((c for c in data.columns if 'Date' in c), None)
            close_col = next((c for c in data.columns if 'Close' in c and ticker in c), None)
            open_col = next((c for c in data.columns if 'Open' in c and ticker in c), None)
            high_col = next((c for c in data.columns if 'High' in c and ticker in c), None)
            low_col = next((c for c in data.columns if 'Low' in c and ticker in c), None)
            
            if not date_col or not close_col:
                 # Fallback for simpler structures
                 date_col = 'Date'
                 close_col = 'Close'
                 open_col = 'Open'
                 high_col = 'High'
                 low_col = 'Low'
        else:
            date_col = 'Date'
            close_col = 'Close'
            open_col = 'Open'
            high_col = 'High'
            low_col = 'Low'

        df = data[[date_col, close_col]].rename(columns={date_col: 'ds', close_col: 'y'})
        
        # Add OHLC data if available
        if open_col in data.columns: df['Open'] = data[open_col]
        if high_col in data.columns: df['High'] = data[high_col]
        if low_col in data.columns: df['Low'] = data[low_col]
        
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None) # Remove timezone for Prophet
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def train_prophet_model(df):
    """Trains a Prophet model and forecasts 30 days into the future."""
    print("Training Prophet model...")
    try:
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        return forecast
    except Exception as e:
        print(f"Error training Prophet model: {e}")
        return None

def calculate_technical_indicators(df):
    """Adds RSI and Bollinger Bands to the DataFrame."""
    try:
        # RSI (14-day)
        delta = df['y'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-day, 2 std dev)
        df['SMA_20'] = df['y'].rolling(window=20).mean()
        df['BB_Upper'] = df['SMA_20'] + (df['y'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['y'].rolling(window=20).std() * 2)
        
        return df
    except Exception as e:
        print(f"Error calculating technicals: {e}")
        return df

def generate_market_summary(ticker, sentiment_score, headlines):
    """Generates a concise market summary using Gemini."""
    if not GEMINI_API_KEY:
        return "AI Summary unavailable (Missing API Key)."
        
    try:
        # Switching to stable alias
        try:
             model = genai.GenerativeModel('gemini-flash-latest')
        except:
             model = genai.GenerativeModel('gemini-pro')

        sentiment_desc = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
        
        prompt = f"""
        You are a professional Wall Street Analyst. 
        Ticker: {ticker}
        Detected Sentiment Score: {sentiment_score:.2f} ({sentiment_desc})
        Recent Headlines:
        {headlines}
        
        Write a very concise (max 3 sentences) executive summary explaining the current market sentiment and why the price might move.
        Focus on the "Why". Use professional financial tone.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

def fetch_news_sentiment(ticker):
    """Fetches news via RSS and analyzes sentiment using Gemini.
    Returns: (score, headline_text, error_message)
    """
    if not GEMINI_API_KEY:
        return 0.0, "", "Missing API Key"

    print(f"Fetching news and analyzing sentiment for {ticker}...")
    try:
        # Use Google News RSS
        encoded_ticker = ticker.replace("-", "+") # Handle crypto pairs like BTC-USD -> BTC+USD
        rss_url = f"https://news.google.com/rss/search?q={encoded_ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        headlines = [entry.title for entry in feed.entries[:10]] # Get top 10 headlines
        if not headlines:
            print(f"No headlines found for {ticker}")
            return 0.0, "", "No headlines found"
            
        headline_text = "\n".join(f"- {h}" for h in headlines)
        
        # Call Gemini
        try:
             # Switching to stable alias to hope for better quota
             model = genai.GenerativeModel('gemini-flash-latest')
        except:
             print("Falling back to gemini-pro")
             model = genai.GenerativeModel('gemini-pro')

        prompt = f"""
        Analyze the sentiment of the following news headlines for {ticker}.
        Return a single floating-point number between -1.0 (Extremely Negative) and 1.0 (Extremely Positive).
        Do not explain. Just return the number.
        
        Headlines:
        {headline_text}
        """
        response = model.generate_content(prompt)
        try:
            sentiment_score = float(response.text.strip())
            return sentiment_score, headline_text, None
        except ValueError:
             msg = f"Could not parse sentiment from: {response.text}"
             print(msg)
             return 0.0, headline_text, msg
             
    except Exception as e:
        msg = str(e)
        print(f"Error in sentiment analysis for {ticker}: {msg}")
        return 0.0, "", msg

def main():
    results = []
    
    # Create plots directory
    if not os.path.exists("plots"):
        os.makedirs("plots")

    for ticker in TARGET_STOCKS:
        print(f"\nProcessing {ticker}...")
        
        # 1. Fetch Data
        df = fetch_stock_data(ticker)
        if df is None:
            continue
            
        # 2. Train Prophet
        forecast = train_prophet_model(df)
        if forecast is None:
            continue
            
        # 3. Get Sentiment
        sentiment_score, headlines, error_msg = fetch_news_sentiment(ticker)
        if error_msg:
            print(f"Sentiment Error: {error_msg}")
        print(f"Sentiment Score: {sentiment_score}")
        
        # 4. Hybrid Calculation
        # Get the forecast for the next 30 days (last 30 rows)
        last_30_days = forecast.tail(30).copy()
        
        # Apply sentiment adjustment: 
        # If sentiment is positive (e.g., 0.5), we boost the forecast by 0.5 * weight (e.g., 2.5%)
        # If negative, we reduce it.
        adjustment_factor = 1 + (sentiment_score * SENTIMENT_WEIGHT)
        last_30_days['hybrid_yhat'] = last_30_days['yhat'] * adjustment_factor
        
        # Store result
        current_price = df['y'].iloc[-1]
        predicted_price_raw = last_30_days['yhat'].iloc[-1]
        predicted_price_hybrid = last_30_days['hybrid_yhat'].iloc[-1]
        
        results.append({
            'Ticker': ticker,
            'Current Price': current_price,
            'Forecast (30d)': predicted_price_raw,
            'Hybrid Forecast (30d)': predicted_price_hybrid,
            'Sentiment': sentiment_score
        })
        
        # 5. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['ds'], df['y'], label='Historical')
        plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', linestyle='--')
        plt.plot(last_30_days['ds'], last_30_days['hybrid_yhat'], label='Hybrid Forecast (Sentiment Adjusted)', color='red')
        plt.title(f"{ticker} - Hybrid Forecast (Sentiment: {sentiment_score})")
        plt.legend()
        plt.savefig(f"plots/{ticker}_forecast.png")
        plt.close()
        print(f"Plot saved to plots/{ticker}_forecast.png")

    # Summary
    print("\n=== Hybrid Model Summary ===")
    summary_df = pd.DataFrame(results)
    print(summary_df)

if __name__ == "__main__":
    main()
