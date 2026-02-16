import hybrid_model as hm
import os
from dotenv import load_dotenv

load_dotenv()

print(f"API KEY Present: {bool(os.getenv('GEMNI_API_KEY_2'))}")
print(f"API KEY (GEMNI_API_KEY_2): {os.getenv('GEMNI_API_KEY_2')[:5]}...")

ticker = "AAPL"
print(f"Testing sentiment for {ticker}...")
score = hm.fetch_news_sentiment(ticker)
print(f"Score for {ticker}: {score}")

ticker = "BTC-USD"
print(f"Testing sentiment for {ticker}...")
score = hm.fetch_news_sentiment(ticker)
print(f"Score for {ticker}: {score}")
