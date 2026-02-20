# Hybrid Time Sesries Model 

A sophisticated forecasting tool combining historical data analysis (Prophet) with real-time news sentiment (Gemini AI) to predict stock prices. Now featuring enhanced visualizations including Candlestick charts, technical indicators, and volatility analysis.

## Features

*   **Hybrid Forecasting**: Blends time-series trends with AI-driven sentiment analysis.
*   **Interactive Dashboard**: Built with Streamlit for a responsive user experience.
*   **Advanced Charts**:
    *   **Candlestick View**: Analyze OHLC data with zoom and pan capabilities.
    *   **Technical Indicators**: Overlay SMA 50, SMA 200, Bollinger Bands, and RSI.
    *   **Confidence Intervals**: Visualize model uncertainty.
*   **Deep Insights**:
    *   **Trend & Seasonality**: Decompose forecasts into global trends and seasonal patterns.
    *   **Volatility Analysis**: Understand risk through daily return distributions.
*   **AI Market Analyst**: Generates concise textual summaries of market conditions using Gemini.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/palash9e-stato/hybrid-timeseries-model.git
    cd hybrid-timeseries-model
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up environment variables:
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```
    GEMNI_API_KEY_2=your_api_key_here
    ```

## Usage

Run the dashboard:
```bash
streamlit run dashboard.py
```
Or if `streamlit` is not in your PATH:
```bash
python -m streamlit run dashboard.py
```

## Technologies Used

*   **Python**: Core language.
*   **Streamlit**: Web framework.
*   **Prophet**: Time-series forecasting.
*   **Google Gemini (GenAI)**: Sentiment analysis and text generation.
*   **Plotly**: Interactive visualizations.
*   **yfinance**: Market data retrieval.

