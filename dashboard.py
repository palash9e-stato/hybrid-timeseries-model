import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hybrid_model as hm
import time

# Page Config
st.set_page_config(page_title="Hybrid Stock Forecaster", page_icon="📈", layout="wide")

# Title and Description
st.title("📈 Hybrid Time-Series & Sentiment Forecaster")
st.markdown("""
This dashboard combines **Prophet** (Historical Trend Analysis) with **Gemini 2.5 Flash** (Real-time News Sentiment) 
to predict stock prices.
""")

# Sidebar
st.sidebar.header("Configuration")
selected_ticker = st.sidebar.selectbox("Select Asset", hm.TARGET_STOCKS)

# Date Picker (Limit to next 60 days for reasonable Prophet accuracy)
min_date = datetime.now().date()
max_date = min_date + timedelta(days=60)
selected_date = st.sidebar.date_input("Prediction Date", min_value=min_date, max_value=max_date, value=min_date + timedelta(days=7))

# --- Caching Functions for Performance ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_cached_data(ticker):
    # Reduced period to 1y for faster dashboard performance
    # Original hybrid_model use '2y', we override here or modify fetch_stock_data to accept period
    # To avoid changing hybrid_model signature globally and breaking scripts, 
    # we'll just monkeypath or better yet, create a local variant or modify fetch_stock_data to take kwargs
    return hm.fetch_stock_data(ticker) 

@st.cache_data(ttl=3600)  # Cache forecast for 1 hour
def get_cached_forecast(data, _ticker): 
    return hm.train_prophet_model(data)

@st.cache_data(ttl=1800) # Cache sentiment for 30 mins
def get_cached_sentiment_v3(ticker):
    return hm.fetch_news_sentiment(ticker)

@st.cache_data(ttl=3600)
def get_cached_summary(ticker, score, headlines):
    return hm.generate_market_summary(ticker, score, headlines)

# --- Main App Logic ---

# Sidebar Enhancements
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Advanced Tools")
show_technicals = st.sidebar.multiselect("Technical Indicators", ["Bollinger Bands", "RSI", "SMA 50", "SMA 200"], default=[])
enable_scenario = st.sidebar.checkbox("Enable 'What-If' Simulation")
scenario_sentiment = 0.0
if enable_scenario:
    scenario_sentiment = st.sidebar.slider("Simulate Sentiment", -1.0, 1.0, 0.0, 0.1, help="Override actual sentiment to see potential price impact.")

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared! Rerunning...")
    time.sleep(1)
    st.rerun()

if st.sidebar.button("Run Analysis", type="primary"):
    with st.status(f"Analyzing {selected_ticker}...", expanded=True) as status:
        try:
            # 1. Fetch Data
            status.write("Fetching historical data...")
            df = get_cached_data(selected_ticker)
            
            if df is None or df.empty:
                status.update(label="Error fetching data", state="error")
                st.error(f"Could not fetch data for {selected_ticker}")
            else:
                # Calculate Technicals (Feature 3)
                df = hm.calculate_technical_indicators(df)

                # 2. Get Forecast
                status.write("Training Prophet model (this can take a few seconds)...")
                forecast = get_cached_forecast(df, selected_ticker)
                
                # 3. Get Sentiment
                status.write("Analyzing news sentiment with Gemini...")
                actual_sentiment_score, headlines_text, error_msg = get_cached_sentiment_v3(selected_ticker)
                
                if error_msg:
                    status.write(f"⚠️ Sentiment Warning: {error_msg}")
                    st.warning(f"Sentiment Analysis Failed: {error_msg}. Using 0.0 (Neutral).")
                
                # 'What-If' Logic (Feature 2)
                used_sentiment = scenario_sentiment if enable_scenario else actual_sentiment_score
                
                # 4. AI Summary (Feature 1)
                status.write("Generating AI Market Commentary...")
                ai_summary = get_cached_summary(selected_ticker, used_sentiment, headlines_text)

            # 5. Process Forecast
            status.write("Calculating hybrid forecast...")
            target_ds = pd.to_datetime(selected_date)
            
            future_mask = forecast['ds'] > df['ds'].max()
            forecast['hybrid_yhat'] = forecast['yhat']
            
            # Apply sentiment adjustment
            adjustment_factor = 1 + (used_sentiment * hm.SENTIMENT_WEIGHT)
            forecast.loc[future_mask, 'hybrid_yhat'] = forecast.loc[future_mask, 'yhat'] * adjustment_factor

            # Find predicted price for selected date
            closest_row = forecast.iloc[(forecast['ds'] - target_ds).abs().argsort()[:1]]
            predicted_price = closest_row['hybrid_yhat'].values[0]
            raw_prediction = closest_row['yhat'].values[0]
            current_price = df['y'].iloc[-1]
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # --- Advanced Results Display ---
            
            # Feature 1: AI Commentary
            st.subheader(f"🧠 AI Market Analyst: {selected_ticker}")
            st.info(ai_summary)

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Current Price", f"${current_price:,.2f}")
            with col2:
                sentiment_label = "Neutral 😐"
                if used_sentiment > 0.1: sentiment_label = "Positive 🚀"
                elif used_sentiment < -0.1: sentiment_label = "Negative 📉"
                label_suffix = " (Simulated)" if enable_scenario else ""
                st.metric(f"Sentiment Score{label_suffix}", f"{used_sentiment:.2f}", sentiment_label)
            with col3:
                diff = predicted_price - current_price
                st.metric(f"Prediction ({selected_date})", f"${predicted_price:,.2f}", f"{diff:,.2f}")
            with col4:
                impact = predicted_price - raw_prediction
                st.metric("News Impact", f"${impact:,.2f}", help="Difference caused by sentiment analysis")

            # --- Visualizations ---
            tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "🔍 Model Components", "📊 Volatility Analysis"])

            with tab1:
                st.subheader("Price Forecast & Technicals")
                
                # Chart Type Selector
                chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
                
                fig = go.Figure()
                
                # Historical Price
                if chart_type == "Candlestick" and 'Open' in df.columns:
                    fig.add_trace(go.Candlestick(x=df['ds'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['y'],
                                    name='Historical OHLC'))
                else:
                    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Price', line=dict(color='gray', width=1)))
                
                # Confidence Interval (Prophet)
                # We use the raw Prophet confidence interval but shift it by the hybrid adjustment factor for consistency
                # Note: This is a simplification. Rigorous hybrid CI would require retraining.
                lower_bound = forecast.loc[future_mask, 'yhat_lower'] * adjustment_factor
                upper_bound = forecast.loc[future_mask, 'yhat_upper'] * adjustment_factor
                
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast.loc[future_mask, 'ds'], forecast.loc[future_mask, 'ds'][::-1]]),
                    y=pd.concat([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=True,
                    name='Confidence Interval'
                ))

                # SMA Lines
                # Calculate on the fly for visualization
                if "SMA 50" in show_technicals:
                    sma50 = df['y'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(x=df['ds'], y=sma50, mode='lines', name='SMA 50', line=dict(color='orange', width=1)))
                
                if "SMA 200" in show_technicals:
                    sma200 = df['y'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=df['ds'], y=sma200, mode='lines', name='SMA 200', line=dict(color='blue', width=1)))

                # Bollinger Bands
                if "Bollinger Bands" in show_technicals:
                    fig.add_trace(go.Scatter(x=df['ds'], y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(width=1, color='rgba(173, 216, 230, 0.5)')))
                    fig.add_trace(go.Scatter(x=df['ds'], y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(width=1, color='rgba(173, 216, 230, 0.5)'), fill='tonexty'))

                # Forecasts
                future_df = forecast[future_mask]
                fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat'], mode='lines', name='Prophet Baseline', line=dict(dash='dash', color='blue', width=1)))
                fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['hybrid_yhat'], mode='lines', name='Hybrid Forecast', line=dict(color='red', width=2)))
                
                # Markers
                fig.add_vline(x=datetime.combine(selected_date, datetime.min.time()).timestamp() * 1000, line_dash="dot", line_color="green", annotation_text="Selected Date")
                fig.add_trace(go.Scatter(x=[target_ds], y=[predicted_price], mode='markers', marker=dict(color='green', size=12), name='Your Prediction'))

                fig.update_layout(title=f"{selected_ticker} Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Chart (if selected)
                if "RSI" in show_technicals:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df['ds'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=300, yaxis_title="RSI", yaxis_range=[0, 100])
                    st.plotly_chart(fig_rsi, use_container_width=True)

            with tab2:
                st.subheader("Decomposition: Trend & Seasonality")
                
                # Extract components from Prophet forecast
                # Trend
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')))
                fig_trend.update_layout(title="Global Trend Component", height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                col_seas1, col_seas2 = st.columns(2)
                
                # Weekly Seasonality (if available)
                if 'weekly' in forecast.columns:
                    # Create a dummy dataframe to plot one week cycle
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    # We can avg the weekly component per day of week roughly or just plot the component
                    # A better way is to slice the forecast for one week, but let's just plot the component over time? NO.
                    # Standard way: Aggregation.
                    forecast['day_of_week'] = forecast['ds'].dt.day_name()
                    weekly_seasonality = forecast.groupby('day_of_week')['weekly'].mean().reindex(days)
                    
                    with col_seas1:
                        fig_weekly = go.Figure()
                        fig_weekly.add_trace(go.Bar(x=weekly_seasonality.index, y=weekly_seasonality.values, name='Weekly'))
                        fig_weekly.update_layout(title="Weekly Seasonality Effect", height=300)
                        st.plotly_chart(fig_weekly, use_container_width=True)

                # Yearly Seasonality (if available)
                if 'yearly' in forecast.columns:
                    forecast['month'] = forecast['ds'].dt.month_name()
                    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                    yearly_seasonality = forecast.groupby('month')['yearly'].mean().reindex(months)
                     
                    with col_seas2:
                         fig_yearly = go.Figure()
                         fig_yearly.add_trace(go.Bar(x=yearly_seasonality.index, y=yearly_seasonality.values, name='Yearly'))
                         fig_yearly.update_layout(title="Yearly Seasonality Effect", height=300)
                         st.plotly_chart(fig_yearly, use_container_width=True)

            with tab3:
                st.subheader("Risk & Volatility")
                # Return Distribution
                df['daily_return'] = df['y'].pct_change()
                clean_returns = df['daily_return'].dropna()
                
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Histogram(x=clean_returns, nbinsx=50, name='Daily Returns', histnorm='probability density', marker_color='blue'))
                
                # Add KDE (Kernel Density Estimate) - simplified as a normal distribution fit for visualization 
                # (since we don't have scipy/seaborn explicitly in all envs, let's stick to histogram for robustness or simple normal curve)
                mean_ret = clean_returns.mean()
                std_ret = clean_returns.std()
                x_range = pd.Series(clean_returns).sort_values()
                # pdf = (1 / (std_ret * (2 * 3.14159)**0.5)) * np.exp(-0.5 * ((x_range - mean_ret) / std_ret)**2) # Needs numpy
                # Let's import numpy inside or use pandas math for basic curve if needed, but Histogram is often enough.
                # Actually, plain Histogram is great.
                
                fig_vol.add_vline(x=mean_ret, line_dash="dash", line_color="green", annotation_text="Mean Return")
                fig_vol.add_vline(x=mean_ret - 2*std_ret, line_dash="dot", line_color="red", annotation_text="-2 Std Dev")
                fig_vol.add_vline(x=mean_ret + 2*std_ret, line_dash="dot", line_color="red", annotation_text="+2 Std Dev")

                fig_vol.update_layout(title="Daily Return Distribution", xaxis_title="Daily Return", yaxis_title="Density", height=400)
                st.plotly_chart(fig_vol, use_container_width=True)
                
                st.metric("Annualized Volatility", f"{std_ret * (252**0.5):.2%}")

            # --- News Section ---
            with st.expander("📰 Raw Headlines"):
                st.text(headlines_text)
        except Exception as e:
            status.update(label="Analysis Failed", state="error")
            st.error(f"An error occurred during analysis: {e}")
            st.code(str(e)) # Show detailed error for debugging


else:
    st.info("Select a ticker and date from the sidebar, then click 'Run Analysis'.")
