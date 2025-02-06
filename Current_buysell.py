import sys
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from fpdf import FPDF

# Ensure Streamlit is installed
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Streamlit is not installed. Please install it using 'pip install streamlit' and try again.")

# Page Configuration
st.set_page_config(page_title="AI-Powered Stock Analysis Tool", layout="wide")
st.title("AI-Powered Stock Analysis Tool")
st.sidebar.header("Configure Analysis")

# User Input for Stock Symbol and Timeframe
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., TCS.BO for BSE):", "RELIANCE.BO")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=4)

# Initialize session state for trade log
if 'trade_log' not in st.session_state:
    st.session_state['trade_log'] = []

@st.cache_data
def fetch_stock_data(symbol, period):
    try:
        # Add a longer minimum period to ensure data retrieval
        if period in ['1d', '5d', '1mo']:
            data = yf.download(symbol, period='3mo')
        else:
            data = yf.download(symbol, period=period)
        
        # If data is empty, try with a longer period
        if data.empty:
            st.warning(f"No data found for {symbol} with period {period}. Trying with a longer period.")
            data = yf.download(symbol, period='max')
        
        # Ensure we have at least some data
        if data.empty:
            st.error(f"Unable to fetch any data for {symbol}. Please check the stock symbol.")
            return pd.DataFrame()
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Fetch stock data
data = fetch_stock_data(stock_symbol, timeframe)

if data.empty:
    st.error("Unable to fetch data for the given symbol.")
    st.stop()

# Technical Indicator Calculations
try:
    # Create a clean DataFrame with just the close prices
    df = pd.DataFrame(data['Close'])
    df.columns = ['close']  # Rename column to 'close' for ta library
    
    # Calculate RSI
    rsi = RSIIndicator(close=df['close'])
    df['RSI'] = rsi.rsi()
    
    # Calculate MACD
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    
    # Calculate Bollinger Bands
    bb = BollingerBands(close=df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    # Merge indicators back into original data
    data['RSI'] = df['RSI']
    data['MACD'] = df['MACD']
    data['BB_High'] = df['BB_High']
    data['BB_Low'] = df['BB_Low']
    
    # Drop NaN values
    data = data.dropna()
    
except Exception as e:
    st.error(f"Error calculating indicators: {str(e)}")
    st.stop()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def generate_stock_response(query, stock_data, symbol):
    """
    Generate response based on stock data and query.
    """
    current_price = stock_data['Close'].iloc[-1]  # Access the last closing price as a scalar
    avg_price = stock_data['Close'].mean()
    rsi_value = stock_data['RSI'].iloc[-1] if 'RSI' in stock_data else None
    
    # Simple response generation based on keywords
    response = ""
    query = query.lower()
    
    if 'price' in query:
        response = f"The current price of {symbol} is ${current_price:.2f}"
    elif 'average' in query:
        response = f"The average price over the selected period is ${avg_price:.2f}"
    elif 'rsi' in query:
        response = f"The current RSI value is {rsi_value:.2f}" if rsi_value is not None else "RSI data is not available"
    elif 'trend' in query:
        trend = "upward" if stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[-5] else "downward"
        response = f"The stock is showing a {trend} trend over the last 5 periods"
    else:
        response = "I can help you analyze stock data. Try asking about price, average, RSI, or trend."
    
    return response


def export_chat_to_pdf(chat_history, selected_messages=None):
    """
    Export chat history to PDF
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Stock Analysis Chat History", ln=True)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    
    messages_to_export = selected_messages if selected_messages else chat_history
    
    for msg in messages_to_export:
        # Add timestamp
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, msg['timestamp'], ln=True)
        
        # Add role and message
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"{'User  ' if msg['is_user'] else 'Assistant'}:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, msg['message'])
        pdf.ln(5)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_export_{timestamp}.pdf"
    pdf.output(filename)
    return filename

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Technical Analysis", "Predictions", "Sentiment", "Chat Assistant"])

# Tab 1: Recent Stock Data
with tab1:
    st.subheader("Recent Stock Data")
    st.dataframe(data.tail(10))
    
    # Basic Statistics
    st.subheader("Basic Statistics")
    st.dataframe(data['Close'].describe())
    
    # Buy and Sell Buttons
    current_price = data['Close'].iloc[-1]
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Buy"):
            x = f"BUY action for {stock_symbol.upper()} at ${current_price.iloc[0]:.2f} USD on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            print(x)
            st.session_state['trade_log'].append(x)
            st.success("BUY action recorded.")
    
    with col2:
        if st.button("Sell"):
            st.session_state['trade_log'].append(
                # f"SELL action for {stock_symbol.upper()} at ${current_price:.2f} USD on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                f"SELL action for {stock_symbol.upper()} at ${current_price.iloc[0]:.2f} USD on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            st.warning("SELL action recorded.")
    
    # Display trade logs
    st.subheader("Trade Log")
    if st.session_state['trade_log']:
        for log in st.session_state['trade_log']:
            st.write(log)
    else:
        st.info("No trades recorded yet.")

# Tab 2: Technical Analysis
with tab2:
    st.subheader("Price Chart")
    fig_price = plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'])
    plt.title('Stock Price')
    st.pyplot(fig_price)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RSI")
        st.line_chart(data['RSI'])
        
    with col2:
        st.subheader("MACD")
        st.line_chart(data['MACD'])
    
    st.subheader("Bollinger Bands")
    bb_chart = pd.DataFrame({
        'Upper Band': data['BB_High'].values.flatten(),
        'Lower Band': data['BB_Low'].values.flatten(),
        'Price': data['Close'].values.flatten()
    }, index=data.index)
    st.line_chart(bb_chart)

# Tab 3: Predictions
with tab3:
    st.subheader("Trading Signals")
    data['Signal'] = 'Hold'
    data.loc[(data['RSI'] < 30) & (data['MACD'] > 0), 'Signal'] = 'Buy'
    data.loc[(data['RSI'] > 70) & (data['MACD'] < 0), 'Signal'] = 'Sell'
    
    signals_df = data[['Close', 'RSI', 'MACD', 'Signal']].tail(10)
    st.dataframe(signals_df)
    
    # ML Model
    st.subheader("ML Price Direction Prediction")
    
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    features = ['RSI', 'MACD', 'BB_High', 'BB_Low']
    X = data[features].dropna()
    y = data['Target'].dropna().iloc[:len(X)]
    
    if len(X) > 10:  # Only run if we have enough data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        })
        st.bar_chart(importance_df.set_index('Feature'))

# Tab 4: Market Sentiment Analysis
with tab4:
    st.subheader("Market Sentiment Analysis")
    sentiment_score = np.random.randint(1, 101)
    sentiment_category = "Positive" if sentiment_score > 60 else "Neutral" if sentiment_score > 40 else "Negative"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment Score", sentiment_score)
    with col2:
        st.metric("Sentiment Category", sentiment_category)
    
    sample_text = f"{stock_symbol} stock market analysis trading investment finance shares equity bull bear trend analysis technical fundamental research"
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sample_text)
    st.image(wordcloud.to_array(), caption="News Sentiment Word Cloud")

# Tab 5: Chat Assistant
with tab5:
    st.subheader("Stock Analysis Chat Assistant")
    
    # Chat interface
    st.write("Ask me anything about the stock analysis!")
    
    # User input
    user_message = st.text_input("Your question:", key="chat_input")
    
    # Process user message
    if user_message:
        st.session_state.chat_history.append({
            'message': user_message,
            'is_user': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        response = generate_stock_response(user_message, data, stock_symbol)
        
        st.session_state.chat_history.append({
            'message': response,
            'is_user': False,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Display chat history
    st.subheader("Chat History")
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['is_user']:
                st.write(f"🙋‍♂ You: {msg['message']}")
            else:
                st.write(f"🤖 Assistant: {msg['message']}")
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Full Chat History"):
            if st.session_state.chat_history:
                filename = export_chat_to_pdf(st.session_state.chat_history)
                with open(filename, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f,
                        file_name=filename,
                        mime="application/pdf"
                    )
            else:
                st.warning("No chat history to export.")
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Custom selection for export
    st.subheader("Export Selected Messages")
    selected_indices = st.multiselect(
        "Select messages to export:",
        range(len(st.session_state.chat_history)),
        format_func=lambda i: f"{st.session_state.chat_history[i]['message'][:50]}..."
    )
    
    if selected_indices:
        if st.button("Export Selected Messages"):
            selected_messages = [st.session_state.chat_history[i] for i in selected_indices]
            filename = export_chat_to_pdf(st.session_state.chat_history, selected_messages)
            with open(filename, "rb") as f:
                st.download_button(
                    "Download Selected Messages PDF",
                    f,
                    file_name=filename,
                    mime="application/pdf"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by LumiSync Team")