import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
from transformers import pipeline

# Fetch stock data using yfinance for NSE and BSE
def fetch_stock_data(stock_symbol, start_date='2023-09-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"Error: No data found for {stock_symbol}. Please try another stock.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Sentiment Analysis using VADER
def analyze_sentiment(news_data):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for headline in news_data:
        sentiment = sid.polarity_scores(headline)
        sentiment_scores.append(sentiment['compound'])
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Prepare stock data for LSTM model
def prepare_stock_data(stock_data, hours_ahead=1):
    close_prices = stock_data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Ensure there are at least 60 data points for LSTM input
    if len(stock_data) < 60:
        print(f"Not enough data for {len(stock_data)} points. At least 60 data points are required.")
        return None, None, None

    x_data, y_data = [], []
    for i in range(60, len(stock_data) - hours_ahead):
        x_data.append(scaled_data[i-60:i, 0])
        y_data.append(scaled_data[i + hours_ahead, 0])

    x_data, y_data = np.array(x_data), np.array(y_data)

    if x_data.shape[0] == 0:  # Check if no data was generated
        print(f"Not enough data after preparation. Available data points: {len(stock_data)}")
        return None, None, None

    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data, scaler

# Create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict stock price for the next few hours using LSTM
def predict_stock_price(model, scaler, stock_data, hours_ahead=1):
    inputs = stock_data[['Close']].tail(60).values
    inputs = scaler.transform(inputs)
    inputs = np.reshape(inputs, (1, 60, 1))

    predicted_prices = []
    for _ in range(hours_ahead):
        prediction = model.predict(inputs, verbose=0)
        predicted_prices.append(prediction[0, 0])

        # Reshape prediction to match the dimensions of inputs
        prediction = prediction.reshape(1, 1, 1)
        inputs = np.append(inputs[:, 1:, :], prediction, axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices

# Fetch news headlines for sentiment analysis
def fetch_news_headlines(stock_symbol):
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey=21b8a91903ef44c88b237241dff5f9ff'
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json().get('articles', [])
        headlines = [article['title'] for article in news_data]
        if not headlines:
            print(f"No news articles found for {stock_symbol}.")
        return headlines[:5]  # Return the top 5 headlines
    except Exception as e:
        print(f"Error fetching news for {stock_symbol}: {e}")
        return []

# Summarize recommendations using Hugging Face
def summarize_recommendations(stock_symbol, sentiment_score, final_prediction, long_term_recommendation):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    prompt = (
        f"Stock analysis for {stock_symbol}: The sentiment score is {sentiment_score:.2f}, "
        f"the predicted price is ₹{final_prediction:.2f}, "
        f"and the long-term recommendation is '{long_term_recommendation}'. "
        f"Provide concise investment advice."
    )

    try:
        summary = summarizer(prompt, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        print("\nInvestment Advice Summary:")
        print(summary)
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary unavailable."

# Integrate sentiment score and stock prediction for a stock
def integrate_prediction_and_sentiment(stock_symbol, hours_ahead=1):
    stock_data = fetch_stock_data(stock_symbol)
    if stock_data is None:
        return None, None, None

    # Ensure there is enough data
    if len(stock_data) < 60:  # You can change this to any other number like 20
        print(f"Not enough data for {stock_symbol}. Prediction cannot be made.")
        return None, None, None

    x_data, y_data, scaler = prepare_stock_data(stock_data, hours_ahead=hours_ahead)

    model = create_lstm_model((x_data.shape[1], 1))
    model.fit(x_data, y_data, epochs=10, batch_size=32, verbose=0)

    predicted_prices = predict_stock_price(model, scaler, stock_data, hours_ahead)
    news_headlines = fetch_news_headlines(stock_symbol)
    sentiment_score = analyze_sentiment(news_headlines)

    print(f"Predicted Stock Prices for {stock_symbol} for the next {hours_ahead} hour(s):")
    for i, price in enumerate(predicted_prices):
        print(f"Hour {i + 1}: ₹{price[0]:.2f}")

    print("\nRecent News Headlines:")
    if not news_headlines:
        print(f"- No news available for {stock_symbol}.")
    else:
        for headline in news_headlines:
            print(f"- {headline}")

    print(f"\nSentiment Score: {sentiment_score:.2f}")
    final_prediction = predicted_prices[-1][0] + sentiment_score * 5
    print(f"Final Stock Prediction with Sentiment Adjustment: ₹{final_prediction:.2f}")

    long_term_recommendation = "Hold for more than 1 year for stable growth." if sentiment_score > 0 else "Consider alternative investments."
    summarize_recommendations(stock_symbol, sentiment_score, final_prediction, long_term_recommendation)

    return stock_data, predicted_prices, final_prediction



# Plot stock data and predictions
def plot_stock_data(stock_symbol, stock_data, predicted_prices, hours_ahead=5, final_prediction=None):
    plt.figure(figsize=(12, 6))

    # Historical data
    plt.subplot(2, 1, 1)
    plt.plot(stock_data.index, stock_data['Close'], label='Historical Data', color='blue')
    plt.title(f'{stock_symbol} Historical Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()

    # Predicted data
    last_time = stock_data.index[-1]
    future_dates = [last_time + timedelta(hours=i) for i in range(1, hours_ahead + 1)]

    plt.subplot(2, 1, 2)
    plt.plot(future_dates, predicted_prices, marker='o', label='Predicted Prices', color='red')
    plt.title(f'{stock_symbol} Predicted Stock Prices (Next {hours_ahead} Hours)')
    plt.xlabel('Date and Time')
    plt.ylabel('Price (INR)')
    plt.legend()

    # Suggestion for buying or selling based on predicted price
    if final_prediction is not None:
        last_price = stock_data['Close'].iloc[-1].item()  # Ensure it's a scalar value
        print(f"\nLast Known Price: ₹{last_price:.2f}")  # Format the price properly
        print(f"Final Predicted Price: ₹{final_prediction:.2f}")
        
        # Reverse logic: Sell when predicted price is higher than last known price, Buy when it's lower
        if final_prediction > last_price:
            suggestion = "Sell"  # Stock price is predicted to go up, sell now
        elif final_prediction < last_price:
            suggestion = "Buy"  # Stock price is predicted to go down, buy now
        else:
            suggestion = "Hold"  # No significant change expected
        
        print(f"\nYou Should: {suggestion}")
        plt.figtext(0.5, 0.01, f"You Should: {suggestion}", ha="center", fontsize=12, color="black")

    plt.tight_layout()
    plt.show()


    # Suggestion for buying or selling based on predicted price
    if final_prediction is not None:
        last_price = stock_data['Close'].iloc[-1].item()  # Convert to a scalar value

        print(f"\nLast Known Price: ₹{last_price:.2f}")
        print(f"Final Predicted Price: ₹{final_prediction:.2f}")
        
        if final_prediction > last_price:
            suggestion = "Sell"
        elif final_prediction < last_price:
            suggestion = "Buy"
        else:
            suggestion = "Hold"
        
        print(f"\nYou Should: {suggestion}")
        plt.figtext(0.5, 0.01, f"You Should: {suggestion}", ha="center", fontsize=12, color="black")

    plt.tight_layout()
    plt.show()

# Main program
def main():
    major_stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 'BAJAJFINSV.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'MARUTI.NS', 'TATAMOTORS.NS']
    
    while True:
        print("Select a stock from the following list:")
        for i, stock in enumerate(major_stocks, 1):
            print(f"{i}. {stock}")
        
        try:
            user_choice = int(input("Enter the number corresponding to your choice (0 to exit): "))
            if user_choice == 0:
                print("Exiting the program. Goodbye!")
                break

            if user_choice < 1 or user_choice > len(major_stocks):
                print("Invalid choice. Please try again.")
                continue

            stock_symbol = major_stocks[user_choice - 1]
            print(f"Processing predictions for {stock_symbol}...")

            stock_data, predicted_prices, final_prediction = integrate_prediction_and_sentiment(stock_symbol, hours_ahead=5)
            if stock_data is None or predicted_prices is None:
                print("Prediction could not be completed. Please try another stock.")
                continue

            plot_stock_data(stock_symbol, stock_data, predicted_prices, hours_ahead=5, final_prediction=final_prediction)
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
