from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go

app = Flask(__name__)

# Define a list of stocks with symbols and names
stocks = [
    {'symbol': 'AAPL', 'name': 'Apple Inc.'},
    {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
    {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
    {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
    {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
    {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
    {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
    {'symbol': 'PYPL', 'name': 'PayPal Holdings Inc.'},
    {'symbol': 'ADBE', 'name': 'Adobe Inc.'},
    {'symbol': 'CRM', 'name': 'Salesforce.com Inc.'}
]

def analyze_stock(symbol, start_date, end_date):
    # Fetch stock data using yfinance
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Create a Plotly figure for the historical closing prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price',
                             hoverinfo='x+y', hovertemplate='%{x|%Y-%m-%d %H:%M:%S}<br>Closing Price: $%{y:.2f}'))
    fig.update_layout(title=f'Historical Closing Prices for {symbol}', xaxis_title='DateTime', yaxis_title='Price')

    # Convert the Plotly graph to HTML
    plot_html = fig.to_html(full_html=False)

    # Calculate summary statistics
    summary_stats = stock_data.describe()

    # Calculate trend analysis
    trend = 'Bullish' if stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[0] else 'Bearish'

    # Extract volume data
    volume_data = [(date.strftime('%Y-%m-%d %H:%M:%S'), volume) for date, volume in zip(stock_data.index, stock_data['Volume'])]

    return {
        'plot': plot_html,
        'mean': summary_stats.loc['mean', 'Close'],
        'std': summary_stats.loc['std', 'Close'],
        'trend': trend,
        'volume': volume_data
    }


def predict_stock_prices(symbol):
    # Fetch historical stock data using yfinance
    stock_data = yf.download(symbol, start='2024-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    stock_data.dropna(inplace=True)

    # Extract features and target variable
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = np.array(stock_data['Close'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict next week's closing prices
    next_week = np.array(range(len(stock_data), len(stock_data) + 7)).reshape(-1, 1)
    next_prices = model.predict(next_week)

    # Create a list of dates for next week
    current_date = datetime.now()
    next_week_dates = [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]

    return next_week_dates, next_prices


@app.route('/')
def index():
    return render_template('index.html', stocks=stocks)

@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Perform analysis
    analysis_results = analyze_stock(symbol, start_date, end_date)

    return render_template('analysis.html', symbol=symbol, analysis_results=analysis_results)

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']

    # Predict next week's closing prices
    next_week_dates, next_prices = predict_stock_prices(symbol)

    # Pass the data to the template
    return render_template('predict.html', symbol=symbol, next_week_dates=next_week_dates, next_prices=next_prices)


if __name__ == '__main__':
    app.run(debug=True)
