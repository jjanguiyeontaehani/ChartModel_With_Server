import argparse
import yfinance as yf
import pandas as pd
import time
import os

def fetch_data(args):

    from_date = pd.to_datetime(args.start_date).tz_localize('UTC')
    to_date = pd.to_datetime(args.end_date).tz_localize('UTC')
    directory = 'data/raw'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fetch_data_from_yfinance(args.ticker, args.resolution, from_date, to_date, directory)


def fetch_data_from_yfinance(symbol, resolution, from_date, to_date, directory):
    data = yf.download(symbol, start=from_date, end=to_date, interval=resolution)

    if data[('Volume', symbol)].isnull().all():
        print("No data found for the specified date range and resolution.")
        if os.path.exists(f'{directory}/{symbol}_{resolution}_{from_date}_{to_date}.csv'):
            print(f"Loading existing data from {directory}/{symbol}_{resolution}_{from_date}_{to_date}.csv")
            df = pd.read_csv(f'{directory}/{symbol}_{resolution}_{from_date}_{to_date}.csv')
        else:
            print("No existing data found.")
            exit(1)
    else:
        def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
            short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
            long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=signal_window, adjust=False).mean()
            return macd, signal_line
        
        def calculate_sma(data, window=20):
            return data['Close'].rolling(window=window).mean()
        
        def calculate_bollinger_bands(data, window=20, num_std_dev=2):
            sma = calculate_sma(data, window)
            rolling_std = data['Close'].rolling(window=window).std()
            upper_band = sma + (rolling_std * num_std_dev)
            lower_band = sma - (rolling_std * num_std_dev)
            return upper_band, lower_band
        
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        # def z_score_normalization(series):
        #     return (series - series.mean()) / series.std()
        
        # data['Volume'] = z_score_normalization(data['Volume'])
        data['MACD'], data['Signal_Line'] = calculate_macd(data)
        data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
        data['SMA_10'] = calculate_sma(data, window=10)
        data['SMA_20'] = calculate_sma(data, window=20)
        data['SMA_50'] = calculate_sma(data, window=50)
        data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
        data['Bollinger_Width'] = data['Upper_Band'] - data['Lower_Band']
        data['RSI'] = calculate_rsi(data)
        data.fillna(method='bfill', inplace=True)

        print("Data retrieved successfully")
        data.reset_index(inplace=True)
        if resolution == '1h':
            data['Date'] = pd.to_datetime(data['Datetime']).dt.floor('h')
        else:
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'MACD', 'Signal_Line', 'MACD_Hist',
                    'SMA_10', 'SMA_20', 'SMA_50',
                    'Upper_Band', 'Lower_Band', 'Bollinger_Width',
                    'RSI']]
        data.columns = ['Date', 'open', 'high', 'low', 'close', 'volume',
                        'macd', 'signal_line', 'macd_hist',
                        'sma_10', 'sma_20', 'sma_50',
                        'upper_band', 'lower_band', 'bollinger_width',
                        'rsi']
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[(df.index >= from_date) & (df.index <= to_date)]
        df = df[['open', 'high', 'low', 'close', 'volume',
                    'macd', 'signal_line', 'macd_hist',
                    'sma_10', 'sma_20', 'sma_50',
                    'upper_band', 'lower_band', 'bollinger_width',
                    'rsi']]
        df = df.astype({
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'int64',
            'macd': 'float32',
            'signal_line': 'float32',
            'macd_hist': 'float32',
            'sma_10': 'float32',
            'sma_20': 'float32',
            'sma_50': 'float32',
            'upper_band': 'float32',
            'lower_band': 'float32',
            'bollinger_width': 'float32',
            'rsi': 'float32'
        })
        df.reset_index(inplace=True)
        if resolution == '1h':
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.sort_values(by='Date')
        df.reset_index(drop=True, inplace=True)

        df.to_csv(f'{directory}/{symbol}_{resolution}_{from_date.strftime("%Y-%m-%d")}_{to_date.strftime("%Y-%m-%d")}.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch and process financial data.")
    parser.add_argument(
        '--ticker',
        type=str,
        default='AAPL',
        help='Ticker symbol of the stock')
    parser.add_argument(
        '--resolution',
        type=str,
        default='1h',
        help='Data resolution (e.g., 1m, 5m, 1h, 1d)')
    parser.add_argument(
        '--start_date',
        type=str,
        # 729 days ago
        default=time.strftime('%Y-%m-%d', time.gmtime(time.time() - 729*24*60*60)),
        help='Start date in YYYY-MM-DD format')
    parser.add_argument(
        '--end_date',
        type=str,
        default=time.strftime('%Y-%m-%d', time.gmtime()),
        help='End date in YYYY-MM-DD format')

    args = parser.parse_args()

    fetch_data(args)