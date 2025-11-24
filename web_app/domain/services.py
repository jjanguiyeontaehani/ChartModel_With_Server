import pandas as pd
from collections import namedtuple
import glob
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.config.config import ModelConfig
from model.transformer import TransformerModel
from data.fetch_data import fetch_data
from model.predict import load_model, dataloader_for_prediction, predict

def add_technical_indicators(data):
    # data['Volume'] = z_score_normalization(data['Volume'])

    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']

    data['SMA_10'] = calculate_sma(data, window=10)
    data['SMA_20'] = calculate_sma(data, window=20)
    data['SMA_50'] = calculate_sma(data, window=50)

    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    data['Bollinger_Width'] = data['Upper_Band'] - data['Lower_Band']

    data['RSI'] = calculate_rsi(data)

    return data


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


def z_score_normalization(series):
    return (series - series.mean()) / series.std()


def set_data_index_and_format(data, resolution):
    data.reset_index(inplace=True)
    if resolution == '1h':
        data['Date'] = pd.to_datetime(data['Datetime']).dt.floor('h')
    if resolution == '1d':
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
    return data

def set_df_index_and_format(df, resolution):
    df.reset_index(inplace=True)
    if resolution == '1h':
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    return df

def convert_data_to_dataframe(data, from_date, to_date, resolution):
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
    print(df.head())
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
    df = set_df_index_and_format(df, resolution)
    df = df.sort_values(by='Date')
    df.reset_index(drop=True, inplace=True)

    return df


def predict_price(model_path, dataframe, config: ModelConfig):
    config.batch_size = 1

    model = TransformerModel(config).to(config.device)
    model = load_model(model, model_path)

    dataloader = dataloader_for_prediction(dataframe, config)

    freqs_cis = None
    predictions = predict(model, dataloader, freqs_cis)

    return predictions