import torch
import numpy as np
import yfinance as yf
import pandas as pd

import random
import os
import sys

from domain.services import add_technical_indicators, set_data_index_and_format, convert_data_to_dataframe
from repositories.models import Stock, StockData, ModelStatus

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.config.config import ModelConfig


def fetch_stock_data(symbol, resolution, from_date, to_date):
    data = retrieve_stock_data_from_yfinance(symbol, resolution, from_date, to_date)
    if data[('Volume', symbol)].isnull().all():
        print("No existing data found.")
        return pd.DataFrame()

    data = add_technical_indicators(data)

    data.bfill(inplace=True)

    data = set_data_index_and_format(data, resolution)

    df = convert_data_to_dataframe(data, from_date, to_date, resolution)

    return df


def save_stock_data_to_db(symbol, df, visible_amount=300):
    stock, created = Stock.objects.get_or_create(symbol=symbol)
    for index, row in df.iterrows():
        stock_data, created = StockData.objects.get_or_create(
            stock=stock,
            time=row['Date'],
            defaults={
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'is_predicted': False
            }
        )
        if not created:
            stock_data.open = row['open']
            stock_data.high = row['high']
            stock_data.low = row['low']
            stock_data.close = row['close']
            stock_data.volume = row['volume']
            stock_data.save()

    stock_data_qs = StockData.objects.filter(stock=stock).order_by('-time')
    if stock_data_qs.count() > visible_amount:
        for stock_data in stock_data_qs[visible_amount:]:
            stock_data.delete()


def update_stock_datas(symbol, resolution='1h'):
    from datetime import datetime, timedelta

    for stock in Stock.objects.filter(symbol=symbol):
        last_data = StockData.objects.filter(stock=stock).order_by('-time').first()
        if last_data:
            from_date = pd.to_datetime(last_data.time + timedelta(hours=1)).tz_localize('UTC')
        else:
            from_date = pd.to_datetime(datetime.now() - timedelta(days=30)).tz_localize('UTC')

        to_date = pd.to_datetime(datetime.now()).tz_localize('UTC')

        if from_date >= to_date:
            print(f"No new data to update for {symbol}.")
            return

        df = fetch_stock_data(symbol, resolution, from_date, to_date)
        if not df.empty:
            save_stock_data_to_db(symbol, df)
            print(f"Updated stock data for {symbol} from {from_date} to {to_date}.")
        else:
            print(f"No new data fetched for {symbol} from {from_date} to {to_date}.")


def save_stock_data_to_csv(df, symbol, resolution, from_date, to_date, directory):
    save_dir = f'{directory}/{symbol}_{resolution}_{from_date}_{to_date}.csv'
    os.makedirs(directory, exist_ok=True)

    df.to_csv(save_dir, index=False)

    if not os.path.exists(save_dir):
        raise FileNotFoundError("[ERROR] Failed to save data to CSV.")
    
    return save_dir


def retrieve_stock_data_from_yfinance(symbol, resolution, from_date, to_date):
    data = yf.download(symbol, start=from_date, end=to_date, interval=resolution)

    if data[('Volume', symbol)].isnull().all():
        raise ValueError("[ERROR] No data found for the specified date range and resolution.")

    return data


def retrieve_stock_data_from_db(symbol, from_date, to_date, resolution='1h'):
    stock = Stock.objects.get(symbol=symbol)
    stock_data_qs = StockData.objects.filter(
        stock=stock,
        time__range=(from_date, to_date)
    ).order_by('time')

    data = {
        'Date': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
        'is_predicted': []
    }

    for stock_data in stock_data_qs:
        data['Date'].append(stock_data.time)
        data['open'].append(stock_data.open)
        data['high'].append(stock_data.high)
        data['low'].append(stock_data.low)
        data['close'].append(stock_data.close)
        data['volume'].append(stock_data.volume)
        data['is_predicted'].append(stock_data.is_predicted)

    data = add_technical_indicators(data)

    data.fillna(method='bfill', inplace=True)

    data = set_data_index_and_format(data, resolution)

    df = convert_data_to_dataframe(data, from_date, to_date, resolution)

    return df


def retrieve_model_status_from_db(symbol):
    stock = Stock.objects.get(symbol=symbol)
    model_status_qs = stock.modelstatus_set.order_by('-last_trained')

    if model_status_qs.exists():
        model_status = model_status_qs.first()
        return {
            'is_trained': model_status.is_trained,
            'last_trained': model_status.last_trained,
            'model_path': model_status.model_path
        }
    else:
        return {
            'is_trained': False,
            'last_trained': None,
            'model_path': None
        }
    

def retrieve_stock_list_from_db():
    stocks = Stock.objects.all()
    stock_list = [stock.symbol for stock in stocks]
    return stock_list


def train_model(train_file_paths, model_save_path, config: ModelConfig):
    from model.train import train_model, load_data
    from model.transformer import TransformerModel
    from data.preprocess import preprocessData
    from torch.utils.tensorboard import SummaryWriter

    symbol = train_file_paths[0].split('/')[-1].split('_')[0]

    writer = SummaryWriter(
        log_dir="logs/seed_" + 
        str(config.seed) + 
        "/symbol_" + symbol
    )

    if config.seed:
        setSeed(config.seed)

    train_dataloader, validation_dataloader, test_dataloader = load_data(
        paths=train_file_paths,
        config=config
    )

    model = TransformerModel(config)
    model.to(config.device)
    model.train()
    trained_model, test_accuracy = train_model(
        model, train_dataloader, validation_dataloader, test_dataloader, config, writer
    )

    os.makedirs("model/trained", exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)

    writer.close()

    model_status = ModelStatus(
        model_name=symbol,
        last_trained=pd.Timestamp.now(),
        accuracy=test_accuracy,
        status="Trained"
    )
    model_status.save()

    

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
