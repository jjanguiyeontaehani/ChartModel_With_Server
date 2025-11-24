import sys
import time
import glob
import os
from celery import shared_task
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.controller import fetch_stock_data, retrieve_stock_data_from_db, save_stock_data_to_csv, save_stock_data_to_db
from services.controller import update_stock_datas
from services.controller import train_model
from repositories.models import Stock, StockData
from model.config.config import ModelConfig
from domain.services import predict_price

config = ModelConfig()

@shared_task
def update_models():
    for stock in Stock.objects.all():
        symbol = stock.symbol
        from_date = time.strftime('%Y-%m-%d', time.gmtime(time.time() - config.window_size* 24 * 60 * 60))
        to_date = time.strftime('%Y-%m-%d', time.gmtime())

        model_path = f'model/trained/{symbol}*.pth'
        model_files = glob.glob(model_path,recursive=True)
        model_files.sort()
        if not model_files:
            raise FileNotFoundError(f"No model files found for {model_path}")
        latest_model_file = model_files[-1]

        update_stock_datas(symbol, resolution='1h')

        df_for_prediction = retrieve_stock_data_from_db(symbol, from_date, to_date, resolution='1h')

        latest_date = df_for_prediction['Date'].max()

        predicted_values = predict_price(latest_model_file, df_for_prediction, config)

        predicted_df = pd.DataFrame(predicted_values, columns=['close'])
        # add date column to predicted_df as index + 1 hour from df_for_prediction
        predicted_df['Date'] = [latest_date + pd.Timedelta(hours=i+1) for i in range(len(predicted_df))]
        predicted_df['open'] = df_for_prediction['close'].iloc[-1]
        predicted_df['is_predicted'] = True

        save_stock_data_to_db(symbol, predicted_df, visible_amount=300)

@shared_task
def train_on_request(symbol):
    from_date = time.strftime('%Y-%m-%d', time.gmtime(time.time() - 729*24*60*60))
    to_date = time.strftime('%Y-%m-%d', time.gmtime())

    df = fetch_stock_data(symbol, '1h', from_date, to_date)

    if df.empty:
        print(f"No data available for {symbol}. Skipping training.")
        return
    
    train_data_dir = save_stock_data_to_csv(df, symbol, '1h', from_date, to_date, directory='data/raw')
    valid_data_dir = save_stock_data_to_csv(df, 'QQQ', '1h', from_date, to_date, directory='data/raw')
    test_data_dir = save_stock_data_to_csv(df, '^GSPC', '1h', from_date, to_date, directory='data/raw')
    model_save_path = f'model/trained/{symbol}_model.pth'

    train_model([train_data_dir, valid_data_dir, test_data_dir], model_save_path, config)

    update_models.delay()
