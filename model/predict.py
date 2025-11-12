import argparse
import csv
import glob
import time
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.config.config import ModelConfig
from model.transformer import TransformerModel
from data.fetch_data import fetch_data


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

class DatasetForPrediction(Dataset):
    def __init__(self, data, config):
        self.config = config
        self.all_cols = config.all_cols
        self.feature_cols = config.feature_cols
        self.target_cols = config.target_cols
        self.target_val_idx = config.all_cols.index(config.target_cols[0])
        self.window_size = config.window_size
        self.future_offset = config.future_offset

        self.np = data

        self.index = pd.to_datetime(self.np[:, 0], unit='s')
        
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        
        last_valid_start_index = len(self.np) - self.window_size - max(self.future_offset)
        
        if last_valid_start_index >= 0:
            start = last_valid_start_index
            val_range = np.arange(start, start + self.window_size)
            
            samples.append(val_range)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices = self.samples[idx]

        exclude_indices = []
        for i, col in enumerate(self.target_cols):
            if i != 0:
                exclude_indices.append(self.all_cols.index(col))

        data = self.np[indices, 1:]
        data = np.delete(data, exclude_indices, axis=1)

        data_tensor = torch.from_numpy(data)
        
        norm_val = data_tensor[-1, self.target_val_idx].item()
        
        data_tensor = torch.cat([data_tensor, data_tensor[:, self.target_val_idx].unsqueeze(1) / norm_val], dim=1)

        return data_tensor, torch.tensor(norm_val, dtype=torch.float32)
    

def dataloader_for_prediction(file_path, config):
    def read_file(path):
        if path is None:
            return []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            all_cols = config.all_cols

            return [
                [row['Date']] + [float(row[col]) for col in all_cols]
                for row in reader if all(row[col] != '' for col in all_cols)
            ]

    data_list = read_file(file_path)

    if not data_list:
        raise ValueError(f"No data found in {file_path}")

    all_cols_manual = config.feature_cols.copy()
    for col in config.target_cols:
        if col not in all_cols_manual:
            all_cols_manual.append(col)
        
    if data_list:
        df = pd.DataFrame(data_list, columns=['Date'] + all_cols_manual)
        df['Date'] = pd.to_datetime(df['Date']).astype(int) / 10**9 # Convert to UNIX timestamp (float)
        numpyData = df.sort_values(by='Date').to_numpy(dtype=np.float32)
    else:
        numpyData = np.empty((0, len(all_cols_manual)), dtype=np.float32)


    dataset = DatasetForPrediction(numpyData, config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def predict(model, dataloader, freqs_cis):
    with torch.no_grad():
        for batch, norm_val in dataloader:
            input = batch.to(config.device, dtype=torch.float32)
            norm_val = norm_val.to(config.device, dtype=torch.float32)

            out_prices = model(input, freqs_cis=freqs_cis)
            out_prices = out_prices * norm_val
            predicted = out_prices.squeeze(-1).cpu().numpy()

    return predicted


if __name__ == "__main__":


    config = ModelConfig()
    config.batch_size = 1

    model = TransformerModel(config).to(config.device)

    parser = argparse.ArgumentParser(description="Fetch and process financial data.")
    parser.add_argument(
        '--ticker',
        type=str,
        default='GOOGL',
        help='Ticker symbol of the stock')
    
    config.ticker = parser.parse_args().ticker

    model_path = f'model/trained/{config.ticker}*.pth'

    model_files = glob.glob(model_path,recursive=True)
    model_files.sort()
    if not model_files:
        raise FileNotFoundError(f"No model files found for {model_path}")

    model = load_model(model, model_files[-1])

    args = namedtuple('Args', ['ticker', 'resolution', 'start_date', 'end_date'])(
        ticker=config.ticker,
        resolution='1h',
        start_date=time.strftime('%Y-%m-%d', time.gmtime(time.time() - config.window_size* 24 * 60 * 60)), # window_size + buffer hours ago
        end_date=time.strftime('%Y-%m-%d', time.gmtime()),
    )

    fetch_data(args)

    file_path = f'data/raw/{config.ticker}_{args.resolution}_{args.start_date}_{args.end_date}.csv'
    print(f"Using data file: {file_path}")

    dataloader = dataloader_for_prediction(file_path, config)

    freqs_cis = None

    predictions = predict(model, dataloader, freqs_cis)
    print(predictions)