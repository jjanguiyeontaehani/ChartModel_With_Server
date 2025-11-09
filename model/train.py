import os
import sys
import time
import math
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.config.config import ModelConfig
from data.preprocess import preprocessData
from transformer import TransformerModel, precompute_freqs_cis, create_attention_mask
config = ModelConfig()

def collate_fn_global(batch, config):
    x, y_price = zip(*batch)

    x = pad_sequence(x, batch_first=True, padding_value=float(config.pad_token_id))
    y_price = pad_sequence(y_price, batch_first=True, padding_value=float(config.pad_token_id))

    is_positive = (y_price > 1).float()
    input_tensors = x

    attention_mask = create_attention_mask(x, float(config.pad_token_id))

    return {
        'input': input_tensors,
        'labels': {
            'price': y_price,
        },
        'is_positive': is_positive,
        'attention_mask': attention_mask
    }

class PreprocessedDataset(Dataset):
    def __init__(self, path, config):
        self.config = config
        self.all_cols = config.all_cols
        self.feature_cols = config.feature_cols
        self.target_cols = config.target_cols
        self.target_val_idx = config.all_cols.index(config.target_cols[0])
        self.window_size = config.window_size
        self.future_offset = config.future_offset
        self.path = path

        self.np = np.load(path, mmap_mode='r')

        self.index = pd.to_datetime(self.np[:, 0], unit='s')
        
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for start in range(len(self.np) - self.window_size - max(self.future_offset)):
            val_range = np.arange(start, start + self.window_size)
            
            future_indices = np.arange(start + self.window_size, start + self.window_size + max(self.future_offset))
            samples.append((val_range, future_indices))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, future_indices = self.samples[idx]

        exclude_indices = []
        for i, col in enumerate(self.target_cols):
            if i != 0:
                exclude_indices.append(self.all_cols.index(col))

        data = self.np[indices, 1:]
        data = np.delete(data, exclude_indices, axis=1)

        data_tensor = torch.from_numpy(data)
        
        norm_val = data_tensor[-1, self.target_val_idx].item()
        
        data_tensor = torch.cat([data_tensor, data_tensor[:, self.target_val_idx].unsqueeze(1) / norm_val], dim=1)

        targets = []
        for i, col in enumerate(self.all_cols):
            if col in self.target_cols:
                if i == self.target_val_idx:
                    target_vals = self.np[future_indices, i + 1] / norm_val # +1 for exclude date
                    targets.append(target_vals)
                else:
                    targets.append([self.np[indices[-1], i + 1]]) # +1 for exclude date
        
        targets = [torch.tensor(t, dtype=torch.float32) for t in targets]
        target_price = targets[0]

        return data_tensor, target_price
    
def load_data(paths, config):
    dataloaders = []

    preprocessData(paths, config)

    for i, path in enumerate(paths):
        split = ['train', 'valid', 'test'][i]

        npy_path = os.path.join('data/preprocessed', f'{split}.npy')
        dataset = PreprocessedDataset(npy_path, config)

        collator = partial(collate_fn_global, config=config)
        
        dl = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True
        )
        dataloaders.append(dl)

    train_dl, valid_dl, test_dl = dataloaders
    return train_dl, valid_dl, test_dl

def train_model(model, train_dataloader, validation_dataloader, test_dataloader, config, writer=None):
    startTime = time.time()
    avg_loss = 0.0
    trainLossHistory = []
    validationLossHistory = []
    validationAccuracyHistory = []
    validationAccuracyHistory_ud = []
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, min_lr= config.min_learning_rate)
    freqs_cis = precompute_freqs_cis(config.window_size, config.dim_head, device=config.device)

    for epoch in range(config.epoch):
        epochStartTime = time.time()
        epoch_loss = 0.0
        batch_count = 0

        for i, batch in enumerate(train_dataloader):
            input = batch['input'].to(config.device, dtype=torch.float32)
            labels = batch['labels']
            labels['price'] = labels['price'].to(config.device, dtype=torch.float32)
            attention_mask = batch['attention_mask'].to(config.device, dtype=torch.bool)

            # outputs = model(input, freqs_cis=freqs_cis, attention_mask=attention_mask)
            out_prices = model(input, freqs_cis=freqs_cis)
            loss = F.mse_loss(
                out_prices.view(-1),
                labels['price'].view(-1),
                reduction='mean'
            )

            loss.backward()
            epoch_loss += loss.item()
            batch_count += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = epoch_loss / batch_count
        trainLossHistory.append(avg_loss)

        val_loss, val_acc, val_acc_ud, val_accuracies, error_ratios, error_trends = validate_model(model, validation_dataloader, freqs_cis, config)
        scheduler.step(val_loss)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.6f}, Validation Accuracy: {val_acc:.2f}")
        model.train()
        validationLossHistory.append(val_loss)
        validationAccuracyHistory.append(val_acc)
        validationAccuracyHistory_ud.append(val_acc_ud)

        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            if val_loss is not None:
                writer.add_scalar('Loss/validation', val_loss, epoch+1)
            if val_acc is not None:
                writer.add_scalar('Accuracy/validation', val_acc, epoch+1)
            if val_acc_ud is not None:
                writer.add_scalar('Accuracy/validation_ud', torch.tensor(val_acc_ud).mean(), epoch+1)
            if val_accuracies is not None:
                for i, val_acc in enumerate(val_accuracies):
                    if i+1 in config.future_offset:
                        writer.add_scalar(f'Accuracy/validation_offset_{i+1}', val_acc, epoch+1)
            if error_ratios is not None:
                for i, error_ratio in enumerate(error_ratios):
                    if i+1 in config.future_offset:
                        writer.add_scalar(f'Error Ratio/validation_offset_{i+1}', error_ratio, epoch+1)
            if error_trends is not None:
                for i, error_trend in enumerate(error_trends):
                    if i+1 in config.future_offset:
                        writer.add_scalar(f'Error Trend/validation_offset_{i+1}', error_trend, epoch+1)


        epochEndTime = time.time()
        print(f"Epoch {epoch+1}/{config.epoch}, train Loss: {avg_loss:.6f} completed in {epochEndTime - epochStartTime:.2f} seconds")

    elapsedTime = time.time() - startTime
    print(f"Training completed in {elapsedTime:.2f} seconds")

    test_loss, test_accuracy, test_accuracy_ud, test_accuracies, error_ratios, error_trends = validate_model(model, test_dataloader, freqs_cis, config)
    print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}, Test Accuracy (UD): {torch.tensor(test_accuracy_ud).mean():.2f}")
    if writer:
        writer.add_scalar('Loss/test', test_loss, config.epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, config.epoch)
        if test_accuracy_ud is not None:
            for i, test_acc in enumerate(test_accuracy_ud):
                writer.add_scalar(f'Accuracy/test_ud', test_acc, i+1)
        if test_accuracies is not None:
            for i, test_acc in enumerate(test_accuracies):
                writer.add_scalar(f'Accuracy/test_offset', test_acc, i+1)
        if error_ratios is not None:
            for i, error_ratio in enumerate(error_ratios):
                writer.add_scalar(f'Error Ratio/test_offset', error_ratio, i+1)
        if error_trends is not None:
            for i, error_trend in enumerate(error_trends):
                writer.add_scalar(f'Error Trend/test_offset', error_trend, i+1)

    return model, trainLossHistory, validationLossHistory, validationAccuracyHistory, elapsedTime

def validate_model(model, dataloader, freqs_cis, config):
    model.eval()
    total_loss = 0.0
    total_accuracies = torch.zeros(max(config.future_offset), device=config.device)
    total_error_ratios = torch.zeros(max(config.future_offset), device=config.device)
    total_error_trends = torch.zeros(max(config.future_offset), device=config.device)
    total_ud_accuracies = torch.zeros(max(config.future_offset), device=config.device)
    batch_count = 0
    input_target_idx = config.all_cols.index(config.target_cols[0])

    inputs = []

    with torch.no_grad():
        for batch in dataloader:
            input = batch['input'].to(config.device, dtype=torch.float32)
            labels = batch['labels']
            labels['price'] = labels['price'].to(config.device, dtype=torch.float32)

            attention_mask = batch['attention_mask'].to(config.device)
            is_positive = batch['is_positive'].to(config.device)

            out_prices = model(input, freqs_cis=freqs_cis)
            
            inputs.append(input[:,-1,input_target_idx])

            price_loss = F.mse_loss(
                out_prices.view(-1),
                labels['price'].view(-1),
                reduction='mean'
            )

            predicted = out_prices.squeeze(-1)
            target = labels['price'].squeeze(-1)

            error_ratio = torch.abs(predicted - target) / (target + 1e-6)
            total_error_ratios += error_ratio.mean(dim=0)
            
            total_error_trends += ((predicted - target) / (target + 1e-6)).mean(dim=0)

            batch_accuracies = (error_ratio < config.error_threshold).float().mean(dim=0)
            total_accuracies += batch_accuracies

            if is_positive is not None:
                batch_ud_accuracies = (is_positive == (predicted > 1)).float().mean(dim=0)
                total_ud_accuracies += batch_ud_accuracies

            total_loss += price_loss.item()
            batch_count += 1


    running_loss = total_loss / batch_count
    
    accuracies = (total_accuracies / batch_count).cpu().numpy() * 100
    error_ratios = (total_error_ratios / batch_count).cpu().numpy() * 100
    error_trends = (total_error_trends / batch_count).cpu().numpy() * 100
    
    running_accuracy_ud = (total_ud_accuracies / batch_count).cpu().numpy() * 100
    running_accuracy = accuracies.mean()

    return running_loss, running_accuracy, running_accuracy_ud, accuracies, error_ratios, error_trends

if __name__ == "__main__":
    start = time.time()
    trainfilenames = [
        # "data/testh.csv",
        "data/QQQ_1h_2023-08-03_2025-08-01.csv",
        # "data/test.csv",
        "data/QQQ_1d_2020-01-01_2025-08-01.csv"
    ]

    datafiles = [
        "data/raw/GOOGL_1h_2023-08-15_2025-08-14.csv",
        "data/raw/AAPL_1h_2023-08-15_2025-08-14.csv",
        "data/raw/TQQQ_1h_2023-08-15_2025-08-14.csv",
    ]

    print("Start reading data")

    writer = SummaryWriter(log_dir="logs/seed_" + str(config.seed) + "/window_" + str(config.window_size) + "/")

    if config.seed:
        def setSeed(seed):
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        setSeed(config.seed)

    train_dataloader, validation_dataloader, test_dataloader = load_data(
        paths=datafiles,
        config=config
    )

    print("Data loaded successfully")

    model = TransformerModel(config)
    model.to(config.device)
    model.train()
    trained_model, trainLossHistory, validationLossHistory, validationAccuracyHistory, elapsedTime = train_model(
        model, train_dataloader, validation_dataloader, test_dataloader, config, writer
    )

    writer.close()

    with open("performance.txt", "a") as f:
        f.write("model config: ")
        f.write(str(config) + "\n")
        f.write("train loss history: " + str(trainLossHistory) + "\n")
        f.write("validation loss history: " + str(validationLossHistory) + "\n")
        f.write("best validation loss: " + str(min([loss for loss in validationLossHistory if loss is not None])) + "\n")
        f.write("validation accuracy history: " + str(validationAccuracyHistory) + "\n")
        f.write("best validation accuracy: " + str(max([acc for acc in validationAccuracyHistory if acc is not None])) + "\n")
        f.write(f"training time: {elapsedTime:.2f} seconds\n")

    print("Performance saved to performance.txt")