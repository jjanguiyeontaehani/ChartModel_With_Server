import pandas as pd
import csv
import os

def load_multiscale_stock_data(paths, config):
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

    preprocessed_dir = 'data/preprocessed'
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    all_cols_manual = config.feature_cols.copy()
    for col in config.target_cols:
        if col not in all_cols_manual:
            all_cols_manual.append(col)


    for i, h_path in enumerate(paths):
        split = ['train', 'valid', 'test'][i]
        
        npy_path = os.path.join(preprocessed_dir, f'{split}.npy')


        print(f"Preprocessing {split} data...")
        
        data_list = read_file(h_path)
            
        if data_list:
            df = pd.DataFrame(data_list, columns=['Date'] + all_cols_manual)
            df['Date'] = pd.to_datetime(df['Date']).astype(int) / 10**9 # Convert to UNIX timestamp (float)
            np = df.sort_values(by='Date').to_numpy(dtype=np.float32)
        else:
            np = np.empty((0, len(all_cols_manual)), dtype=np.float32)
        
        np.save(npy_path, np)

if __name__ == "__main__":
    import model.config.config as config_module
    config = config_module.ModelConfig()

    data_paths = [
        'data/raw/AAPL_1h_20231110_20251108.csv',  # train
        'data/raw/AAPL_1h_20231110_20251108.csv',  # valid
        'data/raw/AAPL_1h_20231110_20251108.csv',  # test
    ]

    load_multiscale_stock_data(data_paths, config)
    print("Data preprocessing completed.")