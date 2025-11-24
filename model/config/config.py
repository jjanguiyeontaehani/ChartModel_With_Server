from dataclasses import dataclass, field
import torch

@dataclass
class ModelConfig:
    input_size: int = 14
    input_feature_variants: int = 1
    window_size: int = 100
    dim_model: int = 512
    dim_ff: int = 1024
    num_layers: int = 3 # 6
    num_heads: int = 4 # 8
    dropout: float = 0.15
    batch_size: int = 64
    epoch: int = 10
    future_offset: list = field(default_factory=lambda: [1, 3, 7, 14, 30])
    feature_cols: list = field(default_factory=lambda: [
        'open', 'close',
        'volume',
        'macd', 'signal_line', 'macd_hist',
        'sma_10', 'sma_20', 'sma_50',
        'upper_band', 'lower_band', 'bollinger_width',
        'rsi'
        ])
    target_cols: list = field(default_factory=lambda: [
        'close'
    ])
    error_threshold: float = 0.02
    learning_rate: float = 0.0015
    min_learning_rate: float = 0.0001
    pad_token_id: float = 0.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 218

    def __post_init__(self):
        self.all_cols = self.feature_cols.copy()
        for col in self.target_cols:
            if col not in self.all_cols:
                self.all_cols.append(col)

        self.input_size = len(self.feature_cols) + 1
        self.dim_head = self.dim_model // self.num_heads