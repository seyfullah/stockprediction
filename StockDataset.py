from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import Dataset

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class StockDataset(Dataset):
    """Stock dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        date = self.df.iloc[idx, 0]
        open = self.df.iloc[idx, 1]
        high = self.df.iloc[idx, 2]
        low = self.df.iloc[idx, 3]
        close = self.df.iloc[idx, 4]
        volume = self.df.iloc[idx, 5]
        Name = self.df.iloc[idx, 6]
        stock = {'date': date, 
                'open': open, 
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,                                                
                'Name': Name}

        return stock
