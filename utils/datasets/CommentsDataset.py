from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CommentsDataset(Dataset):
    def __init__(self, csv_file, delimiter=' ', min_prob=0.01):
        self.data = pd.read_csv(csv_file)
        self.delim = delimiter
        self.min_prob = min_prob

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = dict(self.data.iloc()[idx])
        topics = row['topics'].split(self.delim)
        topics = [float(val) for val in topics]
        row['topics'] = topics
        return row


