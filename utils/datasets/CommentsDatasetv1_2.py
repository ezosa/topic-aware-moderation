from torch.utils.data import Dataset
import pandas as pd


class CommentsDataset(Dataset):
    def __init__(self, csv_file, delimiter=' '):
        self.data = pd.read_csv(csv_file)
        self.delim = delimiter

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = dict(self.data.iloc()[idx])
        topic_emb = [float(val) for val in row['topic_embedding'].split(self.delim)]
        row['topic_embedding'] = topic_emb
        if 'sparse_embedding' in row:
            topic_emb = [float(val) for val in row['sparse_embedding'].split(self.delim)]
            row['topic_embedding'] = topic_emb
        return row


