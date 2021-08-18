from torch.utils.data import Dataset
import pandas as pd


class CommentsDatasetv2(Dataset):
    def __init__(self, csv_file, vocab, delimiter=',', min_prob=0.01):
        self.data = pd.read_csv(csv_file)
        self.delim = delimiter
        self.vocab = vocab
        self.min_prob = min_prob

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx, max_text_len=200):
        row = dict(self.data.iloc()[idx])
        topics = [float(val) for val in row['topics'].split(self.delim)]
        row['topics'] = topics
        text = row['content'].lower().split()
        text = [self.vocab[w] if w in self.vocab else self.vocab['OOV'] for w in text]
        if len(text) > max_text_len:
                text = text[:max_text_len]
        else:
            text.extend([self.vocab['OOV']]*(max_text_len-len(text)))
        row['text'] = text
        row['text_len'] = max_text_len
        return row


