
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.MLP import MLP


class LSTM(nn.Module):

    def __init__(self, lstm_args, mlp_args):
        super(LSTM, self).__init__()
        # embedding layer
        self.embedding_dim = lstm_args['emb_dim']
        self.embedding = nn.Embedding(lstm_args['vocab_size'], self.embedding_dim)
        if 'pretrained_emb' in lstm_args:
            print("Initializing with pretrained embeddings")
            self.embedding.weight.data.copy_(lstm_args['pretrained_emb'])
        self.hidden_dim = lstm_args['hidden_size']
        # bi-LSTM layer
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        # fully-connected linear layer
        mlp_input_size = 2 * self.hidden_dim
        self.mlp = MLP(mlp_input_size,
                       mlp_args['hidden_size'])

    def forward(self, text, text_len):
        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)
        mlp_output = self.mlp(text_fea)
        return mlp_output
