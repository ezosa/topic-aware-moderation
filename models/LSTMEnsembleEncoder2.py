
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.MLP import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder2(nn.Module):

    def __init__(self, lstm_args, mlp_args):
        super(LSTMEncoder2, self).__init__()
        # embedding layer
        self.embedding_dim = lstm_args['emb_dim']
        self.embedding = nn.Embedding(lstm_args['vocab_size'], self.embedding_dim)
        # initialize with pretrained word emb if provided
        if 'pretrained_emb' in lstm_args:
            self.embedding.weight.data.copy_(lstm_args['pretrained_emb'])
        # bi-LSTM layer
        self.hidden_dim = lstm_args['hidden_dim']
        self.input_dim = self.embedding_dim + lstm_args['num_topics'] + lstm_args['topic_emb_dim']
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        # MLP classifier
        mlp_input_size = int(self.hidden_dim * 2)
        self.mlp = MLP(mlp_input_size,
                       mlp_args['hidden_size'])

    def forward(self, text, text_len, topics, topic_embeddings):
        text_emb = self.embedding(text)
        doc_size = text_emb.shape[1]
        topic_input = topics.unsqueeze(1).repeat(1, doc_size, 1)
        topic_emb_input = topic_embeddings.unsqueeze(1).repeat(1, doc_size, 1)
        lstm_input_emb = torch.cat((text_emb, topic_input, topic_emb_input), dim=2)
        lstm_input_len = text_len
        # packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_input = pack_padded_sequence(lstm_input_emb, lstm_input_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)
        mlp_output = self.mlp(text_fea)
        return mlp_output
