import torch
import torch.nn as nn
from models.MLP import MLP
from models.LSTMEncoder import LSTMEncoder


class LSTM_MLP_Ensemble2(nn.Module):
    def __init__(self, mlp_args, lstm_args):
        super(LSTM_MLP_Ensemble2, self).__init__()
        self.lstm_encoder = LSTMEncoder(lstm_args)
        mlp_input_size = int(lstm_args['hidden_dim']*2 + mlp_args['num_topics'] + mlp_args['topic_dim'])
        self.mlp = MLP(mlp_input_size, mlp_args['hidden_size'])

    def forward(self, text, text_len, topics1, topics2):
        # topics1 and topics2 refer to the topic vectors and topic embeddings (in any order)
        encoded_sent = self.lstm_encoder(text, text_len)
        concat_topics = torch.cat((encoded_sent, topics1, topics2), dim=1)
        mlp_output = self.mlp(concat_topics)
        return mlp_output
