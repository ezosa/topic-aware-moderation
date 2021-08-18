import torch
import torch.nn as nn
#from models.SenEncoderLSTM import SenEncoderLSTM
from models.MLP import MLP
from models.LSTMEncoder import LSTMEncoder


class LSTM_MLP_Ensemble(nn.Module):
    def __init__(self, mlp_args, lstm_args):
        super(LSTM_MLP_Ensemble, self).__init__()

        self.lstm_encoder = LSTMEncoder(lstm_args)

        mlp_input_size = int(lstm_args['hidden_dim']*2 + mlp_args['num_topics'])
        self.mlp = MLP(mlp_input_size,
                       mlp_args['hidden_size'])

    def forward(self, text, text_len, topics):
        encoded_sent = self.lstm_encoder(text, text_len)
        concat_topics = torch.cat((encoded_sent, topics), dim=1)
        mlp_output = self.mlp(concat_topics)

        fused_encoding= self.mlp.layers[0](concat_topics)
        return mlp_output, fused_encoding
