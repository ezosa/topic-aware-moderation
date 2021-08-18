
import torch.nn as nn
from models.MLP import MLP
from models.LSTMEncoder import LSTMEncoder

class LSTM2(nn.Module):

    def __init__(self, lstm_args, mlp_args):
        super(LSTM2, self).__init__()
        self.lstm_encoder = LSTMEncoder(lstm_args)

        mlp_input_size = int(lstm_args['hidden_dim']*2)
        self.mlp = MLP(mlp_input_size,
                       mlp_args['hidden_size'])

    def forward(self, text, text_len):
        encoded_sent = self.lstm_encoder(text, text_len)
        mlp_output = self.mlp(encoded_sent)
        fused_encoding = self.mlp.layers[0](encoded_sent)

        return mlp_output,fused_encoding
