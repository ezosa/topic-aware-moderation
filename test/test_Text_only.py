
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
import os

import argparse
from models.LSTM2 import LSTM2
from utils.datasets.CommentsDatasetv2 import CommentsDatasetv2
from train.train_utils import load_checkpoint, load_pretrained_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def evaluate(model, test_loader, threshold=0.5):
    y_pred = []
    y_true = []


    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            labels = test_batch['binary_label'].unsqueeze(1).to(device)
            text = test_batch['text']
            text = torch.stack(text, dim=1).to(device)
            text_len = test_batch['text_len'].to('cpu')
            y_score_i = model(text, text_len)

            y_pred_i = (y_score_i > threshold).int()
            y_pred.extend(y_pred_i.tolist())
            y_true.extend(labels.tolist())

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    results = {'f1': macro_f1,
               'recall': recall,
               'precision': precision}

    return results


parser = argparse.ArgumentParser(description='Testing Text-only')
parser.add_argument('--model_name', type=str, default='Text_only', help='name of trained model')
parser.add_argument('--num_epochs', type=int, default=20, help='training epochs')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of embedding layer')
parser.add_argument('--lstm_hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--mlp_hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--data_path', type=str, default='data/', help='path to the train-valid-test sets')
parser.add_argument('--test_path', type=str, default='data/', help='path to test sets')
parser.add_argument('--vocab_file', type=str, default='vocab.pkl', help='filename of vocab dict')
parser.add_argument('--save_path', type=str, default='', help='path to save trained model; same as data_path by default')
parser.add_argument('--batch_size', type=int, default=500, help='batch_size')
parser.add_argument('--pretrained_emb', type=str, default='', help='path to pretrained word embeddings')

args = parser.parse_args()
data_path = args.data_path
num_epochs = args.num_epochs
save_path = args.save_path

vocab = pickle.load(open(data_path + args.vocab_file, 'rb'))
lstm_args = {}
# load pretrained embeddings if provided
if len(args.pretrained_emb) > 0:
    word_emb = load_pretrained_embeddings(args.pretrained_emb, vocab, embedding_dim=args.emb_dim)
    lstm_args['pretrained_emb'] = word_emb

lstm_args['vocab_size'] = len(vocab)
lstm_args['emb_dim'] = args.emb_dim
lstm_args['hidden_dim'] = args.lstm_hidden_size

mlp_args = {}
mlp_args['hidden_size'] = args.mlp_hidden_size


# Prepare each test set and evaluate them for every run of the model
test_files = os.listdir(args.test_path)
model_name = args.model_name
log_file = open(save_path + model_name + "_test_logs.txt", 'a+')

for test_file in test_files:
    test_csv = args.test_path + test_file
    test_data = CommentsDatasetv2(csv_file=test_csv, vocab=vocab, delimiter=' ')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    best_model = LSTM2(lstm_args=lstm_args, mlp_args=mlp_args).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=0.005)
    load_checkpoint(save_path + model_name + ".pt", best_model, optimizer, device, log_file)
    results = evaluate(best_model, test_loader)
