import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
import argparse
import os
import json

from models.MLP import MLP
from utils.datasets.CommentsDatasetv1_2 import CommentsDataset
from train.train_utils import  load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def evaluate(model, test_loader, threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            labels = test_batch['binary_label'].unsqueeze(1).to(device)
            topics = test_batch['topic_embedding']
            topics = torch.stack(topics, dim=1).to(device)
            y_score_i = model(topics.float())

            y_pred_i = (y_score_i > threshold).int()
            y_pred.extend(y_pred_i.tolist())
            y_true.extend(labels.tolist())

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)


    return macro_f1, recall, precision


parser = argparse.ArgumentParser(description='Testing DTE')
parser.add_argument('--model_name', type=str, default='DTE_', help='name of trained model')
parser.add_argument('--test_path', type=str, default='', help='path to test sets')
parser.add_argument('--save_path', type=str, default='', help='path to save trained model')
parser.add_argument('--batch_size', type=int, default=500, help='batch_size')


args = parser.parse_args()
save_path = args.save_path



# Prepare each test set and evaluate them for every run of the model
test_files = os.listdir(args.test_path)
test_files = sorted([f for f in test_files if ".csv" in f])
print("test_files:", test_files)

model_name = args.model_name
log_file = open(save_path + model_name + "_test_logs.txt", 'a+')

model_scores_dict = {}
for test_file in test_files:
    # prepare test loader for each test set
    test_csv = args.test_path + test_file
    test_data = CommentsDataset(csv_file=test_csv, delimiter=' ')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    best_model = MLP(args.topic_dim, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=0.005)

    load_checkpoint(save_path + model_name + ".pt", best_model, optimizer, device, log_file)
    macro_f1, recall, precision = evaluate(best_model, test_loader)

