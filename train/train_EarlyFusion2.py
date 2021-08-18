
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pickle
import numpy as np

from models.LSTMEnsembleEncoder import LSTMEncoder
from utils.datasets.CommentsDatasetv3 import CommentsDatasetv3
from train.train_utils import save_checkpoint, save_metrics, load_pretrained_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          save_path,
          criterion,
          num_epochs=50,
          eval_every=10,
          best_valid_loss=float("Inf"),
          model_name = "model"):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.float()
    model.train()
    for epoch in range(num_epochs):
        print("Epoch", epoch+1, "of", num_epochs, file=log_file)
        for train_batch in train_loader:
            labels = train_batch['binary_label'].unsqueeze(1).to(device)
            topic_embedding = train_batch['topic_embedding']
            topic_embedding = torch.stack(topic_embedding, dim=1).to(device)
            text = train_batch['text']
            text = torch.stack(text, dim=1).to(device)
            text_len = train_batch['text_len'].to('cpu')
            pred = model(text, text_len, topic_embedding.float()).to(device)

            loss = criterion(pred, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for val_batch in valid_loader:
                        labels = val_batch['binary_label'].unsqueeze(1).to(device)
                        topic_embedding = val_batch['topic_embedding']
                        topic_embedding = torch.stack(topic_embedding, dim=1).to(device)
                        text = val_batch['text']
                        text = torch.stack(text, dim=1).to(device)
                        text_len = val_batch['text_len'].to('cpu')
                        pred = model(text, text_len, topic_embedding.float()).to(device)

                        val_loss = criterion(pred, labels.float())
                        valid_running_loss += val_loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss), file=log_file)

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(save_path + model_name + '.pt', model, optimizer, best_valid_loss, log_file)
                    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, log_file)

    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, log_file)
    print('Finished Training!', file=log_file)


parser = argparse.ArgumentParser(description='Early Fusion 2: text and topic embeddings')
parser.add_argument('--num_epochs', type=int, default=50, help='training epochs')
parser.add_argument('--num_topics', type=int, default=100, help='num of topics in the topic model')
parser.add_argument('--topic_dim', type=int, default=100, help='dimensionality of topic embedding')
parser.add_argument('--lstm_hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--mlp_hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--data_path', type=str, default='', help='path to the train-valid-test sets')
parser.add_argument('--train_csv', type=str, default='train.csv', help='filename of train set')
parser.add_argument('--valid_csv', type=str, default='valid.csv', help='filename of valid set')
parser.add_argument('--vocab_file', type=str, default='vocab.pkl', help='filename of vocab dict')
parser.add_argument('--save_path', type=str, default='', help='path to save trained model, same as data_path by default')
parser.add_argument('--batch_size', type=int, default=500, help='batch_size')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--pretrained_emb', type=str, default='', help='path to pretrained word embeddings')


args = parser.parse_args()
num_epochs = args.num_epochs
data_path = args.data_path
save_path = args.save_path


train_csv = data_path + args.train_csv
valid_csv = data_path + args.valid_csv

vocab = pickle.load(open(data_path + args.vocab_file, 'rb'))
train_data = CommentsDatasetv3(csv_file=train_csv, vocab=vocab, delimiter=' ')
valid_data = CommentsDatasetv3(csv_file=valid_csv, vocab=vocab, delimiter=' ')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

lstm_args = {}
# load pretrained embeddings if provided
if len(args.pretrained_emb) > 0:
    word_emb = load_pretrained_embeddings(args.pretrained_emb, vocab, embedding_dim=args.topic_dim)
    lstm_args['pretrained_emb'] = word_emb

lstm_args['vocab_size'] = len(vocab)
lstm_args['emb_dim'] = 300
lstm_args['hidden_dim'] = args.lstm_hidden_size
# in this model, num_topics is the dimensionality of the topic embedding
lstm_args['num_topics'] = args.topic_dim

mlp_args = {}
mlp_args['hidden_size'] = args.mlp_hidden_size


model_name = "EF1_" + str(args.topic_dim) + "topicDim_" + str(args.lstm_hidden_size) + "LSTMhidden_" + str(args.mlp_hidden_size) + "MLPhidden_" + str(args.num_epochs) + "epochs_" + args.model_id
if len(args.pretrained_emb) > 0:
    model_name += "_pretrained_emb"

log_file = open(save_path + model_name + "_logs.txt", 'a+')

print("Args:", args, file=log_file)
print("Device:", device, file=log_file)
print("Model name:", model_name, file=log_file)

model = LSTMEncoder(lstm_args=lstm_args,
                    mlp_args=mlp_args)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.BCELoss()
print("Training", file=log_file)
train(model=model,
      optimizer=optimizer,
      num_epochs=num_epochs,
      eval_every=args.step_size,
      train_loader=train_loader,
      valid_loader=valid_loader,
      criterion=loss_fn,
      save_path=save_path,
      model_name=model_name)

print("Done training! Best model saved at", save_path + model_name + ".pt", file=log_file)
log_file.close()