
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse


from models.MLP import MLP
from utils.datasets.CommentsDataset import CommentsDataset
from train.train_utils import save_checkpoint, save_metrics

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
          model_name="model"):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    print("Start training for", num_epochs, "epochs...", file=log_file)
    model.float()
    model.train()
    for epoch in range(num_epochs):
        print("-"*5, "Epoch", epoch+1, "of", num_epochs, "-"*5, file=log_file)
        for train_batch in train_loader:
            labels = train_batch['binary_label'].unsqueeze(1).to(device)
            topics = train_batch['topics']
            topics = torch.stack(topics, dim=1).to(device)
            pred = model(topics.float())

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
                        topics = val_batch['topics']
                        topics = torch.stack(topics, dim=1).to(device)
                        pred = model(topics.float())

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
                    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list,
                                 global_steps_list, log_file)

    save_metrics(save_path + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, log_file)
    print('Finished Training!', file=log_file)


parser = argparse.ArgumentParser(description='Training MLP classification using topic vectors')
parser.add_argument('--num_epochs', type=int, default=10, help='training epochs')
parser.add_argument('--num_topics', type=int, default=100, help='input size')
parser.add_argument('--topic_dim', type=int, default=100, help='topic embedding dimension')
parser.add_argument('--hidden_size', type=int, default=10, help='hidden size')
parser.add_argument('--data_path', type=str, default='', help='path to the train-valid-test sets')
parser.add_argument('--train_csv', type=str, default='train.csv', help='filename of train set')
parser.add_argument('--valid_csv', type=str, default='valid.csv', help='filename of valid set')
parser.add_argument('--save_path', type=str, default='', help='path to save trained model')
parser.add_argument('--batch_size', type=int, default=500, help='batch_size')
parser.add_argument('--step_size', type=int, default=100, help='step_size')

args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path


train_csv = data_path + args.train_csv
valid_csv = data_path + args.valid_csv


train_data = CommentsDataset(csv_file=train_csv, delimiter=' ')
valid_data = CommentsDataset(csv_file=valid_csv, delimiter=' ')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)


model_name = "DTD_" + str(args.num_topics) + "topics_" + str(args.topic_dim) + "topicDim_" + \
             str(args.hidden_size) + "hidden_" + str(args.num_epochs) + "epochs_" + \
             args.model_id

print("Model name:", model_name)

log_file = open(save_path + model_name + "_logs.txt", 'w')

print("Args:", args, file=log_file)
print("Device:", device, file=log_file)

model = MLP(args.num_topics, args.hidden_size)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.BCELoss()

print("Model name:", model_name, file=log_file)

print("Training", file=log_file)
train(model=model,
      optimizer=optimizer,
      num_epochs=args.num_epochs,
      eval_every=args.step_size,
      train_loader=train_loader,
      valid_loader=valid_loader,
      criterion=loss_fn,
      save_path=save_path,
      model_name=model_name)

print("Done training! Best model saved at", save_path + model_name + ".pt", file=log_file)
log_file.close()

