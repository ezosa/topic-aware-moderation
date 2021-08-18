
import torch
import numpy as np


def save_checkpoint(save_path, model, optimizer, valid_loss, log_file):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print('Model saved to ==>', save_path, file=log_file)


def load_checkpoint(load_path, model, optimizer, device, log_file):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print('Model loaded from <== ', load_path, file=log_file)

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list, log_file):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    # with open(save_path, 'w') as json_file:
    #     json.dump(state_dict, json_file)
    #     json_file.close()

    torch.save(state_dict, save_path)
    print(f'Model metrics saved to ==> {save_path}', file=log_file)


def load_metrics(load_path, device, log_file):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}', file=log_file)

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def load_pretrained_embeddings(path, word2idx, embedding_dim=300):
    print("Loading pretrained embeddings from", path)
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index is not None:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        print("Pretrained emb:",  embeddings.shape)
        return torch.from_numpy(embeddings).float()


def get_seed(run_num):
    return run_num + 1
