import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from load_pre_trained_embeddings import load_dataset,load_dataset_from_file
from torch.nn.functional import softmax
import torch.nn.functional as F
import numpy as np
import h5py

import sys
import csv
import argparse

"""
NOT READY TO RUN YET. Have to adjust CNN parameters based on word embedding dimensions
"""

# Hyper Parameters
input_size = 600
num_classes = 2
# num_epochs = 400
num_epochs = 3
# batch_size = 10000
batch_size = 2048
learning_rate = 0.001
PATH = "cnn_model.pt"

# temporary CNN models in 2D and 1D
class CNN2d(nn.Module):
    def __init__(self):
        super(CNN2d, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding = 2) # p = (h - 1) / 2
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)
        self.fc1 = nn.Linear(128 * 6 * 64, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        self.conv1 = nn.Conv1d(768, 32, 5, padding = 2) # p = (h - 1) / 2
        self.pool = nn.MaxPool1d(2, stride = 2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv1d(64, 64, 3, padding = 1)
        self.fc1 = nn.Linear(64 * 6, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6)  # TODO: errors on BERT
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2 * EMBEDDING_SIZE, output_size)

    def forward(self, input):
        lstm_out = self.lstm(input)
        lstm_out_reshape = lstm_out[0].contiguous().view(BATCH_SIZE, -1)
        y_pred = self.linear(lstm_out_reshape)
        return y_pred
    
    def initHidden(self):
        return torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size)

# Dataset wrapper for the HDF5 files
class HDF5_Dataset(data.Dataset):
    def __init__(self, filepath, target_filepath, averaged=False):
        self.filepath = filepath

        with open(target_filepath) as csv_file:
            y = []
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                y.append(int(row[0]))
            y = torch.from_numpy(np.array(y).astype(int))
            self.targets = y

        self.len = len(self.targets)

        self.averaged = averaged

    # Return (vector_embedding, target)
    def __getitem__(self, index):
        with h5py.File(self.filepath, "r") as h5py_file:
            embedding_group = h5py_file.get(str(index))

            max_sentence_len = 64  # hardcode maximum sentence length

            # pad sentences - CNN 
            # LSTM/CNN1D - padded_inputs.shape: (128, 1024, 50)
            # CNN2D - padded_inputs.shape:  (128, 1, 1024, 50)
            padded_inputs = np.zeros((input_size, max_sentence_len))
            sentence_len = embedding_group.shape[0]
            if max_sentence_len < sentence_len:
                print("WARNING: max_sentence_len = {0}, sentence_len = {1}".format(max_sentence_len, sentence_len))
            else:
                padded_inputs = np.pad(embedding_group, pad_width=((0, max_sentence_len-sentence_len), (0,0)), mode='constant').T
                padded_inputs = torch.from_numpy(padded_inputs).float()

                # compute the average word
                if self.averaged:
                    padded_inputs = np.mean(padded_inputs, axis=0)

            return (padded_inputs, self.targets[index])

    def __len__(self):
        return self.len

    def __get_targets__(self):
        return self.targets

if __name__ == "__main__":

    # use args to specify embedding file
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store", dest="train_filepath", help="This is the training file, with each of the training examples embedded already")
    parser.add_argument("--targets", action="store", dest="target_filepath", help="This is the path to the training file's targets")
    parser.add_argument("--average", action="store_true", dest="average", default=False, help="Use the average word in the sentence instead of the entire sentence vector")
    args = parser.parse_args()

    """
    python cnn.py --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv --average
    """

    print("Loading Data")
    if args.train_filepath == "":
        args.train_filepath == "../train.csv"
        train_dataset = load_dataset_from_file("../train.csv", "glove_mean_avg.pt")
        weights = torch.Tensor([x[1]*9+1 for x in train_dataset])
        # Hyperparameters
        input_size = 600
    else:
        train_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     averaged=args.average)
        targets = train_dataset.__get_targets__()
        weights = torch.Tensor(list(map(lambda x: 15 if x == 0 else 1, targets)))
        # Hyperparameters
        input_size = 768

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    print("Creating Model")
    model = CNN1d()
    # model.load_state_dict(torch.load(PATH))

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Training")
    # Training the Model
    for epoch in range(num_epochs):
        for i, (sentences, labels) in enumerate(train_loader):
            sentences = Variable(sentences)
            labels = Variable(labels)
            print(labels.sum())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i) % 129 == 0:
                torch.save(model.state_dict(), PATH)
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                       % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))


    print("Testing Model")
    # Test the Model
    model.eval()
    correct = 0
    total = 0
    pred_positives = 0.0
    real_positives = 0.0
    true_positives = 0.0
    for sentences, labels in test_loader:
        sentences = Variable(sentences)
        outputs = model(sentences)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        real_positives += labels.sum()
        pred_positives += predicted.sum()
        true_positives += (labels * predicted).sum().item()
        correct += (predicted == labels).sum()
        # break
    print(true_positives,pred_positives,real_positives)
    precision = true_positives*1.0/pred_positives.item()
    recall = true_positives*1.0/real_positives.item()
    print("F1 score: {}".format(2*(precision*recall)/(precision+recall)))
    print('Accuracy of the model: %d %%' % (100 * correct / total))
