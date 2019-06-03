import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from load_pre_trained_embeddings import load_dataset,load_dataset_from_file
from torch.nn.functional import softmax
import numpy as np
import h5py
import torch.nn.functional as F

import sys
import csv
import argparse


# Hyper Parameters
input_size = 1024
num_classes = 2
num_epochs = 3000
batch_size = 512
learning_rate = 0.001
PATH = "lr_model.pt"
# ELMO_PATH = "elmo_lr"
ELMO_PATH = "elmo_dense"

class Dense(nn.Module):
    def __init__(self, input_size, H1, H2, num_classes):
        super(Dense, self).__init__()
        self.linear1 = nn.Linear(input_size, H1)
        self.bn1 = nn.BatchNorm1d(num_features=H1)
        self.linear2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        l1 = self.bn1(F.relu(self.linear1(x)))
        l2 = self.bn2(F.relu(self.linear2(l1)))
        out = F.softmax(self.linear3(l2))
        return out

class LogisticRegression(nn.Module):
    def __init__(self, n_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.softmax(self.linear(x))
        return out

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
            embedding = h5py_file.get(str(index))

            # compute the average word
            if self.averaged:
                embedding = np.mean(embedding, axis=0)
            return (embedding, self.targets[index])

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
    python logistic_regression.py --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv --average
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
        weights = torch.Tensor(list(map(lambda x: (1306120.0/1225310.0) if x == 0 else (1306120.0/80810.0), targets)))
        # Hyperparameters
        input_size = 1024

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    print("Creating Model")

    # Uncomment the below for Logistic Regression
    # model = LogisticRegression(input_size, num_classes)
    # # model.load_state_dict(torch.load(PATH))
    #
    # # Loss and Optimizer
    # # Softmax is internally computed.
    # # Set parameters to be updated.
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Uncomment the below for Dense NN
    model = Dense(input_size, 256, 32, num_classes)
    model.load_state_dict(torch.load(PATH))

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    f1_scores = []

    print("Starting Training")
    # Training the Model
    for epoch in range(num_epochs):
        for i, (sentences, labels) in enumerate(train_loader):

            sentences = Variable(sentences)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i) % 129 == 0:
                # torch.save(model.state_dict(), PATH)
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                       % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))

        if epoch % 50 == 1:
            # eval mode
            model.eval()
            correct = 0
            total = 0
            pred_positives = 0.0
            real_positives = 0.0
            true_positives = 0.0
            for sentences, labels in train_loader:
                sentences = Variable(sentences)
                outputs = model(sentences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                real_positives += labels.sum()
                pred_positives += predicted.sum()
                true_positives += (labels * predicted).sum().item()
                correct += (predicted == labels).sum()
                # break
            f1 = -1
            try:
                precision = true_positives*1.0/pred_positives.item()
                recall = true_positives*1.0/real_positives.item()
                f1 = 2*(precision*recall)/(precision+recall)
                print(f"\tEpoch: {epoch}\tF1: {f1}")
            except:
                print("Error calculating f1 score")
            f1_scores.append(f1)

            torch.save(model.state_dict(), f"{ELMO_PATH}/elmo_{epoch}_{f1}.pt")

            model.train()

    # Save F1 scores for every 50 epochs
    np.save(f"{ELMO_PATH}/elmo_f1.npy", np.array(f1_scores))

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

    try:
        precision = true_positives*1.0/pred_positives.item()
        recall = true_positives*1.0/real_positives.item()
        print(true_positives,pred_positives,real_positives)
        print("F1 score: {}".format(2*(precision*recall)/(precision+recall)))
    except:
        print("Error calculating f1 score")
    print('Accuracy of the model: %d %%' % (100 * correct / total))
