"""
python generate_roc.py --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv --model=cnn1d --embedding=elmo --model_path=elmo_pad/cnn1d/cnn1d_4999.pt
"""


import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
# from load_pre_trained_embeddings import load_dataset,load_dataset_from_file
from torch.nn.functional import softmax
import numpy as np
import h5py
import torch.nn.functional as F

# Stuff for ROC curve
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
import csv
import argparse
from load_pre_trained_embeddings import load_dataset_from_file


# Hyper Parameters
input_size = 1024
num_classes = 2
num_epochs = 3000
batch_size = 512
learning_rate = 0.001
max_sentence_length = 32
PATH = "lr_model.pt"

# Dataset wrapper for the HDF5 files
class HDF5_Dataset(data.Dataset):
    def __init__(self, filepath, target_filepath, embedding, dataset="train", averaged=False, max_sentence_length=64):
        self.filepath = filepath

        with open(target_filepath) as csv_file:
            y = []
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                y.append(int(row[0]))
            y = torch.from_numpy(np.array(y).astype(int))
            self.targets = y

        self.embedding = embedding
        self.averaged = averaged
        self.max_sentence_length = max_sentence_length

        # Use self.idx as a subset of indices that are "train" or "test" respecitively
        #   these indices are then used in the __getitem__ and __len__ and __get_targets__
        #   methods to grab a subset of the data relating only to "train" or "test"
        if dataset in ["train", "validation", "test"]:
            self.dataset_type = dataset
            self.idx = np.load(f"{dataset}_idx.npy")
            self.len = len(self.idx)
        else:
            self.dataset_type = "full"
            self.idx = None
            self.len = len(self.targets)
            self.valsplit = None

    # Return (vector_embedding, target)
    def __getitem__(self, index):
        if self.dataset_type != "full":
            index = int(self.idx[index])

        with h5py.File(self.filepath, "r") as h5py_file:
            if self.embedding == 'elmo' or self.embedding == 'bert_word' or self.embedding == 'glove':
                embedding = h5py_file.get(str(index))
            elif self.embedding == 'bert_sentence':
                embedding = h5py_file.get(str(index))[0:1]

            # compute the average word
            if self.averaged:
                embedding = np.mean(embedding, axis=0)

            # pad sentences
            else:
                """
                Computing percentiles of the lengths of the sentence vectors:
                    percentile  length
                    50%:        11
                    90%:        22
                    95%:        27
                    97%:        31
                    98%:        34
                    99%:        39
                    99.9%:      48
                    99.99%:     53
                    99.999%:    57
                    99.9999%:   66
                    99.99999%:  125
                    99.999999%: 133
                Therefore, having a max sentence length of 32 will cover the vast majority
                (all but less than 100 examples) of training examples, while keeping the
                embedding size relatively small.

                The resulting embedding will be of shape:
                    (embedding_size, max_sentence_length)
                """
                try:
                    sentence_len, word_dim = embedding.shape
                except:
                    # About 6 sentences in the BERT embeddings file have 0 length,
                    # due to an issue on handling newlines when reading CSV to generate BERT embedding
                    # It shouldn't be a big problem and in this case we skip over this line
                    padded_inputs = np.zeros((self.max_sentence_length, 768)).T
                    embedding = torch.from_numpy(padded_inputs).float()
                    return (embedding, self.targets[index])

                # Pad embedding if it is less than self.max_sentence_length
                if self.max_sentence_length > sentence_len:
                    padded_inputs = np.pad(embedding, pad_width = ((0, self.max_sentence_length - sentence_len), (0, 0)), mode = "constant").T
                    embedding = torch.from_numpy(padded_inputs).float()
                # Truncate embedding if it is greater than self.max_sentence_length
                else:
                    embedding = torch.from_numpy(embedding[:self.max_sentence_length, :].T).float()

            return (embedding, self.targets[index])

    def __len__(self):
        return self.len

    def __get_targets__(self):
        if self.dataset_type != "full":
            return self.targets[self.idx]
        return self.targets

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        return out

class Dense(nn.Module):
    def __init__(self, input_size, H1, H2, num_classes):
        super(Dense, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, H1)
        self.bn1 = nn.BatchNorm1d(num_features=H1)
        self.linear2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, int(self.input_size))
        l1 = self.bn1(F.relu(self.linear1(x)))
        l2 = self.bn2(F.relu(self.linear2(l1)))
        out = F.softmax(self.linear3(l2))
        return out

# TODO: CNN2d
class CNN2d(nn.Module):
    def __init__(self, input_shape):
        super(CNN2d, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 32, 5, padding = 2) # p = (h - 1) / 2
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
    def __init__(self, input_shape):
        super(CNN1d, self).__init__()
        self.input_shape = input_shape
        # Here, we treat the embedding dim as the number of channels
        self.conv1 = nn.Conv1d(self.input_shape[0], 32, 5, padding = 2) # p = (h - 1) / 2
        self.pool = nn.MaxPool1d(2, stride = 2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv1d(64, 64, 3, padding = 1)
        self.fc1 = nn.Linear(int(64 * (self.input_shape[1]/8)), 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Here we divide input_shape by 8, since we pool 3 times
        # Each convolutional layer does not change the second dimension
        x = x.view(-1, int(64 * (self.input_shape[1]/8)))
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x

# TODO: rnn
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

if __name__ == "__main__":

    # use args to specify embedding file
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store", dest="model", help="Which model to try: `logreg`, `dense`")
    parser.add_argument("--model_path", action="store", dest="model_path", help="Path to model file")
    parser.add_argument("--embedding", action="store", dest="embedding", help="The word embeddings to use, which can be 'elmo` or `bert`. Default to pretrained embeddings.")
    parser.add_argument("--train", action="store", dest="train_filepath", help="This is the training file, with each of the training examples embedded already")
    parser.add_argument("--targets", action="store", dest="target_filepath", help="This is the path to the training file's targets")
    parser.add_argument("--average", action="store_true", dest="average", default=False, help="Use the average word in the sentence instead of the entire sentence vector")
    args = parser.parse_args()


    """
    TO RUN:

    1) Run the following command:
            python generate_roc.py --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv --model=cnn1d --embedding=elmo --model_path=elmo_pad/cnn1d/cnn1d_499.pt

       You can change the parameters depending on what you want to run:
            --model=[logreg, dense, cnn2d, cnn1d, rnn]
            --embedding=[elmo, bert_sentence, bert_word, glove]
            [--average]
       If you don't include the --average tag, the embeddings will be padded with 0's, and
       longer sequences will be truncated.

    """

    print("Loading Data")

    if args.train_filepath == "" or args.embedding not in ["elmo", "bert_word", "bert_sentence", "glove"]:
        dataset = load_dataset_from_file("../train.csv", "glove_mean_avg.pt")
        datalen = len(dataset)
        train_len = int(0.7*datalen)
        val_len = int(0.1*datalen)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, datalen-train_len-val_len])
        weights = torch.Tensor([x[1]*9+1 for x in train_dataset])
        # Hyperparameters
        input_size = 600
        args.embedding = "glove"
    else:
        train_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     args.embedding, \
                                     dataset="train", \
                                     averaged=args.average, \
                                     max_sentence_length=max_sentence_length)
        val_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     args.embedding, \
                                     dataset="validation", \
                                     averaged=args.average, \
                                     max_sentence_length=max_sentence_length)
        test_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     args.embedding, \
                                     dataset="test", \
                                     averaged=args.average, \
                                     max_sentence_length=max_sentence_length)
        targets = train_dataset.__get_targets__()

        # Hyperparameters
        if args.embedding == 'elmo':
            input_size = 1024
        elif args.embedding == 'bert_sentence' or args.embedding == 'bert_word':
            input_size = 768

        if not args.average:
            input_size = (input_size, max_sentence_length)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    print("Creating Model")

    if args.model == "logreg":
        model = LogisticRegression(input_size, num_classes)

    elif args.model == "dense":
        if len(np.shape(input_size)) > 0:
            input_size = np.prod(input_size)
        model = Dense(input_size, 256, 32, num_classes)

    # Note: must run cnn1d with non averaged embeddings
    elif args.model == "cnn1d":
        model = CNN1d(input_size)

    else:
        print("Model is not valid")
        exit(1)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))


    # Compute on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running on {device}")


    print("Testing Model")
    # Test the Model
    model.eval()
    y_scores = []
    true_labels = []
    i = 0
    for sentences, labels in test_loader:
        sentences = Variable(sentences).to(device)
        labels = labels.to(device)

        outputs = model(sentences)
        y_scores.extend(outputs.data[:, 1].tolist())
        true_labels.extend(labels.tolist())

    y_scores = np.asarray(y_scores)
    true_labels = np.asarray(true_labels)

    fpr, tpr, thresholds = roc_curve(true_labels, y_scores)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, "b", label=f"AUC={roc_auc:0.4f}")
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    plt.savefig("roc.png")
