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
            if self.embedding == 'elmo':
                embedding = h5py_file.get(str(index))
            elif self.embedding == 'bert':
                embedding = h5py_file.get(str(index))[0]

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
                sentence_len, word_dim = embedding.shape
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
    parser.add_argument("--embedding", action="store", dest="embedding", help="The word embeddings to use, which can be 'elmo` or `bert`. Default to pretrained embeddings.")
    parser.add_argument("--train", action="store", dest="train_filepath", help="This is the training file, with each of the training examples embedded already")
    parser.add_argument("--targets", action="store", dest="target_filepath", help="This is the path to the training file's targets")
    parser.add_argument("--average", action="store_true", dest="average", default=False, help="Use the average word in the sentence instead of the entire sentence vector")
    args = parser.parse_args()


    """
    TO RUN:

    1) Make the directories {embedding_name}/{model_name}. We'll store the trained model in there, as well
       as .npy files containing a list of the F1 scores for each of the validation passes, and the .npy file
       containing the final test accuracy. If you want to use the full embeddings (and not the averaged
       ones, then make the directory {embedding_name}_pad/{model_name}.

    2) Run the following command:
            python run_models.py --model=logreg --embedding=elmo --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv --average
python run_models.py --model=cnn1d --embedding=elmo --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv

       You can change the parameters depending on what you want to run:
            --model=[logreg, dense, cnn2d, cnn1d, rnn]
            --embedding=[elmo, bert, glove]
            [--average]
       If you don't include the --average tag, the embeddings will be padded with 0's, and 
       longer sequences will be truncated.

    """

    print("Loading Data")
    if args.train_filepath == "" or args.embedding not in ["elmo", "bert"]:
        train_dataset = load_dataset_from_file("../train.csv", "glove_mean_avg.pt")
        test_dataset = load_dataset_from_file("../train.csv", "glove_mean_avg.pt")
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
        weights = torch.Tensor(list(map(lambda x: (1306120.0/1225310.0) if x == 0 else (1306120.0/80810.0), targets)))
        # Hyperparameters
        if args.embedding == 'elmo':
            input_size = 1024
        elif args.embedding == 'bert':
            input_size = 768

        if not args.average:
            input_size = (input_size, max_sentence_length)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    print("Creating Model")

    if args.model == "logreg":
        model = LogisticRegression(input_size, num_classes)
        # model.load_state_dict(torch.load(PATH))

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    elif args.model == "dense":
        if len(np.shape(input_size)) > 0:
            input_size = np.prod(input_size)
        model = Dense(input_size, 256, 32, num_classes)
        # model.load_state_dict(torch.load("elmo/dense/dense_3001.pt"))

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Note: must run cnn1d with non averaged embeddings
    elif args.model == "cnn1d":
        model = CNN1d(input_size)
        # model.load_state_dict(torch.load("elmo_pad/cnn1d/3000.pt"))

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(model)

    else:
        print("Model is not valid")
        exit(1)

    if not args.average:
        MODEL_PATH = f"{args.embedding}_pad/{args.model}"
    else:
        MODEL_PATH = f"{args.embedding}/{args.model}"

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

        if epoch % 1000 == 999:
            # eval mode
            model.eval()
            correct = 0
            total = 0
            pred_positives = 0.0
            real_positives = 0.0
            true_positives = 0.0
            for sentences, labels in val_loader:
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
            np.save(f"{MODEL_PATH}/f1.npy", np.array(f1_scores))

            # Save model per epoch
            torch.save(model.state_dict(), f"{MODEL_PATH}/{args.model}_{epoch}.pt")

            model.train()

    # Save F1 scores at end
    np.save(f"{MODEL_PATH}/f1.npy", np.array(f1_scores))
    # Save final model
    torch.save(model.state_dict(), f"{MODEL_PATH}/{num_epochs}.pt")


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

    f1 = -1
    try:
        precision = true_positives*1.0/pred_positives.item()
        recall = true_positives*1.0/real_positives.item()
        print(true_positives,pred_positives,real_positives)
        f1 = 2*(precision*recall)/(precision+recall)
        print("F1 score: {}".format(f1))
    except:
        print("Error calculating f1 score")
    accuracy = (100 * correct / total)
    print('Accuracy of the model: %d %%' % accuracy)
    np.save(f"{MODEL_PATH}/accuracy_f1.npy", np.array([accuracy, f1]))
