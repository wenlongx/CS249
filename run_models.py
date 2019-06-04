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
num_epochs = 5000
batch_size = 512
learning_rate = 0.001
PATH = "lr_model.pt"

# Dataset wrapper for the HDF5 files
class HDF5_Dataset(data.Dataset):
    def __init__(self, filepath, target_filepath, embedding, dataset="train", averaged=False):
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
       containing the final test accuracy.

    2) Run the following command:
            python run_models.py --model=logreg --embedding=elmo --train=quora-insincere-questions-classification/train_average.hdf5 --targets=quora-insincere-questions-classification/train_targets.csv --average
       You can change the parameters depending on what you want to run:
            --model=[logreg, dense]
            --embedding=[elmo, bert, glove]

    """

    print("Loading Data")
    if args.train_filepath == "" or args.embedding not in ["elmo", "bert"]:
        train_dataset = load_dataset_from_file("../train.csv", "glove_mean_avg.pt")
        test_dataset = load_dataset_from_file("../train.csv", "glove_mean_avg.pt")
        weights = torch.Tensor([x[1]*9+1 for x in train_dataset])
        # Hyperparameters
        input_size = 600
    else:
        train_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     args.embedding, \
                                     dataset="train", \
                                     averaged=args.average)
        val_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     args.embedding, \
                                     dataset="validation", \
                                     averaged=args.average)
        test_dataset = HDF5_Dataset(args.train_filepath, \
                                     args.target_filepath, \
                                     args.embedding, \
                                     dataset="test", \
                                     averaged=args.average)
        targets = train_dataset.__get_targets__()
        weights = torch.Tensor(list(map(lambda x: (1306120.0/1225310.0) if x == 0 else (1306120.0/80810.0), targets)))
        # Hyperparameters
        if args.embedding == 'elmo':
            input_size = 1024
        elif args.embedding == 'bert':
            input_size = 768

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
        model = Dense(input_size, 256, 32, num_classes)
        model.load_state_dict(torch.load(PATH))

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    else:
        print("Model is not valid")
        exit(1)

    if args.embedding not in ["elmo", "bert"]:
        args.embedding = "glove"
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

        if epoch % 1000 == 1:
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

    try:
        precision = true_positives*1.0/pred_positives.item()
        recall = true_positives*1.0/real_positives.item()
        print(true_positives,pred_positives,real_positives)
        print("F1 score: {}".format(2*(precision*recall)/(precision+recall)))
    except:
        print("Error calculating f1 score")
    accuracy = (100 * correct / total)
    print('Accuracy of the model: %d %%' % accuracy)
    np.save(f"{MODEL_PATH}/accuracy.npy", np.array([accuracy]))
