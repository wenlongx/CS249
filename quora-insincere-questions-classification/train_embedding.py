import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, time
import h5py, csv

NUM_EPOCH = 500
BATCH_SIZE = 128
EMBEDDING_SIZE = 1024
NUM_TRAIN_INPUTS = 800
NUM_TEST_INPUTS = 200

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
        self.conv1 = nn.Conv1d(1024, 32, 5, padding = 2) # p = (h - 1) / 2
        self.pool = nn.MaxPool1d(2, stride = 2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv1d(64, 64, 3, padding = 1)
        self.fc1 = nn.Linear(64 * 6, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# read csv file to get sentence label (0 or 1)
def import_labels(data_path):
    y = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                y.append(row[-1])
            line_count += 1
    y = torch.from_numpy(np.array(y).astype(int))
    return y

# extract embedding matrix from file
def load_input_minibatch(data_path, indices):
    
    with h5py.File(data_path, 'r') as h5py_file:
    
        # extract mini-batch data
        all_lines = np.array(list(h5py_file.values())[:-1])
        max_sentence_len = len(max(all_lines,key=len))
        extracted_inputs = list(map(lambda x: np.array(x), all_lines[indices]))

        # pad sentences
        padded_inputs = np.zeros((BATCH_SIZE, EMBEDDING_SIZE, max_sentence_len))
        for i,input in enumerate(extracted_inputs):
            sentence_len = input.shape[0]
            padded_inputs[i,:,:] = np.pad(input, pad_width=((0,max_sentence_len-sentence_len),(0,0)), mode='constant').T

        # padded_inputs.shape: (128, 1024, 50)
        padded_inputs = torch.from_numpy(padded_inputs).float()
        # FOR CNN2D ONLY -> add dimension: (128, 1, 1024, 50)
        # padded_inputs = padded_inputs[:, None, :, :]
        return padded_inputs

def training(num_epoch, net, optimizer, criterion, labels_all):
    accuracies = []
    losses = []
    for epoch in range(num_epoch): 
        time_epoch_start = time.time()
        running_loss = 0.0

        indices = np.arange(NUM_TRAIN_INPUTS)
        np.random.shuffle(indices)
        for start_idx in range(0, NUM_TRAIN_INPUTS - BATCH_SIZE + 1, BATCH_SIZE):
            # load data
            excerpt = indices[start_idx:start_idx + BATCH_SIZE]
            inputs = load_input_minibatch('./1000_train_average.hdf5', excerpt)
            labels = labels_all[excerpt]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # save loss and accuracies
        current_loss = running_loss / (NUM_TRAIN_INPUTS / BATCH_SIZE)
        losses.append(current_loss)
        time_epoch_end = time.time()
        time_epoch = time_epoch_end - time_epoch_start
        if (epoch+1) % 10 == 0:
            accuracy = testing(net, labels_all)
            accuracies.append(accuracy)
            print('Epoch %d: Loss = %.8f, Time: %.2f, Accuracy = %.2f %%, ' % (epoch, current_loss, time_epoch, accuracy))
        else:
            print('Epoch %d: Loss = %.8f, Time: %.2f' % (epoch, current_loss, time_epoch))

    return accuracies, losses

def testing(net, labels_all):
    correct = 0
    total = 0
    with torch.no_grad():
        indices = np.arange(NUM_TRAIN_INPUTS, NUM_TRAIN_INPUTS+NUM_TEST_INPUTS)
        for start_idx in range(0, NUM_TEST_INPUTS - BATCH_SIZE + 1, BATCH_SIZE):
            # load data
            excerpt = indices[start_idx:start_idx + BATCH_SIZE]
            inputs = load_input_minibatch('./1000_train_average.hdf5', excerpt)
            labels = labels_all[excerpt]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # test
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main():
    # load labels
    labels_all = import_labels('./1000_example_train.csv')

    # define model, loss function and optimizer
    net = CNN1d()
    if torch.cuda.is_available():
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    # train on data
    time_train_start = time.time()
    accuracies, losses = training(NUM_EPOCH, net, optimizer, criterion, labels_all)
    time_train_end = time.time()
    time_train = time_train_end - time_train_start
    print(accuracies)
    print(losses)
    print('training time: %.1f' % (time_train))


if __name__ == "__main__":
    main()


