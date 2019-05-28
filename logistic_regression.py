import torch
import torch.nn as nn
from torch.autograd import Variable
from load_pre_trained_embeddings import load_dataset,load_dataset_from_file
from torch.nn.functional import softmax


# Hyper Parameters
input_size = 600
num_classes = 2
num_epochs = 400
batch_size = 10000
learning_rate = 0.001
PATH = "lr_model.pt"

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        return out

print("Loading Data")
train_dataset = load_dataset_from_file("../train.csv","glove_mean_avg.pt")
weights = torch.Tensor([x[1]*9+1 for x in train_dataset])
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler=sampler)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
print("Creating Model")
model = LogisticRegression(input_size, num_classes)
# model.load_state_dict(torch.load(PATH))

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
precision = true_positives*1.0/pred_positives.item()
recall = true_positives*1.0/real_positives.item()
print(true_positives,pred_positives,real_positives)
print("F1 score: {}".format(2*(precision*recall)/(precision+recall)))
print('Accuracy of the model: %d %%' % (100 * correct / total))
