import torch
import torch.nn as nn
from torch.autograd import Variable
from load_pre_trained_embeddings import load_dataset
from torch.nn.functional import softmax


# Hyper Parameters
input_size = 600
num_classes = 2
num_epochs = 5
batch_size = 100
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
train_dataset = load_dataset("../train.csv","../embeddings/glove.840B.300d/glove.840B.300d.txt")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
print("Creating Model")
model = LogisticRegression(input_size, num_classes)

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

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(sentences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i) % 100 == 0:
            torch.save(model.state_dict(), PATH)
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                   % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))
