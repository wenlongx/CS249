import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


test_percent = 0.2

# percent of total dataset that val makes up
# for example:
#   test_percent = 0.2
#   val_percent = 0.1
# therefore:
#   train_percent = 1 - 0.2 - 0.1
val_percent = 0.1

train_percent = 1.0 - test_percent - val_percent


df = pd.read_csv("train_targets.csv")
n_samples = len(df["0"])
X = np.zeros(n_samples)
y = np.array(df["0"])

sss = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, random_state=0)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(val_percent/(train_percent + val_percent)), random_state=0)

train_idx_file = "train_idx.npy"
val_idx_file = "validation_idx.npy"
test_idx_file = "test_idx.npy"


temp_train_idx = None
train_idx = None
val_idx = None
test_idx = None
for temp_train, test in sss.split(X, y):
    temp_train_idx = temp_train
    test_idx = test

for train, val in sss2.split(X[temp_train_idx], y[temp_train_idx]):
    train_idx = train
    val_idx = val

np.save(train_idx_file, train_idx)
np.save(val_idx_file, val_idx)
np.save(test_idx_file, test_idx)
