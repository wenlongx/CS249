import numpy as np
from torch.utils.data import TensorDataset
from torch import from_numpy
import pandas as pd

def load_embeddings(file):
    return dict(parse_line(line) for line in open(file))

def parse_line(line):
    line = line.split(" ")
    return line[0], np.asarray(line[1:], dtype='float32')

def clean_text(x):
    '''From: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings'''
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&?!':
        x = x.replace(punct, f' {punct} ')
    for punct in '.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

def load_data(file):
    data = pd.read_csv(file).head(100000)
    data["question_text"] = data["question_text"].apply(lambda x: clean_text(x))
    return data

def create_dataset(data, embeddings):
    sentences = data["question_text"].apply(lambda x: sentence2vec(x.split(),embeddings)).values
    return TensorDataset(from_numpy(np.array([*sentences]).astype(np.float32)), from_numpy(data["target"].values.astype(np.long)))

def load_dataset(data_file, embeddings_file):
    data = load_data(data_file)
    embeddings = load_embeddings(embeddings_file)
    return create_dataset(data,embeddings)

def sentence2vec(sentence, embeddings, embedding_dim=300):
    max = np.zeros(embedding_dim)
    avg = np.zeros(embedding_dim)
    valid = 0
    for word in sentence:
        try:
            vec = embeddings[word]
            max = np.maximum(max,vec)
            avg += vec
            valid += 1
        except:
            continue
    if valid:
        avg /= valid
    return np.concatenate((max,avg))
