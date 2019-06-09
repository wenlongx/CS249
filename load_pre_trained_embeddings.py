import numpy as np
from torch.utils.data import TensorDataset
from torch import from_numpy
import torch
import pandas as pd
import h5py

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
    data = pd.read_csv(file)
    data["question_text"] = data["question_text"].apply(lambda x: clean_text(x))
    return data

def create_dataset(data, embeddings):
    with h5py.File("glove_sentence.hdf5", "w") as f:
        for i,j in data.iterrows():
            f.create_dataset(str(i),data=sentence2vec2D(j["question_text"].split(),embeddings),dtype='f4')
        # for i in range(10):
        #     print(np.array(f[str(i)]))
    return None

def load_dataset(data_file, embeddings_file):
    print("loading data")
    data = load_data(data_file)
    print("loading embeddings")
    embeddings = load_embeddings(embeddings_file)
    print("creating")
    return create_dataset(data,embeddings)

def load_dataset_from_file(data_file, data_embeddings_file):
    sentence_tensor = torch.load(data_embeddings_file)
    data = load_data(data_file)
    # print(from_numpy(data["target"].values.astype(np.long)).sum())
    return TensorDataset(sentence_tensor, from_numpy(data["target"].values.astype(np.long)))

def sentence2vec2D(sentence, embeddings):
    count = 0
    out = None
    for word in sentence:
        if count >= 32:
            return out
        try:
            vec = embeddings[word]
            if out is not None:
                out = np.append(out,np.array([vec]),axis=0)
            else:
                out = np.array([vec])
        except KeyError as e:
            continue
        count += 1
    return out


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
# create_dataset(None,None)
#load_dataset("./train.csv","./glove.840B.300d.txt")
# load_dataset("../train.csv","../embeddings/glove.840B.300d/glove.840B.300d.txt")
