from __future__ import print_function, division
from builtins import range
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

import tensorflow.keras.backend as K

import gensim

s3_embeddings_bin = 's3://zomato-autophrase-poc-zvissh/entity-classifier/model/GoogleNews-vectors-negative300.bin'
s3_train_csv = 's3://zomato-autophrase-poc-zvissh/entity-classifier/data/entity_train.csv'

emb_model = None
train = None
word2vec = {}
sentences = None
targets = None
sequences = None
data = None
embedding_matrix = None
num_words = 0
model = None

MAX_SEQUENCE_LENGTH = 300
MAX_VOCAB_SIZE = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 1

def init():
    print('Loading embeddings...')
    emb_model = gensim.models.KeyedVectors.load_word2vec_format(s3_embeddings_bin, binary=True)
    print('Embeddings loaded SUCCESSFYLLY!!!')

    print('Laoding train csv...')
    train = pd.load_csv(s3_train_csv)
    print('Loaded train csv SUCCESSFULLY!!!')

    
    print('Init word2vec...')
    for word, vec in emb_model.vocab.items():
        word2vec[word] = emb_model.get_vector(word)
    print('Word2Vec init SUCCESSFULLY!!!')

def get_one_hot(lbl):
    one_hot_y = np.zeros(9)
    one_hot_y[lbl - 1] = 1
    return one_hot_y

def pre_process_data():
    sentences = train['phrase'].fillna('DUMMY_VALUES').values
    targets = np.array([ get_one_hot(l) for l in list(train['label'].values)])
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    word2idx = tokenizer.word_index
    print('Filling pre-trained embeddings...')
    num_words = min(MAX_VOCAB_SIZE, len(word2vec) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i < MAX_VOCAB_SIZE:
            #embedding_vector = word2vec[word]
            if word in word2vec:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = word2vec[word]

def build_and_run_model():
    print('Building model...')
    embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )

    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = LSTM(15, return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    output = Dense(9, activation="sigmoid")(x)

    model = Model(input_, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.01),
        metrics=['accuracy'],
    )

    print('Training started...')
    r = model.fit(
        data,
        targets,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

def run():
    init()
    pre_process_data()
    build_and_run_model()


if __name__ == "__main__":
    run()