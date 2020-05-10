from __future__ import print_function, division
from builtins import range
import os, argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

import tensorflow.keras.backend as K

import gensim

import os.path
from os import path

def convert_label_to_one_hot(label_key):
    one_hot = np.zeros(4)
    if 'food' in label_key:
        one_hot[0] = 1
        
    if 'ambience' in label_key:
        one_hot[1] = 1
        
    if 'experience' in label_key:
        one_hot[2] = 1
        
    if 'service' in label_key:
        one_hot[3] = 1
        
    if 'staff' in label_key:
        one_hot[3] = 1
        
    return one_hot

def convert_one_hot_to_label(one_hot):
    label = list()
    
    if one_hot[0] == 1:
        label.append('food')
        
    if one_hot[1] == 1:
        label.append('ambience')
        
    if one_hot[2] == 1:
        label.append('experience')
        
    if one_hot[3] == 1:
        label.append('service')
        
    return '_'.join(label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    MAX_SEQUENCE_LENGTH = 300
    MAX_VOCAB_SIZE = 40000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    training_dir = args.training
    validation_dir = args.validation
    model_dir = args.model_dir

    s3_embeddings_bin = 's3://zomato-autophrase-poc-zvissh/entity-classifier/model/GoogleNews-vectors-negative300.bin'
    train_csv = training_dir + '/multi_class_train.csv'

    print(f"{train_csv} exists = {os.path.isfile(train_csv)}")

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

    

    print('\n\nLoading embeddings...')
    emb_model = gensim.models.KeyedVectors.load_word2vec_format(s3_embeddings_bin, binary=True)
    print('Embeddings loaded SUCCESSFYLLY!!!')

    print('Laoding train csv...')
    train = pd.read_csv(train_csv)
    print('Loaded train csv SUCCESSFULLY!!!')


    print('Init word2vec...')
    for word, vec in emb_model.vocab.items():
        word2vec[word] = emb_model.get_vector(word)
    print('Word2Vec init SUCCESSFULLY!!!')



    sentences = train['phrase'].fillna('DUMMY_VALUES').values
    targets = np.array([ convert_label_to_one_hot(l) for l in list(train['label'].values)])
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
    output = Dense(4, activation="sigmoid")(x)

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

    tf.saved_model.save(
        model,
        os.path.join(model_dir, 'model/1')
    )
