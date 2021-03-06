#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:14:31 2019

@author: admin
"""

import itertools
import os

import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras.layers import LSTM, GRU, Flatten, Bidirectional
from keras.layers import Embedding
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    #text = text.lower()
    #text = BeautifulSoup(text, "lxml").text # HTML decoding
    #text = text.lower() # lowercase text
    
    text = re.sub(r" i'm ", " i am ", text)
    text = re.sub(" id ", " i would ", text)
    text = re.sub(r" im ", " i am ", text)
    #text = re.sub(r"u c", "you see", text)
    #text = re.sub(r" i c ", "i see", text)
    text = re.sub(r"he's", " he is ", text)
    text = re.sub(r" she's ", " she is ", text)
    text = re.sub(r" wut ", " what ", text)
    text = re.sub(r" kewl ", " cool ", text)
    text = re.sub(r" omg ", " oh my god ", text)
    text = re.sub(r" it's ", " it is ", text)
    text = re.sub(r" its ", " it is ", text)
    text = re.sub(r" kno ", " know ", text)
    #text = re.sub(r"i kno", " i know ", text)
    text = re.sub(r" ty ", "thank you", text)
    text = re.sub(r" thx ", " thanks ", text)
    text = re.sub(r" u ", " you ", text)
    text = re.sub(r" sry ", " sorry ", text)
    text = re.sub(r" ur ", " your ", text)
    text = re.sub(r" yr ", " your ", text)
    text = re.sub(r" yea ", " yes ", text)
    text = re.sub(r" ya ", " yes ", text)
    text = re.sub(r" u r ", " you are ", text)
    text = re.sub(r" r ", " are ", text)
    text = re.sub(r" b4 ", " before ", text)
    text = re.sub(r" k ", " okay  ", text)
    text = re.sub(r" ok ", " okay  ", text)
    text = re.sub(r" y ", " why ", text)
    text = re.sub(r" mayb ", " maybe ", text)
    text = re.sub(r" nuthin ", " nothing ", text)
    text = re.sub(r" luv ", " love ", text)
    text = re.sub(r" yup ", " yes ", text)
    text = re.sub(r" ppl ", " people ", text)
    text = re.sub(r" cuz ", " because ", text)
    text = re.sub(r" idk ", " i do not know ", text)
    text = re.sub(r" dunno ", " do not know ", text)
    text = re.sub(r" wat ", " what ", text)
    text = re.sub(r" wats ", " what is ", text)
    text = re.sub(r" gonna ", " going to ", text)
    text = re.sub(r" gotta ", " got to ", text)
    text = re.sub(r" goin ", " going ", text)
    text = re.sub(r" wanna ", " want to ", text)
    text = re.sub(r" wanta ", " want to ", text)
    text = re.sub(r" pussy ", " vagina ", text)
    text = re.sub(r" cock ", " penis ", text)
    text = re.sub(r" dick ", " penis ", text)
    text = re.sub(r" prly ", " probably ", text)
    text = re.sub(r" prolly ", " probably ", text)
    text = re.sub(r" pic ", " photo ", text)
    text = re.sub(r" pics ", " photos ", text)
    text = re.sub(r" pix ", " photos ", text)
    text = re.sub(r" lol ", " laugh ", text)
    text = re.sub(r" LOL ", " laugh ", text)
    text = re.sub(r"that's", " that is ", text)
    text = re.sub(r" thats ", " that is ", text)
    text = re.sub(r"what's", " what is ", text)
    text = re.sub(r" where's", " where is ", text)
    text = re.sub(r" wheres ", " where is ", text)
    text = re.sub(r"how's", " how is ", text)
    text = re.sub(r" hows ", " how is ", text)
    text = re.sub(r" howve ", " how have ", text)
    text = re.sub(r"\'ll ", " will ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    #text = re.sub(r"\'d", " would ", text)
    #text = re.sub(r"\'re", " are ", text)
    text = re.sub(r" won't", " will not ", text)
    text = re.sub(r" wont ", " will not ", text)
    text = re.sub(r" dont ", " do not ", text)
    text = re.sub(r" don't ", " do not ", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r" cant ", " cannot ", text)
    #text = re.sub(r"n't ", " not ", text)
    #text = re.sub(r"n' ", "ng ", text)
    #text = re.sub(r"'bout", "about", text)
    #text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    #text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    
    return text 
  
df = pd.read_csv('/Users/admin/Downloads/labeled_pred2.csv')
#Shuffle the data
df = df.reindex(np.random.permutation(df.index))

#Clean the text
df['Text'] = df['Text'].apply(clean_text)
train_size = int(len(df) * .7)
train_posts = df['Text'][:train_size]
train_tags = df['Class'][:train_size]

test_posts = df['Text'][train_size:]
test_tags = df['Class'][train_size:]

max_words = 10000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
vocab_size = len(tokenize.word_index) + 1

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

from keras.preprocessing.sequence import pad_sequences

maxlen = 100
X_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(x_test, padding='post', maxlen=maxlen)


from keras import layers

embedding_dim = 32
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(GRU(units=100, dropout=0.2, recurrent_dropout=0.2))
#model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(num_classes, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=15,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=32)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 100
embedding_matrix = create_embedding_matrix(
        '/Users/admin/Downloads/glove.6B/glove.6B.100d.txt',
        tokenize.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=False))


model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
#model.add(Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(num_classes, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=25,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=100 )
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)