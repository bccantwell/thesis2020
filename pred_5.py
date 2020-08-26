#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:10:46 2019

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
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, MaxPooling1D
from keras.layers import Embedding, LSTM, Flatten, GRU, TimeDistributed, Bidirectional
from keras.preprocessing import text, sequence
from keras import utils

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
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    
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

def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
 
    # Strip escaped quotes
    text = text.replace('\\"', '')
 
    # Strip quotes
    text = text.replace('"', '')
 
    return text
#Read in the data  
df = pd.read_csv('/Users/admin/Downloads/labeled_pred2.csv')

#Shuffle the data
df = df.reindex(np.random.permutation(df.index))

#Clean the text
df['Text'] = df['Text'].apply(clean_review)

#Separate training and test sets
train_size = int(len(df) * .7)
train_posts = df['Text'][:train_size]
train_tags = df['Class'][:train_size]

test_posts = df['Text'][train_size:]
test_tags = df['Class'][train_size:]

#Tokenizer counts all the unique words in our vocab
max_words = 1000
#Limit vocab to top words by passing num_words to tokenizer
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
#Create word index lookup
tokenize.fit_on_texts(train_posts) # only fit on train

#Create training data to pass to model
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
vocab_size = len(tokenize.word_index) + 1
#Convert class labels to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

#Convert class labels to one-hot representation
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)



batch_size = 64
epochs = 3

#First attempt at an MLP (was highest performing)
model = Sequential()
model.add(Dense(300, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()              

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])
plot_history(history)

#Keras doc MLP


y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

text_labels = encoder.classes_     
cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()

#Susan Li model
batch_size = 64
epochs = 5

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
plot_history(history)

#Recurrent convolutional network
# Embedding
max_features = 500
maxlen = 15
embedding_dim = 32

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 64
epochs = 3

from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dim,
                    input_length=1000))
model.add(Dropout(0.5))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
# We add a vanilla hidden layer:
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
plot_history(history)







