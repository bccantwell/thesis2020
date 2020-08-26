#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:43:35 2019

@author: admin
"""
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
#%matplotlib inline

df = pd.read_csv('/Users/admin/Downloads/labeled_pred.csv')
df = df.reindex(np.random.permutation(df.index))
print(df.head(10))

#plt.figure(figsize=(10,4))
#df.Class.value_counts().plot(kind='bar');

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def print_plot(index):
    example = df[df.index == index][['Text', 'Class']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Class:', example[1])



def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    #text = text.lower()
    text = BeautifulSoup(text, "lxml").text # HTML decoding
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
    text = re.sub(r" NP ", " no problem ", text)
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
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    #text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    
    return text 
    
#df['Text'] = df['Text'].apply(clean_text)
print(print_plot(9))

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re

def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled

X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Class, random_state=0, test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(all_data)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    
    

def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors
    
train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_vectors_dbow, y_train)
logreg = logreg.fit(train_vectors_dbow, y_train)
y_pred = logreg.predict(test_vectors_dbow)
my_class = ['victim', 'predator']
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_class))

