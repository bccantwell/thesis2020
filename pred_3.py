#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:54:43 2019

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
    
#df['Text'] = df['Text'].apply(clean_text)
print(print_plot(9))

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
wv = gensim.models.KeyedVectors.load_word2vec_format("crawl-300d-2M.vec", binary=False)
wv.init_sims(replace=True)

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens
    
train, test = train_test_split(df, test_size=0.3, random_state = 42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train['Class'])
y_pred = logreg.predict(X_test_word_average)
my_class = ['victim', 'predator']
print('accuracy %s' % accuracy_score(y_pred, test.Class))
print(classification_report(test.Class, y_pred,target_names=my_class))

