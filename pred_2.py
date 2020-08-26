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
from nltk.stem import WordNetLemmatizer 

# Init the Wordnet Lemmatizer

from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
#%matplotlib inline

df = pd.read_csv('/Users/admin/Downloads/labeled_pred2.csv')
df = df.reindex(np.random.permutation(df.index))
print(df.head(10))

description = df.describe()
print(description)


plt.figure(figsize=(10,4))
df.Class.value_counts().plot(kind='bar');
count = df.Class.value_counts()
print(count)


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
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text.lower())
    return text  

#df['Text'] = df['Text'].apply(clean_text)
print(print_plot(9))
print(print_plot(29))

vect = CountVectorizer(stop_words='english')
# get counts of each token (word) in text data
X = vect.fit_transform(df['Text'])
vect.get_feature_names()
print(X.shape)
# convert sparse matrix to numpy array to view
X.toarray()


# view token vocabulary and counts
print(vect.vocabulary_)

X = df.Text
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer(ngram_range=(3,3))),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)



from sklearn.metrics import classification_report
my_class = ['victim','predator']
y_pred = nb.predict(X_test)


print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_class))


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer(binary=True, ngram_range=(3,3))),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)



y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_class))

from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer(binary=True, ngram_range=(3,3))),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_class))
