# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 23:43:31 2014

@author: madhav

Sample text mining code

"""

import pandas as pd
import re

from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
train = pd.read_table("/Users/madhav/study/programming/repos/TextMining/Data/SMSSpamCollection.txt")


# ==============================
# preprocessing function
# ==============================

# preliminary cleaning 
def prelimClean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()

# some more cleaning
p = re.compile('[.,/:()<>|?*]|(\\\)')
def cleanText(x):
    x = x.replace("\r", "").replace("\\n", "").lower()
    x = p.sub(" ", x)
    return x

# Remove common words
def removeStopwords(x):
    eng_stop = stopwords.words("english")
    tmp = [w for w in x if w.lower() not in eng_stop]
    return tmp
		
# Stemming
def stemIt(word_list, stemmer= PorterStemmer(), encoding= "utf8"):
    tmp = []
    for w in word_list:
        tmp.append(stemmer.stem(w).encode(encoding))
    return tmp

def wrapItUp(x, stemming= True):
    if stemming:
        x = " ".join(stemIt(removeStopwords(word_tokenize(cleanText(x)))))
    else:
        x = " ".join(removeStopwords(word_tokenize(cleanText(x))))
    return x


# ==============================
# Some action
# ==============================

# pre-process
train['text'] = train['text'].apply(prelimClean)
train["text"] = [wrapItUp(t, stemming= True) for t in train["text"]]    


# ==============================
# TF-IDF
# ==============================

# define
tfidf_vec = TfidfVectorizer(min_df= 5,  max_features= 100)
tfidf_vec.fit(train["text"])

print("Transform data sets")
tdm = tfidf_vec.transform(train["text"])
tdm.shape

# eof
