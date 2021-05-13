import os, re
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time


def text_preprocessing(s: str):
    """
    :param s: original text
    :return: processed text
    Procedures:
    1. Lowercase the sentence
    2. Change "'t" to "not"
    3. Isolate and remove punctuations except "?"
    4. Remove other special characters
    5. Remove stop words except "not" and "can"
    6. Remove trailing whitespaces
    """
    if type(s) is not str:
        return ""
    s = s.lower()
    s = re.sub(r"\'t", "not", s)
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    s = re.sub(r'([\;\:\|《\n])', ' ', s)
    s = " ".join([word for word in s.split() if word not in stopwords.words('english') or word in ['not', 'can']])
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def sentiment_analysis_for_review(df):
    review_texts = df[['review_summary', 'review_text']]
    fit_label = df['fit']

    fit_label_embed = [1 if label == 'fit' else 0 for label in fit_label]
    positive_num = sum(fit_label_embed)
    negative_num = len(fit_label_embed)-positive_num
    print("Positive fit label number: " + str(positive_num))
    print("Negative fit label number: " + str(negative_num))

    X_train, X_test, y_train, y_test = train_test_split(review_texts, fit_label, test_size=0.2, random_state=2021)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2021)

    start = time()

    print("Start text preprocessing...")
    X_train_preprocessed_summary = np.array([text_preprocessing(text) for text in X_train['review_summary']])
    X_train_preprocessed_text = np.array([text_preprocessing(text) for text in X_train['review_text']])
    X_val_preprocessed_summary = np.array([text_preprocessing(text) for text in X_val['review_summary']])
    X_val_preprocessed_text = np.array([text_preprocessing(text) for text in X_val['review_text']])
    print("Finish text preprocessing in " + str((time()-start)/60) + "s")

    # TODO: use TF-IDF to do vectorization, temporarily
    tf_idf = TfidfVectorizer(ngram_range=(1, 3), binary=True, smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(np.append(X_train_preprocessed_text, X_train_preprocessed_summary))
    X_train_tfidf_summary = X_train_tfidf[:len(X_train_preprocessed_summary)]
    X_train_tfidf_text = X_train_tfidf[len(X_train_preprocessed_summary):]
    X_val_tfidf_summary = tf_idf.transform(X_val_preprocessed_summary)
    X_val_tfidf_text = tf_idf.transform(X_val_preprocessed_text)