import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import os
from sklearn.preprocessing import MaxAbsScaler
import pickle
import string
import numpy as np
import re

# data_dir = os.path.join('data_reviews/') 
# x_train = pd.read_csv(data_dir+'x_train.csv')

# y_train = pd.read_csv(data_dir+'y_train.csv')['is_positive_sentiment']


def extract_BoW_features(texts):
    processed_texts = [text[1] for text in texts]
    with open ('./vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    features = vectorizer.transform(processed_texts)
    scaler = MaxAbsScaler()
    features = scaler.fit_transform(features)
    return features.toarray()


