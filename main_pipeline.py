# Main pipeline for training and updating the learned model.

import csv
import pickle
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

train_labels = []
train_data = []
with open('model_training/model_training.csv', mode='r', encoding='utf-8', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    _ = next(csvreader)
    for row in csvreader:
        train_labels.append(row[0])
        train_data.append(row[1])

dev_data, dev_labels = train_data, train_labels
test_data, test_labels = train_data, train_labels

# Produce features for dataset given a vectorizer
def GetFeatures(vectorizer=CountVectorizer(binary=True, analyzer='word', ngram_range=(1, 3))):
    train_features = vectorizer.fit_transform(train_data)
    # print(vectorizer.get_feature_names())
    # print(train_features.toarray())

    # Apply the vectorizer to create features for the dev and test data
    dev_features = vectorizer.transform(dev_data)
    test_features = vectorizer.transform(test_data)

    # Check out the data shapes
    print('Model vectorizer vocab size:', len(vectorizer.vocabulary_))
    print('train features shape:', train_features.shape)
    print('dev features shape:', dev_features.shape)
    print('test features shape:', test_features.shape)

    return vectorizer, train_features, dev_features, test_features


# Using full vocab get features
vectorizer, train_features, dev_features, test_features = GetFeatures(CountVectorizer(binary=True))


def update_model(c_value, features, labels, vectorizer):
    model = LogisticRegression(C=c_value, penalty='l2', random_state=0)
    model.fit(features, labels)
    with open('model', 'wb') as f:
        pickle.dump([vectorizer, model], f)
    now = datetime.now()
    print('\nModel updated at', now.strftime('%H:%M:%S'))


update_model(0.1, train_features, train_labels, vectorizer)
