import matplotlib.pyplot as plt
import nltk
from nltk.corpus import movie_reviews
# assert(nltk.download('movie_reviews'))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import models.set_alarm as sa
import models.search as se
import models.start_timer as st
import models.youtube as yt

data = {
    **sa.export,
    **se.export,
    **st.export,
    **yt.export
}

train_data = []
train_labels = []

for q in data:
    train_data.append(q)
    train_labels.append(data[q])

dev_data, dev_labels = train_data, train_labels
test_data, test_labels = train_data, train_labels

# movie_reviews_and_labels = [(movie_reviews.raw(fileid), category)
#                             for category in movie_reviews.categories()
#                             for fileid in movie_reviews.fileids(category)]
# np.random.seed(0)
# np.random.shuffle(movie_reviews_and_labels)
# train_data, train_labels = zip(*movie_reviews_and_labels[:1600])

# # Hist for yt and sa examples by WC
# plt.hist([len(train_data[i].split()) for i in range(len(train_data)) if train_labels[i] == 'youtube'],
#          30, alpha=0.5, label='YT')
# plt.hist([len(train_data[i].split()) for i in range(len(train_data)) if train_labels[i] == 'set_alarm'],
#          30, alpha=0.5, label='SA')
# plt.xlabel('Words per Review')
# plt.legend()
# plt.show()


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


# Using 250 most frequent words
vectorizer, train_features, dev_features, test_features = GetFeatures(CountVectorizer(binary=True, max_features=250))


def evaluate(model, features, labels):
    predictions = model.predict(features)
    print('Accuracy = %4.4f' % (metrics.accuracy_score(labels, predictions)))


print('\nMultinomialNB')
nb_model = MultinomialNB()
nb_model.fit(train_features, train_labels)
print('on Train:',)
evaluate(nb_model, train_features, train_labels)
print('on Dev:',)
evaluate(nb_model, dev_features, dev_labels)

print('\nLogisticRegression')
lr_model = LogisticRegression()
lr_model.fit(train_features, train_labels)
print('on Train:',)
evaluate(lr_model, train_features, train_labels)
print('on Dev:',)
evaluate(lr_model, dev_features, dev_labels)


# Use logistic function applied to the product of parameters (theta) and features (X) to produce predictions vector
def lr_predict(X, theta):
    return 1.0 / (1.0 + np.exp(-np.dot(X, theta.T)))


def lr_sgd(data, labels, learning_rate, num_epochs, batch_size=20):
    # Make sure data is in dense (non-spars) format.
    X = data.toarray()

    # Convert labels to 0/1 values
    Y = np.array([int(label == 'youtube') for label in list(labels)])

    # m = number of samples, n = number of features
    m, n = X.shape

    # Use vector of size n to store learned parameters
    theta = np.zeros(n)

    # Track the training loss and accuracy after each epoch
    losses, accuracies = [], []

    # Main training loop
    for epoch in range(num_epochs):
        # Compute loss, accuracy over all data. Accuracy threshold 0.5
        predictions = lr_predict(X, theta)
        loss = -np.sum((Y * np.log(predictions)) + ((1 - Y) * np.log(1 - predictions))) / m
        accuracy = 1.0 * sum(np.round(predictions) == Y) / m
        losses.append(loss)
        accuracies.append(accuracy)

        # Make updates based on a single batch of training data at a time
        for batch in range(0, m, batch_size):
            X_batch = X[batch:batch + batch_size]
            Y_batch = Y[batch:batch + batch_size]

            # Get current predictions for training in batch give current estimate of theta
            predictions = lr_predict(X_batch, theta)

            # Get the difference between predictions and target
            diff = predictions - Y_batch

            # The gradient with respect to logistic regression's loss is the product of the inputs and diffs
            gradient = np.dot(X_batch.T, diff) / m

            # Update theta, scaling gradient by learning rate
            theta = theta - learning_rate * gradient

    # Plot loss and accuracy for training
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch'), plt.ylabel('Train Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch'), plt.ylabel('Train Accuracy')
    plt.show()

    return theta

theta = lr_sgd(train_features, train_labels, 0.5, 500)


def regularization_experiment(c_value, penalty_type):
    model = LogisticRegression(C=c_value, penalty=penalty_type, random_state=0)
    model.fit(train_features, train_labels)
    predictions = model.predict(dev_features)

    # Show the L2 norm (Euclidean length) of the params
    # Note that model.coef_ contains a vector of learned params for each class
    print('LR RESULTS: accuracy=%4.4f  L2 norm=%4.4f  C=%4.4f \n' % (
        metrics.accuracy_score(dev_labels, predictions),
        np.sqrt((model.coef_ ** 2).sum()),
        c_value))
    print(predictions, '\n')

    return model,\
           metrics.accuracy_score(dev_labels, predictions),\
           np.sqrt((model.coef_ ** 2).sum()),\
           c_value


# Now using full vocab
vectorizer, train_features, dev_features, test_features = GetFeatures(CountVectorizer(binary=True))


def best_c_value():
    accuracy = 0.0
    l2_norm = 0.0
    c_val = 0.0
    for c_value in [100, 10, 1, 0.1, 0.01, 0.001]:
        model, a, l2, c = regularization_experiment(c_value, 'l2')
        if a < accuracy:
            break
        else:
            accuracy = a
            l2_norm = l2
            c_val = c
        print(accuracy, l2_norm, c_val)

    return model, accuracy, l2_norm, c_val


model, accuracy, l2_norm, c_value = best_c_value()


print(f'The best model uses the c_value of {c_value} and has a {l2_norm} L2 Norm value and a accuracy of {accuracy}.')


def predict_intent(c_value, query):
    model = LogisticRegression(C=c_value, penalty='l2', random_state=0)
    model.fit(train_features, train_labels)
    query_data = [query]
    query_features = vectorizer.transform(query_data)

    prediction = model.predict(query_features)
    print(prediction)


predict_intent(c_value, 'Search youtube')