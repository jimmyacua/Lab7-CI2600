import string
from collections import Counter
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, metrics, svm
import pandas as pd
from sklearn.metrics._plot.tests.test_plot_confusion_matrix import y_pred
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pytest


def read_csv(filename, columns):
    data = pd.read_csv(filename, usecols=columns, sep=",", encoding='latin-1')
    label = data[columns[0]]
    message = data[columns[1]]
    return label, message


def getRatio(vector):
    counter = Counter()
    for word in vector:
        counter[word] += 1
    spamCount = counter['spam']
    hamCount = counter['ham']
    ratio = 0
    if spamCount >= hamCount:
        ratio = spamCount/hamCount
    else:
        ratio = hamCount/spamCount

    ratio = str(int(ratio)) + ":1"
    print("ratio: ", ratio)
    return spamCount, hamCount


def countMostCommonWords(spamvector, hamvector, numberOfWords):
    print("SPAM:",  Counter(spamvector).most_common(numberOfWords))
    print("HAM:", Counter(hamvector).most_common(numberOfWords))


if __name__ == "__main__":
    columns = ["v1", "v2"]
    labels, messages = read_csv("spam.csv", columns)
    ''' OBTENER EL RATIO'''
    spam, ham = getRatio(labels)
    fig = plt.figure(u'Gráfica de barras')  # Figure
    ax = fig.add_subplot(111)  # Axes

    nombres = ['spam', 'ham']
    datos = [spam, ham]
    xx = range(len(datos))

    ax.bar(xx, datos, width=0.8, align='center')
    ax.set_xticks(xx)
    ax.set_xticklabels(nombres)
    plt.show()

    spamWords = []
    hamWords = []
    i = 0
    for msg in messages:
        if labels[i] == 'ham':
            hamWords.append(msg)
        else:
            spamWords.append(msg)
        i += 1
    spamVector = "".join(spamWords)
    hamVector = "".join(hamWords)
    hamVector = hamVector.lower()
    hamVector = hamVector.translate(str.maketrans('', '', string.punctuation))

    spamVector = spamVector.lower()
    spamVector = spamVector.translate(str.maketrans('', '', string.punctuation))

    hamVector = hamVector.split()
    print(len(hamVector))
    spamVector = spamVector.split()
    print(len(spamVector))
    countMostCommonWords(spamVector, hamVector, 15)

    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    x = vectorizer.fit_transform(messages)
    print(vectorizer.get_feature_names())
    print(x.toarray())

    labelsBinary = []
    data = pd.read_csv("spam.csv", usecols=columns, sep=",", encoding='latin-1')
    for label in data[columns[0]]:
        if label == 'spam':
            labelsBinary.append(1)
        else:
            labelsBinary.append(0)

    data[columns[0]] = labelsBinary
    x_train, x_test, y_train, y_test = train_test_split(data[columns[1]], data[columns[0]], random_state=42, test_size=0.30, shuffle=True)

    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()
    print("x train:\n", x_train)
    print("y train:\n", y_train)
    print("x test:\n", x_test)
    print("y test:\n", y_test)
    print("###############################################")

    best_score = 0
    best_C = 0
    best_gamma = 0
    '''for C in np.arange(0.05, 2.05, 0.05):
        for gamma in np.arange(0.001, 0.101, 0.002):
            model = SVC(kernel='rbf', gamma=gamma, C=C)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            if score > best_score:
                best_score = score
                best_C = C
                best_gamma = gamma
                svm_pred = model.predict(x_test)
                print(metrics.classification_report(y_test, svm_pred))
                print("Best c: ", best_C, " best gama: ", best_gamma)

        print('Highest Accuracy Score: ', best_score)
        print('##### C: ', C)'''




    '''clf = SVC(kernel="rbf", gamma=0.01, C=0.05)
    clf.fit(x_train, y_train)
    svm_pred = clf.predict(x_test)
    print(metrics.classification_report(y_test, svm_pred))'''

    # tomado de https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()