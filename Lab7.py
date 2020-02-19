import string
from collections import Counter
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  model_selection , metrics, svm
import pandas as pd
import torch
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

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
    for label in labels:
        if label == 'spam':
            labelsBinary.append(1)
        else:
            labelsBinary.append(0)

    data = pd.read_csv("spam.csv", usecols=columns, sep=",", encoding='latin-1')
    data[columns[0]] = labelsBinary
    X_train, X_test, y_train, y_test = train_test_split(data[columns[1]], data[columns[0]], random_state=42, test_size=0.30, shuffle=True)

    print("x train:", X_train)
    print("x test:", X_test)
    print("y train:", y_train)
    print("y test:", y_test)

    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)