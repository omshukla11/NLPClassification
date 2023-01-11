import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("fake-news/train.csv")
test = pd.read_csv("fake-news/test.csv")
submission = pd.read_csv("fake-news/submit.csv")

data = train.copy()
data.replace("[^a-zA-z]", " ", regex=True, inplace=True)
data = data.dropna()
y = data['label']
data = data.iloc[:,[1,2,3]]

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

corpus = []
lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))

for i in range(len(data.index)):
    text = ""
    for t in data.iloc[i,:]:
        t = t.lower()
        temp = t.split()
        temp = [lemmatizer.lemmatize(word) for word in temp if word not in sw]
        t = " ".join(temp)
        text = text + " " + t
    corpus.append(text)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3))

X_cv = cv.fit_transform(corpus).toarray()
X_tfidf = tfidf.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y, test_size = 0.2)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
classifier_cv = MultinomialNB()
classifier_tfidf = MultinomialNB()
classifier_cv.fit(X_train_cv, y_train_cv)
classifier_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_cv = classifier_cv.predict(X_test_cv)
y_pred_tfidf = classifier_tfidf.predict(X_test_tfidf)

from sklearn.metrics import confusion_matrix, accuracy_score
score_cv = accuracy_score(y_test_cv, y_pred_cv)
print("accuracy:   %0.3f" % score_cv)
cm_cv = confusion_matrix(y_test_cv, y_pred_cv)

score_tfidf = accuracy_score(y_test_tfidf, y_pred_tfidf)
print("accuracy:   %0.3f" % score_tfidf)
cm_tfidf = confusion_matrix(y_test_tfidf, y_pred_tfidf)