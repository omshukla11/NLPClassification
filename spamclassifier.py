import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])

stemmer = PorterStemmer()
corpus = []
sw = set(stopwords.words('english'))
sw.add('us')
for i in range(len(messages)):
    review = re.sub('[^a-zA-z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in sw]
    review = ' '.join(review)
    corpus.append(review)  
    
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer(max_features=900)
X = cv.fit_transform(corpus).toarray()

# tfid = TfidfVectorizer(max_features=750)
# X = tfid.fit_transform(corpus).toarray()
# In every case countgives better accuracy then TFIDF

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
# In every case MultinomialNB works better

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
