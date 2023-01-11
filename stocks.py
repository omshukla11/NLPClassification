import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

df=pd.read_csv('StockSentiment.csv', sep=',')

train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df[df.columns[2:]], df['Label'], test_size=0.2)

data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index

headlines = []

for idx in new_Index:
    data[idx] = data[idx].str.lower()

for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

cv = CountVectorizer(ngram_range=(2,2))
X_train = cv.fit_transform(headlines)
y_train = train['Label']
rf_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm_train = confusion_matrix(y_train, rf_classifier.predict(X_train))

# test_transform= []
# for row in range(0,len(test.index)):
#     test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
# test_dataset = cv.transform(test_transform)

test_data = test.iloc[:, 2:27]
test_data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

test_data.columns = new_Index
for idx in new_Index:
    test_data[idx] = test_data[idx].str.lower()

test_headlines = []
for row in range(0,len(test_data.index)):
    test_headlines.append(' '.join(str(x) for x in test_data.iloc[row,0:25]))

X_test = cv.transform(test_headlines)
y_test = test['Label']
y_pred = rf_classifier.predict(X_test)

cm_test = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# With Lemmatization & Stopword removal (NOT MUCH DIFFERENT RESULT)

train_data = train.iloc[:,2:]
test_data = test.iloc[:,2:]

list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
train_data.columns = new_Index
test_data.columns = new_Index

train_data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
test_data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

for idx in new_Index:
    train_data[idx] = train_data[idx].str.lower()
    test_data[idx] = test_data[idx].str.lower()

sw = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

for x in range(len(train_data.index)):
    print(x)
    for nh in train_data.iloc[x,:]:
        if nh is not np.nan:
            temp = nh.split()
            temp = [lemmatizer.lemmatize(word) for word in temp if word not in sw]
            nh = ' '.join(temp)
        else:
            print(nh)

headlines = [] 
for row in range(0,len(train_data.index)):
    headlines.append(' '.join(str(x) for x in train_data.iloc[row,0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

cv = CountVectorizer(ngram_range=(2,2))
X_train = cv.fit_transform(headlines)
y_train = train['Label']
rf_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm_train = confusion_matrix(y_train, rf_classifier.predict(X_train))

for x in range(len(test_data.index)):
    print(x)
    for nh in test_data.iloc[x,:]:
        if nh is not np.nan:
            temp = nh.split()
            temp = [lemmatizer.lemmatize(word) for word in temp if word not in sw]
            nh = ' '.join(temp)
        else:
            print(nh)

test_headlines = [] 
for row in range(0,len(test_data.index)):
    test_headlines.append(' '.join(str(x) for x in test_data.iloc[row,0:25]))

X_test = cv.transform(test_headlines)
cm_test = confusion_matrix(y_test, rf_classifier.predict(X_test))
acc_test = accuracy_score(y_test, y_pred)