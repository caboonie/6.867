import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import svm

df = pd.read_csv('Combined_News_DJIA.csv')

n,d = df.shape #number of dates by number of columns

corpus = [] #Corpus needs to be a list of strings for vectorizer
y = np.zeros(n)
for i in range(n):
    heads = ""
    y[i] = df.iloc[i,1]
    for j in range(2,d): #Don't include labels/dates
        if str(df.iloc[i,j]) != "nan":
            heads += " "+df.iloc[i,j]
        else:
            heads += ""
    corpus.append(heads)


#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
X = vectorizer.fit_transform(corpus)

#clf = svm.SVC(kernel='linear', C=1)
clf = LogisticRegression()
print(cross_val_score(clf, X, y, cv=5))
