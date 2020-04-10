from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


logreg = LogisticRegression()
dataset = pd.read_csv('data_banknote_authentication.txt', names=[
    'p1', 'p2', 'p3', 'p4', 't'])
x = dataset.drop(columns=['t'])
y = dataset['t']
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(logreg.score(X_test, y_test))
print(classification_report(y_test, y_pred))
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
