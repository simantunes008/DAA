from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

# ? T1.
df_contract_data = pd.read_csv('datasets_2/ContractData.csv')
df_calls_data = pd.read_csv('datasets_2/CallsData.csv')

df = pd.merge(df_contract_data, df_calls_data, how='inner', on=['Area Code', 'Phone'])

df['Churn'] = df['Churn'].astype('category')

# ? T2.
x = df.drop(['Churn', 'State', 'Phone'], axis=1)
y = df['Churn'].to_frame()

clf = DecisionTreeClassifier(random_state=2021)
scores = cross_val_score(clf, x, y, cv=10)
print(scores)
print("RESULT: %0.2f accuracy with a standard deviation of %0.2f \n" % (scores.mean(), scores.std()))

scores = cross_val_score(clf, x, y, cv=10, scoring='f1_macro')
print(scores)
print("RESULT: %0.2f f1_macro with a standard deviation of %0.2f \n" % (scores.mean(), scores.std()))

# ? T3.
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay

scores = []
kf = KFold(n_splits=10)
for train, test in kf.split(x):
    # ! Outra forma de fazer a k fold cross validation
    clf.fit(x.loc[train,:], y.loc[train,:])
    score = clf.score(x.loc[test,:], y.loc[test,:])
    print(score)
    scores.append(score)
    y_predicted = clf.predict(x.loc[test,:])
    # ! Matriz de Confusão para cada fold
    print(confusion_matrix(y.loc[test,:], y_predicted))
    ConfusionMatrixDisplay.from_estimator(clf, x.loc[test,:], y.loc[test,:])
    plt.show()

print("RESULT: %0.2f accuracy with a standard deviation of %0.2f \n" % (np.mean(scores), np.std(scores)))