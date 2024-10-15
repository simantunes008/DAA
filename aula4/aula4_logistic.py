import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

df_advertising = pd.read_csv('datasets/advertising.csv')

df_advertising.head()

# sns.heatmap(df_advertising.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# sns.set_style('whitegrid')
# sns.countplot(x="Clicked on Ad", data=df_advertising)
# sns.countplot(x="Clicked on Ad", hue="Male", data=df_advertising, palette='RdBu_r')
# sns.countplot(x="Clicked on Ad", hue="...", data=df_advertising, palette='rainbow')
# sns.histplot(df_advertising['Age'].dropna(), kde=False, color='darkred', bins=30)
# df_advertising['Age'].hist(bins=30, color='darkred', alpha=0.7)
# sns.countplot(x="...", data=df_advertising)

# ! Como já temos a area income não precisamos das colunas city e country
df_advertising.drop(['Ad Topic Line', 'Country', 'City', 'Timestamp'], axis=1, inplace=True)
# print(df_advertising.head())

x = df_advertising.drop(['Clicked on Ad'], axis=1)
y = df_advertising['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2021)

# sns.countplot(x="Clicked on Ad", data=pd.DataFrame(y_train,columns=['Clicked on Ad']))
# sns.countplot(x="Clicked on Ad", data=pd.DataFrame(y_test,columns=['Clicked on Ad']))

solvers = ['newton-cg', 'lbfgs', 'liblinear']

for solver in solvers:
    starttime = time.process_time()
    
    if solver == 'lbfgs':
        logmodel = LogisticRegression(random_state=2022, solver=solver, max_iter=800)
    else:
        logmodel = LogisticRegression(random_state=2022, solver=solver)
    print(logmodel)
    logmodel.fit(x_train, y_train)
    
    endtime = time.process_time()
    print(f"\nTime spent: {endtime - starttime} seconds")
    
    predictions = logmodel.predict(x_test)
    print(f"With '{solver}': \n", classification_report(y_test, predictions))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.show()
