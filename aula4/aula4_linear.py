import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df_EcommerceCustomers = pd.read_csv('datasets/EcommerceCustomers.csv')
df_EcommerceCustomers.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)

# sns.pairplot(df_EcommerceCustomers)
# sns.histplot(df_EcommerceCustomers['Yearly Amount Spent'])
# sns.heatmap(df_EcommerceCustomers.corr())

x = df_EcommerceCustomers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df_EcommerceCustomers['Yearly Amount Spent']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2021)

# sns.histplot(y_test)
# sns.histplot(y_train)

lm = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
lm.fit(x_train, y_train)

# print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(x_test)
# plt.scatter(y_test, predictions) #! Gráfico de dispersão entre os valores reais e os valores previstos
# sns.histplot((y_test-predictions),bins=50)

print('\nMAE:', metrics.mean_absolute_error(y_test, predictions))        #! Média das diferenças entre os valores reais e os valores previstos
print('MSE:', metrics.mean_squared_error(y_test, predictions))           #! O mesmo só que ao quadrado
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) #! Raiz quadrada do MSE

plt.show()