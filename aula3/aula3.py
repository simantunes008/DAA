import sklearn as skl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

df_sentiment_analysis = pd.read_csv('datasets_3/sentiment_analysis.csv')

# print(df_sentiment_analysis.columns)
# print(df_sentiment_analysis.head())
# print(df_sentiment_analysis.tail())
# print(df_sentiment_analysis.shape)
# print(df_sentiment_analysis.dtypes)
# df_sentiment_analysis.info()
print(df_sentiment_analysis.describe())
# print(df_sentiment_analysis.isna().any())
# print(df_sentiment_analysis.isna().sum())

# print(df_sentiment_analysis.duplicated().sum())
df_sentiment_analysis.drop_duplicates(inplace=True)
df_sentiment_analysis.drop(['Sentiment Analysis', 'Products', 'birthday'], axis=1, inplace=True)
df_sentiment_analysis.info()
