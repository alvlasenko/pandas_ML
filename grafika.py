import pandas as pd
import numpy as np

df = pd.read_csv('data_DDF.csv', delimiter=';')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 230)
data_ddf = df.iloc[:, np.r_[33, 40:49, 59:64]]
data_ddf = data_ddf.dropna()
data_ddf['DDF grade'] = data_ddf['DD LV grade']
replace_valuer = {0: 'normal', 1: 'DDF 1', 2: 'DDF 2', 3: 'DDF 3'}
data_ddf = data_ddf.replace({'DD LV grade': replace_valuer})
df['Пол'] = df['Пол'].str.lower()
print(df)

print(df.shape)

print(df.columns)

print(df.info())

print(df.describe(include='object'))

print(df.describe())

print(df['DD LV grade'].value_counts())

print(df['Диагноз'].value_counts(normalize=True))

print(df[df['DD LV grade'] == 0]['Ve/Va'].mean())

df[(df['DD LV grade'] == 1) & (df['Пол'] == 'м')]['Ve/Va'].max()

data_ddf.apply(np.max)

columns_to_show = ['LALSr', 'LALScd', 'LALSct', 'TEF%']
data_ddf.groupby(['DD LV grade'])[columns_to_show].describe(percentiles=[])

data_ddf.groupby(['DD LV grade'])[columns_to_show].agg([np.mean, np.max, np.min])

data_ddf['TEF%'] = data_ddf['TEF%'].astype('int64')
pd.crosstab(data_ddf['DD LV grade'], data_ddf['TEF%'], margins=True)

data_ddf.pivot_table(['LALSr', 'LALScd', 'LALSct'], ['DD LV grade'], aggfunc=['mean', 'max', 'min'])

import warnings

warnings.simplefilter('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 8, 5

df_ls = data_ddf[[x for x in data_ddf.columns if 'LALS' in x] + ['DDF grade']]
df_ls.groupby('DDF grade').mean().plot()

df_ls.groupby('DDF grade').mean().plot(kind='bar', rot=0)

sns_plot = sns.pairplot(df_ls)
# sns_plot.savefig('pairplot.png')

sns.jointplot(df_ls['LALSr'], df_ls['LALScd'])

sns.boxplot(x='LALSr', y='DDF grade', data=data_ddf, orient='h')

df1 = df.pivot_table(index='DD LV grade', columns='Пол', values='LALSr', aggfunc=[np.mean, max]).fillna(0).applymap(
    float)
sns.heatmap(df1, annot=True, fmt='.1f', linewidths=.5)

data_ddf['DDF grade'].value_counts().plot(kind='bar')

corr_matrix = data_ddf.drop(['DD LV grade'], axis=1).corr()
corr_matrix_map = sns.heatmap(corr_matrix)

features = list(set(data_ddf.columns) - set(['DDF grade', 'DD LV grade']))
data_ddf[features].hist(figsize=(20, 12))
plt.show()

sns_pairplot = sns.pairplot(data_ddf[features + ['DDF grade']], hue='DDF grade')
sns_pairplot.savefig('1.png')

fig, axes = plt.subplots(figsize=(8, 5))
sns.boxplot(x='DDF grade', y='LALSr', data=data_ddf, ax=axes)

import math

print(-(9 / 20) * (math.log(9 / 20, 2)) - (11 / 20) * (math.log(11 / 20, 2)))
