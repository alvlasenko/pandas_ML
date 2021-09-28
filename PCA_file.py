import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_ddf = pd.read_csv('data_DDF.csv', delimiter=';')
data_ddf = data_ddf.iloc[:, np.r_[33, 40:49, 59:64]]
data_ddf = data_ddf.dropna()
# data_ddf['DDF grade'] = data_ddf['DD LV grade']
replace_valuer = {0: 'normal', 1: 'DDF 1', 2: 'DDF 2', 3: 'DDF 3'}
data_ddf = data_ddf.replace({'DD LV grade': replace_valuer})

x = data_ddf.loc[0:, 'LALSr':'TEF%']
y = data_ddf.loc[0:, 'DD LV grade']
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)

pca = PCA()
x_pca = pca.fit(x)
evr = x_pca.explained_variance_ratio_
cvr = np.cumsum(x_pca.explained_variance_ratio_)
pca_df = pd.DataFrame()
print(x_pca)
pca_df['cum_var_rat'] = cvr
pca_df['exp_var_rat'] = evr
a = []
for x in range(0, len(pca_df)):
    a.append('pca_comp {}'.format(x))
r = pd.DataFrame(x_pca.components_, index=a)

print(r.T)
