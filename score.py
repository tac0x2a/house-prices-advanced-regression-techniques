
# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from time import localtime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pd.options.display.max_columns = None

# %%
train = pd.read_csv("train.csv")

print(train.describe())
print(train.shape)

# %% 欠損値の確認


def show_missing_values(train):
    print(train.isnull().sum()[train.isnull().sum() != 0].sort_values())
    dumm = pd.get_dummies(train)
    print(dumm.isnull().sum()[dumm.isnull().sum() != 0].sort_values())


show_missing_values(train)
#

# %% 欠損値の穴埋め


def fill_missing_data(data):
    filling_columns = data.columns[np.where(data.isnull().sum() > 0)]
    filled_data = data.copy()
    for c in filling_columns:
        v = None
        if data[c].dtype == 'object':
            v = 'None'
        else:
            v = data[c].median()
        filled_data[c] = filled_data.fillna(value=v)
    return filled_data

# fill_missing_data(train)

# %% ラベルを数値に置き換え


def label_encoder(data):
    cleansed_data = data.copy()

    label_columns = filter(lambda c: str(
        cleansed_data[c].dtype) == 'object', cleansed_data)

    for l in label_columns:
        cleansed_data[l] = LabelEncoder().fit_transform(cleansed_data[l])

    cleansed_data = cleansed_data.drop(['LotArea', 'MiscVal'], axis=1)

    return cleansed_data

# %% CSV出力する


def predict_and_output_csv(model, src_file_name, dst_file_name, scalers=[], encoders=[]):
    src = pd.read_csv(src_file_name)
    # show_missing_values(src)

    test = fill_missing_data(src)
    creansed_data = label_encoder(test)

    data = creansed_data.drop(['Id'], axis=1)

    for s in scalers:
        # print('Appling Scaler:', str(s))
        data = s.transform(data)
    predict = model.predict(data)

    result = pd.concat([test['Id'], pd.DataFrame(
        predict, columns=['SalePrice'])], axis=1)
    result.to_csv(dst_file_name, index=False)

    return result

# %% 元データからX,yを抽出する


def split_X_y(original_data):
    data = original_data.copy()
    X = data.drop(['Id', 'SalePrice'], axis=1)
    y = data['SalePrice']
    return (X, y)


# %% [WIP] エンコーダを生成する
d1 = train.drop(['Id', 'SalePrice'], axis=1)
test = pd.read_csv("test.csv")
d2 = test.drop(['Id'], axis=1)

print(d1.shape)
print(d2.shape)
print(pd.concat([d1, d2], ignore_index=True).shape)


# %% クレンジングしてデータをX,yに分解する
filled_train = fill_missing_data(train)

# ラベルを数値に置き換える
cleansed_train = label_encoder(filled_train)

X_org, y = split_X_y(cleansed_train)


# %% スケールを合わせる
scalers = []

scaler = StandardScaler().fit(X_org)
# scaler = MinMaxScaler().fit(X_org)
# scaler = RobustScaler().fit(X_org)
scalers.append(scaler)

X_scaled = scaler.transform(X_org)
X = X_scaled

# pca = PCA(n_components=10).fit(X_scaled)
# scalers.append(pca)
# X_pca = pca.transform(X_scaled)
# X = X_pca

X_train, X_test, y_train, y_test = train_test_split(X, y)


# %% Apply models -------------------------------
models = []

# Ridge
model = Ridge(alpha=.4).fit(X_train, y_train)
models.append(model)

# KNeighbors
model = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
models.append(model)

# Lasso
model = Lasso().fit(X_train, y_train)
models.append(model)

# GradientBoosting
model = GradientBoostingRegressor().fit(X_train, y_train)
models.append(model)

# RandomForest
model = RandomForestRegressor().fit(X_train, y_train)
models.append(model)

# SVM
model = SVR(kernel='rbf', C=10, gamma=.1).fit(X_train, y_train)
models.append(model)

# NN
model = MLPRegressor(solver='lbfgs', random_state=42,
                     hidden_layer_sizes=[10, 20, 10]).fit(X_train, y_train)
models.append(model)

# # %% Print Scores
best_score = 0.0
best_model = models[0]
for m in models:
    score = m.score(X_test, y_test)
    if best_score < score:
        best_score = score
        best_model = m
    print("Train:{}".format(m.score(X_train, y_train)))
    print("Test :{}".format(score, str(m)))
    print("{}\n\n".format(str(m)))

print("Best")
print(best_model.score(X_test, y_test))
print(best_model)

# %% Output 'output.csv' file.
result = predict_and_output_csv(
    best_model, 'test.csv', 'out.csv', scalers, encoders)

# %% Plot coef

"""
for model in models:
    if hasattr(model, 'coef_'):
        print(str(model))
        coef = pd.DataFrame(X.columns, columns=['Name'])
        coef['coef'] = pd.Series(model.coef_)
        coef.sort_values(by='coef')

        sns.set_context("notebook")

        model.score(X_test, y_test)
        plt.title(str(m))
        plt.figure(figsize=(20, 30))
        sns.barplot(y='Name', x='coef', data=coef)
"""
