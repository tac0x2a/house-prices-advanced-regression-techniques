
# %%
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
pd.options.display.max_columns = None

# %%
train = pd.read_csv("train.csv")

print(train.describe())
print(train.shape)
print(train.shape)

# %% 欠損値の確認
print(train.isnull().sum()[train.isnull().sum() != 0].sort_values())

dumm = pd.get_dummies(train)
print(dumm.isnull().sum()[dumm.isnull().sum() != 0].sort_values())

# %% 欠損値を埋める


def cleanse_data(data):
    fill = {
        'PoolQC': 'NoPool',

        # Fence: Fence quality. NA	No Fence
        'Fence': 'NoFence',

        # Alley: Type of alley access to property NA No alley access
        'Alley': 'NoAlleyAccess',

        # FireplaceQu: Fireplace quality NA	No Garage
        'FireplaceQu': 'NoGarage',

        'GarageType': 'NoGarage',
        'GarageYrBlt': 'NoGarage',
        'GarageFinish': 'NoGarage',
        'GarageQual': 'NoGarage',
        'GarageCond': 'NoGarage',

        # Miscellaneous feature not covered in other categories
        'MiscFeature': 'None',

        # BsmtQual: Evaluates the height of the basement
        'BsmtQual': 'NoBasement',
        # BsmtCond: Evaluates the general condition of the basement
        'BsmtCond': 'NoBasement',
        'BsmtExposure': 'NoBasement',
        'BsmtFinType1': 'NoBasement',
        'BsmtFinType2': 'NoBasement',

        # LotFrontage: Linear feet of street connected to property
        'LotFrontage': train.describe()['LotFrontage']['50%'],

        # MasVnrType
        'MasVnrType': 'None',
        'MasVnrArea': 'None',

        # Electrical: Electrical system
        'Electrical': 'Unknown'
    }
    cleansed_data = data.copy().fillna(value=fill)

    label_columns = filter(lambda c: str(
        cleansed_data[c].dtype) == 'object', cleansed_data)
    uniqne_labels = set()
    for l in label_columns:
        uniqne_labels = uniqne_labels.union(set(cleansed_data[l].unique()))
    unique_labels = reversed(list(uniqne_labels))
    replace = {l: i for l, i in zip(uniqne_labels, range(len(uniqne_labels)))}
    cleansed_data = cleansed_data.replace(replace)

    cleansed_data = cleansed_data.drop(['LotArea', 'MiscVal'], axis=1)

    return cleansed_data

# %% CSV出力する
# ---------------------------------------------------------------


def predict_and_output_csv(model, src_file_name, dst_file_name, scalers=[]):
    test = pd.read_csv(src_file_name)
    creansed_data = cleanse_data(test)
    # creansed_data.drop(['Id'], axis=1)
    data = creansed_data.drop(['Id'], axis=1)
    # data.columns
    print(data.shape)
    if len(scalers) > 0:
        for s in scalers:
            print('Appling Scaler:', str(s))
            data = s.transform(data)
    print(data.shape)
    predict = model.predict(data)

    result = pd.concat([test['Id'], pd.DataFrame(
        predict, columns=['SalePrice'])], axis=1)
    result.to_csv(dst_file_name, index=False)

    return result


# %% 欠損値を埋める
cleansed_train = cleanse_data(train)
# cleansed_train.describe()

# %% prepare train data
# -----------------------------------------------------------------------------------------
# cleansed_train[['OverallQual', 'OverallCond','GarageCars', 'Fireplaces']].head()
X_org = cleansed_train.drop(['Id', 'SalePrice'], axis=1)
y = cleansed_train['SalePrice']

# %% スケールを合わせる
scalers = []

scaler = StandardScaler().fit(X_org)
# scaler = MinMaxScaler().fit(X_org)
# scaler = RobustScaler().fit(X_org)
scalers.append(scaler)

X_scaled = scaler.transform(X_org)

pca = PCA(n_components=10).fit(X_scaled)
# scalers.append(pca)
# X_pca = pca.transform(X_scaled)


X = X_scaled
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

# %% Print Scores
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
result = predict_and_output_csv(best_model, 'test.csv', 'out.csv', scalers)


# %% Plot coef

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
