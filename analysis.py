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

# %% 欠損値の穴埋め
def fill_missing_data(data):
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
    return data.copy().fillna(value=fill)


# %% ラベルを数値に置き換え
def cleanse_data(data):
    cleansed_data = data.copy()

    label_columns = filter(lambda c: str(
        cleansed_data[c].dtype) == 'object', cleansed_data)
    uniqne_labels = set()
    for l in label_columns:
        uniqne_labels = uniqne_labels.union(set(cleansed_data[l].unique()))
    unique_labels = reversed(list(uniqne_labels))
    replace = {l: i for l, i in zip(uniqne_labels, range(len(uniqne_labels)))}
    cleansed_data = cleansed_data.replace(replace)

    #cleansed_data = cleansed_data.drop(['LotArea', 'MiscVal'], axis=1)

    return cleansed_data


# %% 欠損値の確認
print(train.isnull().sum()[train.isnull().sum() != 0].sort_values())
dumm = pd.get_dummies(train)
print(dumm.isnull().sum()[dumm.isnull().sum() != 0].sort_values())

# %% 欠損値の穴埋めとラベルの置き換え
filled_data = fill_missing_data(train)
cleansed_data = cleanse_data(filled_data)

print(cleansed_data.isnull().sum()[
      cleansed_data.isnull().sum() != 0].sort_values())
dumm = pd.get_dummies(cleansed_data)
print(dumm.isnull().sum()[dumm.isnull().sum() != 0].sort_values())

# %% 主成分分析して効いてそうなパラメータを探す
X = cleansed_data.drop(['Id', 'SalePrice'], axis=1)
y = cleansed_data['SalePrice']

# スケールを合わせる
# scaler = StandardScaler().fit(X)
scaler = MinMaxScaler().fit(X)
# scaler = RobustScaler().fit(X)

X_scaled = scaler.transform(X)

# 主成分分析してみる
pca = PCA(n_components=5).fit(X_scaled)
X_pca = pca.transform(X_scaled)

# ひーとまっぷで表示してみる
feature_columns = train.columns.drop(['Id', 'SalePrice'])

plt.matshow(pca.components_)
plt.colorbar()
plt.xticks(range(len(feature_columns)), feature_columns, rotation=90)

plt.savefig("heatmap.png")




# %%
import seaborn as sns

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(cleansed_data.astype(float).corr(), linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.savefig("heatmap_sns.png")

# %%
