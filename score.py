
import pandas as pd
pd.options.display.max_columns = None
train = pd.read_csv("train.csv")
# %matplotlib inline


# train.head()
# train.describe()

# 欠損値を埋める
def cleanse_data(data):
    fill = {
        # Fence: Fence quality. NA	No Fence
        'Fence'       : 'NoFence',

        # Alley: Type of alley access to property NA No alley access
        'Alley'       : 'NoAlleyAccess',

        # FireplaceQu: Fireplace quality NA	No Garage
        'FireplaceQu' : 'NoGarage',

        'GarageType'  : 'NoGarage',
        'GarageYrBlt' : 'NoGarage',
        'GarageFinish': 'NoGarage',
        'GarageQual'  : 'NoGarage',
        'GarageCond'  : 'NoGarage',

        'PoolQC' : 'NoPool',

        # Miscellaneous feature not covered in other categories
        'MiscFeature' : 'None',

        # BsmtQual: Evaluates the height of the basement
        'BsmtQual' : 'NoBasement',
        # BsmtCond: Evaluates the general condition of the basement
        'BsmtCond'     : 'NoBasement',
        'BsmtExposure' : 'NoBasement',
        'BsmtFinType1' : 'NoBasement',
        'BsmtFinType2' : 'NoBasement',

        # LotFrontage: Linear feet of street connected to property
        'LotFrontage' : train.describe()['LotFrontage']['50%'],

        # MasVnrType
        'MasVnrType' : 'None',
        'MasVnrArea' : 'None',

        # Electrical: Electrical system
        'Electrical' : 'Unknown'
    }
    cleansed_data = data.copy().fillna(value = fill)

    label_columns = filter(lambda c: str(cleansed_data[c].dtype) == 'object', cleansed_data)
    uniqne_labels = set()
    for l in label_columns:
        uniqne_labels = uniqne_labels.union(set(cleansed_data[l].unique()))
    unique_labels = reversed(list(uniqne_labels))
    replace = {l : i for l, i in zip(uniqne_labels, range(len(uniqne_labels)))}
    cleansed_data = cleansed_data.replace(replace)

    # cleanse_data = cleanse_data.drop(['LotArea', 'MiscVal'], axis=1)

    return cleansed_data

cleansed_train = cleanse_data(train)

# -----------------------------------------------------------------------------------------

# cleansed_train[['OverallQual', 'OverallCond','GarageCars', 'Fireplaces']].head()
X = cleansed_train.drop(['Id', 'SalePrice'],axis=1)
y = cleansed_train['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# X_train.head()
# y_train.head()
# X_test.head()
# y_test.head()

# -------------------------------
# from sklearn.linear_model import Ridge
# model = Ridge(alpha=.4).fit(X_train, y_train)
# model.score(X_test, y_test)
#
# # -------------------------------
# from sklearn.neighbors.regression import KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
# model.score(X_test, y_test)

# -------------------------------
from sklearn.linear_model import Lasso
model = Lasso().fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)

# -------------------------------
# Plot coef
coef = pd.DataFrame(X.columns, columns=['Name'])
coef['coef'] = pd.Series(model.coef_)
coef.sort_values(by='coef')

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("notebook")

model.score(X_test, y_test)
plt.figure(figsize=(20, 30))
sns.barplot(y='Name', x='coef', data=coef )

# ---------------------------------------------------------------

def predict_and_output_csv(model, src_file_name, dst_file_name):
    src_file_name ='test.csv'
    test = pd.read_csv(src_file_name)
    creansed_data = cleanse_data(test)
    # creansed_data.drop(['Id'], axis=1)
    data = creansed_data.drop(['Id'], axis=1)
    data.columns

    predict = model.predict(data)

    result = pd.concat([test['Id'], pd.DataFrame(predict, columns=['SalePrice'])], axis=1)
    result.to_csv(dst_file_name, index=False)
    # result.to_csv(file_name, index=False)

    return result

result = predict_and_output_csv(model, 'test.csv', 'out.csv')
