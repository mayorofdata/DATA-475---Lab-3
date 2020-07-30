# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
sns.boxplot(x="CentralAir", y="SalePrice", data=train)

# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
sns.boxplot(x="BldgType", y="SalePrice", data=train)

# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
sns.boxplot(x="Neighborhood", y="SalePrice", data=train)


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
sns.boxplot(x="OverallQual", y="SalePrice", data=train)

# %% SalePrice distribution w.r.t YearBuilt / Neighborhood
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
ax = sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45
)

# %%
plt.figure(figsize=(16,8))
ax = sns.boxplot(x="YearBuilt", y="SalePrice", data=train)
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45,fontsize=5
)

# %%
plt.figure(figsize=(16,8))
built_after_2000 = train["YearBuilt"] > 2000
ax = sns.boxplot(x="YearBuilt", y="SalePrice", data=train[built_after_2000])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45,fontsize=5
)

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# %% your solution to the regression problem

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

numeric_cols = [
    "1stFlrSF",
    "2ndFlrSF",
    "BsmtFullBath",
    "FullBath"
]

cat_cols = ["Neighborhood"]
selected_columns = numeric_cols + cat_cols

train_x = train[selected_columns]
train_y = train["SalePrice"]

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

imp = SimpleImputer()
enc = OneHotEncoder(handle_unknown='ignore')

ct = ColumnTransformer(
    [
        ("Neighborhood_ohe", enc, ["Neighborhood"]),
        ("fill_na", imp, numeric_cols),
    ],
remainder="passthrough"
)


train_x_np = ct.fit_transform(train_x)

reg.fit(train_x_np, train_y)

# %%
print("Training Set Performance")
print(reg, train_x, train_y)

# %%
truth = pd.read_csv("truth_house_prices.csv")
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x = ct.transform(test_x)

print("Test Set Performance")
print(evaluate(reg, test_x, test_y))


# %%
