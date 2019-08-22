import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# find the nan attributes
def find_nan_attr(data):
    cols = []
    nums = []
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            num = data[col].isnull().sum()
            cols.append(col)
            nums.append(num)
        else:
            cols.append(col)
            nums.append(0)
    delete_attr = pd.DataFrame({'待删除属性': cols, '空值个数': nums})
    # delete_attr.to_csv('/Users/martin_yan/Desktop/1111.csv', index=False, encoding="utf_8_sig")
    print(delete_attr)


def numerical_to_bin(data, attr, val_map):
    result = data.copy(deep=True)
    for the_map in val_map:
        lower = the_map['lower']
        upper = the_map['upper']
        val = the_map['val']
        result.loc[np.logical_and(data[attr] >= lower, data[attr] < upper), attr] = val
    return result


# one-hot coding
def onehot(data, attr):
    data = pd.concat([data, pd.get_dummies(data[attr], prefix=attr)], axis=1)
    data.drop(attr, axis=1, inplace=True)
    return data


# calculate bmi
def bmi(height, weight):
    bmi = weight / pow(height / 100, 2)
    return bmi


# change cm to foot
def waistline_transform(data):
    result = data.copy(deep=True)
    for i in range(len(result)):
        if result.loc[i]['F_1042'] == 1:
            result.loc[i, 'F_1044'] = result.loc[i]['F_327'] / 33.33
    result.drop('F_1042', axis=1, inplace=True)
    result.drop('F_327', axis=1, inplace=True)
    return result


# fill nan with given value
def fill_na(data, attributes, value):
    for attribute in attributes:
        data[attribute].fillna(value, inplace=True)


def change_values(data, attributes):
    for attribute in attributes:
        data.loc[data[attribute] == 1, attribute] = 0
        data.loc[data[attribute] == 2, attribute] = 1


def one_year(a, b, c):
    return a.astype(int) | b.astype(int) | c.astype(int)


# get rid of columns which have all same values
def drop_same(data):
    sameValues = data.loc[:, data.apply(pd.Series.nunique, axis=0) == 1]
    dropsame = list(sameValues)
    data.drop(dropsame, axis=1, inplace=True)
    return data


def delete_all_nan_attr(data, sample_num=2000):
    cols = []
    nums = []
    for col in data.columns:
        if data[col].isnull().sum() == sample_num:
            nums.append(data[col].isnull().sum())
            cols.append(col)
    data.drop(cols, axis=1, inplace=True)
    return data


def fillna_values(data, attr, value=0):
    if value == 'int':
        data[attr].fillna(data[attr].mean(), inplace=True)
    else:
        data[attr].fillna(value, inplace=True)
        data[attr] = data[attr].astype(int)
    return data


# single cdss_model
def single_model(x_train, y_train, x_test, y_test, algorithms):
    for alg in algorithms:
        alg.fit(x_train, y_train)
        prediction = alg.predict(x_test.astype(float))
        print(prediction)
        print('Accuracy of classifier on test set: {:.3f}'.format(alg.score(x_test, y_test)))
    return prediction


# combine several models
def voting_blending_models(x_train, y_train, x_test, algorithms):
    full_predictions = np.zeros(shape=(len(x_test), 5))
    for alg in algorithms:
        alg.fit(x_train, np.ravel(y_train))
        predictions = alg.predict(x_test)
        print(predictions)
        for i in range(len(x_test)):
            full_predictions[i, int(predictions[i] - 1)] += 1
    y_predictions = np.argmax(full_predictions, axis=1) + 1
    return y_predictions


# evaluate models
def evaluate_models(y_test, y_predictions):
    print(f1_score(y_true=y_test, y_pred=y_predictions, labels=[1, 2, 3, 4, 5], average='micro'))
    precision, recall, fscore, support = score(y_test, y_predictions)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


def chi2(train, y):
    variable = []
    score = []
    p = []
    # X = train.select_dtypes(exclude=[np.number])
    X = train
    for i in X.columns.values.tolist():
        table = pd.crosstab(X[i], y['Attrition'])
        variable.append(i)
        score.append(chi2_contingency(table)[0])
        p.append(chi2_contingency(table)[1])
    result = pd.DataFrame({'Variable': variable, 'Score': score, 'P': p})
    print(result)


def label_encoder(data, attrs):
    for attr in attrs:
        data[attr] = LabelEncoder().fit_transform(data[attr])
    return data


# discretize age
age_map = [
    {'lower': 0, 'upper': 50, 'val': 0},
    {'lower': 50, 'upper': 60, 'val': 1},
    {'lower': 60, 'upper': 70, 'val': 2},
    {'lower': 70, 'upper': 80, 'val': 3},
    {'lower': 80, 'upper': 99, 'val': 4},
]

# discretize bmi
bmi_map = [
    {'lower': 0, 'upper': 20, 'val': 1},
    {'lower': 20, 'upper': 24, 'val': 2},
    {'lower': 24, 'upper': 28, 'val': 3},
    {'lower': 28, 'upper': 32, 'val': 4},
    {'lower': 32, 'upper': 99, 'val': 5},
]

# discretize waistline
waistline_map = [
    {'lower': 0, 'upper': 2, 'val': 0},
    {'lower': 2, 'upper': 2.4, 'val': 1},
    {'lower': 2.4, 'upper': 2.8, 'val': 2},
    {'lower': 2.8, 'upper': 3.2, 'val': 3},
    {'lower': 3.2, 'upper': 99, 'val': 4},
]

# discretize sbp 收缩压 高
sbp_map = [
    {'lower': 0, 'upper': 120, 'val': 0},
    {'lower': 120, 'upper': 140, 'val': 1},
    {'lower': 140, 'upper': 160, 'val': 2},
    {'lower': 160, 'upper': 180, 'val': 3},
    {'lower': 180, 'upper': 999, 'val': 4},
]

# discretize dbp 舒张压 低
dbp_map = [
    {'lower': 0, 'upper': 70, 'val': 0},
    {'lower': 70, 'upper': 80, 'val': 1},
    {'lower': 80, 'upper': 90, 'val': 2},
    {'lower': 90, 'upper': 100, 'val': 3},
    {'lower': 100, 'upper': 999, 'val': 4},
]

# discretize heart
heart_map = [
    {'lower': 0, 'upper': 60, 'val': 0},
    {'lower': 60, 'upper': 70, 'val': 1},
    {'lower': 70, 'upper': 80, 'val': 2},
    {'lower': 80, 'upper': 90, 'val': 3},
    {'lower': 90, 'upper': 200, 'val': 4},
]

# discretize NIHSS
nihss_map = [
    {'lower': 0, 'upper': 1, 'val': 0},
    {'lower': 1, 'upper': 5, 'val': 1},
    {'lower': 5, 'upper': 16, 'val': 2},
    {'lower': 16, 'upper': 21, 'val': 3},
    {'lower': 21, 'upper': 99, 'val': 4},
]

# discretize sugar
sugar_map = [
    {'lower': 0, 'upper': 0.1, 'val': 0},
    {'lower': 0.1, 'upper': 3.9, 'val': 1},
    {'lower': 3.9, 'upper': 6.1, 'val': 2},
    {'lower': 6.1, 'upper': 10, 'val': 3},
    {'lower': 10, 'upper': 99, 'val': 4},
]
