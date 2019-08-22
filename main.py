import pandas as pd
from helper import find_nan_attr, drop_same, chi2, label_encoder
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

category = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
if __name__ == "__main__":
    # read data
    train = pd.read_csv('data/pfm_train.csv')
    test = pd.read_csv('data/pfm_test.csv')

    y = train.loc[:, train.columns == 'Attrition']
    train.drop('Attrition', axis=1, inplace=True)

    combined = pd.concat([train, test])
    combined = combined.reset_index(drop=True)

    # drop same ['Over18', 'StandardHours']
    data = drop_same(combined)
    combined.drop('EmployeeNumber', axis=1, inplace=True)

    combined = label_encoder(combined, category)

    for i in list(combined):
        if i in category:
            combined[i] = combined[i].astype('category')

    train = combined[:1100]
    test = combined[1100:]

    cw = 'balanced'
    algorithms = [
        LogisticRegressionCV(random_state=1),
        RandomForestClassifier(n_estimators=200, max_depth=5),
        GradientBoostingClassifier(random_state=1, n_estimators=200, max_depth=5)
    ]

    full_predictions = []
    for alg in algorithms:
        # Fit the algorithm using the full training data.
        alg.fit(train, y['Attrition'])
        # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
        predictions = alg.predict_proba(test)[:, 1]
        full_predictions.append(predictions)
    predictions = (full_predictions[0] + full_predictions[1] + full_predictions[2]) / 3
    print(predictions)
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    predictions = predictions.astype(int)
    submission = pd.DataFrame({
            "result": predictions,
        })
    print(submission)
    submission.to_csv('/Users/martin_yan/Desktop/submission1.csv', index=False)