from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re


class NanDealing(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X_local = X.copy()
        print('local_x:', X_local)
        X_local['Embarked'].fillna(X_local['Embarked'].mode()[0], inplace=True)
        X_local['Fare'].fillna(X_local['Fare'].median(), inplace=True)
        X_local.drop(['Cabin'], axis=1, inplace=True, errors='ignore')
        X_local['Age'].fillna(X_local['Age'].median(), inplace=True)
        print('Number of NaN in each columns: ')
        print(X_local.isnull().sum())
        return X_local


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X_local = X.copy()
        X_local['FamilySize'] = X_local['SibSp'] + X_local['Parch'] + 1

        def get_title(name):
            title_search = re.search('([A-Za-z]+)\.', name)
            # If the title exists, extract and return it.
            if title_search:
                return title_search.group(1)
            return ""

        X_local['Title'] = X_local['Name'].apply(get_title)

        X_local['Title'] = X_local['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        X_local['Title'] = X_local['Title'].replace('Mlle', 'Miss')
        X_local['Title'] = X_local['Title'].replace('Ms', 'Miss')
        X_local['Title'] = X_local['Title'].replace('Mme', 'Mrs')

        X_local['Age_bin'] = pd.cut(X_local['Age'], bins=[0, 12, 20, 40, 120],
                                    labels=['Children', 'Teenage', 'Adult', 'Elder'])

        X_local['Fare_bin'] = pd.cut(X_local['Fare'], bins=[0, 7.91, 14.45, 31, 120],
                                     labels=['Low_fare', 'median_fare',
                                             'Average_fare', 'high_fare'])

        drop_column = ['Age', 'Fare', 'Name', 'Ticket']
        X_local.drop(drop_column, axis=1, inplace=True, errors='ignore')

        drop_column = ['PassengerId']
        X_local.drop(drop_column, axis=1, inplace=True, errors='ignore')

        return X_local


class GetDummies(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X_local = X.copy()
        dummies_list = ["Sex", "Title", "Age_bin", "Embarked", "Fare_bin"]
        dummies_prefix = ["Sex", "Title", "Age_type", "Em_type", "Fare_type"]
        X_local = pd.get_dummies(X_local, columns=dummies_list, prefix=dummies_prefix)
        all_cols = ['Pclass', 'SibSp', 'Parch', 'FamilySize', 'Sex_female', 'Sex_male', 'Title_',
                    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                    'Age_type_Children', 'Age_type_Teenage', 'Age_type_Adult',
                    'Age_type_Elder', 'Em_type_C', 'Em_type_Q', 'Em_type_S',
                    'Fare_type_Low_fare', 'Fare_type_median_fare', 'Fare_type_Average_fare',
                    'Fare_type_high_fare']
        for feature in all_cols:
            if feature not in X_local.columns:
                X_local[feature] = [0] * len(X_local.index)
        print(X_local)
        return X_local


class GetValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.values