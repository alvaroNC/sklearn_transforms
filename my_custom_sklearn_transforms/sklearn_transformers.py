import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns2(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
