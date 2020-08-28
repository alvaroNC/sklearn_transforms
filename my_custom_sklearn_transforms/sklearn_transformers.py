import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
class ImputerValuesDesafio4(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.si0= 0
        self.si1= 0
        self.sif= 0
        self.y_label_0=[]
        
    def fit(self, X, y=None):
        self.si0.fit(X=X[self.y_label_0=='Aceptado'])
        self.si1.fit(X=X[self.y_label_0=='Sospechoso'])
        self.sif.fit(X=X[self.y_label_0=='Aceptado']) #doesnt matter whicch you choose.It's only to get the number of features.
        self.sif.statistics_=(self.si0.statistics_+self.si1.statistics_)/2
        return self

    
    def transform(self, X):
        data=self.sif.transform(X.copy())
        return data
