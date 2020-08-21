from sklearn.base import BaseEstimator, TransformerMixin


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
class ImputerMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self=self

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Filling missing HOURS_BACKEND values with mean
        data['HOURS_BACKEND'] = data['HOURS_BACKEND'].fillna(data['HOURS_BACKEND'].median())
        # Filling missing HOURS_DATASCIENCE values with most common value
        data['HOURS_DATASCIENCE'] = data['HOURS_DATASCIENCE'].fillna(data['HOURS_DATASCIENCE'].median())
        # Filling missing HOURS_FRONTEND values with most common value
        data['HOURS_FRONTEND'] = data['HOURS_FRONTEND'].fillna(data['HOURS_FRONTEND'].median())
        # Filling missing NUM_COURSES_BEGINNER_DATASCIENCE values with most common value
        data['NUM_COURSES_BEGINNER_DATASCIENCE'] = data['NUM_COURSES_BEGINNER_DATASCIENCE'].fillna(data['NUM_COURSES_BEGINNER_DATASCIENCE'].mode()[0])
        # Filling missing NUM_COURSES_BEGINNER_BACKEND values with most common value
        data['NUM_COURSES_BEGINNER_BACKEND'] = data['NUM_COURSES_BEGINNER_BACKEND'].fillna(data['NUM_COURSES_BEGINNER_BACKEND'].mode()[0])
        # Filling missing NUM_COURSES_BEGINNER_FRONTEND values with most common value
        data['NUM_COURSES_BEGINNER_FRONTEND'] = data['NUM_COURSES_BEGINNER_FRONTEND'].fillna(data['NUM_COURSES_BEGINNER_FRONTEND'].mode()[0])
        # Filling missing NUM_COURSES_ADVANCED_DATASCIENCE values with mean
        data['NUM_COURSES_ADVANCED_DATASCIENCE'] = data['NUM_COURSES_ADVANCED_DATASCIENCE'].fillna(data['NUM_COURSES_ADVANCED_DATASCIENCE'].mode()[0])
        # Filling missing NUM_COURSES_ADVANCED_BACKEND values with mean
        data['NUM_COURSES_ADVANCED_BACKEND'] = data['NUM_COURSES_ADVANCED_BACKEND'].fillna(data['NUM_COURSES_ADVANCED_BACKEND'].mode()[0])
        # Filling missing NUM_COURSES_ADVANCED_FRONTEND values with mean
        data['NUM_COURSES_ADVANCED_FRONTEND'] = data['NUM_COURSES_ADVANCED_FRONTEND'].fillna(data['NUM_COURSES_ADVANCED_FRONTEND'].mode()[0])
        #AVG_SCORE_DATASCIENCE with mean
        data['AVG_SCORE_DATASCIENCE'] = data['AVG_SCORE_DATASCIENCE'].fillna(data['AVG_SCORE_DATASCIENCE'].median())
        #AVG_SCORE_BACKEND with mean
        data['AVG_SCORE_BACKEND'] = data['AVG_SCORE_BACKEND'].fillna(data['AVG_SCORE_BACKEND'].median())
        #AVG_SCORE_FRONTEND with mean
        data['AVG_SCORE_FRONTEND'] = data['AVG_SCORE_FRONTEND'].fillna(data['AVG_SCORE_FRONTEND'].median())
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data
