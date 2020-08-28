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
class ImputerMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_hb=0
        self.median_hd=0
        self.median_hf=0
        self.moda_cbd=0
        self.moda_cbb=0
        self.moda_cbf=0
        self.moda_cad=0
        self.moda_cab=0
        self.moda_caf=0
        self.median_ad=0
        self.median_ab=0
        self.median_af=0

    def fit(self, X, y=None):
        self.median_hb=X['HOURS_BACKEND'].median()
        self.median_hd=X['HOURS_DATASCIENCE'].median()
        self.median_hf=X['HOURS_FRONTEND'].median()
        self.moda_cbd=X['NUM_COURSES_BEGINNER_DATASCIENCE'].mode()[0]
        self.moda_cbb=X['NUM_COURSES_BEGINNER_BACKEND'].mode()[0]
        self.moda_cbf=X['NUM_COURSES_BEGINNER_FRONTEND'].mode()[0]
        self.moda_cad=X['NUM_COURSES_ADVANCED_DATASCIENCE'].mode()[0]
        self.moda_cab=X['NUM_COURSES_ADVANCED_BACKEND'].mode()[0]
        self.moda_caf=X['NUM_COURSES_ADVANCED_FRONTEND'].mode()[0]
        self.median_ad=X['AVG_SCORE_DATASCIENCE'].median()
        self.median_ab=X['AVG_SCORE_BACKEND'].median()
        self.median_af=X['AVG_SCORE_FRONTEND'].median()
        return self

    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Filling missing HOURS_BACKEND values with mean
        data['HOURS_BACKEND'] = data['HOURS_BACKEND'].fillna(self.median_hb)
        # Filling missing HOURS_DATASCIENCE values with most common value
        data['HOURS_DATASCIENCE'] = data['HOURS_DATASCIENCE'].fillna(self.median_hd)
        # Filling missing HOURS_FRONTEND values with most common value
        data['HOURS_FRONTEND'] = data['HOURS_FRONTEND'].fillna(self.median_hf)
        # Filling missing NUM_COURSES_BEGINNER_DATASCIENCE values with most common value
        data['NUM_COURSES_BEGINNER_DATASCIENCE'] = data['NUM_COURSES_BEGINNER_DATASCIENCE'].fillna(self.moda_cbd)
        # Filling missing NUM_COURSES_BEGINNER_BACKEND values with most common value
        data['NUM_COURSES_BEGINNER_BACKEND'] = data['NUM_COURSES_BEGINNER_BACKEND'].fillna(self.moda_cbb)
        # Filling missing NUM_COURSES_BEGINNER_FRONTEND values with most common value
        data['NUM_COURSES_BEGINNER_FRONTEND'] = data['NUM_COURSES_BEGINNER_FRONTEND'].fillna(self.moda_cbf)
        # Filling missing NUM_COURSES_ADVANCED_DATASCIENCE values with mean
        data['NUM_COURSES_ADVANCED_DATASCIENCE'] = data['NUM_COURSES_ADVANCED_DATASCIENCE'].fillna(self.moda_cad)
        # Filling missing NUM_COURSES_ADVANCED_BACKEND values with mean
        data['NUM_COURSES_ADVANCED_BACKEND'] = data['NUM_COURSES_ADVANCED_BACKEND'].fillna(self.moda_cab)
        # Filling missing NUM_COURSES_ADVANCED_FRONTEND values with mean
        data['NUM_COURSES_ADVANCED_FRONTEND'] = data['NUM_COURSES_ADVANCED_FRONTEND'].fillna(self.moda_caf)
        #AVG_SCORE_DATASCIENCE with mean
        data['AVG_SCORE_DATASCIENCE'] = data['AVG_SCORE_DATASCIENCE'].fillna(self.median_ad)
        #AVG_SCORE_BACKEND with mean
        data['AVG_SCORE_BACKEND'] = data['AVG_SCORE_BACKEND'].fillna(self.median_ab)
        #AVG_SCORE_FRONTEND with mean
        data['AVG_SCORE_FRONTEND'] = data['AVG_SCORE_FRONTEND'].fillna(self.median_af)
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data

class ImputerValuesDesafio4(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.si0= SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='mean',  # la estrategia elegida es cambiar el valor faltante por una constante
            fill_value=None,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
        self.si1= SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='mean',  # la estrategia elegida es cambiar el valor faltante por una constante
            fill_value=None,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
        self.sif= SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='mean',  # la estrategia elegida es cambiar el valor faltante por una constante
            fill_value=None,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
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
