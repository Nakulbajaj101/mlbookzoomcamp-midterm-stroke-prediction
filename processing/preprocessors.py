from typing import List

from sklearn.base import BaseEstimator, TransformerMixin


def clean_strings(val):
    """Function to make data values consistent"""

    return val.replace(" ","_").replace("-", "_")



class BinaryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, column_name: str):
        """Instatiating the class"""

        self.column_name = column_name


    def fit(self, X):
        """Custom fit function"""

        return self

    def transform(self, X):
        """Custom transformer that converts yes and no to 1 and 0"""

        X[self.column_name] = X[self.column_name].apply(lambda val: 1 if val.lower() == 'yes' else 0)
        return X


class CleanStrings(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: List[str]):

        self.column_list = column_list


    def fit(self, X):
        """Custom fit function"""

        return self

    def transform(self, X):
        """Custom transformer to clean all strings and make them lower case"""

        for col in self.column_list:
            if X[col].dtype == 'object':
                X[col] = X[col].apply(lambda val: val.lower())
                X[col] = X[col].apply(lambda val: clean_strings(val))

        return X


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,column_list: List[str]):

        self.column_list = column_list

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self,X,y=None):
        """Custom transformer to drop specific columns"""

        return X.drop(self.column_list,axis=1)

