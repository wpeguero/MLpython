"""
Spacy DataFrame
---------------
---------------
An efficient container that allows one to utilize spacy data in a trainable format. Contains useful set of functions that can be used to process the spacy data.
"""
from pandas import DataFrame

class Spata(object):
    def __init__(self, df):
        self.df = df
        columns = df.columns
        columns = columns.tolist()
        self.columns = columns
        return self

    def israw(self,columns):
        """
        Extracts the columns within the file and sees whether the data is already in a labeled format or not.
        Labeled Format:
        ---------------
        
        Index | Raw Data |                    Entities                           | entity 1       | entity 2 | etc...
        ---------------------------------------------------------------------------------------------------------------------
           1  | xxxxxxxx | [1, {'Entities':[1,[(n1,n2, label),(n2,n3, label)]]}] | (n2,n3, label) | etc...
        """
        for column in columns:
            if 'entity' in column:
                break
        self.isRaw = True
        return self
    
    def _rename_columns(self):
        df = self.df