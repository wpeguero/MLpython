"""PROJECT: DAI
Diagnostic Artificial Intelligence, or DAI, is a program that will provide diagnoses based on the information provided for it, be it an image or an transcription.
1. DescriptionDiagnosis - Provides a diagnosis based on the description provided by extracting keywords in the description and categorizing it under a medical specialty and sample name.
2. ImagingDiagnosis
"""
from spformat import *
import pandas as pd

def main():
    #%% Load DataFrame
    df__transcriptions = pd.read_csv(r'C:\Users\wpegu\Documents\Github\MLpython\Sample_Data\mtsamples_train__test.csv')
    df__transcriptions = df__transcriptions.dropna(axis=0, how='any', subset=['transcription', 'keywords'])
    df__transcriptions.reset_index(drop=True)
    ldata = load_data(df__transcriptions, 'transcription','keywords') #May need to create a dataframe that contains 1 label per column.
    #print(ldata[0])
    #nlp = train_data(ldata)
    #Extract data into a sample file for reviewing
    #with open(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\training_datav4.txt', 'w') as file:
    #    text_train = str(train_data)
    #    file.write(text_train)
    #    file.close()
    #return nlp


def __load_specialty_data(df,dcolumn,lcolumn):
    """
    Create Simple Training Data from List
    ------------------------------------

    Creates training data based on the keywords provided and the specialty associated with it.
    """
    #Set initial conditions
    traindata = []
    df[str(dcolumn)].astype(str)
    df[str(lcolumn)].astype(str)
    for row in df.itertuples():
        entities__list = []
        specialty = row.lcolumn
        sample__str = str(row.dcolumn)
        if ',' in sample__str:
            sample__list = sample__str.split(',')
            for sample in sample__list:
                sample = sample.strip()
                start = 0
                end = len(sample)
                entities = (start, end, specialty)
                entities__list.append(entities)
            entities__dict = {'entities': entities__list}
        else:
            sample = sample__str
            sample = sample.strip()
            start = 0
            end = len(sample)
            entities = (start, end, specialty)
            entities__dict = {'entities': [entities]}
        trainsample = (row[str(dcolumn)], entities__dict)
        traindata.append(trainsample)
    return traindata


if __name__ == "__main__":
    main()
    #nlp.to_disk(r'C:\Users\wpegu\Documents\Github\MLpython\mtner2')