"""PROJECT: DAI
Diagnostic Artificial Intelligence, or DAI, is a program that will provide diagnoses based on the information provided for it, be it an image or an transcription.
1. DescriptionDiagnosis - Provides a diagnosis based on the description provided by extracting keywords in the description and categorizing it under a medical specialty and sample name.

2. ImagingDiagnosis
"""
import pandas as pd
import spacy as sp

def main():
    #%% Load DataFrame
    df__transcriptions = pd.read_csv(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\mtsamples.csv')
    df__transcriptions.dropna(subset=['keywords'], inplace=True)


def load_diagnostic_data(df,dcolumn,lcolumn):
    """
    Labels all of the transcription field based on the keywords and the medical specialty.
    """
    traindata = []
    for row in df.itertuples():
        entities__list = []
        for kword in row.lcolumn:
            transcription = row.dcolumn
            start = transcription.find(kword)
            end = start + len(kword)
            entity = (start, end, kword)
            entities__list.append(entity)
        entity__dict = {'entities': entities__list}
        trainsample = (row.dcolumn, entity__dict)
        traindata.append(trainsample)
    return traindata



def train_data(ldata):
    """
    Trains the loaded and parsed data into an nlp.
    """
    pass


if __name__ == "__main__":
    main()
