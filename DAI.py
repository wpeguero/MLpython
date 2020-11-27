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
    df__transcriptions = df__transcriptions.dropna(axis=0, how='any', subset=['transcription', 'keywords'])
    df__transcriptions.reset_index(drop=True)
    train_data = load_diagnostic_data(df__transcriptions, 'transcription','keywords')
    with open(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\training_datav3.txt', 'w') as file:
        text_train = str(train_data)
        file.write(text_train)
        file.close()


def load_diagnostic_data(df,dcolumn,lcolumn):
    """
    Labels all of the transcription field based on the keywords and the medical specialty.
    """
    traindata = []
    df[str(dcolumn)].astype(str)
    df[str(lcolumn)].astype(str)
    for index, row in df.iterrows():
        entities__list = []
        kword__str = row[str(lcolumn)]
        kword__list = kword__str.split(',')
        for kword in kword__list:
            transcription = row[str(dcolumn)]
            transcription = transcription.lower()
            if kword in transcription:
                start = transcription.find(kword)
                end = start + len(kword)
                entity = (start, end, kword)
                entities__list.append(entity)
        entity__dict = {'entities': entities__list}
        trainsample = (row[str(dcolumn)], entity__dict)
        traindata.append(trainsample)
    return traindata



def train_data(ldata):
    """
    Trains the loaded and parsed data into an nlp.
    """
    pass


if __name__ == "__main__":
    main()
