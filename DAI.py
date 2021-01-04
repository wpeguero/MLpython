"""PROJECT: DAI
Diagnostic Artificial Intelligence, or DAI, is a program that will provide diagnoses based on the information provided for it, be it an image or an transcription.
1. DescriptionDiagnosis - Provides a diagnosis based on the description provided by extracting keywords in the description and categorizing it under a medical specialty and sample name.
2. ImagingDiagnosis
"""
from spacy.util import compounding, minibatch
from spformat import *
from tqdm import tqdm
import pandas as pd
import spacy as sp
import warnings
import random

def main():
    #%% Load DataFrame
    df__transcriptions = pd.read_csv(r'C:\Users\wpegu\Documents\Github\MLpython\Sample_Data\mtsamples.csv')
    df__transcriptions = df__transcriptions.dropna(axis=0, how='any', subset=['transcription', 'keywords'])
    df__transcriptions.reset_index(drop=True)
    ldata = load_data(df__transcriptions, 'transcription','keywords') #Use EntityRuler to remove overlapping information.
    #print(ldata[0])
    nlp__DAI = train_data(ldata)
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

def train_data(ldata):
    """
    Data Trainer
    ------------
    Trains the loaded and parsed data into an nlp.
    
    Parameters:
    ldata (list): contains labeled data in the spacy format.
    """
    nlp = sp.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    for _,annotations in tqdm(ldata, desc='Annotating Data'):
        for ent in annotations.get('entities'):
            ner.add_label(ent [2])
    ##
    # Disable unneeded pipes.
    ##
    pipe_exceptions = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    ##
    # Train the ner pipe only
    ##
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        #Reset and initialize the weights randomly.
        nlp.begin_training()
        n_iter = 30
        for itn in tqdm(range(n_iter), desc='Iterating'):
            random.shuffle(ldata)
            losses = {}
            #batch up the examples using spacys minibatch
            batches = minibatch(ldata, size=compounding(4.0, 32.0, 1.001))
            for batch in tqdm(batches, desc='Loading Batches', leave=False):
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop= 0.15,
                    losses = losses
                )
            print("losses", (losses['ner']/len(train_data)) * 100)
    return nlp


if __name__ == "__main__":
    main()
    #nlp.to_disk(r'C:\Users\wpegu\Github\MLpython\DAIvB1')
