"""PROJECT: DAI
Diagnostic Artificial Intelligence, or DAI, is a program that will provide diagnoses based on the information provided for it, be it an image or an transcription.
1. DescriptionDiagnosis - Provides a diagnosis based on the description provided by extracting keywords in the description and categorizing it under a medical specialty and sample name.

2. ImagingDiagnosis
"""
from spacy.util import compounding, minibatch
from alive_progress import alive_bar
from rapidfuzz import fuzz
import pandas as pd
import spacy as sp
import itertools
import warnings
import random

def main():
    #%% Load DataFrame
    df__transcriptions = pd.read_csv(r'D:\Users\wpeguerorosario\Documents\Github\MLpython\Sample_Data\mtsamples.csv')
    df__transcriptions = df__transcriptions.dropna(axis=0, how='any', subset=['transcription', 'keywords'])
    df__transcriptions.reset_index(drop=True)
    ldata = load_diagnostic_data(df__transcriptions, 'transcription','keywords') #Use EntityRuler to remove overlapping information.
    nlp = train_data(ldata)
    #Extract data into a sample file for reviewing
    #with open(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\training_datav4.txt', 'w') as file:
    #    text_train = str(train_data)
    #    file.write(text_train)
    #    file.close()
    return nlp


def load_diagnostic_data(df,dcolumn,lcolumn):
    """
    Create Training Data from DataFrame
    -----------------------------------
    Function that extracts labels from a body of text based on the keywords provided on a separate field.
    
    Sample training data format:
    
    train_data = [
    ("Uber blew through $1 million a week", [(0, 4, 'ORG')]),
    ("Android Pay expands to Canada", [(0, 11, 'PRODUCT'), (23, 30, 'GPE')]),
    ("Spotify steps up Asia expansion", [(0, 8, "ORG"), (17, 21, "LOC")]),
    ("Google Maps launches location sharing", [(0, 11, "PRODUCT")]),
    ("Google rebrands its business apps", [(0, 6, "ORG")]),
    ("look what i found on google! 😂", [(21, 27, "PRODUCT")])]
    """
    #Deal with bad inputs
    if not isinstance(dcolumn, str):
        raise TypeError(f'dcolumn must be a string, but is a {type(dcolumn)}')
    if not isinstance(lcolumn, str):
        raise TypeError(f'dcolumn must be a string, but is a {type(lcolumn)}')
    #Initial conditions/formatting of the data
    traindata = []
    df[str(dcolumn)].astype(str)
    df[str(lcolumn)].astype(str)
    for index, row in df.iterrows():
        entities__list = []
        label__str = row[str(lcolumn)]
        transcription = row[str(dcolumn)]
        transcription = transcription.lower()
        transcription = transcription.strip()
        #Conditional statements for the fields provided
        if ',' in label__str: #There are multiple labels in a field.
            label__list = label__str.split(',') #Work on this portion to apply the load_diagnostic_data function in a general sense (i.e single label vs multiple labels)
            label__list = remove_duplicate_labels(label__list)
            for label in label__list:
                label = label.lower()
                label = label.strip()
                if label in transcription:
                    start = transcription.find(label)
                    end = start + len(label)
                    entities = (start, end, label)
                    entities__list.append(entities)
            entities__dict = {'entities': entities__list}
        else: #There is only one label in the field.
            label = label__str
            label = label.lower()
            label = label.strip()
            if label in transcription:
                start = transcription.find(label)
                end = start + len(label)
                entities = (start, end, label)
            else:
                start = 0
                end = len(transcription)
                entities = (start, end, label)
            entities__dict = {'entities': [entities]}
        trainsample = (row[str(dcolumn)], entities__dict)
        traindata.append(trainsample)
    return traindata

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

def remove_duplicate_labels(label__list):
    """Label Cleaner
    ----------------
    Removes duplicates in the label lists by comparing whether one item is in another.
    
    Parameters:
    label__list (list): a list filled with labels that will be used to label information.
    """
    #Removes empty labels
    if '' in label__list:
        i = label__list.index('')
        del label__list[i]
    else:
        pass
    #Removes labels with '-'.
    for label in label__list:
        if '-' in label:
            i = label__list.index(label)
            del label__list[i]
        else:
            pass
    #Removes labels that are generalized versions of other labels
    label__list.sort(key=len)
    label__list = list(map(lambda it: it.strip(), label__list))
    for a,b in itertools.combinations(label__list,2):
        ratio = fuzz.ratio(a,b)
        if a in b:
            try:
                i = label__list.index(a)
                del label__list[i]
            except ValueError:
                pass
        elif b in a:
            try:
                i = label__list.index(b)
                del label__list[i]
            except ValueError:
                pass
        elif ratio >= 70:
            if len(a) > len(b): #Removes overlapping labels based on similarity
                i = label__list.index(b)
                del label__list[i]
            elif len(a) < len(b):
                i = label__list.index(a)
                del label__list[i]
    return label__list

def train_data(ldata):
    """
    Trains the loaded and parsed data into an nlp.
    """
    nlp = sp.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    with alive_bar(len(ldata)) as bar:
        for _,annotations in ldata:
            for ent in annotations.get('entities'):
                ner.add_label(ent [2])
            bar.text('extracting entities.')
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
        with alive_bar(n_iter) as bar:
            for itn in range(n_iter):
                random.shuffle(ldata)
                losses = {}
                #batch up the examples using spacys minibatch
                batches = minibatch(ldata, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,
                        annotations,
                        drop= 0.3,
                        losses = losses
                    )
                bar.text("training the data. Iteration {}".format(itn))
    return nlp


if __name__ == "__main__":
    nlp = main()
    nlp.to_disk(r'C:\Users\wpegu\Github\MLpython\DAIvB1')
