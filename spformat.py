"""
Spacy Data Formatting Tool
--------------------------
Grabs data from CSV and converts it into a format that can then be utilized to train NER using the Spacy module.
"""
from spacy.util import compounding, minibatch
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import matplotlib.pyplot as plt
from spacy import blank, load
from rapidfuzz import fuzz
from tqdm import tqdm
import pandas as pd
import itertools
import warnings
import random
import json
import tqdm
import re

def main():
    df__transcriptions = pd.read_csv(r'C:\Users\wpegu\Documents\Github\MLpython\Sample_Data\mtsamples_train__test.csv')
    df__transcriptions = df__transcriptions.dropna(axis=0, how='any', subset=['transcription', 'keywords'])
    df__transcriptions.reset_index(drop=True)
    ldata, _labels = load_data(df__transcriptions, 'transcription','keywords')
    df_spata = create_dataframe(ldata)
    #df_spata.to_csv(r'data in spacy format.csv')
    fdups = []
    for label in _labels:
        count = 0
        for x in _labels:
            if x == label:
                count += 1
        fdups.append({'Label': label, 'repeats': count})
    df_repeating = pd.DataFrame(fdups)
    print("Number of columns before dropping duplicates: ", df_repeating.count() + 1)
    df_repeating.sort_values('Label', inplace=True)
    df_repeating = df_repeating.drop_duplicates(subset='Label')
    print("\nNumber of columns after dropping duplicates: ", df_repeating.count() + 1)
    ax = df_repeating.plot.bar(x='Label', y = 'repeats', rot=0)
    plt.show()
    #df_repeating.to_csv(r'repeating labels.csv')

class RangeError(ValueError):
    pass


def evaluate(nlp, examples):
    """
    Scores the accuracy of the nlp model
    ------------------------------------
    """
    scorer = Scorer()

def isinrange(value, low, high):
    """
    Checks if Integer is In Range
    -----------------------------
    Desc: Function evaluates whether the value is within the low-high range.
    
    Parameters:
    value (int): Integer value that is being compared
    low (int): Low end of the range
    high (int): High end of the range.
    """
    #Error Handling
    if not isinstance(value, int):
        raise TypeError(f'dcolumn must be a string, but is a {type(value)}')
    if not isinstance(low, int):
        raise TypeError(f'low must be an integer, but is a {type(low)}')
    if not isinstance(high, int):
        raise TypeError(f'high must be an integer, but is a {type(high)}')
    if low >= high:
        raise RangeError(f'Lowest number in range must be low and highest number in range must be high.')
    #Conditional Statement
    if low <= value <= high:
        return True
    else:
        return False

def clean_punctuation(words):
    """
    Punctuation Cleaner
    -------------------
    Removes off punctuation, such as tabs in words, newlines, spaces in locations where there are not meant to be any, etc.
    Parameters:
    words (str) : Set of words or single word that requires modifications in punctuation.
    """
    words = str(words)
    words = words.lower()
    words = words.strip()
    words = words.replace('\n','')
    words = words.replace('\r', '')
    words = words.replace('\t', '')
    words = words.replace(',', ', ')
    words = words.replace(':', '')
    for word in words:
        if word.isalnum() is not True or word != '-':
            word = word.replace(word, '')
    words = words.replace(' ,', ', ')
    words = words.replace(', , ', ',')
    words = words.replace('.,', '.')
    words = re.sub(r' {2,}', ' ', words)
    return words

def remove_bad_labels(label__list):
    """
    Remove Bad Labels
    -----------------
    Function that removes all of the bad labels.
    Parameters:
    Label__list (list): A list containing all of the labels for sample data.
    """
    if None in label__list:
        i = label__list.index(None)
        del label__list[i]
    elif ' ' in label__list:
        i = label__list.index(' ')
        del label__list[i]
    elif '' in label__list:
        i = label__list.index('')
        del label__list[i]
    else:
        for label in label__list:
            if '-' in label:
                i = label__list.index(label)
                del label__list[i]
    return label__list

def remove_duplicate_labels(label__list):
    """Label Cleaner
    ----------------
    Removes duplicates in the label lists by comparing whether one item is in another.
    
    Parameters:
    label__list (list): a list filled with labels that will be used to label information.
    """
    #Removes labels that are generalized versions of other labels
    label__list.sort(key=len)
    label__list = list(map(lambda it: it.strip(), label__list))
    for a,b in itertools.combinations(label__list,2):
        ratio = fuzz.ratio(a,b)
        a_list = a.split(' ')
        for a_var in a_list:
            if a_var in b and len(a) < len(b):
                try:
                    i = label__list.index(a)
                    del label__list[i]
                except ValueError:
                    continue
                break
            elif a_var in b and len(a) > len(b):
                i = label__list.index(b)
                del label__list[i]
                break
            else:
                continue
        if a in b:
            try:
                i = label__list.index(a)
                del label__list[i]
            except ValueError:
                continue
        elif b in a:
            try:
                i = label__list.index(b)
                del label__list[i]
            except ValueError:
                continue
        elif ratio >= 70:
            if len(a) > len(b): #Removes overlapping labels based on similarity
                i = label__list.index(b)
                del label__list[i]
                continue
            elif len(a) < len(b):
                try:
                    i = label__list.index(a)
                    del label__list[i]
                except ValueError:
                    continue
    return label__list

def remove_overlapping_entities(entities__dict):
    """
    Overlapping Entity Remover
    ---------------
    Removes any overlapping entities by confirming whether the highest or lower value in an entity is within the range of values of another entity.
    Parameters:
    entities__dict (dict): a dictionary made up of one key containing all of the entities associated with sample data (the key name is 'entities').
    """
    entities = entities__dict['entities']
    for a,b in itertools.combinations(entities, 2):
        #Initial Parameters
        a_low = a[0]
        a_high = a[1]
        a_len = len(a[2])
        b_low = b[0]
        b_high = b[1]
        b_len = len(b[2])
        #See if  a and b overlap
        if isinrange(a_high, b_low, b_high) or isinrange(b_low, a_low, a_high) or isinrange(b_high, a_low, a_high) or isinrange(a_low, b_low, b_high):
            try:
                if a_len >= b_len:
                    i = entities.index(b)
                    del entities__dict['entities'][i]
                elif a_len < b_len:
                    i = entities.index(a)
                    del entities__dict['entities'][i]
            except IndexError:
                print('a: ', a, '\nb: ', b, '\na_len: ', a_len, '\nb_len: ', b_len)
        else:
            a__list = a[2].split(' ')
            for a_var in a__list:
                if a_var in b and a_len > b_len:
                    pass
                elif a_var in b and a_len < b_len:
                    pass
    return entities__dict

def load_data(df,dcolumn,lcolumn):
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
    
    Parameters:
    df (DataFrame): Pandas dataframe with the raw unlabeled data and related labels.
    dcolumn (str): Column containing the raw data.
    lcolumn (str): Column containing all of the labels.
    """
    #Deal with bad inputs
    if not isinstance(dcolumn, str):
        raise TypeError(f'dcolumn must be a string, but is a {type(dcolumn)}')
    if not isinstance(lcolumn, str):
        raise TypeError(f'dcolumn must be a string, but is a {type(lcolumn)}')
    #Initial conditions/formatting of the data
    traindata = []
    _labels = []
    df[str(dcolumn)].astype(str)
    df[str(lcolumn)].astype(str)
    for index, row in df.iterrows():
        entities__list = []
        label__str = row[str(lcolumn)]
        label__str = clean_punctuation(label__str)
        sample_text = row[str(dcolumn)]
        sample_text = clean_punctuation(sample_text)
        #Conditional statements for the fields provided
        if ',' in label__str: #There are multiple labels in a field.
            label__list = label__str.split(',') #Work on this portion to apply the load_diagnostic_data function in a general sense (i.e single label vs multiple labels)
            label__list = remove_duplicate_labels(label__list)
            for label in label__list:
                _labels.append(label)
                if label in sample_text:
                    start = sample_text.find(label)
                    end = start + len(label) - 1
                    entities = (start, end, label)
                    entities__list.append(entities)
            entities__dict = {'entities': entities__list}
        else: #There is only one label in the field.
            label = label__str
            label = label.lower()
            label = label.strip()
            _labels.append(label)
            if label in sample_text:
                start = sample_text.find(label)
                end = start + len(label) - 1
                entities = (start, end, label)
            else:
                start = 0
                end = len(sample_text) - 1
                entities = (start, end, label)
            entities__dict = {'entities': [entities]}
        entities__dict = remove_overlapping_entities(entities__dict)
        trainsample = (sample_text, entities__dict)
        traindata.append(trainsample)
    with open('mtsamplesdata.json', 'w', encoding='utf-8') as f:
        y = json.dump(traindata, f, ensure_ascii=False, indent=4)
    return traindata, _labels

def train_data(ldata):
    """
    Data Trainer
    ------------
    Trains the loaded and parsed data into an nlp.
    
    Parameters:
    ldata (list): contains labeled data in the spacy format.
    """
    nlp = blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    for _,annotations in ldata:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    ##
    # Disable unneeded pipes.
    ##
    pipe_exceptions = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    ##
    # Train the ner pipe only
    ##
    nlp.Defaults.stop_words.remove("'s")
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        #Reset and initialize the weights randomly.
        nlp.begin_training()
        n_iter = 30
        x = 0
        b_iter = []
        losses_list = []
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
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
                    drop= 0.4,
                    losses = losses
                )
                print("losses: ", (losses['ner']/len(ldata)) * 100, '\nLost: ', losses, '\n')
                x += 1
                b_iter.append(x)
                losses_list.append(( losses['ner'] / len(ldata) ) * 100)
                #ax1.clear()
                #ax1.plot(b_iter, losses_list)
                #plt.pause(.001)
    return nlp

def create_dataframe(ldata):
    """Creates dataframe from ldata."""
    spdata__list = []
    for sample in ldata:
        spdata__dict = {}
        entities = sample[1]['entities']
        transcription = sample[0]
        spdata__dict['sample_string'] = transcription
        spdata__dict['entities'] = entities
        labels = [entity[2] for entity in entities]
        labels = sorted(labels)
        for label in labels:
            i = labels.index(label)
            spdata__dict[str('label' + str(i))] = label
        spdata__list.append(spdata__dict)
    df = pd.DataFrame(spdata__list)
    return df


if __name__ == "__main__":
    main()