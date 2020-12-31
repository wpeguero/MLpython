"""
Spacy Data Formatting Tool
--------------------------
Grabs data from CSV and converts it into a format that can then be utilized to train NER using the Spacy module.
"""
from rapidfuzz import fuzz
import itertools
import re

def main():
    pass


class RangeError(ValueError):
    pass


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
    if value <= low or value >= high:
        raise ValueError(f'value {value} is outside of the range {low}-{high}')
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
    words = words.lower()
    words = words.strip()
    words = re.sub(r' {2,}', ' ', words)
    words = words.replace('\n','').replace('\r', '').replace('\t', '')
    return words

def remove_bad_labels(label__list):
    """
    Remove Bad Labels
    -----------------
    Function that removes all of the bad labels.
    Parameters:
    Label__list (list): A list containing all of the labels for sample data.
    """
    if '' in label__list:
        i = label__list.index('')
        del label__list[i]
    elif ' ' in label__list:
        i = label__list.index(' ')
        del label__list[i]
    else:
        pass
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
            if a_len > b_len:
                i = entities.index(b)
                del entities__dict['entities'][i]
                del entities[i]
            elif a_len < b_len:
                i = entities.index(a)
                del entities__dict['entities'][i]
                del entities[i]
        else:
            a__list = a.split(' ')
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
                if label in sample_text:
                    start = sample_text.find(label)
                    end = start + len(label)
                    entities = (start, end, label)
                    entities__list.append(entities)
            entities__dict = {'entities': entities__list}
        else: #There is only one label in the field.
            label = label__str
            label = label.lower()
            label = label.strip()
            if label in sample_text:
                start = sample_text.find(label)
                end = start + len(label)
                entities = (start, end, label)
            else:
                start = 0
                end = len(sample_text)
                entities = (start, end, label)
            entities__dict = {'entities': [entities]}
        entities__dict = remove_overlapping_entities(entities__dict)
        trainsample = (row[str(dcolumn)], entities__dict)
        traindata.append(trainsample)
    return traindata


if __name__ == "__main__":
    main()