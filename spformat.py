"""
Spacy Data Formatting Tool
--------------------------
Grabs data from CSV and converts it into a format that can then be utilized to train NER using the Spacy module.
"""
from rapidfuzz import fuzz
import itertools

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
                try:
                    i = label__list.index(a)
                    del label__list[i]
                except ValueError:
                    pass
    return label__list

def remove_duplicate_entities(entities__dict):
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
            pass
    return entities__dict


if __name__ == "__main__":
    main()