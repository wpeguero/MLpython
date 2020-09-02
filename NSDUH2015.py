import re
from re import match
import pandas as pd
import spacy as sp
from tika import parser
from alive_progress import alive_bar

def main():
    """Uses the National Survey on Drug use and Health 2015 (NSDUH2015) to..."""
    df = pd.read_table(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\NSDUH_2015__clsremoved.csv')
    df = remove_empty_columns(df, 0.005)
    column_names = df.columns
    column_definitions = []
    PDFObject = parser.from_file(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\NSDUH-2015-DS0001-info-codebook.pdf')
    CodeBook = PDFObject['content']
    print("\n",find_definitions(CodeBook,'TXPAYOTSP2'))


def remove_empty_columns(df, percentage):
    """Removes all of the columns that are mainly empty based on a threshold value.
    ----------
    Parameters
    df = DataFrame (Object)
    percentage = value from 0 to 1 (int)
    - Percent of the column that is missing."""
    columns = df.columns
    percent_missing = (df.isnull().sum()) / len(df)
    df__missing_value = pd.DataFrame({'column_name': columns, 'percent_missing': percent_missing})
    drop_missing = list(df__missing_value[df__missing_value.percent_missing>percentage].column_name)
    df = df.drop(drop_missing, axis=1)
    return df

def find_definitions(text, column_name):
    """Finds the definition of the column from the codebook pdf."""
    text = text.replace('\n', '|')
    pattern = str(column_name)  + '.*?' + '\|99'
    regex = re.compile(pattern)
    match = re.search(regex, text)
    result = match.group(0)
    result = result.replace(".", "")
    results = result.split('|')
    column_definition__list = re.split('Len : \d', results[0])
    column_definition = column_definition__list[1]
    column_definition = column_definition.strip()
    code_definitions = []
    for result in results:
        if re.match(r'\d{1,2} = .*', result) is not None:
            result = result.strip()
            result_list = result.split(" = ")
            code_definitions.append({str(result_list[0]) + '(' + column_name + ')'  :str(result_list[1])})
        else:
            print('Index: ',results.index(result), '\n', 'The result did not match the regex. \n \n')
    print(column_definition)
    # Need to extract the definition behind the code now that the defined codes are split up.
    return code_definitions


if __name__ == "__main__":
    main()