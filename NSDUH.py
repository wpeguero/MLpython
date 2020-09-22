import re
from re import match
import pandas as pd
import spacy as sp
from tika import parser
from alive_progress import alive_bar

def main():
    """Uses the National Survey on Drug use and Health 2015 (NSDUH2015) to..."""
    df = pd.read_csv(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\NSDUH_2015__clsremoved.csv')
    df = remove_empty_columns(df, 0.005)
    PDFObject = parser.from_file(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\NSDUH-2015-DS0001-info-codebook.pdf')
    CodeBook = PDFObject['content']
    #print("\n\nTotal result:\n",find_definitions(CodeBook,'FLURPDAPYU')) #This is a sample column that was skipped.
    extract_definitions(df, CodeBook)
    #code_definitions, col_def=find_definitions(CodeBook, 'CABINGEVR')
    #print(code_definitions)


def extract_definitions(df, CodeBook):
    """Extracts all of the definitions from the codebook based on the columns from the DataFrame."""
    columns = df.columns
    col_def__list = []
    code_def__list = []
    col_list = []
    with alive_bar(len(columns)) as bar:
        for col in columns:
            try:
                code_def, col_def = find_definitions(CodeBook, col)
                col_def__list.append(col_def)
                code_def__list.extend(code_def)
            except(IndexError, AttributeError) as e:
                col_list.append(col)
            bar()
    df__col = pd.DataFrame(col_def__list) # DataFrame of column and its definition
    df__col.to_csv('col_name_and_definitions.csv', index=False)
    df__code = pd.DataFrame(code_def__list) # DataFrame of code with the column name and the code definition.
    df__code.to_csv('code_and_definitions.csv',index=False)
    print('\n\n\n', col_list, '\n\n','columns missing: ', len(col_list))

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
    # text = text.replace('\n', '|')
    pattern = '(\n'+ str(column_name) + '\d{0,} Len : \d.*?\n\n \n)' #+ '|(\n' + str(column_name) + '.*?\n \n)' #Test this string further with the following columns: COUTYP4, SERVICE, VEREP
    regex = re.compile(pattern)
    match = re.search(str(pattern), text, re.S)
    result = match.group(0)
    #print('raw match:\n', result)
    result = result.replace(".", "")
    result = result.replace('NOTE', '')
    result = result.strip()
    results = result.split('\n')
    r_def = ''
    desired_result__string = str(column_name) + '\d{0,} Len : \d{0,}'
    for r in results: #This for loop is failing to identify the string
        if re.match(str(desired_result__string), r) is not None:
            r_def = r
        else:
            pass
    #print('r_def: ',r_def, '\n\n')
    #print('Raw result: \n', results, '\n\nnumber of lines in results: ', len(results))
    column_definition__list = re.split('Len : \d{0,}', r_def)
    column_definition = column_definition__list[1]
    column_definition = column_definition.strip()
    col_def = {'columnName': str(column_name), 'Definition': column_definition}
    code_definitions = []
    for result in results:
        if '  Freq Pct ' in results[1]:
            if re.match(r'\d{1,} = .*', result) is not None:
                result = result.strip()
                result_list = result.split(" = ")
                definition = result_list[1]
                definition = re.sub(r'\s{2,}.*', '',definition)
                code_definitions.append({'Code': str(result_list[0]), 'columnName': column_name, 'Definition': str(definition)})
            else:
                pass
                #print('Index: ',results.index(result), '\n', 'The result did not match the regex. \n \n')
        else:
            pass
    #print('\n\nTesting definition split:\n',column_definition)
    return code_definitions, col_def


if __name__ == "__main__":
    main()
