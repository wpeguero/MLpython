import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def main():
    """Creates visuals for data that provide better insight as to what information the data can provide."""
    #%%Load the information
    path_to_file = os.path.abspath('Sample_Data/Breast Cancer Diagnostic (Wisconsin).csv')
    df__diabetics = pd.read_csv(path_to_file)
    #/mnt/c/Users/Wpeguero/Documents/GitHub/MLpython/Sample_Data/estimated_inpatient_all_20200921_0928.csv
    columns = df__diabetics.columns
    columns = columns.tolist()
    print(columns)


def extract_groups(df, group_variable):
    """Extracts groups from dataframe"""
    dict__groups = {}
    groups = df.groupby(group_variable)
    for name, group in groups:
        dict__groups["df__{}".format(name)] = group
    return dict__groups

def avg_col_value(dict_groups, col):
    """Exxtracts the columns."""
    dict__averages = {}
    for key,df__group in dict_groups.items():
        df__group = df__group.replace(',', '', regex=True)
        df__group[col] = df__group[col].astype(float)
        avg = df__group[col].mean()
        avg = round(avg)
        state = key[-2] + key[-1]
        dict__averages['{}'.format(state)] = avg
    return dict__averages


if __name__ == "__main__":
    main()