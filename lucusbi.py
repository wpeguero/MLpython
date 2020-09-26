import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def main():
    """Creates visuals for data that provide better insight as to what information the data can provide."""
    #%%Load the information
    path_to_file = os.path.abspath('Sample_Data/estimated_inpatient_all_20200921_0928.csv')
    df__estimated_inpatient = pd.read_csv(path_to_file)
    #/mnt/c/Users/Wpeguero/Documents/GitHub/MLpython/Sample_Data/estimated_inpatient_all_20200921_0928.csv
    #%% Clean the Data
    datatype = {
        'state': 'category',
        'Inpatient Beds Occupied Estimated': 'int',
        'Count LL': 'int',
        'Count UL': 'int',
        'Percentage of Inpatient Beds Occupied Estimated': 'float',
        'Percentage LL': 'float',
        'Percentage UL': 'float',
        'Total Inpatient Beds': 'int',
        'Total LL': 'int',
        'Total UL': 'int'
    }
    df__estimated_inpatient = df__estimated_inpatient.replace(',','',regex=True)
    df__estimated_inpatient = df__estimated_inpatient.astype(datatype)
    #%%Plot the Data
    plt.figure(figsize=(20,10))
    ax1 = sns.barplot(x='state',y='Inpatient Beds Occupied Estimated', data=df__estimated_inpatient)
    try:
        plt.show(ax1)
    except TypeError as e:
        fig = ax1.get_figure()
        fig.savefig('Estimation of Inpatiend Beds Occupied by State.png')


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

def clean_data(df):
    """Converts numerical data into proper format, as well as remove any unwanted characters."""
    pass


if __name__ == "__main__":
    main()