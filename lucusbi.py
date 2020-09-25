import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    """Creates visuals for data that provide better insight as to what information the data can provide."""
    #%%Load the information
    df__estimated_inpatient = pd.read_csv(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\estimated_inpatient_all_20200921_0928.csv')
    groups__state = extract_groups(df__estimated_inpatient, 'state')
    avg__IBOE__state__raw = avg_col_value(groups__state, 'Inpatient Beds Occupied Estimated')
    print('Length of dictionary: ', len(avg__IBOE__state__raw))
    x = []
    y = []
    avg__IBOE__state = {}
    for key, avg in avg__IBOE__state__raw.items():
        x.append(key)
        y.append(avg)
        avg__IBOE__state['{}'.format(key)] = avg
    ax1 = sns.barplot(x,y)
    df__estimated_inpatient['Inpatient Beds Occupied Estimated'] = df__estimated_inpatient['Inpatient Beds Occupied Estimated'].astype(float)
    ax2 = sns.barplot(x='state',y='Inpatient Beds Occupied Estimated', data=df__estimated_inpatient)
    plt.show(ax1)
    plt.show(ax2)


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