import pandas as pd
import seaborn as sns

def main():
    """Creates visuals for data that provide better insight as to what information the data can provide."""
    #%%Load the information
    df__estimated_inpatient = pd.read_csv(r'/mnt/c/Users/Wpeguero/Documents/GitHub/MLpython/Sample_Data/estimated_inpatient_all_20200921_0928.csv')
    grp__estimated_inpatient__by_state = df__estimated_inpatient.groupby('state')
    print(grp__estimated_inpatient__by_state.get_group('AK'))

def extract_groups(df):
    """Extracts groups from dataframe"""
    pass

if __name__ == "__main__":
    main()