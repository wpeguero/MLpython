import pandas as pd
from datetime import datetime

def main():
    """Trains data to predict the weather."""
    df__weather__raw = pd.read_csv(r'D:\Github\MLpython\Sample_Data\weather.csv', low_memory=False) 
    df__weather = remove_empty_columns(df__weather__raw, 0.95)
    df__weather.to_csv("sample.csv")


def test(): 
    df__weather__raw = pd.read_csv(r'D:\Github\MLpython\Sample_Data\weather.csv', low_memory=False) 
    df__weather = remove_empty_columns(df__weather__raw, 0.95)
    time = []
    time_difference(df__weather)


def remove_empty_columns(df, percentage):
    """Removes all of the columns that are mainly empty based on a threshold value."""
    columns = df.columns
    percent_missing = (df.isnull().sum()) / len(df)
    df__missing_value = pd.DataFrame({'column_name': columns, 'percent_missing': percent_missing})
    drop_missing = list(df__missing_value[df__missing_value.percent_missing>percentage].column_name)
    df = df.drop(drop_missing, axis=1)
    return df

def time_difference(df):
    """Calculates time difference based on datetime objects.
    - Only works with dataframes that have a datetime string."""
    dt = [] # stores all of the datetime objects
    for row in df.itertuples():
        d = datetime.strptime(row.DATE,"%Y-%m-%dT%H:%M:%S")
        dt.append(d)
        relative_time = []
    for i in range(len(dt)):
        if i > 0:
            t_relative = dt[i] - dt[i-1]
            relative_time.append(t_relative.total_seconds())
    print(relative_time)


if __name__ == "__main__":
    #main()
    test()
