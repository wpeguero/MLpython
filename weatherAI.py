import pandas as pd

def main():
    """Trains data to predict the weather."""
    df__weather__raw = pd.read_csv(r'D:\Github\Sample_Data\weather.csv', low_memory=False)
    df__weather = remove_empty_columns(df__weather__raw, 0.5)
    print('Raw data: \n', df__weather__raw)
    print('processed data: \n', df__weather)


def remove_empty_columns(df, percentage):
    """Removes all of the columns that are mainly empty based on a threshold value."""
    columns = df.columns
    percent_missing = (df.isnull().sum() * 100) / len(df)
    df__missing_value = pd.DataFrame({'column_name': columns, 'percent_missing': percent_missing})
    drop_missing = list(df__missing_value[df__missing_value.percent_missing>percentage].column_name)
    df = df.drop(drop_missing, axis=1)
    return df


if __name__ == "__main__":
    main()