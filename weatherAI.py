import pandas as pd
from datetime import datetime
import numpy as np

def main():
    """Trains data to predict the weather."""
    df__weather__raw = pd.read_csv(r'D:\Github\MLpython\Sample_Data\weather.csv', low_memory=False) 
    df__weather = remove_empty_columns(df__weather__raw, 0.95)
    relative_time = time_difference(df__weather)
    relative_humidity = []
    for row in df__weather.itertuples():
        T_wet = row.HourlyWetBulbTemperature
        T_dry = row.HourlyDryBulbTemperature
        humidity = calculate_humidity(float(T_dry), float(T_wet))
        relative_humidity.append(humidity)
    df__weather['Time'] = relative_time
    df__weather['RelativeHumidity'] = relative_humidity
    df__weather.to_csv("sample__6_26_2020.csv")


def test(): 
    pass


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
        else:
            t_relative = 0
            relative_time.append(t_relative)
    return relative_time

def calculate_humidity(T__dry_bulb, T__wet_bulb):
    """Calculates the relative humidity.
    ------------------------------------
    *Temperature is in 'Celcius*"""
    N = 0.6687451584
    e_dry = 6.112 * np.e ** ((17.502 * T__dry_bulb) / (240.97 *T__dry_bulb))
    e_wet = 6.112 * np.e ** ((17.502 * T__wet_bulb) / (240.97 *T__wet_bulb))
    relative_humidity = ((e_wet - N * (1 + .00115 * T__wet_bulb) * (T__dry_bulb - T__wet_bulb)) / e_dry) * 100
    return relative_humidity


if __name__ == "__main__":
    #main()
    test()
