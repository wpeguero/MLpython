import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Trains data to predict the weather."""
    df__weather__raw = pd.read_csv(r'D:\Github\MLpython\Sample_Data\weather.csv', low_memory=False) 
    df__weather = remove_empty_columns(df__weather__raw, 0.80)
    relative_time = time_difference(df__weather)
    relative_humidity = []
    for row in df__weather.itertuples():
        T_wet = row.HourlyWetBulbTemperature #degrees Farenheit
        T_dry = row.HourlyDryBulbTemperature #degrees Farenheit
        humidity = calculate_humidity(float(T_dry), float(T_wet))
        relative_humidity.append(humidity)
    df__weather['Time'] = relative_time
    df__weather['RelativeHumidity'] = relative_humidity
    print('Before dropping the NaN values: \n \n',len(df__weather['RelativeHumidity']))
    df__weather.dropna(subset=['RelativeHumidity'], inplace=True)
    print('\n After dropping the NaN values: \n \n',len(df__weather['RelativeHumidity']))
    print('\n Columns Left: \n \n', df__weather.columns, '\n \n Number of columns \n \n', len(df__weather.columns))
    df__weather.plot(x='Time', y='RelativeHumidity')
    plt.show()


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
        delta = dt[i] - dt[0]
        t_seconds = delta.total_seconds()
        relative_time.append(t_seconds)
    return relative_time

def calculate_humidity(T__dry_bulb, T__wet_bulb):
    """Calculates the relative humidity.
    ------------------------------------
    *Temperature is in Celsius*"""
    T__dry_bulb__C = (T__dry_bulb - 32) * 5 / 9
    T__wet_bulb__C = (T__wet_bulb - 32) * 5 / 9
    N = 0.6687451584
    try:
        e_dry = 6.112 * np.e ** ((17.502 * T__dry_bulb__C) / (240.97 *T__dry_bulb__C))
        e_wet = 6.112 * np.e ** ((17.502 * T__wet_bulb__C) / (240.97 *T__wet_bulb__C))
        relative_humidity = ((e_wet - N * (1 + .00115 * T__wet_bulb__C) * (T__dry_bulb__C - T__wet_bulb__C)) / e_dry) * 100
        if relative_humidity < 0:
            relative_humidity = None
    except ZeroDivisionError:
        relative_humidity = None
    return relative_humidity

def stat_values(input_list):
    """Calculates statistical values from a list of data points and returns them as a dictionary."""
    n = len(input_list)
    X_barr = sum(input_list) / n
    X_i_sq__list = []
    for i in input_list:
        value = i**2
        X_i_sq__list.append(value)
    svariance = ( 1 / n * (n - 1))(n * sum(X_i_sq__list) - sum(input_list) ** 2)
    #Calculate the Standard deviation
    top_sum = []
    for i in input_list:
        difference = i - X_barr
        difference ** 2
        top_sum.append(difference)
    stdeviation = np.sqrt(sum(top_sum) / (n - 1))
    svalues = {'mean': X_barr, 'SampleVariance': svariance, 'StandardDeviation': stdeviation}
    return svalues


if __name__ == "__main__":
    main()