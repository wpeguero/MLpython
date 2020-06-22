import pandas as pd

def main():
    """Trains data to predict the weather."""
    df__weather__raw = pd.read_csv(r'D:\Github\Sample_Data\weather.csv')
    print(df__weather__raw)

if __name__ == "__main__":
    main()