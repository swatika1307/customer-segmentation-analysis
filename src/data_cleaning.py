import pandas as pd


def load_data(path):
    """
    Load dataset from CSV file.
    """

    return pd.read_csv(path)


def inspect_data(df):
    """
    Display dataset information.
    """

    print("Dataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nStatistical Summary:")
    print(df.describe())


def clean_data(df):
    """
    Clean dataset by removing duplicates.
    """

    df = df.drop_duplicates()

    return df


def check_unique_values(df):
    """
    Display unique categorical values.
    """

    print("Unique Gender Values:")
    print(df['Gender'].unique())