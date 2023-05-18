import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def encode_categorical_variables(df, column1, column2, column3, column4):
    """
    Encodes categorical variables in a DataFrame using one-hot encoding.

    The function encodes the specified categorical variables in the DataFrame using one-hot encoding.
    It creates dummy variables for each unique value in the categorical columns and drops the original columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the categorical columns.
        column1 (str): The name of the first categorical column.
        column2 (str): The name of the second categorical column.
        column3 (str): The name of the third categorical column.
        column4 (str): The name of the fourth categorical column.

    Returns:
        pandas.DataFrame: The modified DataFrame with encoded categorical variables.

    Example:
        df = pd.DataFrame({'color': ['red', 'blue', 'green'],
                           'size': ['small', 'medium', 'large'],
                           'shape': ['circle', 'square', 'triangle'],
                           'material': ['metal', 'wood', 'plastic']})
        df = encode_categorical_variables(df, 'color', 'size', 'shape', 'material')
    """

    encoded_df = pd.get_dummies(df[[column1, column2, column3, column4]], prefix=[column1, column2, column3, column4],
                                drop_first=True)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop([column1, column2, column3, column4], axis=1, inplace=True)

    return df


def convert_time_to_float(df, time_column):
    """
    Convert a time column in a DataFrame to float representation.
    Assumes the time column is in the format 'HH:MM:SS'.
    Adds a new column 'float_time' with the float representation of the time.

    Parameters:
    df (DataFrame): The DataFrame containing the time column.
    time_column (str): The name of the time column.

    Returns:
    df (DataFrame): The modified DataFrame with the float representation of the time.
    """
    df['float_time'] = df[time_column].apply(lambda x: x.hour * 60 + x.minute + x.second / 60)
    return df


def generate_day_month_columns(df, datetime_column):
    """
    Generates separate columns for day of the week and month from a datetime column in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the datetime column.
        datetime_column (str): The name of the datetime column.

    Returns:
        pandas.DataFrame: The modified DataFrame with additional day of the week and month columns.

    Raises:
        ValueError: If the specified datetime_column does not exist in the DataFrame.

    Example:
        df = pd.DataFrame({'datetime': ['2023-01-01', '2023-02-01', '2023-03-01']})
        df = generate_day_month_columns(df, 'datetime')
    """

    if datetime_column not in df.columns:
        raise ValueError(f"Column '{datetime_column}' does not exist in the DataFrame.")

    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['day_of_week'] = df[datetime_column].dt.day_name()
    df['month'] = df[datetime_column].dt.month_name()

    return df


def scale_numerical_columns(df):
    """
    Scale numerical columns in a DataFrame to the range [0, 1] using MinMaxScaler.

    Args:
        df (pandas.DataFrame): The DataFrame to be scaled.

    Returns:
        pandas.DataFrame: The modified DataFrame with scaled numerical columns.

    Example:
        df = pd.DataFrame({'col1': [1, 2, 3],
                           'col2': [4, 5, 6],
                           'col3': ['A', 'B', 'C']})
        scaled_df = scale_numerical_columns(df)
    """
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include='number').columns

    if len(numerical_columns) == 0:
        raise ValueError("No numerical columns found in the DataFrame.")

    # Create a scaler object
    scaler = MinMaxScaler()

    # Scale numerical columns
    df_scaled = df.copy()
    df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df_scaled
