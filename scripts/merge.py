import pandas as pd


def merge_cols(origin_col_list: list, target_col_name: str, dataframe: pd.DataFrame):
    """
    - Merges columns of a dataframe as : origin_col_list values to dataframe[target_col_name]

    Args:
    - origin_col_list : The columns where values will be extracted
    - target_col_name : The column where merged values will be stored
    - dataframe : The dataframe containing the data and cols

    Returns:
    - Nothing, modifies dataframe inplace
    """

    col_data = dict.fromkeys(origin_col_list)

    for index, row in dataframe.iterrows():
        for col in origin_col_list:
            col_data[col] = row[col]
        dataframe.at[index, target_col_name] = str(col_data)
