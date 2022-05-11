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


def complete_df(df_one: pd.DataFrame, df_two: pd.DataFrame, on_col: str = None) -> pd.DataFrame:
    # Not yet functionnal
    """
    Void function - Takes df_one and df_two, df_one as the most up to date, df_two as an older df
    function will try to fill the nan values from df_one if value is non nan in df_two
    on_col can be passed as parameter to override the default index

    Args:
    - df_one : dataframe to complete
    - df_two : dataframe used to complete df_one
    - on_col : name of a column, used to override the use of an index, using on_col as index
    (ex: primary keys, unique id, other identifiers)

    Returns:
    - Void function
    """

    for index, series in df_one.iterrows():
        if on_col is not None:
            idx = series[on_col]
        else:
            idx = index

        for row in list(series.index):
            if (series.size > 0 or pd.isna(series[row])):
                try:
                    if on_col is not None:
                        df_two_sers = df_two[df_two["OSEBuildingID"] == idx]
                    elif on_col is None:
                        df_two_sers = df_two.iloc[idx]

                    if (df_two_sers[row].values.size > 0 and not pd.isna(df_two_sers[row].values)):
                        df_one.at[index, row] = df_two_sers[row].values
                    else:
                        pass
                except KeyError:
                    pass  # Key of df_one not in df_two
