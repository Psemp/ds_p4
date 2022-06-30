import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def one_hot_dataframe(dataframe: pd.DataFrame, subset: list, prefix: list = None, drop_og: bool = False):

    """
    Takes in argument a dataframe and a subset (the col names to encode), and encodes the subset with one hot encoder\
    , uses numeric index by default for new cols but can be overriden with a prefix list.\
    Can drop original cols or keep them with parameter "drop_og"

    Args:
    - dataframe : pandas.DataFrame object containing the columns to encode
    - subset : list of names of the columns to encode
    - prefix : list (default is None), if not None, provides a prefix for each col in subset, hence :\
        len(subset) must be == to len(prefix), or leave prefix to None and it will provide an index
    - drop_og : boolean, False by default. Determines whether or not to keep the original columns in\
        the returned dataframe

    Returns:
    - pandas.DataFrame object with columns encoded with OHE and args preferences.
    """

    if prefix is not None and len(subset) != len(prefix):
        raise Exception("Lenght of subset and prefix must be equal (if prefix not None)")

    ohe_idx = 0
    df_ohe = dataframe.copy(deep=True)

    for original in subset:

        ohe_col = OneHotEncoder()

        transformed = ohe_col.fit_transform(df_ohe[[original]])

        df_ohe[ohe_col.categories_[0]] = transformed.toarray()

        col_names = list(ohe_col.categories_[0])

        rename_cols = dict.fromkeys(col_names)

        if prefix is not None:
            for key in rename_cols.keys():
                rename_cols[key] = f"ohe_{str(prefix[ohe_idx])}_{key}"
        else:
            for key in rename_cols.keys():
                rename_cols[key] = f"ohe_{ohe_idx}_{key}"
        df_ohe.rename(columns=rename_cols, inplace=True)

        ohe_idx += 1

    if drop_og:
        df_ohe.drop(columns=subset, inplace=True)
        return df_ohe

    elif not drop_og:
        return df_ohe


def scale_df(dataframe_to_scale: pd.DataFrame, constant_col: list = None) -> pd.DataFrame:
    """
    Function : Uses sklearn's StandardScaler to scale data in dataframe to scale, returns scaled dataframe

    Args :
    - dataframe_to_scale : Dataframe object containing data to scale
    - constant_col - Optionnal : col_name list in dataframe_to_scale to keep as is in scaled_df return

    Returns :
    - Dataframe object containing scaled data
    """
    scaler = StandardScaler()

    if constant_col is None:
        columns = dataframe_to_scale.columns
    elif constant_col is not None:
        columns = [col for col in dataframe_to_scale if col not in constant_col]

    scaled_df = pd.DataFrame(index=dataframe_to_scale.index)

    scaled_data = scaler.fit_transform(dataframe_to_scale[columns]).T

    for arr in range(0, len(scaled_data)):
        scaled_df[f"scaled_{columns[arr]}"] = scaled_data[arr]

    if constant_col is not None:
        for col in constant_col:
            scaled_df[col] = dataframe_to_scale[col]

    return scaled_df
