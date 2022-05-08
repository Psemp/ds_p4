import pandas as pd
import numpy as np


def remove_outliers(column_eval: str,  df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the IQR (Inter Quartile Range) for each column value of the column_subset of a DataFrame using
     the column to evaluate.
    Applies the method to clean the data for each value in the subset and returns a cleaned dataframe
    Args:
     - column_eval : name of the column to evaluate, dtype must be int or float
     - df : the dataframe to clean
    Returns : Void function
    """
    dtype = str(df[column_eval].dtype)
    if not (dtype.startswith("int") or dtype.startswith("float")):
        print(not dtype.startswith("int") or not dtype.startswith("float"))
        print(dtype, "trying using column.astype(float)")

        try:
            df[column_eval] = df[column_eval].astype(float)

        except Exception:
            raise Exception("Error : Data Type not a number")

    q3, q1 = np.percentile(df[column_eval], [75, 25])
    iqr = (q3 - q1) * 1.5
    q3_max = q3 + iqr
    q1_min = q1 - iqr

    for index, row in df.iterrows():
        if (row[column_eval] < q1_min) or (row[column_eval] > q3_max):
            df.drop(index=index, axis=1, inplace=True)
