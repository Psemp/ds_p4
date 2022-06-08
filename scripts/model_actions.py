import json


def freeze_model(model, save_file: bool = False, file_path: str = None):
    """
    Takes a model as parameter and saves the split by using train=[ids], test=[ids]
    takes only the index so splitting the dataset manually will be possible.
    if not save_file (default), returns the dict of ids
    if save_file, saves the file by default in cwd or file_path
    Args :
    - model (custom model using DataFrames to do the split)
    - save_file : bool, default = False, whether to return ids or saving ids to a file
    - file_path : str, default = None, relative path to save the file. Default will be : ./freeze_split.txt
    Returns:
    - ids_train, ids_test
    """

    ids_train = model.df_train.index.values.astype(int)
    ids_test = model.df_test.index.values.astype(int)
    if not save_file:
        return ids_train, ids_test
    elif save_file:
        ids_dict = {"train": ids_train.tolist(), "test": ids_test.tolist()}
        if file_path is None:
            with open("splits.json", "w") as json_file:
                json.dump(ids_dict, json_file)
        elif file_path is not None:
            with open(file_path, "w") as json_file:
                json.dump(ids_dict, json_file)
