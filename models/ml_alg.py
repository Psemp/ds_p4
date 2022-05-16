import os
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import KFold, train_test_split
from time import perf_counter


load_dotenv()

cores = int(os.getenv('CORES'))


class Ml_alg():
    def __init__(self, name: str, dataframe: pd.DataFrame, method: callable, fold: bool) -> None:
        self.name = name
        self.df = dataframe
        self.method = method
        self.df_test = []
        self.df_train = []
        self.fold = fold

        if not fold:
            self.df_train, self.df_test = train_test_split(self.df)
        elif fold:
            kf = KFold(n_splits=cores)

            for train, test in kf.split(self.df):
                self.df_train.append(train)
                self.df_test.append(test)

    def execute(self):

        start = perf_counter()
        if not self.fold:
            pass
        elif self.fold:
            pass

        stop = perf_counter()
        self.exec_time = stop - start

    def get_perf(self):
        pass
        perf_dict = {
            "execution_time": self.exec_time,
        }

        return perf_dict

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name
