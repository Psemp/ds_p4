import numpy as np
from sklearn import metrics


def rmse(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
    return np.sqrt(mse)


def residual_sum_squared(y_true: list, y_pred: list):
    rss = 0
    for idx in range(len(y_pred)):
        rss += np.square(y_true[idx] - y_pred[idx])

    return rss
