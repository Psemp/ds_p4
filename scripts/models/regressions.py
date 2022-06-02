# import time

import pandas as pd
import numpy as np

# from collections import namedtuple
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import ElasticNetCV, Ridge, Lasso, RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Timescore = namedtuple("Timescore", ["fit", "predict"])


class Regressions():

    common_parameters = {
        "scoring": "r2",
        "cv": 50,
        "n_jobs": -1
    }

    def __init__(self, dataframe: pd.DataFrame, target_col: str, split_size: float = 0.3) -> None:

        self.df_origin = dataframe

        ##
        # Train test split
        self.df_train, self.df_test = train_test_split(self.df_origin, test_size=split_size)
        self.X_train = self.df_train.drop(columns=target_col).to_numpy()
        self.X_test = self.df_test.drop(columns=target_col).to_numpy()

        self.y_train = self.df_train[[target_col]].to_numpy()
        self.y_test = self.df_test[[target_col]].to_numpy()
        #
        ##

        self.std_calc, self.ridge_calc, self.lasso_calc, self.enet_calc = False, False, False, False
        self.ridge_cv_calc = False
        listed_ytest = [value[0] for value in self.y_test]
        self.df_predictions = pd.DataFrame({"True": listed_ytest})

    def get_metrics(self, y_pred):

        metric_list = ["rss", "r2_score", "rsme", "mae"]
        metric_dict = dict.fromkeys(metric_list)

        rse = metrics.mean_squared_error(y_true=self.y_test, y_pred=y_pred)
        metric_dict["rss"] = (((y_pred) - self.y_test) ** 2).sum()
        metric_dict["r2_score"] = metrics.r2_score(y_true=self.y_test, y_pred=y_pred)
        metric_dict["rsme"] = np.sqrt(rse)
        metric_dict["mae"] = metrics.mean_absolute_error(y_true=self.y_test, y_pred=y_pred)
        return metric_dict

    def standard_regression(self, override_default: dict = None):

        std_parameters = Regressions.common_parameters

        if override_default is not None:
            try:
                for key in override_default.keys():
                    std_parameters[key] = override_default[key]
            except KeyError:
                pass
        else:
            pass

        if not self.std_calc:
            print("Step : standard regression")

            self.lin_reg = LinearRegression()
            self.lin_reg.fit(self.X_train, self.y_train)

            if override_default is not None:
                try:
                    for key in override_default.keys():
                        std_parameters[key] = override_default[key]
                except KeyError:
                    pass

            elif override_default is None:

                scores_regression = cross_val_score(
                    self.lin_reg,
                    X=self.X_train,
                    y=self.y_train,
                    scoring=std_parameters["scoring"],
                    cv=std_parameters["cv"],
                    n_jobs=std_parameters["n_jobs"],
                )

            self.std_reg_mean_r2 = scores_regression.mean()
            y_pred_basic = self.lin_reg.predict(self.X_test)

            self.df_predictions["basic_regression"] = y_pred_basic

            self.std_reg_metrics = self.get_metrics(y_pred=y_pred_basic)
            self.std_calc = True

        elif self.std_calc:
            print("Baseline regression already calculated for this model, use self.std_reg_metrics to get the results")

    def ridge_regression(self, override_default: dict = None, alphas_override=None):

        ridge_parameters = Regressions.common_parameters

        if override_default is not None:
            try:
                for key in override_default.keys():
                    ridge_parameters[key] = override_default[key]
            except KeyError:
                pass

        if not self.ridge_calc:
            print("Step : Ridge")

            ridge = Ridge(fit_intercept=False)

            if alphas_override is None:
                step_alphas = 0.05
                alphas = np.arange(0.01, 50, step_alphas)
            elif alphas_override is not None:
                alphas = alphas_override

            parameter = {"alpha": alphas}
            self.clf_ridge = GridSearchCV(
                estimator=ridge,
                param_grid=parameter,
                scoring=ridge_parameters["scoring"],
                cv=ridge_parameters["cv"],
                n_jobs=ridge_parameters["n_jobs"],
            )

            self.clf_ridge.fit(
                X=self.X_train,
                y=self.y_train
            )

            self.ridge_best_alpha = self.clf_ridge.best_params_["alpha"]
            y_pred_ridge = self.clf_ridge.predict(self.X_test)

            self.df_predictions["ridge"] = y_pred_ridge

            self.ridge_metrics = self.get_metrics(y_pred=y_pred_ridge)
            self.ridge_calc = True

        elif self.ridge_calc:
            print("Ridge regression optimal parameters already calculated for this model, \
use self.ridge_metrics to get the results")

    def ridge_cv_regression(self, override_default: dict = None, alphas_override=None):

        ridge_parameters = Regressions.common_parameters

        if override_default is not None:
            try:
                for key in override_default.keys():
                    ridge_parameters[key] = override_default[key]
            except KeyError:
                pass

        if alphas_override is None:
            step_alphas = 0.05
            alphas = np.arange(0.01, 50, step_alphas)
        elif alphas_override is not None:
            alphas = alphas_override

        if not self.ridge_cv_calc:
            print("Step : Ridge")

            self.ridge_cv = RidgeCV(
                fit_intercept=False,
                alphas=alphas,
                store_cv_values=True,
            )

            self.ridge_cv.fit(
                X=self.X_train,
                y=self.y_train
            )

            self.ridge_best_alpha = self.clf_ridge.best_params_["alpha"]
            y_pred_ridge = self.clf_ridge.predict(self.X_test)

            self.df_predictions["ridge"] = y_pred_ridge

            self.ridge_metrics = self.get_metrics(y_pred=y_pred_ridge)
            self.ridge_calc = True

        elif self.ridge_calc:
            print("Ridge regression optimal parameters already calculated for this model, \
use self.ridge_metrics to get the results")

    def lasso_regression(self, override_default: dict = None):

        lasso_parameters = Regressions.common_parameters
        if override_default is not None:
            try:
                for key in override_default.keys():
                    lasso_parameters[key] = override_default[key]
            except KeyError:
                pass

        if not self.lasso_calc:
            print("Step : Lasso")

            lasso = Lasso(fit_intercept=False)
            parameter = {"alpha": np.arange(0.1, 10, 0.01)}

            self.clf_lasso = GridSearchCV(
                estimator=lasso,
                param_grid=parameter,
                scoring=lasso_parameters["scoring"],
                cv=lasso_parameters["cv"],
                n_jobs=lasso_parameters["n_jobs"],
            )

            self.clf_lasso.fit(
                    X=self.X_train,
                    y=self.y_train
                )
            self.lasso_best_alpha = self.clf_lasso.best_params_["alpha"]
            y_pred_lasso = self.clf_lasso.predict(self.X_test)

            self.df_predictions["lasso"] = y_pred_lasso

            self.lasso_metrics = self.get_metrics(y_pred=y_pred_lasso)
            self.lasso_calc = True
        elif self.lasso_calc:
            print("Lasso regression optimal parameters already calculated for this model, \
use self.lasso_metrics to get the results")

    def elastic_net_reg(self, override_default: dict = None):

        enet_parameters = Regressions.common_parameters

        if override_default is not None:
            try:
                for key in override_default.keys():
                    enet_parameters[key] = override_default[key]
            except KeyError:
                pass

        if not self.enet_calc:
            print("Step : Elastic Net")
            l1_range = np.arange(0.01, 0.99, 0.05)
            self.clf_enet = ElasticNetCV(
                l1_ratio=l1_range,
                n_alphas=150,
                cv=enet_parameters["cv"],
                n_jobs=enet_parameters["n_jobs"],
                fit_intercept=False
            )

            self.clf_enet.fit(
                X=self.X_train,
                y=self.y_train
            )

            self.enet_best_l1_ratio = self.clf_enet.l1_ratio_
            self.enet_best_alpha = self.clf_enet.alpha_

            y_pred_enet = self.clf_enet.predict(self.X_test)

            self.df_predictions["Elastic_Net"] = y_pred_enet

            self.elastic_net_metrics = self.get_metrics(y_pred=y_pred_enet)
            self.enet_calc = True
        elif self.enet_calc:
            print("Elastic net regression optimal parameters already calculated for this model, \
use self.lasso_metrics to get the results")

    def execute_all(self):
        self.elastic_net_reg()
        self.lasso_regression()
        self.ridge_regression()
        self.standard_regression()

    def display_all_metrics(self):
        if not (self.enet_calc and self.lasso_calc and self.std_calc and self.ridge_calc):
            self.execute_all()
            self.display_all_metrics()
        else:

            hashes = "\n###############\n"

            print(hashes)

            print("Standard :")
            for key in self.std_reg_metrics.keys():
                print(" -", key, " = ", self.std_reg_metrics[key])

            print(hashes)

            print("Ridge :")
            print(f" - Ridge best alpha = {self.ridge_best_alpha}")
            for key in self.ridge_metrics.keys():
                print(" -", key, " = ", self.ridge_metrics[key])

            print(hashes)

            print("LASSO : ")
            print(f" - Best LASSO alpha : {self.lasso_best_alpha}")
            for key in self.lasso_metrics.keys():
                print(" -", key, " = ", self.lasso_metrics[key])

            print(hashes)

            print("Elastic Net :")
            print(f" - Elastic net best l1 ratio = {self.enet_best_l1_ratio}")
            print(f" - Elastic net best alpha = {self.enet_best_alpha}")
            for key in self.elastic_net_metrics.keys():
                print(" -", key, " = ", self.elastic_net_metrics[key])
