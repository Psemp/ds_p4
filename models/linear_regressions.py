import pandas as pd
import seaborn as sns
import numpy as np

import time

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn import metrics
from matplotlib import pyplot as plt

from models.time_card import Time_card


class Linear_reg():
    """
    Class for easier regressions using OLS, Lasso, Ridge or Elnet. Used to find best parameters

    Args:
        - dataframe = dataframe on which regression will be performed
        - target = str : column of the dataframe to set as regression target
        - override_common = dict : default is None, can be set with correct keys to change regression, keys are :
            - "scoring": "neg_mean_squared_error",
            - "cv": None,  # Default = Leave One Out
            - "n_jobs": -1  # Use all cores
        - split = float : default is .3, determines propotion test/train (0>split>1)
        - ingore_cols = list : default is None. input columns that will not make it in the X_matrices
    """

    common_parameters = {
        "scoring": "neg_mean_squared_error",
        "cv": None,  # Default = Leave One Out
        "n_jobs": -1  # Use all cores
    }

    def __init__(
            self, dataframe: pd.DataFrame, target: str,
            split: float = 0.3, ignore_columns: list = None
            ):

        self.df_origin = dataframe
        self.target = target
        ##
        # Train test split
        self.df_train, self.df_test = train_test_split(self.df_origin, test_size=split)

        if ignore_columns is not None:
            self.df_train.drop(columns=ignore_columns, inplace=True)
            self.df_test.drop(columns=ignore_columns, inplace=True)

        self.X_train = self.df_train.drop(columns=self.target).to_numpy()
        self.X_test = self.df_test.drop(columns=self.target).to_numpy()

        self.y_train = self.df_train[[self.target]].to_numpy()
        self.y_test = self.df_test[[self.target]].to_numpy()
        #
        ##

        self.std_calc, self.ridge_calc, self.lasso_calc, self.enet_calc = False, False, False, False
        self.listed_ytrain = [value[0] for value in self.y_train]
        self.listed_ytest = [value[0] for value in self.y_test]
        self.df_predictions = pd.DataFrame({"True": self.listed_ytest})

    def force_split(self, df_train_ovr: pd.DataFrame, df_test_ovr: pd.DataFrame):
        """
        Provides an override possibility for train test split. Resets the columns
        to recreate the matrices. Also recreates the y_test/train vectors and resets the df_predictions
        with expected values. (If col has to be dropped, need to recall the drop_col method)

        Args :
        - df_train_ovr : training set, with target included
        - df_test_ovr : testing set, with target included

        Returns :
        - Void
        """

        self.df_train = df_train_ovr
        self.df_test = df_test_ovr
        self.reset_cols()
        self.y_train = self.df_train[[self.target]].to_numpy()
        self.y_test = self.df_test[[self.target]].to_numpy()
        self.listed_ytrain = [value[0] for value in self.y_train]
        self.listed_ytest = [value[0] for value in self.y_test]
        self.df_predictions = pd.DataFrame({"True": self.listed_ytest})

    def drop_col(self, col_list: list):
        """
        can be used to drop a column of the train/test set and keep the same data repartition
         to minimize split randomization
        """
        dropcols = []
        dropcols.append(self.target)
        [dropcols.append(col) for col in col_list]
        self.X_train = self.df_train.drop(columns=dropcols).to_numpy()
        self.X_test = self.df_test.drop(columns=dropcols).to_numpy()

    def reset_cols(self):
        """
        resets train test matrices if col has been dropped, use for preservation of initial split
        """
        self.X_train = self.df_train.drop(columns=self.target).to_numpy()
        self.X_test = self.df_test.drop(columns=self.target).to_numpy()

    def get_table(scores_train, scores_test):
        # Tuple shape : [0] = RMSE, [1] = R2
        return pd.DataFrame(
            columns=["RMSE", "R2"],
            data=[
                [scores_train[0], scores_train[1]],
                [scores_test[0], scores_test[1]],
                ],
            index=["Train", "Test"]
        )

    def get_plot(rmse_list, alpha_list, fig_params: dict = None, labels: dict = None):
        if fig_params is not None:
            dpi = fig_params["dpi"]
            w = fig_params["w"]
            h = fig_params["h"]
        else:
            dpi = 155
            w = 5
            h = 5
        if Linear_reg.common_parameters["cv"] is None:
            fig, (ax1) = plt.subplots(
                ncols=1,
                nrows=1,
                figsize=(w, h),
                dpi=dpi,
            )
            ax1.plot(alpha_list, rmse_list, color="navy", linewidth=1)

            ax1.set_xlabel("Alpha")
            ax1.set_ylabel("RMSE")
            fig.suptitle("Evolution de la RMSE en fonction de l'hyperparametre")

            if labels is not None:
                try:
                    ax1.set_xlabel(labels["x_label"])
                    ax1.set_ylabel(labels["y_label"])
                    fig.suptitle(labels["title"])
                except KeyError:
                    print("Please provide all keys values in labels : x_label, y_label, title")
                    pass

            plt.tight_layout()
            plt.show()

        else:
            print("""Cannot use plot on ridge regression with CV not None, cf. sklearn doc\
 on "store_cv_values" on RidgeCV""")

    def scatter_true_pred(self, regression_name: str, fig_params: dict = None):
        """
        Plots values from regression as y=true, x=predicted, using x=y as guideline

        Args :
        - regression_name : the name of the column of df_predictions to plot against y_true
        - fig_params : not defined atm
        """

        if regression_name not in self.df_predictions.columns:
            raise KeyError(f"regression name must be part of {self.df_predictions.columns}")

        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(4, 4),
            dpi=155,
        )

        ax1 = sns.scatterplot(
            data=self.df_predictions,
            x="True",
            y=f"{regression_name}"
            )

        x_min = round(min(self.df_predictions["True"].values)) - 1
        x_max = round(max(self.df_predictions["True"].values)) + 1
        x = np.arange(x_min, x_max, 1)
        y = x
        ax1.plot(x, y, color="navy", linewidth=.9)
        ###
        # Titles/Lables
        ax1.set_xlabel(f"{regression_name}")
        ax1.set_ylabel("Y_True")
        ax1.set_ylim(x_min, x_max)
        ax1.set_xlim(x_min, x_max)
        #
        ###
        plt.show()

    def dummy_regression(self):
        dummy_reg = DummyRegressor()
        scores_dummy = cross_validate(
            estimator=dummy_reg,
            X=self.X_train,
            y=self.y_train,
            scoring=["neg_mean_squared_error", "r2"],
            n_jobs=-1,
            cv=Linear_reg.common_parameters["cv"]
        )

        t_zero_fit = time.perf_counter()
        dummy_reg.fit(X=self.X_train, y=self.y_train)
        t_f_fit = time.perf_counter()

        t_zero_predict = time.perf_counter()
        y_pred_dummy = dummy_reg.predict(self.X_test)
        dummy_reg.fit(
                X=self.X_train,
                y=self.y_train
            )
        t_f_predict = time.perf_counter()

        self.dummy_time_card = Time_card(
            t_fit=t_f_fit - t_zero_fit,
            t_predict=t_f_predict - t_zero_predict,
            )

        # Metrics Train
        dummy_mean_mse_train = scores_dummy["test_neg_mean_squared_error"].mean()
        rmse_dummy_train = np.sqrt(abs(dummy_mean_mse_train))
        r2_dummy_train = scores_dummy["test_r2"].mean()
        # /Metrics Train

        # Metrics Test
        dummy_mse_test = metrics.mean_squared_error(y_pred=y_pred_dummy, y_true=self.y_test)
        dummy_rmse_test = np.sqrt(abs(dummy_mse_test))
        std_r2_test = metrics.r2_score(y_pred=y_pred_dummy, y_true=self.y_test)
        self.table_dummy = Linear_reg.get_table(
            scores_train=(rmse_dummy_train, r2_dummy_train),
            scores_test=(dummy_rmse_test, std_r2_test)
            )

    def standard_regression(self):
        """
        Standard ols regression
        """

        self.lin_reg = LinearRegression(fit_intercept=False)

        scores_regression = cross_validate(
                    self.lin_reg,
                    X=self.X_train,
                    y=self.y_train,
                    scoring=["neg_mean_squared_error", "r2"],
                    cv=Linear_reg.common_parameters["cv"],
                    n_jobs=Linear_reg.common_parameters["n_jobs"],
                )
        t_zero_fit = time.perf_counter()
        self.lin_reg.fit(
            X=self.X_train,
            y=self.y_train
        )
        t_f_fit = time.perf_counter()

        t_zero_predict = time.perf_counter()

        y_pred_basic = self.lin_reg.predict(self.X_test)

        t_f_predict = time.perf_counter()

        self.ols_time_card = Time_card(
            t_fit=t_f_fit - t_zero_fit,
            t_predict=t_f_predict - t_zero_predict,
            )

        # Metrics Train
        std_reg_mean_mse_train = scores_regression["test_neg_mean_squared_error"].mean()
        std_mean_rmse_train = np.sqrt(abs(std_reg_mean_mse_train))
        std_mean_r2_train = scores_regression["test_r2"].mean()
        # /Metrics Train
        # Metrics Test
        std_mse_test = metrics.mean_squared_error(y_pred=y_pred_basic, y_true=self.y_test)
        std_rmse_test = np.sqrt(abs(std_mse_test))
        std_r2_test = metrics.r2_score(y_pred=y_pred_basic, y_true=self.y_test)
        # /Metrics Test
        self.std_table = Linear_reg.get_table(
            scores_train=(std_mean_rmse_train, std_mean_r2_train),
            scores_test=(std_rmse_test, std_r2_test)
        )
        self.df_predictions["ols_regression"] = y_pred_basic
        self.std_calc = True

    def use_ridge_cv(self, alphas):

        if Linear_reg.common_parameters["cv"] is None:
            self.ridge_cv = RidgeCV(
                fit_intercept=False,
                alphas=alphas,
                store_cv_values=True,
            )
        else:
            self.ridge_cv = RidgeCV(
                fit_intercept=False,
                alphas=alphas,
                cv=Linear_reg.common_parameters["cv"]
            )

        t_zero_fit = time.perf_counter()

        self.ridge_cv.fit(
            X=self.X_train,
            y=self.y_train
        )

        t_f_fit = time.perf_counter()

        self.ridge_best_alpha = self.ridge_cv.alpha_

        t_zero_predict = time.perf_counter()

        y_predict = self.ridge_cv.predict(X=self.X_test)
        t_f_predict = time.perf_counter()

        self.ridge_time_card = Time_card(
            t_fit=t_f_fit - t_zero_fit,
            t_predict=t_f_predict - t_zero_predict,
            )

        # Metrics Train
        predict_train = self.ridge_cv.predict(X=self.X_train)
        if Linear_reg.common_parameters["cv"] is None:
            mses_ridge = np.mean(self.ridge_cv.cv_values_, axis=0)[0]
            rmses_ridge = np.sqrt(abs(mses_ridge))
        rmse_train = np.sqrt(abs(self.ridge_cv.best_score_))
        r2_train = metrics.r2_score(y_pred=predict_train, y_true=self.y_train)
        # /Metrics Train

        self.df_predictions["Ridge"] = y_predict

        # Metrics Test
        mse_test = metrics.mean_squared_error(y_pred=y_predict, y_true=self.y_test)
        rmse_test = np.sqrt(abs(mse_test))
        r2_test = metrics.r2_score(y_pred=y_predict, y_true=self.y_test)
        # /Metrics Test

        self.ridge_table = Linear_reg.get_table(
            scores_train=(rmse_train, r2_train),
            scores_test=(rmse_test, r2_test)
        )
        if Linear_reg.common_parameters["cv"] is None:
            def get_ridge_plot():
                Linear_reg.get_plot(rmse_list=rmses_ridge, alpha_list=self.ridge_cv.alphas)

            self.ridge_plot = get_ridge_plot

    def use_lasso_cv(self, alphas):
        self.lasso_cv = LassoCV(
            fit_intercept=False,
            alphas=alphas,
            cv=Linear_reg.common_parameters["cv"],
            n_jobs=Linear_reg.common_parameters["n_jobs"]
        )
        t_zero_fit = time.perf_counter()
        self.lasso_cv.fit(
                X=self.X_train,
                y=self.listed_ytrain
            )
        t_f_fit = time.perf_counter()

        self.lasso_best_alpha = self.lasso_cv.alpha_

        t_zero_predict = time.perf_counter()
        y_predict = self.lasso_cv.predict(X=self.X_test)
        t_f_predict = time.perf_counter()

        self.lasso_time_card = Time_card(
            t_fit=t_f_fit - t_zero_fit,
            t_predict=t_f_predict - t_zero_predict
            )

        # Metrics Train
        predict_train = self.lasso_cv.predict(X=self.X_train)
        mses = self.lasso_cv.mse_path_
        mse_avg = []

        for mse_list in mses:
            mse_avg.append(np.mean(mse_list))

        rmses_lasso = np.sqrt(mse_avg)
        rmse_train = np.min(rmses_lasso)
        r2_train = metrics.r2_score(y_pred=predict_train, y_true=self.listed_ytrain)
        # /Metrics Train

        # Metrics Test
        mse_test = metrics.mean_squared_error(y_pred=y_predict, y_true=self.y_test)
        rmse_test = np.sqrt(abs(mse_test))
        r2_test = metrics.r2_score(y_pred=y_predict, y_true=self.y_test)
        # /Metrics Test

        self.lasso_table = Linear_reg.get_table(
            scores_train=(rmse_train, r2_train),
            scores_test=(rmse_test, r2_test)
        )

        def get_lasso_plot():
            Linear_reg.get_plot(rmse_list=mse_avg, alpha_list=self.lasso_cv.alphas_)

        self.lasso_plot = get_lasso_plot

        self.df_predictions["Lasso"] = y_predict

    def use_elnet(self, alphas: list):

        l1_range = np.arange(0.01, 0.99, 0.05)
        self.elnet_cv = ElasticNetCV(
            fit_intercept=False,
            l1_ratio=l1_range,
            alphas=alphas,
            n_jobs=Linear_reg.common_parameters["n_jobs"],
            cv=Linear_reg.common_parameters["cv"]
        )

        t_zero_fit = time.perf_counter()
        self.elnet_cv.fit(
            X=self.X_train,
            y=self.listed_ytrain
        )
        t_f_fit = time.perf_counter()

        t_zero_predict = time.perf_counter()
        y_predict = self.elnet_cv.predict(X=self.X_test)
        t_f_predict = time.perf_counter()

        self.elnet_time_card = Time_card(
            t_fit=t_f_fit - t_zero_fit,
            t_predict=t_f_predict - t_zero_predict
            )

        self.enet_best_l1_ratio = self.elnet_cv.l1_ratio_
        self.enet_best_alpha = self.elnet_cv.alpha_
        y_pred_enet = self.elnet_cv.predict(self.X_test)
        # Metrics Train
        predict_train = self.elnet_cv.predict(X=self.X_train)
        mses = self.elnet_cv.mse_path_
        mse_avg = []

        for mse_list in mses:
            mse_avg.append(np.mean(mse_list))

        rmses_elnet = np.sqrt(mse_avg)
        rmse_train = np.min(rmses_elnet)
        r2_train = metrics.r2_score(y_pred=predict_train, y_true=self.listed_ytrain)
        # /Metrics Train

        # Metrics Test
        mse_test = metrics.mean_squared_error(y_pred=y_predict, y_true=self.y_test)
        rmse_test = np.sqrt(abs(mse_test))
        r2_test = metrics.r2_score(y_pred=y_predict, y_true=self.y_test)
        # /Metrics Test

        self.elnet_table = Linear_reg.get_table(
            scores_train=(rmse_train, r2_train),
            scores_test=(rmse_test, r2_test)
        )

        self.df_predictions["Elastic_Net"] = y_pred_enet

        self.enet_calc = True

    def execute_all(self, alphas_ridge=None, alphas_elnet=None, alphas_lasso=None):
        self.dummy_regression()
        self.standard_regression()
        if alphas_elnet is None:
            elnet_alpha_start = float(input("Elastic Net alpha start : "))
            elnet_alpha_end = float(input("Elastic Net alpha end : "))
            elnet_alpha_step = float(input("Elastic Net alpha step : "))
            elnet_range = np.arange(elnet_alpha_start, elnet_alpha_end, elnet_alpha_step)
        elif alphas_elnet is not None:
            elnet_range = alphas_elnet
        self.use_elnet(alphas=elnet_range)

        if alphas_ridge is None:
            ridge_alpha_start = float(input("Ridge alpha start : "))
            ridge_alpha_end = float(input("Ridge alpha end : "))
            ridge_alpha_step = float(input("Ridge alpha step : "))
            ridge_range = np.arange(ridge_alpha_start, ridge_alpha_end, ridge_alpha_step)
        elif alphas_ridge is not None:
            ridge_range = alphas_ridge
        self.use_ridge_cv(ridge_range)

        if alphas_lasso is None:
            lasso_alpha_start = float(input("Lasso alpha start : "))
            lasso_alpha_end = float(input("Lasso alpha end : "))
            lasso_alpha_step = float(input("Lasso alpha step : "))
            lasso_range = np.arange(lasso_alpha_start, lasso_alpha_end, lasso_alpha_step)
        elif alphas_lasso is not None:
            lasso_range = alphas_lasso
        self.use_lasso_cv(lasso_range)

    def format_all_metrics(self, include_dummy: bool = False):
        """
        Joins all metrics of linear regression in current state.
        Renames columns from metric_name to metric_name_regression_method
        - Assigns to self.all_metrics

        Args :
        - include_dummy : bool, default False : Include dummy_regression stats in the table

        Returns :
        - self.all_metrics
        """
        ols_t = self.std_table.rename(columns={"RMSE": "RMSE_OLS", "R2": "R2_OLS"})
        ridge_t = self.ridge_table.rename(columns={"RMSE": "RMSE_Ridge", "R2": "R2_Ridge"})
        lasso_t = self.lasso_table.rename(columns={"RMSE": "RMSE_LASSO", "R2": "R2_LASSO"})
        elnet_t = self.elnet_table.rename(columns={"RMSE": "RMSE_Elastic_Net", "R2": "R2_Elastic_Net"})
        if include_dummy:
            dummy_t = self.table_dummy.rename(columns={"RMSE": "RMSE_Dummy", "R2": "R2_Dummy"})

        all_metrics = ols_t.join(ridge_t, how="right")
        all_metrics = all_metrics.join(lasso_t, how="right")
        all_metrics = all_metrics.join(elnet_t, how="right")

        if include_dummy:
            all_metrics = all_metrics.join(dummy_t, how="right")

        self.all_metrics = all_metrics
        return self.all_metrics

    def get_common_params(self):
        """
        Prints current state of common parameters
        """
        for key in Linear_reg.common_parameters.keys():
            print(f"{key} = {Linear_reg.common_parameters[key]}")
