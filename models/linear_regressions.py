import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn import metrics
from matplotlib import pyplot as plt


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
            override_common: dict = None, split: float = 0.3,
            ignore_columns: list = None
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

        self.override = override_common
        self.std_calc, self.ridge_calc, self.lasso_calc, self.enet_calc = False, False, False, False
        self.listed_ytrain = [value[0] for value in self.y_train]
        self.listed_ytest = [value[0] for value in self.y_test]
        self.df_predictions = pd.DataFrame({"True": self.listed_ytest})

        # Dummy
        dummy_reg = DummyRegressor()
        scores_dummy = cross_validate(
            estimator=dummy_reg,
            X=self.X_train,
            y=self.y_train,
            scoring=["neg_mean_squared_error", "r2"],
            n_jobs=-1,
        )

        dummy_reg.fit(X=self.X_train, y=self.y_train)

        y_pred_dummy = dummy_reg.predict(self.X_test)
        dummy_reg.fit(
                X=self.X_train,
                y=self.y_train
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
        # /Dummy

    def drop_col(self, col_list):
        """
        can be used to drop a column of the train/test set and keep the same data repartition
         to minimize split randomization
        """
        self.X_train = self.df_train.drop(columns=[self.target + col_list]).to_numpy()
        self.X_test = self.df_test.drop(columns=[self.target + col_list]).to_numpy()

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
                [scores_train[0], scores_test[0]],
                [scores_train[1], scores_test[1]],
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
            ax1.plot(alpha_list, rmse_list)

            plt.show()

            if labels is not None:
                try:
                    ax1.set_xlabel = (labels["x_label"])
                    ax1.set_ylabel = (labels["y_label"])
                    fig.suptitle(labels["title"])
                except KeyError:
                    print("Please provide all keys values in labels : x_label, y_label, title")
                    pass

        else:
            print("""Cannot use plot on ridge regression with CV not None, cf. sklearn doc\
 on "store_cv_values" on RidgeCV""")

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
        self.lin_reg.fit(
            X=self.X_train,
            y=self.y_train
        )
        # Metrics Train
        std_reg_mean_mse_train = scores_regression["test_neg_mean_squared_error"].mean()
        std_mean_rmse_train = np.sqrt(abs(std_reg_mean_mse_train))
        std_mean_r2_train = scores_regression["test_r2"].mean()
        # /Metrics Train
        y_pred_basic = self.lin_reg.predict(self.X_test)
        # Metrics Test
        std_mse_test = metrics.mean_squared_error(y_pred=y_pred_basic, y_true=self.y_test)
        std_rmse_test = np.sqrt(abs(std_mse_test))
        std_r2_test = metrics.r2_score(y_pred=y_pred_basic, y_true=self.y_test)
        # /Metrics Test
        self.std_table = Linear_reg.get_table(
            scores_train=(std_mean_rmse_train, std_mean_r2_train),
            scores_test=(std_rmse_test, std_r2_test)
        )
        self.df_predictions["basic_regression"] = y_pred_basic
        self.std_calc = True

    def use_ridge_cv(self, alphas):

        if self.override is None:
            self.ridge_cv = RidgeCV(
                fit_intercept=False,
                alphas=alphas,
                store_cv_values=True,
            )
        else:
            self.ridge_cv = RidgeCV(
                fit_intercept=False,
                alphas=alphas,
                cv=self.overide["cv"]
            )

        self.ridge_cv.fit(
            X=self.X_train,
            y=self.y_train
        )

        self.ridge_best_alpha = self.ridge_cv.alpha_

        # Metrics Train
        predict_train = self.ridge_cv.predict(X=self.X_train)
        if self.override is None or self.override["cv"] is None:
            mses_ridge = np.mean(self.ridge_cv.cv_values_, axis=0)[0]
            rmses_ridge = np.sqrt(abs(mses_ridge))
        rmse_train = np.sqrt(abs(self.ridge_cv.best_score_))
        r2_train = metrics.r2_score(y_pred=predict_train, y_true=self.y_train)
        # /Metrics Train

        y_predict = self.ridge_cv.predict(X=self.X_test)

        # Metrics Test
        mse_test = metrics.mean_squared_error(y_pred=y_predict, y_true=self.y_test)
        rmse_test = np.sqrt(abs(mse_test))
        r2_test = metrics.r2_score(y_pred=y_predict, y_true=self.y_test)
        # /Metrics Test

        self.ridge_table = Linear_reg.get_table(
            scores_train=(rmse_train, r2_train),
            scores_test=(rmse_test, r2_test)
        )
        if self.override is None or self.override["cv"] is None:
            def get_ridge_plot():
                Linear_reg.get_plot(rmse_list=rmses_ridge, alpha_list=self.ridge_cv.alphas)

            self.ridge_plot = get_ridge_plot

    def use_lasso_cv(self, alphas, cv_method=10):
        self.lasso_cv = LassoCV(
            fit_intercept=False,
            alphas=alphas,
            cv=cv_method,
            n_jobs=Linear_reg.common_parameters["n_jobs"]
        )
        self.lasso_cv.fit(
                X=self.X_train,
                y=self.listed_ytrain
            )

        self.lasso_best_alpha = self.lasso_cv.alpha_

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

        y_predict = self.lasso_cv.predict(X=self.X_test)

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

    def use_elnet(self, override_default: dict = None, alphas: list):

        enet_parameters = Linear_reg.common_parameters
        print("Step : Elastic Net")
        l1_range = np.arange(0.01, 0.99, 0.05)
        self.elnet_cv = ElasticNetCV(
            l1_ratio=l1_range,
            alphas=alphas,
            n_jobs=enet_parameters["n_jobs"],
            fit_intercept=False,

        )

        self.elnet_cv.fit(
            X=self.X_train,
            y=self.listed_ytrain
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

        y_predict = self.elnet_cv.predict(X=self.X_test)

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

    def execute_all(self):
        self.standard_regression()
        elnet_alpha_start = float(input("Elastic Net alpha start : "))
        elnet_alpha_end = float(input("Elastic Net alpha end : "))
        elnet_alpha_step = float(input("Elastic Net alpha step : "))
        elnet_range = np.arange(elnet_alpha_start, elnet_alpha_end, elnet_alpha_step)
        self.use_elnet(alphas=elnet_range)
        ridge_alpha_start = float(input("Ridge alpha start : "))
        ridge_alpha_end = float(input("Ridge alpha end : "))
        ridge_alpha_step = float(input("Ridge alpha step : "))
        ridge_range = np.arange(ridge_alpha_start, ridge_alpha_end, ridge_alpha_step)
        self.use_ridge_cv(ridge_range)
        lasso_alpha_start = float(input("Lasso alpha start : "))
        lasso_alpha_end = float(input("Lasso alpha end : "))
        lasso_alpha_step = float(input("Lasso alpha step : "))
        lasso_range = np.arange(lasso_alpha_start, lasso_alpha_end, lasso_alpha_step)
        self.use_lasso_cv(lasso_range)
