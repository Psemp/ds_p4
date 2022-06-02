import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn import metrics
from matplotlib import pyplot as plt


class Linear_reg():

    common_parameters = {
        "scoring": "neg_mean_squared_error",
        "cv": None,  # Default = Leave One Out
        "n_jobs": -1  # Use all cores
    }

    def __init__(self, dataframe: pd.DataFrame, target: str, override_common: dict = None, split: float = None):
        self.df_origin = dataframe

        ##
        # Train test split
        self.df_train, self.df_test = train_test_split(self.df_origin, test_size=split)
        self.X_train = self.df_train.drop(columns=target).to_numpy()
        self.X_test = self.df_test.drop(columns=target).to_numpy()

        self.y_train = self.df_train[[target]].to_numpy()
        self.y_test = self.df_test[[target]].to_numpy()
        #
        ##

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

    def get_table(scores_train, scores_test):
        # Tuple shape : [0] = RMSE, [1] = R2
        return pd.DataFrame(
            columns=["RMSE", "R2"],
            data=[[scores_train[0], scores_test[0]], [scores_train[1], scores_test[1]]],
            index=["Train", "Test"]
        )

    def get_plot(rmse_list, alpha_list):
        ax = plt.plot(alpha_list, rmse_list)
        return ax

    def standard_regression(self):

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

    def ridge_cv_oneout(self, alphas):
        """
        Used to cross validate and find best hyperparameters
        for ridge and lasso !only!, elasticNet is too different
        to be generalized
        """
        self.ridge_cv = RidgeCV(
            fit_intercept=False,
            alphas=alphas,
            store_cv_values=True,
        )

        self.ridge_cv.fit(
            X=self.X_train,
            y=self.y_train
        )

        self.ridge_best_alpha = self.ridge_cv.alpha_

        # Metrics Train
        predict_train = self.ridge_cv.predict(X=self.X_train)
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
        self.ridge_ax = Linear_reg.get_plot(rmse_list=rmses_ridge, alpha_list=self.ridge_cv.alphas)
        self.df_predictions["Ridge"] = y_predict

    def lasso_cv_oneout(self, alphas, cv_method=10):
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
        self.lasso_ax = Linear_reg.get_plot(rmse_list=mse_avg, alpha_list=self.lasso_cv.alphas_)
        self.df_predictions["Lasso"] = y_predict

    def elastic_net_cv_oneout(self, override_default: dict = None):

        enet_parameters = Linear_reg.common_parameters
        print("Step : Elastic Net")
        l1_range = np.arange(0.01, 0.99, 0.05)
        self.elnet_cv = ElasticNetCV(
            l1_ratio=l1_range,
            n_alphas=150,
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
        self.elastic_net_cv_oneout()
        ridge_alpha_start = input("Ridge alpha start : ")
        ridge_alpha_end = input("Ridge alpha end : ")
        ridge_alpha_step = input("Ridge alpha step : ")
        ridge_range = np.arange(ridge_alpha_start, ridge_alpha_end, ridge_alpha_step)
        self.ridge_cv_oneout(ridge_range)
        lasso_alpha_start = input("Lasso alpha start : ")
        lasso_alpha_end = input("Lasso alpha end : ")
        lasso_alpha_step = input("Lasso alpha step : ")
        lasso_range = np.arange(lasso_alpha_start, lasso_alpha_end, lasso_alpha_step)
        self.lasso_cv_oneout(lasso_range)
