# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:44:16 2022

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 50 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

plt.show()

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import neural_network, linear_model, metrics, tree, ensemble
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

df=pd.read_excel('Taxi_Trip_Data.xlsx')
df=df.drop(['store_and_fwd_flag','PULocationID','DOLocationID','fare_amount','extra','mta_tax','tip_amount','tolls_amount'], axis=1)

df=pd.get_dummies(df,'DOBorough')
df=pd.get_dummies(df,'PUBorough')

df['PU_year'] = df['lpep_pickup_datetime'].dt.year
df['PU_month'] = df['lpep_pickup_datetime'].dt.month
df['PU_day'] = df['lpep_pickup_datetime'].dt.day
df['PU_hour'] = df['lpep_pickup_datetime'].dt.hour
df['PU_minute'] = df['lpep_pickup_datetime'].dt.minute
df['PU_dayofweek'] = df['lpep_pickup_datetime'].dt.dayofweek

df['DO_year'] = df['lpep_dropoff_datetime'].dt.year
df['DO_month'] = df['lpep_dropoff_datetime'].dt.month
df['DO_day'] = df['lpep_dropoff_datetime'].dt.day
df['DO_hour'] = df['lpep_dropoff_datetime'].dt.hour
df['DO_minute'] = df['lpep_dropoff_datetime'].dt.minute
df['DO_dayofweek'] = df['lpep_dropoff_datetime'].dt.dayofweek

df=df.drop(['lpep_pickup_datetime','lpep_dropoff_datetime'], axis=1)

corr=df.corr()
df=(df-df.min())/(df.max()-df.min())
x=np.array(df.drop(['total_amount'], axis=1))
y=np.array(df['total_amount'])
(x_train, x_test, y_train, y_test) = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=22222)

# Linear Regression
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(x_train,y_train)
y_pred_lrm=linear_regression_model.predict(x_test)
R2_lrm=linear_regression_model.score(x_test,y_test)
mae_lrm=metrics.mean_absolute_error(y_test,y_pred_lrm)
evs_lrm=metrics.explained_variance_score(y_test,y_pred_lrm)
mse_lrm=metrics.mean_squared_error(y_test,y_pred_lrm)

# Neural Network Regression
neural_network_model = neural_network.MLPRegressor(hidden_layer_sizes=(10,10,10), activation='identity',
learning_rate='adaptive', tol=0.0001, solver='adam', alpha=0.0001, early_stopping=True, validation_fraction=0.1)
neural_network_model.fit(x_train, y_train)
y_pred_cnn=neural_network_model.predict(x_test)
R2_cnn=neural_network_model.score(x_test,y_test)
mae_cnn=metrics.mean_absolute_error(y_test, y_pred_cnn)
evs_cnn=metrics.explained_variance_score(y_test,y_pred_cnn)
mse_cnn=metrics.mean_squared_error(y_test,y_pred_cnn)

# Regression Decision Tree
regression_tree_model=tree.DecisionTreeRegressor(max_depth=3)
regression_tree_model.fit(x_train,y_train)
y_pred_rtm=regression_tree_model.predict(list(x_test),list(y_test))
R2_rtm=metrics.r2_score(y_test,y_pred_rtm)
mae_rtm=metrics.mean_absolute_error(y_test,y_pred_rtm)
evs_rtm=metrics.explained_variance_score(y_test,y_pred_rtm)
mse_rtm=metrics.mean_squared_error(y_test,y_pred_rtm)

x=np.column_stack((x_test,y_pred_lrm,y_pred_cnn,y_pred_rtm))
y=y_test
(x_train, x_test, y_train, y_test) = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=22222)

#Final Stage
random_forest_model=ensemble.RandomForestRegressor(random_state=22222)
plot_learning_curve(estimator=random_forest_model,X=x_train,y=y_train,cv=5, title="Multistage Model")
random_forest_model.fit(x_train,y_train)
y_pred_rfr=random_forest_model.predict(x_test)
R2_rfr=metrics.r2_score(y_test,y_pred_rfr)
mae_rfr=metrics.mean_absolute_error(y_test,y_pred_rfr)
mse_rfr=metrics.mean_squared_error(y_test,y_pred_rfr)
evs_rfr=metrics.explained_variance_score(y_test, y_pred_rfr)


# plt.plot(y_pred_rfr,x_test[:,5], alpha=0.3)
# plt.scatter(y_test,x_test[:,5], alpha =0.09)
# # plt.scatter(y_test,y_pred_rfr)
# plt.show()

# plt.figure(figsize=(10,10))
# plt.scatter(y_test, y_pred_rfr, c='crimson')

# p1 = max(max(y_pred_rfr), max(y_test))
# p2 = min(min(y_pred_rfr), min(y_test))
# plt.plot([p1,p2], [p1,p2], 'b-')
# plt.xlabel('True Values', fontsize=15)
# plt.ylabel('Predictions', fontsize=15)
# plt.axis('equal')
# plt.show()


# x=np.column_stack((x_test,y_pred_lrm,y_pred_cnn,y_pred_rtm))
# y=y_test
# (x_train, x_test, y_train, y_test) = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=22222)
# cv = ShuffleSplit(test_size=0.2, random_state=0)
# random_forest_model=ensemble.RandomForestRegressor(random_state=22222)




