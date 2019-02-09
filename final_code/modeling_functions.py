# -*- coding: utf-8 -*-

"""
Created Feb 8, 2019
@author: Alex Truesdale

Container for modeling functions to be used in automated ensembling.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, Normalizer

class modeler(object):
    """
    Class containing modeling functions for XGBoost, Random Forest, and a Neural Network.

    Attributes:
        self.X_train: predictor features for testing of the model.
        self.X_test: predictor features for evaulation of the model.
        self.y_train: target for testing of the model.
        self.y_test: target for evaulation of the model.
        unknown: unknown data.
    """

    def __init__(self, X, y, unknown):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
        self.unknown = unknown

    def predictor_nn(self, batch_size = 128, capacity = 156, activation = 'sigmoid'):
        """Neural network model wrapper; returns prediction array and iteration AUC score."""

        # Define optimizer algorithm and callback(s).
        optimizer = keras.optimizers.Adam()
        callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 6, verbose = 0,
                                                 min_delta = 1e-5, restore_best_weights = True)

        # Initialise network (as Sequential); add layers.
        network = Sequential()
        network.add(Dense(capacity, activation = activation, input_dim = len(self.X_test.columns)))
        network.add(Dense(45, activation = activation))
        network.add(Dense(1, activation = activation))

        # Compile network.
        network.compile(optimizer = optimizer,
                        loss = 'binary_crossentropy',
                        metrics = ['accuracy'])

        # And fit it.
        network.fit(self.X_train, self.y_train, epochs = 40, batch_size = batch_size,
                    callbacks = [callback], validation_split = 0.12,
                    shuffle = True, verbose = 0)

        # Test block for NN model on test data.
        predictions_nn_real = network.predict(self.unknown)
        predictions_test_nn = network.predict(self.X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, predictions_test_nn)
        roc_auc_nn = auc(false_positive_rate, true_positive_rate)
        predictions_nn_out = np.asarray(predictions_test_nn)

        print()
        print('Network done. Test fit:{}'.format(roc_auc_nn))
        return (predictions_nn_out.flatten(), predictions_nn_real.flatten(), roc_auc_nn)

    def predictor_xgb(self, n_estimators = 500, max_depth = 3, learning_rate = 0.01,
                      colsample_bytree = 0.5, min_child_weight = 7):

        """XGBoost model wrapper; returns prediction array and iteration AUC score."""

        hyper_parameters = {
            'silent': False,
            'min_child_weight': min_child_weight,
            'scale_pos_weight': 1,
            'learning_rate': learning_rate,
            'colsample_bytree': colsample_bytree,
            'subsample': 0.8,
            'objective': 'binary:logistic',
            'n_estimators': n_estimators,
            'reg_alpha': 0.3,
            'max_depth': max_depth,
            'gamma': 0.3,
            'nthread': 1,
            'verbose': 0
        }

        xg_classifier = xgb.XGBClassifier(**hyper_parameters)
        xg_classifier.fit(self.X_train, self.y_train)

        predictions_xbg_real = xg_classifier.predict_proba(self.unknown)[:, 1]
        predictions_xgb_out = xg_classifier.predict_proba(self.X_test)[:, 1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, predictions_xgb_out)
        roc_auc_xgb = auc(false_positive_rate, true_positive_rate)

        print()
        print('XGBoost done. Test fit:{}'.format(roc_auc_xgb))
        return (predictions_xgb_out, predictions_xbg_real, roc_auc_xgb)

    def predictor_rf(self, n_estimators, min_samples_split, min_samples_leaf, max_depth):
        """Random Forest model wrapper; returns prediction array and iteration AUC score."""

        hyper_parameters = {
            'n_estimators': n_estimators,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': 'auto',
            'max_depth': max_depth,
            'bootstrap': True,
            'verbose': 0
        }

        random_forest = RandomForestClassifier(**hyper_parameters)
        random_forest.fit(self.X_train, self.y_train)

        predictions_rf_real = random_forest.predict_proba(self.unknown)[:, 1]
        predictions_rf_out = random_forest.predict_proba(self.X_test)[:, 1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, predictions_rf_out)
        roc_auc_rf = auc(false_positive_rate, true_positive_rate)

        print()
        print('Random Forest done. Test fit:{}'.format(roc_auc_rf))
        return (predictions_rf_out, predictions_rf_real, roc_auc_rf)
