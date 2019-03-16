# -*- coding: utf-8 -*-

"""
Created Feb 8, 2019
@author: Alex Truesdale
@email: alex.truesdale@colorado.edu

Container for modeling functions to be used in automated ensembling.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import operator

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, Normalizer

def predictor_nn(data_in, unknown, batch_size = 128, capacity = 156, activation = 'sigmoid'):
    """Neural network model wrapper; returns prediction array and iteration AUC score."""

    # Define splits.
    X_train = data_in[0]
    X_test = data_in[1]
    y_train = data_in[2]
    y_test = data_in[3]

    # Define optimizer algorithm and callback(s).
    optimizer = keras.optimizers.Adam()
    callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 6, verbose = 0,
                                             min_delta = 1e-5, restore_best_weights = True)

    # Initialise network (as Sequential); add layers.
    network = Sequential()
    network.add(Dense(capacity, activation = activation, input_dim = len(X_test.columns)))
    network.add(Dense(45, activation = activation))
    network.add(Dense(1, activation = activation))

    # Compile network.
    network.compile(optimizer = optimizer,
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

    # And fit it.
    network.fit(X_train, y_train, epochs = 40, batch_size = batch_size,
                callbacks = [callback], validation_split = 0.12,
                shuffle = True, verbose = 0)

    # Test block for NN model on test data.
    predictions_nn_real = network.predict(unknown)
    predictions_test_nn = network.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_test_nn)
    roc_auc_nn = auc(false_positive_rate, true_positive_rate)
    predictions_nn_out = np.asarray(predictions_test_nn)

    print('Network done. Test fit: {}'.format(roc_auc_nn))
    return (predictions_nn_out.flatten(), predictions_nn_real.flatten(), roc_auc_nn)

def predictor_xgb(data_in, unknown, n_estimators = 500, max_depth = 3, learning_rate = 0.01,
                  colsample_bytree = 0.5, min_child_weight = 7):

    """XGBoost model wrapper; returns prediction array and iteration AUC score."""

    # Define splits.
    X_train = data_in[0]
    X_test = data_in[1]
    y_train = data_in[2]
    y_test = data_in[3]

    # And hyperparameters.
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
        'verbosity': 0,
        'verbose': 0,
        'silent': 1,
        'verbose_eval': False
    }

    # Define classifier and fit it.
    xg_classifier = xgb.XGBClassifier(**hyper_parameters)
    xg_classifier.fit(X_train, y_train)

    # Create predictions for fold and unknown data, calculating fold AUC.
    predictions_xbg_real = xg_classifier.predict_proba(unknown)[:, 1]
    predictions_xgb_out = xg_classifier.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_xgb_out)
    roc_auc_xgb = auc(false_positive_rate, true_positive_rate)

    print('XGBoost done. Test fit: {}'.format(roc_auc_xgb))
    return (predictions_xgb_out, predictions_xbg_real, roc_auc_xgb)

def predictor_rf(data_in, unknown, n_estimators, min_samples_split, min_samples_leaf, max_depth):
    """Random Forest model wrapper; returns prediction array and iteration AUC score."""

    # Define splits.
    X_train = data_in[0]
    X_test = data_in[1]
    y_train = data_in[2]
    y_test = data_in[3]

    # And hyperparameters.
    hyper_parameters = {
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': 'auto',
        'max_depth': max_depth,
        'bootstrap': True,
        'verbose': 0
    }

    # Define classifier and fit it.
    random_forest = RandomForestClassifier(**hyper_parameters)
    random_forest.fit(X_train, y_train)

    # Create predictions for fold and unknown data, calculating fold AUC.
    predictions_rf_real = random_forest.predict_proba(unknown)[:, 1]
    predictions_rf_out = random_forest.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_rf_out)
    roc_auc_rf = auc(false_positive_rate, true_positive_rate)

    print('Random Forest done. Test fit: {}'.format(roc_auc_rf))
    return (predictions_rf_out, predictions_rf_real, roc_auc_rf)

def model_aggregator(split, unknown):
    """Runs multiple algorithms and save results in dict."""

    # Initialise empty model_dict.
    model_dict = {}

    # Run NNs
    model_dict.update({'nn_01': predictor_nn(split, unknown)})
    model_dict.update({'nn_02': predictor_nn(split, unknown, batch_size = 180, capacity = 80)})
    model_dict.update({'nn_03': predictor_nn(split, unknown, batch_size = 180, capacity = 124)})
    model_dict.update({'nn_04': predictor_nn(split, unknown, batch_size = 256, capacity = 124)})
    model_dict.update({'nn_05': predictor_nn(split, unknown, batch_size = 256, capacity = 192)})

    # Run XGBs
    model_dict.update({'xgb_00': predictor_xgb(split, unknown, n_estimators = 550, max_depth = 4, learning_rate = .03, colsample_bytree = .5)})
    model_dict.update({'xgb_01': predictor_xgb(split, unknown, n_estimators = 450, max_depth = 5, learning_rate = .05, colsample_bytree = .7)})
    model_dict.update({'xgb_02': predictor_xgb(split, unknown, n_estimators = 500, max_depth = 5, learning_rate = .05, colsample_bytree = .6)})
    model_dict.update({'xgb_03': predictor_xgb(split, unknown, n_estimators = 550, max_depth = 5, learning_rate = .05, colsample_bytree = .6)})
    model_dict.update({'xgb_04': predictor_xgb(split, unknown, n_estimators = 500, max_depth = 5, learning_rate = .05, colsample_bytree = .7)})
    model_dict.update({'xgb_05': predictor_xgb(split, unknown, n_estimators = 500, max_depth = 5, learning_rate = .05, colsample_bytree = .8)})
    model_dict.update({'xgb_06': predictor_xgb(split, unknown, n_estimators = 550, max_depth = 5, learning_rate = .05, colsample_bytree = .8)})
    model_dict.update({'xgb_07': predictor_xgb(split, unknown, n_estimators = 600, max_depth = 5, learning_rate = .05, colsample_bytree = .8)})
    model_dict.update({'xgb_08': predictor_xgb(split, unknown, n_estimators = 600, max_depth = 5, learning_rate = .05, colsample_bytree = .9)})

    # Run RFs
    model_dict.update({'rf_00': predictor_rf(split, unknown, n_estimators = 700, min_samples_split = 2, min_samples_leaf = 2, max_depth = 15)})
    model_dict.update({'rf_01': predictor_rf(split, unknown, n_estimators = 650, min_samples_split = 3, min_samples_leaf = 3, max_depth = 15)})
    model_dict.update({'rf_02': predictor_rf(split, unknown, n_estimators = 650, min_samples_split = 3, min_samples_leaf = 3, max_depth = 20)})
    model_dict.update({'rf_03': predictor_rf(split, unknown, n_estimators = 650, min_samples_split = 3, min_samples_leaf = 3, max_depth = 25)})
    model_dict.update({'rf_04': predictor_rf(split, unknown, n_estimators = 600, min_samples_split = 4, min_samples_leaf = 4, max_depth = 20)})
    model_dict.update({'rf_05': predictor_rf(split, unknown, n_estimators = 650, min_samples_split = 4, min_samples_leaf = 4, max_depth = 15)})

    return model_dict

def ensembler(data_in, model_dict, winner):
    """Ensembling pipeline."""

    # ID y_test.
    y_test = data_in[3]

    # Perform ensembling.
    for i in range(0, 100):
        print(i)

        # Initialise dict for all ensembling permutations.
        # Average winning model with all models and store AUC and predictor vectors.
        ensembling_dict = {}
        for model, data in list(model_dict.items()):
            ensemble_name = winner + '_' + model
            ensemble_scores = np.mean(np.array([model_dict[winner][0], data[0]]), axis = 0)
            unknown_scores = np.mean(np.array([model_dict[winner][1], data[1]]), axis = 0)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ensemble_scores)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            ensembling_dict.update({ensemble_name: [ensemble_scores, unknown_scores, roc_auc]})

        # Re-ID winning model or ensemble from last round of local ensembling.
        # Add winner to aggregate model / ensemble dict.
        winner = max(ensembling_dict, key = lambda k: operator.itemgetter(2)(ensembling_dict[k]))
        model_dict[winner] = tuple(ensembling_dict[winner])
        winner_total = max(model_dict, key = lambda k: operator.itemgetter(2)(model_dict[k]))

        # If the iteration does not improve AUC, record winner data and exit process.
        if winner == winner_total:
            print(winner_total)
            print(model_dict[winner_total][2])
            print('continuing..')
            continue
        else:
            print(winner_total)
            print(model_dict[winner_total][2])
            predictions_out = model_dict[winner_total][0]
            predictions_real = model_dict[winner_total][1]
            roc_ensemble = model_dict[winner_total][2]
            out_tuple = (predictions_out, predictions_real, roc_ensemble)
            break

    return out_tuple
