# -*- coding: utf-8 -*-

"""
Created Feb 8, 2019
@author: Alex Truesdale

Jupyter Notebook Kernel (via Hydrogen / Atom) for modeling on prepared challenge data.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import operator

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

# Simplify DataFrame build function.
df = pd.DataFrame

# Read in saved features / frames.
known = pd.read_csv('modified_features/known_modified_final.csv')
unknown = pd.read_csv('modified_features/unknown_modified_final.csv')

# Store order IDs and remove columns.
known_order_id = known['order_item_id']
unknown_order_id = unknown['order_item_id']

known = known.drop(['order_item_id'], axis = 1)
unknown = unknown.drop(['order_item_id'], axis = 1)

known = known.drop(['item_id'], axis = 1)
unknown = unknown.drop(['item_id'], axis = 1)

known = known.drop(['brand_id'], axis = 1)
unknown = unknown.drop(['brand_id'], axis = 1)

known = known.drop(['user_id'], axis = 1)
unknown = unknown.drop(['user_id'], axis = 1)

known = known.drop(['Unnamed: 0'], axis = 1)
unknown = unknown.drop(['Unnamed: 0'], axis = 1)

# Reorder columns.
columns = list(known.columns.values)
columns.pop(columns.index('return'))
known = known[columns + ['return']]
columns = list(known.columns.values)
columns

# Split target.
X, y = known.iloc[:, :-1], known.iloc[:, -1].values

# Split test / train data; create dmatrices.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

######################################################################################################
####
######## XGBoost
####
######################################################################################################

# Define parametres for RandomSearch.
folds = 5
parameter_iteration = 20
parameter_grid = {
     'learning_rate' : [0.01, 0.03, 0.05, 0.10, 0.15, 0.20],
     'max_depth' : [3, 4, 5, 6, 8],
     'min_child_weight' : [1, 3, 5, 7],
     'gamma' : [0.0, 0.1, 0.2, 0.3, 0.4],
     'colsample_bytree' : [0.3, 0.4, 0.5, 0.7]
 }

# Define stratified K-folds fold-er.
skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 1001)

# Define XGB classifier.
xgb_c = xgb.XGBClassifier(n_estimators = 500, objective = 'binary:logistic',
                          silent = True, subsample = 0.8, nthread = 1)

# Define random search model.
random_search_xgb = RandomizedSearchCV(xgb_c, param_distributions = parameter_grid,
                                       n_iter = parameter_iteration, scoring = 'roc_auc',
                                       n_jobs = 4, cv = skf.split(X, y), verbose = 3,
                                       random_state = 1001)
# And fit it.
random_search_xgb.fit(X, y)

# Show output.
random_search_xgb.best_score_
random_search_xgb.best_params_
random_search_xgb.cv_results_

# Define final model w/ best parametres.
xg_classifier = xgb.XGBClassifier(silent = False,
                                  min_child_weight = 7,
                                  scale_pos_weight = 1,
                                  learning_rate = 0.01,
                                  colsample_bytree = 0.5,
                                  subsample = 0.8,
                                  objective = 'binary:logistic',
                                  n_estimators = 500,
                                  reg_alpha = 0.3,
                                  max_depth = 3,
                                  gamma = 0.3,
                                  nthread = 1)

# And fit it.
xg_classifier.fit(X_train, y_train)

# Test block for XGB model on test data.
predictions_test_xgb = xg_classifier.predict_proba(X_test)[:, 1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_test_xgb)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# Prediction block for unknown data points.
predictions_real_xgb = xg_classifier.predict_proba(unknown)[:, 1]
prediction_frame = df(unknown_order_id, columns = ['order_item_id'])
prediction_frame['return'] = predictions_real_xgb
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/predictions/single_predictors/prediction_xgb.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)

# Feature importance table.
print('Feature importance Table\n',
      pd.DataFrame(xg_classifier.feature_importances_[:],
      unknown.columns[:]).sort_values(by = 0, ascending = False)[:32])

######################################################################################################
####
######## RANDOM FOREST
####
######################################################################################################

# Define parametres for RandomSearch.
folds = 5
parameter_iteration = 20

n_estimators = [int(x) for x in np.linspace(start = 400, stop = 1000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [5, 10, 15, 20, 25]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Define stratified K-folds fold-er.
skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 1001)

# Define RF classifier.
rf = RandomForestClassifier()

# Define random search model.
random_search_rf = RandomizedSearchCV(rf, param_distributions = random_grid,
                                      n_iter = parameter_iteration, scoring = 'roc_auc',
                                      n_jobs = 4, cv = skf.split(X, y), verbose = 3,
                                      random_state = 1001)

# And fit it.
random_search_rf.fit(X, y)

# Show output.
random_search_rf.best_score_
random_search_rf.best_params_
random_search_rf.cv_results_

# Define final model w/ best parametres.
random_forest = RandomForestClassifier(n_estimators = 800,
                                       min_samples_split = 2,
                                       min_samples_leaf = 2,
                                       max_features = 'auto',
                                       max_depth = 15,
                                       bootstrap = True,
                                       verbose = 3)

# And fit it.
random_forest.fit(X_train, y_train)

# Test block for RF model on test data.
predictions_test_rf = random_forest.predict_proba(X_test)[:, 1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_test_rf)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# Prediction block for unknown data points.
prediction_real_rf = random_forest.predict_proba(unknown)[:, 1]
prediction_frame = df(unknown_order_id, columns = ['order_item_id'])
prediction_frame['return'] = prediction_real_rf
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/predictions/single_predictors/prediction_rf.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)

# Feature importance table.
print('Feature importance Table\n',
      pd.DataFrame(random_forest.feature_importances_[:],
      unknown.columns[:]).sort_values(by = 0, ascending = False)[:32])

######################################################################################################
####
######## Neural Network
####
######################################################################################################

# Define optimizer algorithm and callback(s).
optimizer = keras.optimizers.Adam()
callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 6, verbose = 2,
                                         min_delta = 1e-5, restore_best_weights = True)

# Initialise network (as Sequential); add layers.
network = Sequential()
network.add(Dense(156, activation = 'sigmoid', input_dim = len(X_test.columns)))
network.add(Dense(45, activation = 'sigmoid'))
network.add(Dense(1, activation = 'sigmoid'))

# Compile network.
network.compile(optimizer = optimizer,
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

# And fit it.
network.fit(X_train, y_train, epochs = 40, batch_size = 128,
            callbacks = [callback], validation_split = 0.12,
            shuffle = True, verbose = 2)

# Test block for NN model on test data.
predictions_test_nn = network.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_test_nn)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# Prediction block for unknown data points.
prediction_real_nn = network.predict(unknown)
prediction_frame = df(unknown_order_id, columns = ['order_item_id'])
prediction_frame['return'] = prediction_real_nn
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/predictions/single_predictors/prediction_nn.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)

######################################################################################################
####
######## SIMPLE ENSEMBLING
####
######################################################################################################

# Take predictions.
predictions_test_xgb = df(list(xg_classifier.predict_proba(X_test)[:, 1]))
predictions_test_rf = df(list(random_forest.predict_proba(X_test)[:, 1]))
predictions_test_nn = df(list(network.predict(X_test)))

# Join predictions.
ensemble = predictions_test_xgb.merge(predictions_test_rf, on = predictions_test_xgb.index)
ensemble['0_z'] = predictions_test_nn
ensemble['score'] = (ensemble['0_x'] + ensemble['0_y'] + ensemble['0_z']) / 3
ensemble = np.asarray(ensemble['score'])
ensemble

# Check AUC.
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ensemble)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# Final ensembling.
xgboost_predictions = pd.read_csv('predictions/single_predictors/prediction_xgb.csv')
forest_predictions = pd.read_csv('predictions/single_predictors/prediction_rf.csv')
network_predictions = pd.read_csv('predictions/single_predictors/prediction_nn.csv')

joined = xgboost_predictions.merge(forest_predictions, on = xgboost_predictions.order_item_id)
joined['return_z'] = network_predictions['return']
joined = joined[['key_0', 'return_x', 'return_y', 'return_z']]
joined.columns = ['order_item_id', 'score_xgb', 'score_rf', 'score_nn']
joined['score_avg'] = (joined['score_xgb'] + joined['score_rf'] + joined['score_nn']) / 3
joined = joined[['order_item_id', 'score_avg']]
joined.columns = ['order_item_id', 'return']

# Save to .csv file.
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/predictions/ensemble_predictors/prediction_03.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)

######################################################################################################
####
######## AUTOMATED ENSEMBLING
####
######################################################################################################

import sys
sys.path.insert(0, '/Users/alextruesdale/Documents/business-analytics/term_project/final_code')

import modeling_functions as mf
models = mf.modeler(X, y, unknown)

# Initialise empty model_dict.
model_dict = {}

# Run NNs
model_dict.update({'nn_01': models.predictor_nn()})
model_dict.update({'nn_02': models.predictor_nn(batch_size = 180, capacity = 80)})
model_dict.update({'nn_03': models.predictor_nn(batch_size = 256, capacity = 124)})

# Run XGBs
model_dict.update({'xgb_00': models.predictor_xgb()})
model_dict.update({'xgb_01': models.predictor_xgb(n_estimators = 600, max_depth = 3, learning_rate = .01, colsample_bytree = .5)})
model_dict.update({'xgb_02': models.predictor_xgb(n_estimators = 550, max_depth = 4, learning_rate = .03, colsample_bytree = .5)})
model_dict.update({'xgb_03': models.predictor_xgb(n_estimators = 450, max_depth = 5, learning_rate = .05, colsample_bytree = .7)})

# Run RFs
model_dict.update({'rf_01': models.predictor_rf(n_estimators = 800, min_samples_split = 2, min_samples_leaf = 2, max_depth = 15)})
model_dict.update({'rf_02': models.predictor_rf(n_estimators = 650, min_samples_split = 3, min_samples_leaf = 3, max_depth = 20)})
model_dict.update({'rf_03': models.predictor_rf(n_estimators = 500, min_samples_split = 6, min_samples_leaf = 3, max_depth = 25)})

winner = max(model_dict, key = lambda k: operator.itemgetter(2)(model_dict[k]))
winner

for i in range(0, 100):
    print(i)
    ensembling_dict = {}
    for model, data in list(model_dict.items())[:10]:
        ensemble_name = winner + '_' + model
        ensemble_scores = np.mean(np.array([model_dict[winner][0], data[0]]), axis = 0)
        unknown_scores = data[1]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ensemble_scores)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ensembling_dict.update({ensemble_name: [ensemble_scores, unknown_scores, roc_auc]})

    winner = max(ensembling_dict, key = lambda k: operator.itemgetter(2)(ensembling_dict[k]))
    model_dict[winner] = tuple(ensembling_dict[winner])
    winner_total = max(model_dict, key = lambda k: operator.itemgetter(2)(model_dict[k]))

    if winner == winner_total:
        continue
    else:
        print(winner_total)
        print(model_dict[winner_total][2])
        out_scores = model_dict[winner_total][1]
        break

prediction_frame = df(unknown_order_id, columns = ['order_item_id'])
prediction_frame['return'] = out_scores
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/predictions/ensemble_predictors/prediction_04.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)
