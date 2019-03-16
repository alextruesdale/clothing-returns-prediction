# -*- coding: utf-8 -*-

"""
Created Feb 8, 2019
@author: Alex Truesdale
@email: alex.truesdale@colorado.edu

Jupyter Notebook Kernel (via Hydrogen / Atom) for modeling on prepared challenge data.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import operator

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

# Split target.
X, y = known.iloc[:, :-1], known.iloc[:, -1].values

######################################################################################################
####
######## XGBoost
####
######################################################################################################

# Define parameters for RandomSearch.
folds = 5
parameter_iteration = 20
parameter_grid = {
     'n_estimators':[500, 550, 600, 650, 700, 750],
     'learning_rate': [0.05, 0.06, 0.07, 0.08],
     'max_depth': [5, 6, 7, 8],
     'min_child_weight': [7],
     'gamma': [0.3],
     'colsample_bytree': [0.6, 0.7, 0.8]
 }

# Define stratified K-folds fold-er.
skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 1001)

# Define XGB classifier.
xgb_c = xgb.XGBClassifier(objective = 'binary:logistic', silent = True, subsample = 0.8, nthread = 1)

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

# Define final model w/ best parameters.
xg_classifier = xgb.XGBClassifier(silent = 1,
                                  min_child_weight = 7,
                                  scale_pos_weight = 1,
                                  learning_rate = 0.05,
                                  colsample_bytree = 0.8,
                                  subsample = 0.8,
                                  objective = 'binary:logistic',
                                  n_estimators = 500,
                                  reg_alpha = 0.3,
                                  max_depth = 5,
                                  gamma = 0.3,
                                  nthread = 1,
                                  early_stopping_rounds = 75)

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

# Define parameters for RandomSearch.
folds = 5
parameter_iteration = 20

n_estimators = [int(x) for x in np.linspace(start = 600, stop = 900, num = 8)]
max_features = ['auto']
max_depth = [12, 15, 18, 21, 24]
min_samples_split = [2, 3, 3]
min_samples_leaf = [2, 3, 4]
bootstrap = [True]

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

# Define final model w/ best parameters.
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
sys.path.insert(0, '/Users/alextruesdale/Documents/business-analytics/final_code')

import modeling_functions as mf

# Create CV dictionary with indexed folds; test / train splits.
split_dict = {}
kf = KFold(n_splits = 10, shuffle = True)
for i, split in enumerate(kf.split(X)):
    train_index = split[0]
    test_index = split[1]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    split_dict.update({i: [X_train, X_test, y_train, y_test]})

# Run each fold through ensembler and store fold known predictions, unknown
# predictions and ensemble test AUC score.

ensemble_dict_final = {}
for fold, split in split_dict.items():
    print('FOLD:', fold)

    model_dict = mf.model_aggregator(split, unknown)
    winner = max(model_dict, key = lambda k: operator.itemgetter(2)(model_dict[k]))
    ensemble_dict_final.update({fold: mf.ensembler(split, model_dict, winner)})

# Print ensembled AUC scores.
for x in ensemble_dict_final.values():
    print(x[2])

# Identify winning ensemble and define unknown prediction vector.
ultimate_winner = max(ensemble_dict_final, key = lambda k: operator.itemgetter(2)(ensemble_dict_final[k]))
out_scores_single = ensemble_dict_final[ultimate_winner][1]

# Average all ensembles into final unknown prediction vector.
out_scores_aggregate = np.mean(np.array([item[1] for item in ensemble_dict_final.values()]), axis = 0)

# Output as submission .csv file.
prediction_frame = df(unknown_order_id, columns = ['order_item_id'])
prediction_frame['return'] = out_scores_aggregate
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/predictions/ensemble_predictors/prediction_09.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)

######################################################################################################
####
######## Cost Calculations
####
######################################################################################################

# Extract prediction vectors for folds from ensemble dict.
# Get fold indeces in order to re-combine into 100% known data.
prediction_vectors = [item[0] for item in ensemble_dict_final.values()]
split_dict_indeces = {i: list(df(split[1]).index) for i, split in split_dict.items()}

# Create data frames with indeces for folds.
df_list = []
for i, indeces in split_dict_indeces.items():
    frame = df(prediction_vectors[i], indeces)
    df_list.append(frame)

# Join fold dfs and average across rows for reconstructed known prediction.
df_all = pd.concat([df for df in df_list], axis = 1)
cost_all = df(df_all.mean(axis = 1), columns = ['prediction'])
cost_all['item_price'] = X['item_price']
cost_all['return'] = y

# Cost-optimal (minimiser) ratio calculation.
c_bG = 0.5 * -cost_all['item_price']
c_gB = 0.5 * 5 * -(3 + 0.1 * cost_all['item_price'])
cost_all['τ'] = c_bG / (c_bG + c_gB)

# Define 'warn' if prediction value is greater than τ; reorder columns.
cost_all['warn'] = cost_all.apply(lambda row: 1 if (row['τ'] * 1) <= row['prediction'] else 0, axis = 1)
cost_all = cost_all.iloc[:, [1, 3, 0, 2, 4]]
len(cost_all[cost_all['warn'] == 1])
cost_all.head(10)

def revenue_calculator(row):
    """Calculate type 1 and 2 error costs or revenue earned."""

    if row['return'] > row['warn']:
        revenue = 0.5 * 5 * -(3 + 0.1 * row['item_price'])
    elif row['warn'] > row['return']:
        revenue = 0.5 * -row['item_price']
    elif row['return'] == 1 and row['warn'] == 1:
        revenue = 0
    else:
        revenue = row['item_price']

    return revenue

def cost_minimiser(row):
    """Calculate type 1 and 2 error costs."""

    if row['return'] > row['warn']:
        cost = 0.5 * 5 * -(3 + 0.1 * row['item_price'])
    elif row['warn'] > row['return']:
        cost = 0.5 * -row['item_price']
    else:
        cost = 0

    return cost

# Apply cost calculator at τ = τ1.
cost_all['revenue'] = cost_all.apply(revenue_calculator, axis = 1)
cost_all['revenue'].sum()
cost_all.head(5)

cost_all['cost'] = cost_all.apply(cost_minimiser, axis = 1)
cost_all['cost'].sum()
cost_all.head(10)

# Treat τ values with scale of multipliers to examine decision threshold effects on revenue.
rev_dict = {}
for i, val in enumerate(np.linspace(.8, 1.7, num = 100)):
    print('{} of 100 ... treatment multiplier: {}'.format(i + 1, round(val, 2)), end = '\r')
    cost_all['warn'] = cost_all.apply(lambda row: 1 if (row['τ'] * val) <= row['prediction'] else 0, axis = 1)
    cost_all['revenue'] = cost_all.apply(revenue_calculator, axis = 1)
    sum_rev = cost_all['revenue'].sum()

    rev_dict.update({val: sum_rev})

cost_all['cost'].mean()

# Build df from dict; plot revenue as function of τ treatment.
rev_frame = df(list(rev_dict.values()), list(rev_dict.keys()), columns = ['revenue'])
rev_frame[rev_frame['revenue'] == rev_frame.max()[0]]

rev_frame.plot.line()

# Treat τ values with scale of multipliers to examine decision threshold effects on revenue.
cost_dict = {}
for i, val in enumerate(np.linspace(.98, 1.1, num = 100)):
    print('{} of 100 ... treatment multiplier: {}'.format(i + 1, round(val, 2)), end = '\r')
    cost_all['warn'] = cost_all.apply(lambda row: 1 if (row['τ'] * val) <= row['prediction'] else 0, axis = 1)
    cost_all['cost'] = cost_all.apply(cost_minimiser, axis = 1)
    sum_cost = cost_all['cost'].sum()

    cost_dict.update({val: sum_cost})

# Build df from dict; plot cost as function of τ treatment.
cost_frame = df(list(cost_dict.values()), list(cost_dict.keys()), columns = ['cost'])
cost_frame[cost_frame['cost'] == cost_frame.max()[0]]

cost_frame.plot.line()

######################################################################################################
####
######## Calibration Curve
####
######################################################################################################

cost_all.head(5)

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots()
labels = cost_all['return']

plt_y, plt_x = calibration_curve(labels, cost_all['prediction'], n_bins = 50)
plt.plot(plt_x, plt_y, marker = 'o', markersize = 4, label = 'All')

ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability in each bin')
plt.plot([0,1], [0,1], c = 'black', linewidth = 0.5, label = 'Ideal')

plt.legend()
plt.show()
