# -*- coding: utf-8 -*-

"""
Created Dec 25, 2018
@author: Alex Truesdale

Jupyter Notebook Kernel (via Hydrogen / Atom) for working on raw challenge data.
"""

import os
import string
import numpy as np
import pandas as pd
import datetime as dt
import xgboost as xgb
from pprint import pprint

from statsmodels import robust
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.DataFrame

# Read in data sets.
known_data = pd.read_csv('term_project/BADS_WS1819_known.csv')
unknown_data = pd.read_csv('term_project/BADS_WS1819_unknown.csv')

print(len(known_data))
print(len(unknown_data))

# Initial data exploration.
known_data.dtypes

unknown_data['item_id'].value_counts()[:20]
known_data['item_id'].value_counts()[:20]

unknown_data['brand_id'].value_counts()[:20]
known_data['brand_id'].value_counts()[:20]

unknown_data['user_id'].value_counts()[:20]
known_data['user_id'].value_counts()[:20]

# User State feature inspection.
state_group = df(known_data.groupby('user_state')['return'].sum())
state_group['count'] = known_data.groupby('user_state')['user_id'].count()
state_group['ratio'] = state_group['return'] / state_group['count']
state_group

# Stack data frames.
known_data['key'] = 1
unknown_data['key'] = 0

joined = pd.concat([known_data, unknown_data], sort = False)
joined.head(5)

# Deemed uninformative; drop feature.
joined = joined.drop(['user_state'], axis = 1)
joined = joined.drop(['user_title'], axis = 1)
joined.dtypes

# Null value inspection.
columns = list(joined.columns.values)
null_count = df(joined.count(), columns = ['count_null'])
null_count['count_null'] = len(joined) - null_count['count_null']
null_count

# Fix misspells in colours.
colours = df(joined['item_color'].unique())
colours.sort_values(0)

joined.loc[joined['item_color'] == 'blau', 'item_color'] = 'blue'
joined.loc[joined['item_color'] == 'brwon', 'item_color'] = 'brown'
joined.loc[joined['item_color'] == 'oliv', 'item_color'] = 'olive'

# Fix lower-case upper-case issue in sizes.
joined['item_size'] = joined['item_size'].str.lower()

# Fix sizes with plus signs.
joined['item_size'] = joined['item_size'].str.replace('+', '.5', regex = False)

# Derive 'item types' by item.
joined['type'] = None
for item in joined['item_id'].unique():
    item_df = joined.loc[joined['item_id'] == item]
    item_sizes = [size for size in item_df['item_size']]
    if all(['.5' in size for size in item_sizes]):
        item_type = 1
    elif all([size.isdigit() for size in item_sizes]):
        item_sizes = [float(size) for size in item_df['item_size']]
        max_size = max(item_sizes)
        min_size = min(item_sizes)
        if min_size < 15 and max_size < 25:
            item_type = 3
        elif min_size > 2000 and max_size > 2000:
            item_type = 4
        elif min_size > 80 and max_size < 200:
            item_type = 5
        else:
            item_type = 6
    else:
        item_type = 2

    joined.loc[joined['item_id'] == item, 'type'] = item_type

# Create item_count, type_count, brand_counts, and users_counts dfs.
item_counts = df(joined.loc[joined['key'] == 1]['item_id'].value_counts())
type_counts = df(joined.loc[joined['key'] == 1]['type'].value_counts())
brand_counts = df(joined.loc[joined['key'] == 1]['brand_id'].value_counts())
users_counts = df(joined.loc[joined['key'] == 1]['user_id'].value_counts())

# Define base_ce value for items.
items_under = list(item_counts.loc[item_counts['item_id'] < 40].index)
base_ce_items = joined[joined['item_id'].isin(items_under)]['return'].mean()
base_ce_items

# Define base_ce value for brands.
brands_under = list(brand_counts.loc[brand_counts['brand_id'] < 100].index)
base_ce_brands = joined[joined['item_id'].isin(brands_under)]['return'].mean()
base_ce_brands

# Define base_ce value for users.
users_under = list(users_counts.loc[users_counts['user_id'] < 3].index)
base_ce_users = joined[joined['item_id'].isin(users_under)]['return'].mean()
base_ce_users

# Compute Conditional Expectation values for colour.
for item in joined.loc[joined['key'] == 1]['item_id'].unique():
    print(item)
    if item_counts.loc[item, 'item_id'] >= 20:
        print('YES:', item)
        item_df = joined.loc[joined['item_id'] == item]
        item_colours = [colour for colour in item_df['item_color']]

        colours_df = df(item_df.groupby('item_color')['return'].sum())
        colours_df['count'] = item_df.groupby('item_color')['return'].count()
        colours_df['ratio'] = colours_df['return'] / colours_df['count']

        for colour in colours_df.index:
            conditional_expectation = float(colours_df.loc[colours_df.index == colour]['ratio'])
            joined.loc[(joined['item_id'] == item) & (joined['item_color'] == colour) & (joined['key'] == 1), 'ce_colour_item'] = conditional_expectation
    else:
        joined.loc[(joined['item_id'] == item) & (joined['key'] == 1), 'ce_colour_item'] = base_ce_items

for type in joined.loc[joined['key'] == 1]['type'].unique():
    type_df = joined.loc[(joined['key'] == 1) & (joined['type'] == type)]
    type_colours = [colour for colour in type_df['item_color']]

    colours_df = df(type_df.groupby('item_color')['return'].sum())
    colours_df['count'] = type_df.groupby('item_color')['return'].count()
    colours_df['ratio'] = colours_df['return'] / colours_df['count']

    for colour in colours_df.index:
        conditional_expectation = float(colours_df.loc[colours_df.index == colour]['ratio'])
        joined.loc[(joined['type'] == type) & (joined['item_color'] == colour) & (joined['key'] == 1), 'ce_colour_type'] = conditional_expectation

joined = joined.drop(['item_color'], axis = 1)

# Compute Conditional Expectation valus for size.
for item in joined.loc[joined['key'] == 1]['item_id'].unique():
    print(item)
    if item_counts.loc[item, 'item_id'] >= 20:
        print('YES', item)
        item_df = joined.loc[joined['item_id'] == item]
        item_sizes = [size for size in item_df['item_size']]

        sizes_df = df(item_df.groupby('item_size')['return'].sum())
        sizes_df['count'] = item_df.groupby('item_size')['return'].count()
        sizes_df['ratio'] = sizes_df['return'] / sizes_df['count']

        for size in sizes_df.index:
            conditional_expectation = float(sizes_df.loc[sizes_df.index == size]['ratio'])
            joined.loc[(joined['item_id'] == item) & (joined['item_size'] == size) & (joined['key'] == 1), 'ce_size_item'] = conditional_expectation
    else:
        joined.loc[(joined['item_id'] == item) & (joined['key'] == 1), 'ce_colour_item'] = base_ce_items

for type in joined['type'].unique():
    type_df = joined.loc[(joined['key'] == 1) & (joined['type'] == type)]
    type_sizes = [size for size in type_df['item_size']]

    sizes_df = df(type_df.groupby('item_size')['return'].sum())
    sizes_df['count'] = type_df.groupby('item_size')['return'].count()
    sizes_df['ratio'] = sizes_df['return'] / sizes_df['count']

    for size in sizes_df.index:
        conditional_expectation = float(sizes_df.loc[sizes_df.index == size]['ratio'])
        joined.loc[(joined['type'] == type) & (joined['item_size'] == size) & (joined['key'] == 1), 'ce_size_type'] = conditional_expectation

# Change date columns to datetime features.
joined['order_date'] = pd.to_datetime(joined['order_date'])
joined['delivery_date'] = pd.to_datetime(joined['delivery_date'])
joined['user_dob'] = pd.to_datetime(joined['user_dob'])
joined['user_reg_date'] = pd.to_datetime(joined['user_reg_date'])

# Create 'was_delivered' feature.
joined['was_delivered'] = joined.apply(lambda row: 0 if pd.isnull(row['delivery_date']) else 1, axis = 1)

# Create 'membership_age' feature.
joined['membership_age_days'] = (joined['order_date'] - joined['user_reg_date']).dt.days
joined.loc[joined['membership_age_days'] == -1, 'membership_age_days'] = 0

bins = [-.01, 14, 182, 365, 548, 800]
labels = list(range(1, len(bins)))
joined['membership_age_binned'] = pd.cut(joined['membership_age_days'], bins = bins, labels = labels, include_lowest = True)
joined = joined.drop(['membership_age_days'], axis = 1)

# Bin prices.
bins = [-.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150, 250, 500]
labels = list(range(1, len(bins)))
joined['price_binned'] = pd.cut(joined['item_price'], bins = bins, labels = labels, include_lowest = True)
joined = joined.drop(['item_price'], axis = 1)

# Create 'time_to_delivery' feature.
joined['time_to_delivery_days'] = (joined['delivery_date'] - joined['order_date']).dt.days
joined['time_to_delivery_days'].fillna(9999, inplace = True)

bins = [-9999, -.01, 1, 3, 7, 14, 31, 62, 160, 10000]
labels = list(range(1, len(bins)))
joined['delivery_date_binned'] = pd.cut(joined['time_to_delivery_days'], bins = bins, labels = labels, include_lowest = True)
joined = joined.drop(['time_to_delivery_days'], axis = 1)

# Create 'assumed_age' feature.
joined['assumed_age'] = (joined['order_date'] - joined['user_dob']).dt.days / 365
joined['assumed_age'].fillna(0, inplace = True)

bins = [-.01, 1, 18, 27, 35, 55, 75, 125]
labels = list(range(1, len(bins)))
joined['age_binned'] = pd.cut(joined['assumed_age'], bins = bins, labels = labels, include_lowest = True)
joined = joined.drop(['assumed_age'], axis = 1)

# Remove datetime date features.
joined = joined.drop(['order_date', 'delivery_date', 'user_reg_date', 'user_dob'], axis = 1)

# Conditional expectation for item_id.
items_df = df(joined.loc[joined['key'] == 1].groupby('item_id')['return'].sum())
items_df['count'] = joined.loc[joined['key'] == 1].groupby('item_id')['return'].count()
items_df['ratio'] = items_df['return'] / items_df['count']

joined.loc[joined['key'] == 1, 'ce_item_id'] = base_ce_items
for title in items_df.index:
    if item_counts.loc[title, 'item_id'] >= 40:
        conditional_expectation = float(items_df.loc[items_df.index == title]['ratio'])
        joined.loc[(joined['item_id'] == title) & (joined['key'] == 1), 'ce_item_id'] = conditional_expectation

# Conditional expectation for brand_id.
brands_df = df(joined.loc[joined['key'] == 1].groupby('brand_id')['return'].sum())
brands_df['count'] = joined.loc[joined['key'] == 1].groupby('brand_id')['return'].count()
brands_df['ratio'] = brands_df['return'] / brands_df['count']

joined.loc[joined['key'] == 1, 'ce_brand_id'] = base_ce_brands
for title in brands_df.index:
    if brand_counts.loc[title, 'brand_id'] >= 100:
        conditional_expectation = float(brands_df.loc[brands_df.index == title]['ratio'])
        joined.loc[(joined['brand_id'] == title) & (joined['key'] == 1), 'ce_brand_id'] = conditional_expectation

# Conditional expectation for user_id.
user_df = df(joined.loc[joined['key'] == 1].groupby('user_id')['return'].sum())
user_df['count'] = joined.loc[joined['key'] == 1].groupby('user_id')['return'].count()
user_df['ratio'] = user_df['return'] / user_df['count']

joined.loc[joined['key'] == 1, 'ce_user_id'] = base_ce_users
for title in user_df.index:
    if users_counts.loc[title, 'user_id'] >= 3:
        conditional_expectation = float(user_df.loc[user_df.index == title]['ratio'])
        joined.loc[(joined['user_id'] == title) & (joined['key'] == 1), 'ce_user_id'] = conditional_expectation

path_joined_type = '/Users/alextruesdale/Documents/business-analytics/term_project/joined_type.csv'
joined.to_csv(path_joined_type)

joined = pd.read_csv('term_project/joined_type.csv')

# Dummy encoding of binned categoricals
# joined = joined.join(pd.get_dummies(joined['type'], prefix = 'type'))
joined = joined.join(pd.get_dummies(joined['was_delivered'], prefix = 'delivered'))
joined = joined.join(pd.get_dummies(joined['membership_age_binned'], prefix = 'membership_age'))
joined = joined.join(pd.get_dummies(joined['delivery_date_binned'], prefix = 'delivery_date'))
joined = joined.join(pd.get_dummies(joined['age_binned'], prefix = 'customer_age'))
joined = joined.join(pd.get_dummies(joined['price_binned'], prefix = 'price'))

# Drop stacked categoricals.
# joined = joined.drop(['type'], axis = 1)
joined = joined.drop(['was_delivered'], axis = 1)
joined = joined.drop(['membership_age_binned'], axis = 1)
joined = joined.drop(['delivery_date_binned'], axis = 1)
joined = joined.drop(['age_binned'], axis = 1)
joined = joined.drop(['price_binned'], axis = 1)

# Assign string sizes numerical values.
joined.loc[joined['item_size'] == 'unsized', 'item_size'] = '0'
joined.loc[joined['item_size'] == 'xs', 'item_size'] = '1'
joined.loc[joined['item_size'] == 's', 'item_size'] = '2'
joined.loc[joined['item_size'] == 'm', 'item_size'] = '3'
joined.loc[joined['item_size'] == 'l', 'item_size'] = '4'
joined.loc[joined['item_size'] == 'xl', 'item_size'] = '5'
joined.loc[joined['item_size'] == 'xxl', 'item_size'] = '6'
joined.loc[joined['item_size'] == 'xxxl', 'item_size'] = '7'

# Change item_size string column to float.
joined['item_size'] = pd.to_numeric(joined.item_size)

# Normalise size data by item.
def mad_normalise(initial_size, sizes_median, mad):
    return (float(initial_size) - sizes_median) / mad

for item in joined['item_id'].unique():
    item_df = joined.loc[joined['item_id'] == item]
    item_sizes = [float(size) for size in item_df['item_size']]
    sizes_median = np.median(item_sizes)
    mad = robust.mad(item_sizes)
    if mad == 0:
        replacement_dict = {initial_size: 0 for initial_size in item_sizes}
    else:
        replacement_dict = {initial_size: mad_normalise(initial_size, sizes_median, mad) for initial_size in item_sizes}

    for size in item_sizes:
        joined.loc[(joined['item_id'] == item) & (joined['item_size'] == size), 'item_size'] = replacement_dict[size]

# Split data once more.
known_data = df(joined[:100000])
unknown_data = df(joined[100000:])

# Preparation for unknown_data.
unknown_data['item_id'] = unknown_data.apply(lambda row: (row['item_id'] * 2) + 2, axis = 1)
unknown_data['brand_id'] = unknown_data.apply(lambda row: row['brand_id'] + 100, axis = 1)

# Identify known data points.
known_ids = list(known_data['item_id'].unique())
known_brands = list(known_data['brand_id'].unique())
known_users = list(known_data['user_id'].unique())

# IDs
matched_ids = []
unmatched_ids = []
for item in unknown_data['item_id'].unique():
    if item in known_ids:
        matched_ids.append(item)
    else:
        unmatched_ids.append(item)

for item in unknown_data['item_id'].unique():
    if item in unmatched_ids:
        # unknown_data.loc[unknown_data['item_id'] == item, 'ce_colour_item'] = .48
        # unknown_data.loc[unknown_data['item_id'] == item, 'ce_size_item'] = .48
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_item_id'] = .48

# matched_item_dict_colour = {}
# matched_item_dict_size = {}
matched_item_dict_item = {}
for item in matched_ids:
    # matched_item_dict_colour.update({item: known_data.loc[known_data['item_id'] == item]['ce_colour_item'].unique()[0]})
    # matched_item_dict_size.update({item: known_data.loc[known_data['item_id'] == item]['ce_size_item'].unique()[0]})
    matched_item_dict_item.update({item: known_data.loc[known_data['item_id'] == item]['ce_item_id'].unique()[0]})

for item in unknown_data['item_id'].unique():
    if item in matched_ids:
        # unknown_data.loc[unknown_data['item_id'] == item, 'ce_colour_item'] = matched_item_dict_colour[item]
        # unknown_data.loc[unknown_data['item_id'] == item, 'ce_size_item'] = matched_item_dict_size[item]
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_item_id'] = matched_item_dict_item[item]

# for type in unknown_data['type'].unique():
#     unknown_data.loc[unknown_data['type'] == type, 'ce_colour_type'] = known_data.loc[known_data['type'] == type]['ce_colour_type'].unique()[0]
#     unknown_data.loc[unknown_data['type'] == type, 'ce_size_type'] = known_data.loc[known_data['type'] == type]['ce_size_type'].unique()[0]

# Brands
matched_brands = []
unmatched_brands = []
for brand in unknown_data['brand_id'].unique():
    if brand in known_brands:
        matched_brands.append(brand)
    else:
        unmatched_brands.append(brand)

for brand in unknown_data['brand_id'].unique():
    if brand in unmatched_brands:
        unknown_data.loc[unknown_data['brand_id'] == brand, 'ce_brand_id'] = .48

matched_brand_dict = {}
for brand in matched_brands:
    matched_brand_dict.update({brand: known_data.loc[known_data['brand_id'] == brand]['ce_brand_id'].unique()[0]})

for brand in unknown_data['brand_id'].unique():
    if brand in matched_brands:
        unknown_data.loc[unknown_data['brand_id'] == brand, 'ce_brand_id'] = matched_brand_dict[brand]

# Users
matched_users = []
unmatched_users = []
for user in unknown_data['user_id'].unique():
    if user in known_users:
        matched_users.append(user)
    else:
        unmatched_users.append(user)

unknown_data['ce_user_id'] = .48
# for user in unknown_data['user_id'].unique():
#     if user in unmatched_users:
#         unknown_data.loc[unknown_data['user_id'] == user, 'ce_user_id'] = .48

matched_users_dict = {}
for user in matched_users:
    matched_users_dict.update({user: known_data.loc[known_data['user_id'] == user]['ce_user_id'].unique()[0]})

for user in unknown_data['user_id'].unique():
    if user in matched_users:
        unknown_data.loc[unknown_data['user_id'] == user, 'ce_user_id'] = matched_users_dict[user]

# Output modified data frames to .csv files.
path_known = '/Users/alextruesdale/Documents/business-analytics/term_project/known_modified_4.csv'
path_unknown = '/Users/alextruesdale/Documents/business-analytics/term_project/unknown_modified_4.csv'

known_data.to_csv(path_known)
unknown_data.to_csv(path_unknown)

######################################################################################################
####
######## END DATA PREP
####
######################################################################################################

# Read in data; check features; remove features.
known_data = pd.read_csv('term_project/known_modified_4.csv')
unknown_data = pd.read_csv('term_project/unknown_modified_4.csv')

list(known_data.columns.values)
known_data = known_data.drop(['Unnamed: 0'], axis = 1)
known_data = known_data.drop(['Unnamed: 0.1'], axis = 1)
known_data = known_data.drop(['key'], axis = 1)

list(unknown_data.columns.values)
unknown_data = unknown_data.drop(['return'], axis = 1)
unknown_data = unknown_data.drop(['Unnamed: 0'], axis = 1)
unknown_data = unknown_data.drop(['Unnamed: 0.1'], axis = 1)
unknown_data = unknown_data.drop(['key'], axis = 1)

# Store order IDs and remove columns.
known_data_order_id = known_data['order_item_id']
unknown_data_order_id = unknown_data['order_item_id']

known_data = known_data.drop(['order_item_id'], axis = 1)
unknown_data = unknown_data.drop(['order_item_id'], axis = 1)

known_data = known_data.drop(['brand_id'], axis = 1)
unknown_data = unknown_data.drop(['brand_id'], axis = 1)

known_data = known_data.drop(['item_id'], axis = 1)
unknown_data = unknown_data.drop(['item_id'], axis = 1)

known_data = known_data.drop(['user_id'], axis = 1)
unknown_data = unknown_data.drop(['user_id'], axis = 1)

known_data = known_data.drop(['item_size'], axis = 1)
unknown_data = unknown_data.drop(['item_size'], axis = 1)

known_data = known_data.drop(['item_price'], axis = 1)
unknown_data = unknown_data.drop(['item_price'], axis = 1)

# Check for null values
columns = list(known_data.columns.values)
null_count = df(known_data.count(), columns=['count_null'])
null_count['count_null'] = len(known_data) - null_count['count_null']
null_count

columns = list(unknown_data.columns.values)
null_count = df(unknown_data.count(), columns=['count_null'])
null_count['count_null'] = len(unknown_data) - null_count['count_null']
null_count

# Define train data / target.
train_data = known_data
target = df(train_data['return'])
target = target.values.ravel()
train_data.pop('return')

train_data.head(10)
len(list(train_data))
len(train_data)
len(target)

unknown_data.head(10)
len(list(unknown_data))

######################################################################################################
####
######## GRID TEST FOR RF PARAMETERS
####
######################################################################################################

# Create split sets for cross-validation.
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size = 0.25)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
               }

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose = 5, random_state = 42,
                               n_jobs = -1)

rf_random.fit(x_train, y_train)

rf_random.best_params_

######################################################################################################
####
######## END GRID TEST
####
######################################################################################################

# Define classifier and fit.
random_forest = RandomForestClassifier(n_estimators = 200,
                                       max_features = x,
                                       max_depth = 15,
                                       min_samples_split = x,
                                       min_samples_leaf = x,
                                       bootstrap = x,
                                       verbose = 3)

random_forest.fit(train_data, target)

# Make predictions and evaluate cross-validation score.
prediction = random_forest.predict(unknown_data)
prediction_roc = random_forest.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, prediction_roc)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# Save prediction_frame as submit-able file
prediction_frame = df(unknown_data_order_id, columns = ['order_item_id'])
prediction_frame['return'] = prediction
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/prediction_1.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)
