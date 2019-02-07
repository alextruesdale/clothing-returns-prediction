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

known_data = known_data.drop(['user_state'], axis = 1)
unknown_data = unknown_data.drop(['user_state'], axis = 1)

# Null value inspection.
columns = list(known_data.columns.values)
null_count = df(known_data.count(), columns=['count_null'])
null_count['count_null'] = len(known_data) - null_count['count_null']
null_count

colours = df(known_data['item_color'].unique())
colours.sort_values(0)



# Fix misspells in colours.
known_data.loc[known_data['item_color'] == 'blau', 'item_color'] = 'blue'
known_data.loc[known_data['item_color'] == 'brwon', 'item_color'] = 'brown'
known_data.loc[known_data['item_color'] == 'oliv', 'item_color'] = 'olive'

unknown_data.loc[unknown_data['item_color'] == 'blau', 'item_color'] = 'blue'
unknown_data.loc[unknown_data['item_color'] == 'brwon', 'item_color'] = 'brown'
unknown_data.loc[unknown_data['item_color'] == 'oliv', 'item_color'] = 'olive'

known_data['item_size'].value_counts()



# Fix lower-case upper-case issue in sizes.
known_data['item_size'] = known_data['item_size'].str.lower()
unknown_data['item_size'] = unknown_data['item_size'].str.lower()



# Fix sizes with plus signs.
known_data['item_size'] = known_data['item_size'].str.replace('+', '.5', regex = False)
unknown_data['item_size'] = unknown_data['item_size'].str.replace('+', '.5', regex = False)



# Derive 'item types' by item.
known_data['type'] = None
for item in known_data['item_id'].unique():
    item_df = known_data.loc[known_data['item_id'] == item]
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

    known_data.loc[known_data['item_id'] == item, 'type'] = item_type

unknown_data['type'] = None
for item in unknown_data['item_id'].unique():
    item_df = unknown_data.loc[unknown_data['item_id'] == item]
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

    unknown_data.loc[unknown_data['item_id'] == item, 'type'] = item_type



# Compute Conditional Expectation values for colour.
for item in known_data['item_id'].unique():
    item_df = known_data.loc[known_data['item_id'] == item]
    item_colours = [colour for colour in item_df['item_color']]

    colours_df = df(item_df.groupby('item_color')['return'].sum())
    colours_df['count'] = item_df.groupby('item_color')['return'].count()
    colours_df['ratio'] = colours_df['return'] / colours_df['count']

    for colour in colours_df.index:
        conditional_expectation = float(colours_df.loc[colours_df.index == colour]['ratio'])
        known_data.loc[(known_data['item_id'] == item) & (known_data['item_color'] == colour), 'ce_colour_item'] = conditional_expectation

for type in known_data['type'].unique():
    type_df = known_data.loc[known_data['type'] == type]
    type_colours = [colour for colour in type_df['item_color']]

    colours_df = df(type_df.groupby('item_color')['return'].sum())
    colours_df['count'] = type_df.groupby('item_color')['return'].count()
    colours_df['ratio'] = colours_df['return'] / colours_df['count']

    for colour in colours_df.index:
        conditional_expectation = float(colours_df.loc[colours_df.index == colour]['ratio'])
        known_data.loc[(known_data['type'] == type) & (known_data['item_color'] == colour), 'ce_colour_type'] = conditional_expectation

known_data = known_data.drop(['item_color'], axis = 1)
unknown_data = unknown_data.drop(['item_color'], axis = 1)

# Compute Conditional Expectation valus for size.
for item in known_data['item_id'].unique():
    item_df = known_data.loc[known_data['item_id'] == item]
    item_sizes = [size for size in item_df['item_size']]

    sizes_df = df(item_df.groupby('item_size')['return'].sum())
    sizes_df['count'] = item_df.groupby('item_size')['return'].count()
    sizes_df['ratio'] = sizes_df['return'] / sizes_df['count']

    for size in sizes_df.index:
        conditional_expectation = float(sizes_df.loc[sizes_df.index == size]['ratio'])
        known_data.loc[(known_data['item_id'] == item) & (known_data['item_size'] == size), 'ce_size_item'] = conditional_expectation

for type in known_data['type'].unique():
    type_df = known_data.loc[known_data['type'] == type]
    type_sizes = [size for size in type_df['item_size']]

    sizes_df = df(type_df.groupby('item_size')['return'].sum())
    sizes_df['count'] = type_df.groupby('item_size')['return'].count()
    sizes_df['ratio'] = sizes_df['return'] / sizes_df['count']

    for size in sizes_df.index:
        conditional_expectation = float(sizes_df.loc[sizes_df.index == size]['ratio'])
        known_data.loc[(known_data['type'] == type) & (known_data['item_size'] == size), 'ce_size_type'] = conditional_expectation



### Date delta feature engineering.

# Change date columns to datetime features.
known_data['order_date'] = pd.to_datetime(known_data['order_date'])
known_data['delivery_date'] = pd.to_datetime(known_data['delivery_date'])
known_data['user_dob'] = pd.to_datetime(known_data['user_dob'])
known_data['user_reg_date'] = pd.to_datetime(known_data['user_reg_date'])

unknown_data['order_date'] = pd.to_datetime(unknown_data['order_date'])
unknown_data['delivery_date'] = pd.to_datetime(unknown_data['delivery_date'])
unknown_data['user_dob'] = pd.to_datetime(unknown_data['user_dob'])
unknown_data['user_reg_date'] = pd.to_datetime(unknown_data['user_reg_date'])



# Create 'was_delivered' feature.
known_data['was_delivered'] = known_data.apply(lambda row: 0 if pd.isnull(row['delivery_date']) else 1, axis = 1)
unknown_data['was_delivered'] = unknown_data.apply(lambda row: 0 if pd.isnull(row['delivery_date']) else 1, axis = 1)



# Create 'membership_age' feature.
known_data['membership_age_days'] = (known_data['order_date'] - known_data['user_reg_date']).dt.days
known_data.loc[known_data['membership_age_days'] == -1, 'membership_age_days'] = 0

unknown_data['membership_age_days'] = (unknown_data['order_date'] - unknown_data['user_reg_date']).dt.days
unknown_data.loc[unknown_data['membership_age_days'] == -1, 'membership_age_days'] = 0

known_data['membership_age_days'].max()
known_data['membership_age_days'].min()
known_data['membership_age_days'].max() - known_data['membership_age_days'].min()) / 5

bins = [-.01, 14, 182, 365, 548, 800]
labels = list(range(1, len(bins)))
known_data['membership_age_binned'] = pd.cut(known_data['membership_age_days'], bins = bins, labels = labels, include_lowest = True)
unknown_data['membership_age_binned'] = pd.cut(unknown_data['membership_age_days'], bins = bins, labels = labels, include_lowest = True)

known_data.groupby('membership_age_binned')['return'].count()
known_data.groupby('membership_age_binned')['return'].sum()
known_data.hist('membership_age_binned')

known_data = known_data.drop(['membership_age_days'], axis = 1)
unknown_data = unknown_data.drop(['membership_age_days'], axis = 1)



# Bin prices.
bins = [-.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150, 250, 500]
known_data['item_price'].max()
known_data['item_price'].min()
known_data['item_price'].hist(bins = 30)

labels = list(range(1, len(bins)))
known_data['price_binned'] = pd.cut(known_data['item_price'], bins = bins, labels = labels, include_lowest = True)
unknown_data['price_binned'] = pd.cut(unknown_data['item_price'], bins = bins, labels = labels, include_lowest = True)

known_data = known_data.drop(['time_to_delivery_days'], axis = 1)
unknown_data = unknown_data.drop(['time_to_delivery_days'], axis = 1)

# Create 'time_to_delivery' feature.
known_data['time_to_delivery_days'] = (known_data['delivery_date'] - known_data['order_date']).dt.days
known_data['time_to_delivery_days'].fillna(9999, inplace = True)

unknown_data['time_to_delivery_days'] = (unknown_data['delivery_date'] - unknown_data['order_date']).dt.days
unknown_data['time_to_delivery_days'].fillna(9999, inplace = True)

known_data.loc[known_data['time_to_delivery_days'] < -5]

bins = [-9999, -.01, 1, 3, 7, 14, 31, 62, 160, 10000]
labels = list(range(1, len(bins)))
known_data['delivery_date_binned'] = pd.cut(known_data['time_to_delivery_days'], bins = bins, labels = labels, include_lowest = True)
unknown_data['delivery_date_binned'] = pd.cut(unknown_data['time_to_delivery_days'], bins = bins, labels = labels, include_lowest = True)

known_data = known_data.drop(['time_to_delivery_days'], axis = 1)
unknown_data = unknown_data.drop(['time_to_delivery_days'], axis = 1)



# Create 'assumed_age' feature.
known_data['assumed_age'] = (known_data['order_date'] - known_data['user_dob']).dt.days / 365
known_data['assumed_age'].fillna(0, inplace = True)

unknown_data['assumed_age'] = (unknown_data['order_date'] - unknown_data['user_dob']).dt.days / 365
unknown_data['assumed_age'].fillna(0, inplace = True)

bins = [-.01, 1, 18, 27, 35, 55, 75, 125]
labels = list(range(1, len(bins)))
known_data['age_binned'] = pd.cut(known_data['assumed_age'], bins = bins, labels = labels, include_lowest = True)
unknown_data['age_binned'] = pd.cut(unknown_data['assumed_age'], bins = bins, labels = labels, include_lowest = True)

known_data = known_data.drop(['assumed_age'], axis = 1)
unknown_data = unknown_data.drop(['assumed_age'], axis = 1)



# Remove datetime date features.
known_data = known_data.drop(['order_date', 'delivery_date', 'user_reg_date', 'user_dob'], axis = 1)
unknown_data = unknown_data.drop(['order_date', 'delivery_date', 'user_reg_date', 'user_dob'], axis = 1)

# Conditional expectation for user_title.
titles_df = df(known_data.groupby('user_title')['return'].sum())
titles_df['count'] = known_data.groupby('user_title')['return'].count()
titles_df['ratio'] = titles_df['return'] / titles_df['count']

known_data['ce_user_title'] = None
for title in titles_df.index:
    conditional_expectation = float(titles_df.loc[titles_df.index == title]['ratio'])
    known_data.loc[known_data['user_title'] == title, 'ce_user_title'] = conditional_expectation



# Conditional expectation for item_id.
items_df = df(known_data.groupby('item_id')['return'].sum())
items_df['count'] = known_data.groupby('item_id')['return'].count()
items_df['ratio'] = items_df['return'] / items_df['count']

known_data['ce_item_id'] = None
for title in items_df.index:
    conditional_expectation = float(items_df.loc[items_df.index == title]['ratio'])
    known_data.loc[known_data['item_id'] == title, 'ce_item_id'] = conditional_expectation



# Conditional expectation for brand_id.
brands_df = df(known_data.groupby('brand_id')['return'].sum())
brands_df['count'] = known_data.groupby('brand_id')['return'].count()
brands_df['ratio'] = brands_df['return'] / brands_df['count']

known_data['ce_brand_id'] = None
for title in brands_df.index:
    conditional_expectation = float(brands_df.loc[brands_df.index == title]['ratio'])
    known_data.loc[known_data['brand_id'] == title, 'ce_brand_id'] = conditional_expectation



# Conditional expectation for user_id.
user_df = df(known_data.groupby('user_id')['return'].sum())
user_df['count'] = known_data.groupby('user_id')['return'].count()
user_df['ratio'] = user_df['return'] / user_df['count']

known_data['ce_user_id'] = None
for title in user_df.index:
    conditional_expectation = float(user_df.loc[user_df.index == title]['ratio'])
    known_data.loc[known_data['user_id'] == title, 'ce_user_id'] = conditional_expectation



# Dummy encoding of binned categoricals
known_data = known_data.join(pd.get_dummies(known_data['type'], prefix = 'type'))
known_data = known_data.join(pd.get_dummies(known_data['was_delivered'], prefix = 'delivered'))
known_data = known_data.join(pd.get_dummies(known_data['membership_age_binned'], prefix = 'membership_age'))
known_data = known_data.join(pd.get_dummies(known_data['delivery_date_binned'], prefix = 'delivery_date'))
known_data = known_data.join(pd.get_dummies(known_data['age_binned'], prefix = 'customer_age'))
known_data = known_data.join(pd.get_dummies(known_data['price_binned'], prefix = 'price'))

unknown_data = unknown_data.join(pd.get_dummies(unknown_data['type'], prefix = 'type'))
unknown_data = unknown_data.join(pd.get_dummies(unknown_data['was_delivered'], prefix = 'delivered'))
unknown_data = unknown_data.join(pd.get_dummies(unknown_data['membership_age_binned'], prefix = 'membership_age'))
unknown_data = unknown_data.join(pd.get_dummies(unknown_data['delivery_date_binned'], prefix = 'delivery_date'))
unknown_data = unknown_data.join(pd.get_dummies(unknown_data['age_binned'], prefix = 'customer_age'))
unknown_data = unknown_data.join(pd.get_dummies(unknown_data['price_binned'], prefix = 'price'))



# Drop stacked categoricals.
known_data = known_data.drop(['type'], axis = 1)
known_data = known_data.drop(['was_delivered'], axis = 1)
known_data = known_data.drop(['membership_age_binned'], axis = 1)
known_data = known_data.drop(['delivery_date_binned'], axis = 1)
known_data = known_data.drop(['age_binned'], axis = 1)
known_data = known_data.drop(['price_binned'], axis = 1)

unknown_data = unknown_data.drop(['type'], axis = 1)
unknown_data = unknown_data.drop(['was_delivered'], axis = 1)
unknown_data = unknown_data.drop(['membership_age_binned'], axis = 1)
unknown_data = unknown_data.drop(['delivery_date_binned'], axis = 1)
unknown_data = unknown_data.drop(['age_binned'], axis = 1)
unknown_data = unknown_data.drop(['price_binned'], axis = 1)



# Assign string sizes numerical values.
known_data.loc[known_data['item_size'] == 'unsized', 'item_size'] = '0'
known_data.loc[known_data['item_size'] == 'xs', 'item_size'] = '1'
known_data.loc[known_data['item_size'] == 's', 'item_size'] = '2'
known_data.loc[known_data['item_size'] == 'm', 'item_size'] = '3'
known_data.loc[known_data['item_size'] == 'l', 'item_size'] = '4'
known_data.loc[known_data['item_size'] == 'xl', 'item_size'] = '5'
known_data.loc[known_data['item_size'] == 'xxl', 'item_size'] = '6'
known_data.loc[known_data['item_size'] == 'xxxl', 'item_size'] = '7'

unknown_data.loc[unknown_data['item_size'] == 'unsized', 'item_size'] = '0'
unknown_data.loc[unknown_data['item_size'] == 'xs', 'item_size'] = '1'
unknown_data.loc[unknown_data['item_size'] == 's', 'item_size'] = '2'
unknown_data.loc[unknown_data['item_size'] == 'm', 'item_size'] = '3'
unknown_data.loc[unknown_data['item_size'] == 'l', 'item_size'] = '4'
unknown_data.loc[unknown_data['item_size'] == 'xl', 'item_size'] = '5'
unknown_data.loc[unknown_data['item_size'] == 'xxl', 'item_size'] = '6'
unknown_data.loc[unknown_data['item_size'] == 'xxxl', 'item_size'] = '7'



# Change item_size string column to float.
known_data['item_size'] = pd.to_numeric(known_data.item_size)
unknown_data['item_size'] = pd.to_numeric(unknown_data.item_size)



# Normalise size data by item.
def mad_normalise(initial_size, sizes_median, mad):
    return (float(initial_size) - sizes_median) / mad

def normal_normalise(initial_size, sizes_mean, sizes_sdev):
    return (float(initial_size) - sizes_mean) / sizes_sdev

for item in known_data['item_id'].unique():
    item_df = known_data.loc[known_data['item_id'] == item]
    item_sizes = [float(size) for size in item_df['item_size']]
    sizes_median = np.median(item_sizes)
    mad = robust.mad(item_sizes)
    if mad == 0:
        replacement_dict = {initial_size: 0 for initial_size in item_sizes}
    else:
        replacement_dict = {initial_size: mad_normalise(initial_size, sizes_median, mad) for initial_size in item_sizes}

    for size in item_sizes:
        known_data.loc[(known_data['item_id'] == item) & (known_data['item_size'] == size), 'item_size'] = replacement_dict[size]

for item in unknown_data['item_id'].unique():
    item_df = unknown_data.loc[unknown_data['item_id'] == item]
    item_sizes = [float(size) for size in item_df['item_size']]
    sizes_median = np.median(item_sizes)
    mad = robust.mad(item_sizes)
    if mad == 0:
        replacement_dict = {initial_size: 0 for initial_size in item_sizes}
    else:
        replacement_dict = {initial_size: mad_normalise(initial_size, sizes_median, mad) for initial_size in item_sizes}

    for size in item_sizes:
        unknown_data.loc[(unknown_data['item_id'] == item) & (unknown_data['item_size'] == size), 'item_size'] = replacement_dict[size]



# Preparation for unknown_data.
unknown_data['item_id'] = unknown_data.apply(lambda row: (row['item_id'] * 2) + 2, axis = 1)
unknown_data['brand_id'] = unknown_data.apply(lambda row: row['brand_id'] + 100, axis = 1)

known_data = pd.read_csv('term_project/known_modified.csv')
unknown_data = pd.read_csv('term_project/unknown_modified.csv')

known_data = known_data.drop(['user_state'], axis = 1)
unknown_data = unknown_data.drop(['user_state'], axis = 1)

known_data = known_data.drop(['Unnamed: 0'], axis = 1)
unknown_data = unknown_data.drop(['Unnamed: 0'], axis = 1)

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
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_colour_item'] = .5
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_size_item'] = .5
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_item_id'] = .5

matched_item_dict_colour = {}
matched_item_dict_size = {}
matched_item_dict_item = {}
for item in matched_ids:
    matched_item_dict_colour.update({item: known_data.loc[known_data['item_id'] == item]['ce_colour_item'].unique()[0]})
    matched_item_dict_size.update({item: known_data.loc[known_data['item_id'] == item]['ce_size_item'].unique()[0]})
    matched_item_dict_item.update({item: known_data.loc[known_data['item_id'] == item]['ce_item_id'].unique()[0]})

for item in unknown_data['item_id'].unique():
    if item in matched_ids:
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_colour_item'] = matched_item_dict_colour[item]
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_size_item'] = matched_item_dict_size[item]
        unknown_data.loc[unknown_data['item_id'] == item, 'ce_item_id'] = matched_item_dict_item[item]

for type in unknown_data['type'].unique():
    unknown_data.loc[unknown_data['type'] == type, 'ce_colour_type'] = known_data.loc[known_data['type'] == type]['ce_colour_type'].unique()[0]
    unknown_data.loc[unknown_data['type'] == type, 'ce_size_type'] = known_data.loc[known_data['type'] == type]['ce_size_type'].unique()[0]

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
        unknown_data.loc[unknown_data['brand_id'] == brand, 'ce_brand_id'] = .5

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

for user in unknown_data['user_id'].unique():
    if user in unmatched_users:
        unknown_data.loc[unknown_data['user_id'] == user, 'ce_user_id'] = .5

matched_users_dict = {}
for user in matched_users:
    matched_users_dict.update({user: known_data.loc[known_data['user_id'] == user]['ce_user_id'].unique()[0]})

for user in unknown_data['user_id'].unique():
    if user in matched_users:
        unknown_data.loc[unknown_data['user_id'] == user, 'ce_user_id'] = matched_users_dict[user]

# Title
for title in unknown_data['user_title'].unique():
    unknown_data.loc[unknown_data['user_title'] == title, 'ce_user_title'] = known_data.loc[known_data['user_title'] == title]['ce_user_title'].unique()[0]



# Prepare Random Forests model.
known_data.head(10)
len(list(known_data))
columns = list(known_data.columns.values)

test_known = df(known_data)
test_unknown = df(unknown_data)
remove_list = [
 'ce_colour_item',
 'ce_colour_type',
 'ce_size_item',
 'ce_size_type',
 'ce_user_title',
]

for remove in remove_list:
    test_known = test_known.drop([remove], axis = 1)
    test_unknown = test_unknown.drop([remove], axis = 1)

test_known['target'] = target
test_known.head(5)
test_unknown.head(5)
len(list(test_known))
len(list(test_unknown))

train_data.head(10)
len(list(train_data))

unknown_data.head(10)
len(list(unknown_data))

# Remove user_title strings.
known_data = known_data.drop(['user_title'], axis = 1)
unknown_data = unknown_data.drop(['user_title'], axis = 1)

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

# Define train data / target.
train_data = test_known
target = df(test_known['target'])
target = target.values.ravel()
train_data.pop('target')

# Check for null values
columns = list(known_data.columns.values)
null_count = df(known_data.count(), columns=['count_null'])
null_count['count_null'] = len(known_data) - null_count['count_null']
null_count

columns = list(unknown_data.columns.values)
null_count = df(unknown_data.count(), columns=['count_null'])
null_count['count_null'] = len(unknown_data) - null_count['count_null']
null_count

# Create split sets for cross-validation.
x_train, x_test, y_train, y_test = train_test_split(test_known, target, test_size = 0.25)

# Define classifier and fit.
random_forest = RandomForestClassifier(n_estimators = 200,
                                       max_depth = 15,
                                       verbose = 3)

random_forest.fit(train_data, target)

# Make predictions and evaluate cross-validation score.
prediction = random_forest.predict(test_unknown)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, prediction)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# Save prediction_frame as submit-able file
prediction_frame = df(unknown_data_order_id, columns = ['order_item_id'])
prediction_frame['return'] = prediction
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/term_project/prediction.csv'
prediction_frame.to_csv(prediction_frame_path, index = False)

######################################################################################################



######################################################################################################


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 30, stop = 1000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
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
               'bootstrap': bootstrap}

pprint(random_grid)

# Output modified data frames to .csv files.
path_known = '/Users/alextruesdale/Documents/business-analytics/term_project/known_modified_2.csv'
path_unknown = '/Users/alextruesdale/Documents/business-analytics/term_project/unknown_modified_2.csv'

known_data.to_csv(path_known)
unknown_data.to_csv(path_unknown)
