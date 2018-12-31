# -*- coding: utf-8 -*-

'''
Created Dec 25, 2018
@author: Alex Truesdale

Jupyter Notebook Kernel (via Hydrogen / Atom) for working on raw challenge data.
'''

import os
import string
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
from statsmodels import robust
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

pd.options.display.max_rows = 999
df = df

# Read in data sets.
known_data = pd.read_csv('term_project/BADS_WS1819_known.csv')
unknown_data = pd.read_csv('term_project/BADS_WS1819_unknown.csv')

# Initial data exploration.
known_data.dtypes

unknown_data['item_id'].value_counts()[:15]
known_data['item_id'].value_counts()[:15]

known_data['brand_id'].value_counts()

# User State feature inspection.
state_group = df(known_data.groupby('user_state')['return'].sum())
state_group['count'] = known_data.groupby('user_state')['user_id'].count()
state_group['ratio'] = state_group['return'] / state_group['count']
state_group

known_data = known_data.drop(['user_state'], axis = 1)

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
known_data['item_size'] = known_data['item_size'].str.lower()
known_data['item_size'] = known_data['item_size'].str.replace('+', '.5', regex = False)

unknown_data['item_size'] = unknown_data['item_size'].str.lower()
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

# Compute conditional_expectation values for colour.
for item in known_data['item_id'].unique():
    item_df = known_data.loc[known_data['item_id'] == item]
    item_colours = [colour for colour in item_df['item_color']]

    colours_df = df(item_df.groupby('item_color')['return'].sum())
    colours_df['count'] = item_df.groupby('item_color')['return'].count()
    colours_df['ratio'] = colours_df['return'] / colours_df['count']

    for colour in colours_df.index:
        conditional_expectation = float(colours_df.loc[colours_df.index == colour]['ratio'])
        known_data.loc[(known_data['item_id'] == item) & (known_data['item_color'] == colour), 'ce_colour_item'] = conditional_expectation

for type in known_data['type'].unique()[:2]:
    print(type)
    type_df = known_data.loc[known_data['type'] == type]
    type_colours = [colour for colour in type_df['item_color']]

    colours_df = df(type_df.groupby('item_color')['return'].sum())
    colours_df['count'] = type_df.groupby('item_color')['return'].count()
    colours_df['ratio'] = colours_df['return'] / colours_df['count']

    for colour in colours_df.index:
        conditional_expectation = float(colours_df.loc[colours_df.index == colour]['ratio'])
        known_data.loc[(known_data['type'] == type) & (known_data['item_color'] == colour), 'ce_colour_type'] = conditional_expectation

known_data = known_data.drop(['item_color'], axis = 1)

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

# Create 'time_to_delivery' feature.
known_data['time_to_delivery_days'] = (known_data['delivery_date'] - known_data['order_date']).dt.days
known_data['time_to_delivery_days'].fillna(known_data['time_to_delivery_days'].median(), inplace = True)

unknown_data['time_to_delivery_days'] = (unknown_data['delivery_date'] - unknown_data['order_date']).dt.days
unknown_data['time_to_delivery_days'].fillna(unknown_data['time_to_delivery_days'].median(), inplace = True)

bins = [-.01, 1, 3, 7, 14, 31, 62, 160]
labels = list(range(1, len(bins)))
known_data['delivery_date_binned'] = pd.cut(known_data['time_to_delivery_days'], bins = bins, labels = labels, include_lowest = True)
unknown_data['delivery_date_binned'] = pd.cut(unknown_data['time_to_delivery_days'], bins = bins, labels = labels, include_lowest = True)

plt.hist(known_data['delivery_date_binned'], bins = None)

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

for title in titles_df.index:
    conditional_expectation = float(titles_df.loc[titles_df.index == title]['ratio'])
    known_data.loc[known_data['user_title'] == title, 'user_title'] = conditional_expectation

# Conditional expectation for item_id.
items_df = df(known_data.groupby('item_id')['return'].sum())
items_df['count'] = known_data.groupby('item_id')['return'].count()
items_df['ratio'] = items_df['return'] / items_df['count']

for title in items_df.index:
    conditional_expectation = float(items_df.loc[items_df.index == title]['ratio'])
    known_data.loc[known_data['item_id'] == title, 'item_id'] = conditional_expectation

# Conditional expectation for brand_id.
brands_df = df(known_data.groupby('brand_id')['return'].sum())
brands_df['count'] = known_data.groupby('brand_id')['return'].count()
brands_df['ratio'] = brands_df['return'] / brands_df['count']

for title in brands_df.index:
    conditional_expectation = float(brands_df.loc[brands_df.index == title]['ratio'])
    known_data.loc[known_data['brand_id'] == title, 'brand_id'] = conditional_expectation

# Conditional expectation for user_id.
user_df = df(known_data.groupby('user_id')['return'].sum())
user_df['count'] = known_data.groupby('user_id')['return'].count()
user_df['ratio'] = user_df['return'] / user_df['count']

for title in user_df.index:
    conditional_expectation = float(user_df.loc[user_df.index == title]['ratio'])
    known_data.loc[known_data['user_id'] == title, 'user_id'] = conditional_expectation

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

    print(item)
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

    print(item)
    for size in item_sizes:
        unknown_data.loc[(unknown_data['item_id'] == item) & (unknown_data['item_size'] == size), 'item_size'] = replacement_dict[size]

# Preparation for unknown_data.


# create_woe_frame(fiveohninesix, 'item_size', 4)
# def create_woe_frame(in_frame, feature, item_id):
#     value_dict_list = []
#     for value in in_frame[feature].unique():
#         value_dict = {}
#         value_dict['item_id'] = item_id
#         value_dict['value'] = value
#         value_dict['count_occurences'] = in_frame[feature].value_counts()[value]
#         value_dict['count_returns'] = in_frame.loc[in_frame[feature] == value, 'return'].sum()
#         value_dict['count_no_returns'] = value_dict['count_occurences'] - value_dict['count_returns']
#         value_dict_list.append(value_dict)
#
#     feature_df = df(value_dict_list)
#     feature_df['return_rate'] = feature_df['count_returns'] / feature_df['count_occurences']
#     feature_df['non_return_rate'] = feature_df['count_no_returns'] / feature_df['count_occurences']
#     feature_df['distribution_item_returns'] = feature_df['count_returns'] / feature_df.sum().count_returns
#     feature_df['distribution_item_no_returns'] = feature_df['count_no_returns'] / feature_df.sum().count_no_returns
#
#     feature_df['WOE'] = np.log(feature_df['distribution_item_returns'] / feature_df['distribution_item_no_returns'])
#     feature_df['IV'] = (feature_df['distribution_item_returns'] - feature_df['distribution_item_no_returns']) * np.log(feature_df['distribution_item_returns'] / feature_df['distribution_item_no_returns'])
#     feature_df = feature_df.replace([np.inf, -np.inf], 0)
#     feature_df['IV'] = feature_df['IV'].sum()
#
#     feature_df.plot(x = 'state', y = 'return_rate', style = 'o')
