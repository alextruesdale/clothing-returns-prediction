# -*- coding: utf-8 -*-

"""
Created Feb 8, 2019
@author: Alex Truesdale
@email: alex.truesdale@colorado.edu

Jupyter Notebook Kernel (via Hydrogen / Atom) for working on raw challenge data.
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels import robust
from pprint import pprint

# Simplify DataFrame build function.
df = pd.DataFrame

known = pd.read_csv('BADS_WS1819_known.csv')
unknown = pd.read_csv('BADS_WS1819_unknown.csv')

# Preparation for unknown_data IDs.
unknown['item_id'] = unknown.apply(lambda row: (row['item_id'] * 2) + 2, axis = 1)
unknown['brand_id'] = unknown.apply(lambda row: row['brand_id'] + 100, axis = 1)

# Brand CE.
brand_counts = df(known['brand_id'].value_counts())
brand_returns = df(known.groupby(['brand_id'])['return'].agg('sum'))
new_df_brand = brand_counts.join(brand_returns, on = brand_counts.index)
new_df_brand.columns = ['count', 'return']
new_df_brand['ce_brand'] = np.round(new_df_brand['return'] / new_df_brand['count'], 3)
new_df_brand.loc[new_df_brand['count'] < 100, 'ce_brand'] = .48

x_brand = list(new_df_brand.index)
y_brand = list(new_df_brand.ce_brand.values)
zip_dict_known_brand = dict(zip(x_brand, y_brand))
zip_dict_unknown_brand = {id: (zip_dict_known_brand[id] if id in zip_dict_known_brand.keys() else .48) for id in list(unknown['brand_id'].unique())}

known['ce_brand'] = known['brand_id'].map(zip_dict_known_brand)
unknown['ce_brand'] = unknown['brand_id'].map(zip_dict_unknown_brand)

known['ce_brand'] = round(known['ce_brand'], 2)
unknown['ce_brand'] = round(unknown['ce_brand'], 2)

# Item CE.
item_counts = df(known['item_id'].value_counts())
item_returns = df(known.groupby(['item_id'])['return'].agg('sum'))
new_df_item = item_counts.join(item_returns, on = item_counts.index)
new_df_item.columns = ['count', 'return']
new_df_item['ce_item'] = np.round(new_df_item['return'] / new_df_item['count'], 3)
new_df_item.loc[new_df_item['count'] < 40, 'ce_item'] = .48

x_item = list(new_df_item.index)
y_item = list(new_df_item.ce_item.values)
zip_dict_known_item = dict(zip(x_item, y_item))
zip_dict_unknown_item = {id: (zip_dict_known_item[id] if id in zip_dict_known_item.keys() else .48) for id in list(unknown['item_id'].unique())}

known['ce_item'] = known['item_id'].map(zip_dict_known_item)
unknown['ce_item'] = unknown['item_id'].map(zip_dict_unknown_item)

known['ce_item'] = round(known['ce_item'], 2)
unknown['ce_item'] = round(unknown['ce_item'], 2)

# User CE.
user_counts = df(known['user_id'].value_counts())
user_returns = df(known.groupby(['user_id'])['return'].agg('sum'))
new_df_user = user_counts.join(user_returns, on = user_counts.index)
new_df_user.columns = ['count', 'return']
new_df_user['ce_user'] = np.round(new_df_user['return'] / new_df_user['count'], 3)
new_df_user.loc[new_df_user['count'] < 5, 'ce_user'] = .48

x_user = list(new_df_user.index)
y_user = list(new_df_user.ce_user.values)
zip_dict_known_user = dict(zip(x_user, y_user))
zip_dict_unknown_user = {id: (zip_dict_known_user[id] if id in zip_dict_known_user.keys() else .48) for id in list(unknown['user_id'].unique())}

known['ce_user'] = known['user_id'].map(zip_dict_known_user)
unknown['ce_user'] = unknown['user_id'].map(zip_dict_unknown_user)

known['ce_user'] = round(known['ce_user'], 2)
unknown['ce_user'] = round(unknown['ce_user'], 2)

# Round price.
known.item_price = known.item_price.round()
unknown.item_price = unknown.item_price.round()

# Price CE.
price_counts = df(known['item_price'].value_counts())
price_returns = df(known.groupby(['item_price'])['return'].agg('sum'))
new_df_price = price_counts.join(price_returns, on = price_counts.index)
new_df_price.columns = ['count', 'return']
new_df_price['ce_price'] = np.round(new_df_price['return'] / new_df_price['count'], 3)
new_df_price.loc[new_df_price['count'] < 90, 'ce_price'] = .48

x_price = list(new_df_price.index)
y_price = list(new_df_price.ce_price.values)
zip_dict_known_price = dict(zip(x_price, y_price))
zip_dict_unknown_price = {id: (zip_dict_known_price[id] if id in zip_dict_known_price.keys() else .48) for id in list(unknown['item_price'].unique())}

known['ce_price'] = known['item_price'].map(zip_dict_known_price)
unknown['ce_price'] = unknown['item_price'].map(zip_dict_unknown_price)

known['ce_price'] = round(known['ce_price'], 2)
unknown['ce_price'] = round(unknown['ce_price'], 2)

# Sale or mark-up detection.
joined = pd.concat([known, unknown], sort = False)
joined['item_avg_price'] = joined.groupby(['item_id'])['item_price'].transform('mean')
joined['price_delta'] = joined['item_avg_price'] - joined['item_price']

known['price_delta'] = joined[:100000]['price_delta']
unknown['price_delta'] = joined[100000:]['price_delta']

known['is_discounted'] = known.apply(lambda row: 1 if row['price_delta'] > 0 else 0, axis = 1)
unknown['is_discounted'] = unknown.apply(lambda row: 1 if row['price_delta'] > 0 else 0, axis = 1)

known = known.drop(['price_delta'], axis = 1)
unknown = unknown.drop(['price_delta'], axis = 1)

# Multiple per-day orders.
joined = pd.concat([known, unknown], sort = False)
joined['orders_per_day'] = joined.groupby(['user_id', 'order_date'])['user_id'].transform('size')

known['orders_per_day'] = joined[:100000]['orders_per_day']
unknown['orders_per_day'] = joined[100000:]['orders_per_day']

# Multiple multi-colour per-day orders.
joined = pd.concat([known, unknown], sort = False)
joined['multi_colour_per_day'] = joined.groupby(['user_id', 'item_id', 'item_color'])['user_id'].transform('size')

known['multi_colour_per_day'] = joined[:100000]['multi_colour_per_day']
unknown['multi_colour_per_day'] = joined[100000:]['multi_colour_per_day']

# Fix misspells in colours.
known.loc[known['item_color'] == 'blau', 'item_color'] = 'blue'
known.loc[known['item_color'] == 'brwon', 'item_color'] = 'brown'
known.loc[known['item_color'] == 'oliv', 'item_color'] = 'olive'

unknown.loc[unknown['item_color'] == 'blau', 'item_color'] = 'blue'
unknown.loc[unknown['item_color'] == 'brwon', 'item_color'] = 'brown'
unknown.loc[unknown['item_color'] == 'oliv', 'item_color'] = 'olive'

# Colour CE by item.
colours_df = df(known.groupby(['item_id', 'item_color'])['return'].sum())
colours_df['count'] = known.groupby(['item_id', 'item_color'])['return'].count()
colours_df['ratio'] = colours_df['return'] / colours_df['count']
colours_df.loc[colours_df['count'] < 45, 'ratio'] = .48

key_colour_composite = list(colours_df.index)
value_colour_ce = list(colours_df.ratio.values)
zip_dict_colour_ce = dict(zip(key_colour_composite, value_colour_ce))
zip_dict_unknown_colour_ce = {id: (zip_dict_colour_ce[id] if id in zip_dict_colour_ce.keys() else .48) for id in list(df(unknown.groupby(['item_id', 'item_color'])['user_id'].count()).index)}

known['colour_item_ce'] = known.set_index(['item_id', 'item_color']).index.map(zip_dict_colour_ce.get)
unknown['colour_item_ce'] = unknown.set_index(['item_id', 'item_color']).index.map(zip_dict_unknown_colour_ce.get)

known['colour_item_ce'] = round(known['colour_item_ce'], 2)
unknown['colour_item_ce'] = round(unknown['colour_item_ce'], 2)

# Create 'count_item_user' feature.
joined = pd.concat([known, unknown], sort = False)
user_item_df = df(joined.groupby(['user_id', 'item_id'])['user_id'].agg('count'))

key_user_item = list(user_item_df.index)
value_user_item = list(user_item_df.user_id.values)
zip_user_item = dict(zip(key_user_item, value_user_item))

known['count_item_user'] = known.set_index(['user_id', 'item_id']).index.map(zip_user_item.get)
unknown['count_item_user'] = unknown.set_index(['user_id', 'item_id']).index.map(zip_user_item.get)

# Drop 'useless' features.
known = known.drop(['user_state'], axis = 1)
unknown = unknown.drop(['user_state'], axis = 1)

known = known.drop(['user_title'], axis = 1)
unknown = unknown.drop(['user_title'], axis = 1)

known = known.drop(['item_color'], axis = 1)
unknown = unknown.drop(['item_color'], axis = 1)

# Create customer_value feature.
joined = pd.concat([known, unknown], sort = False)
customer_value = df(joined.groupby(['user_id'])['item_price'].agg('sum'))

x_value = list(customer_value.index)
y_value = list(customer_value.item_price.values)
zip_dict_value = dict(zip(x_value, y_value))

known['customer_value'] = known['user_id'].map(zip_dict_value)
unknown['customer_value'] = unknown['user_id'].map(zip_dict_value)

# Create count_user feature.
joined = pd.concat([known, unknown], sort = False)
user_counts = df(joined['user_id'].value_counts())

x_count = list(user_counts.index)
y_count = list(user_counts.user_id.values)
zip_dict_count = dict(zip(x_count, y_count))

known['count_user'] = known['user_id'].map(zip_dict_count)
unknown['count_user'] = unknown['user_id'].map(zip_dict_count)

# Create new_user boolean.
known['new_user'] = known.apply(lambda row: 1 if row['count_user'] <= 1 else 0, axis = 1)
unknown['new_user'] = unknown.apply(lambda row: 1 if row['count_user'] <= 1 else 0, axis = 1)

# Fix lower-case upper-case issue in sizes.
known['item_size'] = known['item_size'].str.lower()
unknown['item_size'] = unknown['item_size'].str.lower()

# Fix sizes with plus signs.
known['item_size'] = known['item_size'].str.replace('+', '.5', regex = False)
unknown['item_size'] = unknown['item_size'].str.replace('+', '.5', regex = False)

# Size CE by item.
sizes_df = df(known.groupby(['item_id', 'item_size'])['return'].sum())
sizes_df['count'] = known.groupby(['item_id', 'item_size'])['return'].count()
sizes_df['ratio'] = sizes_df['return'] / sizes_df['count']
sizes_df.loc[sizes_df['count'] < 45, 'ratio'] = .48

key_size_composite = list(sizes_df.index)
value_size_ce = list(sizes_df.ratio.values)
zip_dict_size_ce = dict(zip(key_size_composite, value_size_ce))
zip_dict_unknown_size_ce = {id: (zip_dict_size_ce[id] if id in zip_dict_size_ce.keys() else .48) for id in list(df(unknown.groupby(['item_id', 'item_size'])['user_id'].count()).index)}

known['size_item_ce'] = known.set_index(['item_id', 'item_size']).index.map(zip_dict_size_ce.get)
unknown['size_item_ce'] = unknown.set_index(['item_id', 'item_size']).index.map(zip_dict_unknown_size_ce.get)

known['size_item_ce'] = round(known['size_item_ce'], 2)
unknown['size_item_ce'] = round(unknown['size_item_ce'], 2)

# Derive 'item types' by item.
joined = pd.concat([known, unknown], sort = False)
item_sizes = df(joined.groupby('item_id')['item_size'].apply(list))
x_size = list(item_sizes.index)
y_size = list(item_sizes.item_size.values)
zip_dict_size = dict(zip(x_size, y_size))

item_type_dict = {}
for item in joined['item_id'].unique():
    item_sizes = zip_dict_size[item]
    if any(['.5' in size for size in item_sizes]):
        item_type = 1
    elif all([size.isdigit() for size in item_sizes]):
        item_sizes = [float(size) for size in item_sizes]
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
    elif all([size == 'unsized' for size in item_sizes]):
        item_type = 7
    else:
        item_type = 2

    item_type_dict.update({item: item_type})

known['type'] = known['item_id'].map(item_type_dict)
unknown['type'] = unknown['item_id'].map(item_type_dict)

# Create dummies for type; drop type stacked.
known = known.join(pd.get_dummies(known['type'], prefix = 'type'))
unknown = unknown.join(pd.get_dummies(unknown['type'], prefix = 'type'))

known = known.drop(['type'], axis = 1)
unknown = unknown.drop(['type'], axis = 1)

# Assign string sizes numerical values.
string_replace_dict = {
    'unsized': '0',
    'xs': '1',
    's': '2',
    'm': '3',
    'l': '4',
    'xl': '5',
    'xxl': '6',
    'xxxl': '7'
}

known['item_size'] = known['item_size'].replace(string_replace_dict)
unknown['item_size'] = unknown['item_size'].replace(string_replace_dict)

# Change item_size string column to float.
known['item_size'] = pd.to_numeric(known.item_size)
unknown['item_size'] = pd.to_numeric(unknown.item_size)

# Normalise size data by item.
joined = pd.concat([known, unknown], sort = False)
item_sizes = df(joined.groupby('item_id')['item_size'].apply(list))
x_size = list(item_sizes.index)
y_size = list(item_sizes.item_size.values)
zip_dict_size = dict(zip(x_size, y_size))

def mad_normalise(initial_size, sizes_median, mad):
    return (float(initial_size) - sizes_median) / mad

replacement_dict_master = {}
for item in joined['item_id'].unique():
    item_sizes = zip_dict_size[item]
    item_sizes = [float(size) for size in item_sizes]
    sizes_median = np.median(item_sizes)
    mad = robust.mad(item_sizes)
    if mad == 0:
        replacement_dict = {(item, initial_size): 0 for initial_size in item_sizes}
    else:
        replacement_dict = {(item, initial_size): round(mad_normalise(initial_size, sizes_median, mad), 2) for initial_size in item_sizes}

    replacement_dict_master.update(replacement_dict)

known['item_size'] = known.set_index(['item_id', 'item_size']).index.map(replacement_dict_master.get)
unknown['item_size'] = unknown.set_index(['item_id', 'item_size']).index.map(replacement_dict_master.get)

# Change date columns to datetime features.
known['order_date'] = pd.to_datetime(known['order_date'])
known['delivery_date'] = pd.to_datetime(known['delivery_date'])
known['user_dob'] = pd.to_datetime(known['user_dob'])
known['user_reg_date'] = pd.to_datetime(known['user_reg_date'])

unknown['order_date'] = pd.to_datetime(unknown['order_date'])
unknown['delivery_date'] = pd.to_datetime(unknown['delivery_date'])
unknown['user_dob'] = pd.to_datetime(unknown['user_dob'])
unknown['user_reg_date'] = pd.to_datetime(unknown['user_reg_date'])

# Create 'was_delivered' feature.
known['was_delivered'] = known.apply(lambda row: 0 if pd.isnull(row['delivery_date']) else 1, axis = 1)
unknown['was_delivered'] = unknown.apply(lambda row: 0 if pd.isnull(row['delivery_date']) else 1, axis = 1)

# Create 'latest_order' feature.
joined = pd.concat([known, unknown], sort = False)
user_dates = df(joined.groupby('user_id')['order_date'].apply(list).apply(sorted))
x_date = list(user_dates.index)
y_date = list(user_dates.order_date.values)
zip_dict_date = dict(zip(x_date, y_date))

user_order_dict = {}
for user in joined['user_id'].unique():
    user_orders = zip_dict_date[user]
    diff_dict = {}
    for i, order in enumerate(user_orders):
        if i == 0:
            diff = -1
        else:
            diff = (order - user_orders[i - 1]).days

        if not (user, order) in diff_dict.keys():
            diff_dict.update({(user, order): diff})

    user_order_dict.update(diff_dict)

known['latest_order'] = known.set_index(['user_id', 'order_date']).index.map(user_order_dict.get)
unknown['latest_order'] = unknown.set_index(['user_id', 'order_date']).index.map(user_order_dict.get)

# Create 'ordered_item_recently' feature.
joined = pd.concat([known, unknown], sort = False)
user_item_dates = df(joined.groupby(['user_id', 'item_id'])['order_date'].apply(list).apply(sorted))
x_user_item = list(user_item_dates.index)
y_user_item = list(user_item_dates.order_date.values)
zip_user_item = dict(zip(x_user_item, y_user_item))

user_item_dict = {}
for user in joined['user_id'].unique():
    for item in joined['item_id'].unique():
        key = (user, item)
        if key in zip_user_item.keys():
            user_orders = zip_user_item[key]
            diff_dict = {}
            for i, order in enumerate(user_orders):
                new_key = (user, item, order)
                if i == 0:
                    diff = 0
                else:
                    diff = (order - user_orders[i - 1]).days

                if not new_key in diff_dict.keys():
                    diff_dict.update({new_key: diff})

            user_item_dict.update(diff_dict)

known['ordered_item_recently'] = known.set_index(['user_id', 'item_id', 'order_date']).index.map(user_item_dict.get)
unknown['ordered_item_recently'] = unknown.set_index(['user_id', 'item_id', 'order_date']).index.map(user_item_dict.get)

# Create 'membership_age' feature.
known['membership_age_days'] = (known['order_date'] - known['user_reg_date']).dt.days
known.loc[known['membership_age_days'] == -1, 'membership_age_days'] = 0

unknown['membership_age_days'] = (unknown['order_date'] - unknown['user_reg_date']).dt.days
unknown.loc[unknown['membership_age_days'] == -1, 'membership_age_days'] = 0

# Create 'time_to_delivery' feature.
known['time_to_delivery_days'] = (known['delivery_date'] - known['order_date']).dt.days
known.loc[known['time_to_delivery_days'] < 0, 'time_to_delivery_days'] = -1
known['time_to_delivery_days'].fillna(9999, inplace = True)

unknown['time_to_delivery_days'] = (unknown['delivery_date'] - unknown['order_date']).dt.days
unknown.loc[unknown['time_to_delivery_days'] < 0, 'time_to_delivery_days'] = -1
unknown['time_to_delivery_days'].fillna(9999, inplace = True)

# Create 'assumed_age' feature.
known['assumed_age'] = (known['order_date'] - known['user_dob']).dt.days / 365
known['assumed_age'].fillna(0, inplace = True)
known.assumed_age = known.assumed_age.round()

unknown['assumed_age'] = (unknown['order_date'] - unknown['user_dob']).dt.days / 365
unknown['assumed_age'].fillna(0, inplace = True)
unknown.assumed_age = unknown.assumed_age.round()

# Remove datetime date features.
known = known.drop(['order_date', 'delivery_date', 'user_reg_date', 'user_dob'], axis = 1)
unknown = unknown.drop(['order_date', 'delivery_date', 'user_reg_date', 'user_dob'], axis = 1)

# Output modified data frames to .csv files.
path_known = 'modified_features/known_modified_final.csv'
path_unknown = 'modified_features/unknown_modified_final.csv'

known.to_csv(path_known)
unknown.to_csv(path_unknown)
