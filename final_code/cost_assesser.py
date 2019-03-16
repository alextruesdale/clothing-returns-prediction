# -*- coding: utf-8 -*-

"""
Created Mar 12, 2019
@author: Alex Truesdale
@email: alex.truesdale@colorado.edu

Jupyter Notebook Kernel (via Hydrogen / Atom) for applying cost function to predictions.
"""

import pandas as pd
import numpy as np
import operator

# Simplify DataFrame build function.
df = pd.DataFrame

# Read in saved features / frames.
unknown = pd.read_csv('modified_features/unknown_modified_final.csv')
predictions = pd.read_csv('predictions/ensemble_predictors/prediction_09.csv')

# Create df with item prices and prediction values.
price_df_unknown = df(unknown['item_price'])
price_df_unknown['prediction'] = predictions.iloc[:, 1]

# Calculate cost-minimal τ ratio.
c_bG = 0.5 * -unknown['item_price']
c_gB = 0.5 * 5 * -(3 + 0.1 * unknown['item_price'])
price_df_unknown['τ'] = c_bG / (c_bG + c_gB)

# Define 'return' if prediction value is greater than τ.
multiplier = 1
price_df_unknown['return'] = price_df_unknown.apply(lambda row: 1 if (row['τ'] * multiplier) <= row['prediction'] else 0, axis = 1)
len(price_df_unknown[price_df_unknown['return'] == 1])

# Define output frame.
predictions_final = df(price_df_unknown.iloc[:, -1])
predictions_final['order_item_id'] = list(predictions.iloc[:, 0])
predictions_final = predictions_final.iloc[:, [1, 0]]

# Output final predictions to .csv file.
prediction_frame_path = '/Users/alextruesdale/Documents/business-analytics/predictions/final_prediction_binary.csv'
predictions_final.to_csv(prediction_frame_path, index = False)
