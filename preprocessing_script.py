
import pandas as pd
import numpy as np

def preprocess_data(data):
    # Advanced Feature Engineering
    data['price_per_gram'] = data['unit_price'] / data['product_weight_g']
    data['freight_per_unit'] = data['freight_price'] / (data['qty'] + 1e-5)
    data['price_to_total'] = data['unit_price'] / (data['total_price'] + 1e-5)
    data['qty_to_weight'] = data['qty'] / (data['product_weight_g'] + 1e-5)
    data['comp_1_diff'] = data['unit_price'] - data['comp_1']
    data['comp_2_diff'] = data['unit_price'] - data['comp_2']
    data['comp_3_diff'] = data['unit_price'] - data['comp_3']
    data['comp_1_score_weighted'] = data['ps1'] / (data['comp_1'] + 1e-5)
    data['comp_2_score_weighted'] = data['ps2'] / (data['comp_2'] + 1e-5)
    data['comp_3_score_weighted'] = data['ps3'] / (data['comp_3'] + 1e-5)
    data['is_weekend'] = ((data['weekday'] == 0) & (data['weekend'] > 0)).astype(int)
    data['is_holiday'] = (data['holiday'] > 0).astype(int)
    data['name_desc_ratio'] = data['product_name_lenght'] / (data['product_description_lenght'] + 1e-5)
    data['photos_per_unit'] = data['product_photos_qty'] / (data['qty'] + 1e-5)
    skewed_features = ['qty', 'total_price', 'freight_price', 'product_weight_g', 's', 'volume']
    for feature in skewed_features:
        data[f'log_{feature}'] = np.log1p(data[feature])
    return data
    