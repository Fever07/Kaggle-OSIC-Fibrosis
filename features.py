import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from constants import CATEGORICAL_FEATURES, EPS
from preprocess import encode_categorical, row_per_patient
import sklearn
from sklearn.feature_selection import SelectKBest, RFECV, f_regression, mutual_info_regression
import itertools

NORMALIZER_FEATURES = sklearn.preprocessing.MinMaxScaler()
FIT_COLUMNS_FEATURES = ['FVC_perc']
FEATS_TO_AVERAGE = [
    'FVC_last_first',
    'Weeks_last_first',
    'Coef_last_first',
    'FVC_last',
    'Weeks_last'
]
ADDED_CATEGORICAL_FEATURES = {}
FEATURE_AVERAGE_ALPHA = 5

def add_features_before_normalization(df):
    df = df.copy()

    for patient in df['Patient'].unique():
        fvc = df[df['Patient'] == patient]['FVC']
        percent = df[df['Patient'] == patient]['Percent']

        df.loc[df['Patient'] == patient, 'FVC_perc'] = np.mean(fvc / percent) * 100.

    return df

def add_categorical_features(df):
    df = df.copy()

    age_min, age_max = df['Age'].min(), df['Age'].max() + 1
    # bins = np.linspace(age_min, age_max, num_bins + 1)
    # print(bins)
    bins = [0, 65, 70, 120]
    num_bins = len(bins) - 1
    age_bin = np.digitize(df['Age'], bins) - 1
    print(np.unique(age_bin, return_counts=True))
    for i in range(num_bins):
        df[f'Age_bin_{i}'] = (age_bin == i).astype(float)
    ADDED_CATEGORICAL_FEATURES['Age_bin'] = [f'Age_bin_{i}' for i in range(num_bins)]
    return df

def add_features_after_normalization(df_dest, df_src):
    df_dest = df_dest.copy()
    df_src = df_src.copy()

    df_src['FVC_last_first'] = df_src.groupby('Patient').transform('last')['FVC'] - df_src.groupby('Patient').transform('first')['FVC']
    df_src['Weeks_last_first'] = df_src.groupby('Patient').transform('last')['Weeks'] - df_src.groupby('Patient').transform('first')['Weeks']
    df_src['Coef_last_first'] = df_src['FVC_last_first'] / df_src['Weeks_last_first']
    df_src['FVC_last'] = df_src.groupby('Patient').transform('last')['FVC']
    df_src['Weeks_last'] = df_src.groupby('Patient').transform('last')['Weeks']

    feats_to_average = FEATS_TO_AVERAGE

    pat_df_dest = row_per_patient(df_dest)
    pat_df_src = row_per_patient(df_src)

    categorical_features = {**CATEGORICAL_FEATURES, **ADDED_CATEGORICAL_FEATURES}

    # Averaging by tuple of 1, 2 or 3 categorical features            
    
    for NUM_FEATURES in [1, 2, 3]:
        pairs_categorical_features = list(itertools.combinations(categorical_features, NUM_FEATURES))

        global_average = pat_df_src[feats_to_average].mean()
        for pair_categorical in pairs_categorical_features:
            pair_values = list(itertools.product(*[categorical_features[feature] for feature in pair_categorical]))
            new_feats = np.array([f'{feat_to_average}_average_by_{pair_categorical}' for feat_to_average in feats_to_average])
            for values_tuple in pair_values:
                values_conditions_src = [pat_df_src[value] == 1.0 for value in values_tuple]
                values_tuple_conditions_src = values_conditions_src[0]
                for value_condition in values_conditions_src:
                    values_tuple_conditions_src &= value_condition
                
                values_conditions_dest = [df_dest[value] == 1.0 for value in values_tuple]
                values_tuple_conditions_dest = values_conditions_dest[0]
                for value_condition in values_conditions_dest:
                    values_tuple_conditions_dest &= value_condition

                pair_df_src = pat_df_src[values_tuple_conditions_src]
                K = pair_df_src.shape[0]
                if K == 0:
                    df_dest.loc[values_tuple_conditions_dest, new_feats] = global_average[feats_to_average].values
                else:
                    average_by_category = pair_df_src[feats_to_average].mean()
                    isnan_mask = np.isnan(average_by_category)

                    if sum(isnan_mask) != 0:
                        df_dest.loc[values_tuple_conditions_dest, new_feats[isnan_mask]] = global_average[np.array(feats_to_average)[isnan_mask]].values
                    if sum(isnan_mask) != len(isnan_mask):
                        df_dest.loc[values_tuple_conditions_dest, new_feats[~isnan_mask]] = ((average_by_category[np.array(feats_to_average)[~isnan_mask]] * K + global_average[np.array(feats_to_average)[~isnan_mask]] * FEATURE_AVERAGE_ALPHA) / (K + FEATURE_AVERAGE_ALPHA)).values

    return df_dest

def init_features_normalizer(df):
    numeric_df = df[FIT_COLUMNS_FEATURES]
    NORMALIZER_FEATURES.fit(numeric_df)

def select_best_features(X, y, keep_features=None, k=2):    
    skb = SelectKBest(f_regression, k=k)
    if keep_features:
        keep_inds = np.array(keep_features)
        fit_inds = np.array(list(set(np.arange(0, X.shape[1])).difference(set(keep_inds))))
        skb.fit(X[:, fit_inds], y)
        _transform = skb.transform
        def transform_and_keep(_X):
            _X_transformed = np.zeros(shape=[_X.shape[0], k + len(keep_features)], dtype=_X.dtype)
            _X_transformed[:, :k] = _transform(_X[:, fit_inds])
            _X_transformed[:, k:] = _X[:, keep_inds]
            return _X_transformed
        skb.transform = transform_and_keep
        return skb
    else:
        skb.fit(X, y)
        return skb

def normalize_features(df):
    df = df.copy()
    # df[FIT_COLUMNS_FEATURES] = NORMALIZER_FEATURES.transform(df[FIT_COLUMNS_FEATURES])

    from preprocess import NORMALIZER, FVC_RANGE
    fvc_min = FVC_RANGE[0]
    fvc_max = FVC_RANGE[1]
    df['FVC_perc'] = (df['FVC_perc'] - fvc_min) / (fvc_max - fvc_min)
    return df
