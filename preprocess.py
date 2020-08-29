import numpy as np
import pandas as pd
import os
from constants import MIN_WEEK, MAX_WEEK, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
import sklearn
import sklearn.linear_model as lm
from utils import detect_outlier

NORMALIZER = sklearn.preprocessing.MinMaxScaler()
# Weeks fitted separately
FIT_COLUMNS = list(set(NUMERICAL_FEATURES).difference({'Weeks'}))  
FVC_RANGE = []

def remove_duplicates(df_train):
    df_train['FVC'] = df_train.groupby(['Patient', 'Weeks'])['FVC'].transform('mean')
    df_train['Percent'] = df_train.groupby(['Patient', 'Weeks'])['Percent'].transform('mean')
    df_train.drop_duplicates(inplace=True)
    df_train.reset_index(drop=True)
    return df_train

def remove_outliers(df):
    df = df.copy()
    index = []
    for patient in df['Patient'].unique():
        pat_df = df[df['Patient'] == patient]
        i = detect_outlier(pat_df['Weeks'], pat_df['FVC'])
        if i < len(pat_df) - 3:
            index.append(pat_df.index[i])

    df.drop(index=index, inplace=True)
    return df

def row_per_patient(df):
    return pd.concat([
        pd.Series(df['Patient'].unique(), name='Patient'),
        df.groupby(['Patient']).transform('first').drop_duplicates().reset_index(drop=True)
    ], axis=1)

def drop_first_point(df):
    df = df.copy()
    first_weeks = df.groupby('Patient').transform('first').drop_duplicates()
    df.drop(first_weeks.index, inplace=True)
    return df

def add_first_point(df):
    df = df.copy()
    first_weeks = df.groupby('Patient').transform('first')
    df['Weeks_first'] = first_weeks['Weeks']
    df['FVC_first'] = first_weeks['FVC']
    df['Percent_first'] = first_weeks['Percent']
    return df

def init_normalizer(df):
    numeric_df = df[FIT_COLUMNS]
    NORMALIZER.fit(numeric_df)

    # Store fvc min/max values separately
    FVC_RANGE.append(NORMALIZER.data_min_[FIT_COLUMNS.index('FVC')])
    FVC_RANGE.append(NORMALIZER.data_max_[FIT_COLUMNS.index('FVC')])        

def normalize_df(df):
    df = df.copy()
    df['Weeks'] = (df['Weeks'] - MIN_WEEK) / (MAX_WEEK - MIN_WEEK)
    df[FIT_COLUMNS] = NORMALIZER.transform(df[FIT_COLUMNS])
    return df

def encode_categorical(df):
    df = df.copy()
    for feature in CATEGORICAL_FEATURES:
        values = CATEGORICAL_FEATURES[feature]
        for val in values:
            df[val] = df[feature].map(lambda x: float(x == val))

    df.drop(columns=CATEGORICAL_FEATURES.keys(), inplace=True)
    return df

def decode_categorical(df):
    from features import ADDED_CATEGORICAL_FEATURES
    df = df.copy()
    categorical_features = {**CATEGORICAL_FEATURES, **ADDED_CATEGORICAL_FEATURES}
    for feature in categorical_features:
        values = categorical_features[feature]
        for val in values:
            df.loc[df[val] == 1.0, feature] = values.index(val)
    
        df.drop(columns=values, inplace=True)

    return df

def prepare_df(df):
    df = df.copy()
    encode_categorical(df)
    add_coef(df)
    df_coef = row_per_patient(df)
    return df_coef  

def denormalize(y):
    fvc_min = FVC_RANGE[0]
    fvc_max = FVC_RANGE[1]
    return y * (fvc_max - fvc_min) + fvc_min
