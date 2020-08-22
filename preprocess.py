import numpy as np
import pandas as pd
import os
from constants import MIN_WEEK, MAX_WEEK, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
import sklearn
import sklearn.linear_model as lm
from utils import get_coef

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

def row_per_patient(df):
    return pd.concat([
        pd.Series(df['Patient'].unique(), name='Patient'),
        df.groupby(['Patient']).transform('first').drop_duplicates().reset_index(drop=True)
    ], axis=1)

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

def add_coef(df):
    df = df.copy()
    for patient_id in df['Patient'].unique():
        pat_df = df[df['Patient'] == patient_id]
        coef = get_coef(pat_df['Weeks'].values, pat_df['FVC'].values)
        df.loc[df['Patient'] == patient_id, 'Coef'] = [coef] * (df['Patient'] == patient_id).sum()

    return df
