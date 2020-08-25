import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from constants import DATA_DIR
from preprocess import init_normalizer, remove_duplicates, prepare_df, encode_categorical, normalize_df, row_per_patient, drop_first_point, add_first_point
from train import train
from submit import submit_preds
from features import add_features_before_normalization, add_categorical_features, add_features_after_normalization, init_features_normalizer, normalize_features, select_best_features
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    print(f'Train shape: {df_train.shape}, Test shape: {df_test.shape}')

    df_train = remove_duplicates(df_train)
    print(f'Train shape, removed duplicates: {df_train.shape}')

    df_train = encode_categorical(df_train)

    df_train = add_features_before_normalization(df_train)
    df_train = add_categorical_features(df_train)

    init_normalizer(pd.concat([df_train, df_test]))
    df_train = normalize_df(df_train)
    init_features_normalizer(df_train)
    df_train = normalize_features(df_train)

    df_train = add_first_point(df_train)
    df_train.to_csv('df_train.csv')
    print(drop_first_point(df_train).shape)

    random_states = [
        41,
        81, 
        901, 
        1337,
        2020
    ]

    clfs, skbs, mean_mae = train(df_train, df_train, include_nn=False, random_states=random_states)
    submit_preds(df_test, df_train, clfs, skbs, mean_mae * np.sqrt(2))
