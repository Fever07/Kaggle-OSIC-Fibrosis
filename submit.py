import pandas as pd
import numpy as np
from preprocess import normalize_df, denormalize, add_first_point
from train import getX
from constants import WEEKS, MIN_WEEK, MAX_WEEK
from features import encode_categorical, add_features_before_normalization, add_categorical_features, add_features_after_normalization, normalize_features

def add_weeks_to_test(df):
    df = df.copy()
    patient_ids = df['Patient']

    df = pd.concat([df] * len(WEEKS), ignore_index=True)
    df = add_first_point(df)
    df['Patient_Week'] = [f'{patient_id}_{week}' for (patient_id, week) in zip(df['Patient'], np.repeat(np.arange(MIN_WEEK, MAX_WEEK + 1), len(patient_ids)))]
    df['Weeks'] = np.array([WEEKS.tolist()] * len(patient_ids)).T.flatten()
    return df

def submit_preds(df_test, df_train, clfs, skbs, sigma):
    df_test = df_test.copy()
    df_test = encode_categorical(df_test)

    df_test = add_features_before_normalization(df_test)
    df_test = add_categorical_features(df_test)

    df_test = normalize_df(df_test)
    df_test = normalize_features(df_test)

    df_test = add_weeks_to_test(df_test)
    df_test.to_csv('df_test.csv')

    df_test = add_features_after_normalization(df_test, df_train)

    X_test = getX(df_test)
    preds = []
    for fold_clfs, skb in zip(clfs, skbs):
        X_test_best = skb.transform(X_test)
        for clf in fold_clfs:
            fvc_pred = clf.predict(X_test_best).flatten()
            preds.append(fvc_pred)
    preds = np.mean(preds, axis=0)

    df_test['FVC'] = denormalize(preds)
    submit_df = df_test[['FVC', 'Patient_Week']]
    submit_df['Confidence'] = sigma
    submit_df = submit_df.reindex(columns=['Patient_Week', 'FVC', 'Confidence'])
    submit_df.to_csv('submission.csv', index=False)

    print(submit_df.head())