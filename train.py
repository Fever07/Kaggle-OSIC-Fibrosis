from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import KFold as KF
from model import get_models, is_net
import numpy as np
from validate import validate
from preprocess import normalize_df, drop_first_point, remove_outliers
from features import select_best_features, add_features_after_normalization
from utils import _laplace

def getX(df):
    if 'Patient_Week' in df.columns:
        return df.drop(columns=['Patient', 'FVC', 'Percent', 'Patient_Week']).values
    else:
        return df.drop(columns=['Patient', 'FVC', 'Percent']).values
    
def getY(df):
    return df['FVC'].values

def get_X_y(df):
    return getX(df), getY(df)

def train_val_split(df, random_state=2020):
    skf = KF(n_splits=5, shuffle=True, random_state=random_state)
    patient_ids = df['Patient'].unique()
    for train_idx, val_idx in skf.split(patient_ids):
        train_patients = patient_ids[train_idx]
        val_patients = patient_ids[val_idx]
        yield df[df['Patient'].isin(train_patients)].reset_index(drop=True), \
                df[df['Patient'].isin(val_patients)].reset_index(drop=True), 

def train(df_train_coef, df_train, random_states, include_nn=True):
    preds = []
    maes = []
    clfs = []
    skbs = []

    scores = []

    for random_state in random_states:
        print(f'Seed: {random_state}')
        for fold, (train_df, val_df) in enumerate(train_val_split(df_train_coef, random_state=random_state)):
            print(f'Fold: {fold}')  
            train_df = add_features_after_normalization(train_df, train_df)
            val_df = add_features_after_normalization(val_df, train_df)
            train_df = drop_first_point(train_df)
            val_df = drop_first_point(val_df)

            # train_df.to_csv('train_df.csv')
            # val_df.to_csv('val_df.csv')
            # exit()

            X_train, y_train = get_X_y(train_df)
            X_val, y_val = get_X_y(val_df)

            skb = select_best_features(X_train, y_train, k=2)
            scores.append(skb.scores_)
            X_train = skb.transform(X_train)
            X_val = skb.transform(X_val)
            skbs.append(skb)

            _clfs = get_models(num_features=X_train.shape[1], include_nn=include_nn, random_state=random_state)
            train_fold_preds = []
            val_fold_preds = []
            for clf in _clfs:
                if is_net(clf):
                    clf.fit(X_train, y_train, epochs=20, verbose=0)
                else:
                    clf.fit(X_train, y_train)
                train_fold_preds.append(clf.predict(X_train).flatten())
                val_fold_preds.append(clf.predict(X_val).flatten())

            clfs.append(_clfs)
            train_fold_preds = np.mean(train_fold_preds, axis=0)
            val_fold_preds = np.mean(val_fold_preds, axis=0)

            train_weeks_df = df_train[df_train['Patient'].isin(train_df['Patient'])]
            val_weeks_df = df_train[df_train['Patient'].isin(val_df['Patient'])]

            _mae = validate(val_weeks_df, val_fold_preds)
            maes.append(_mae)

    # print(f'MAES: {maes}') 
    val_mae = np.mean(maes)
    print(f'MAE: {val_mae}')
    print(f'STD: {np.std(maes)}')
    print(f'Laplace: {_laplace(val_mae, val_mae * np.sqrt(2))}')

    # import pickle
    # with open('scores.pkl', 'wb') as file:
    #     pickle.dump(train_df.drop(columns=['Patient', 'FVC', 'Percent']).columns, file)
    #     pickle.dump(scores, file)

    return clfs, skbs, val_mae