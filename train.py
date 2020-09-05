from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import KFold as KF
from model import get_models, predict_confidence
import numpy as np
from validate import validate
from preprocess import normalize_df, drop_first_point, remove_outliers, decode_categorical, row_per_patient
from features import select_best_features, add_features_after_normalization
from utils import _laplace, laplace_optimal
import itertools
import pickle
from sklearn import tree

X_DROP_COLUMNS = ['Patient', 'FVC', 'FVC_best_diff', 'Percent']

def getX(df):
    if 'Patient_Week' in df.columns:
        return df.drop(columns=X_DROP_COLUMNS + ['Patient_Week']).values
    else:
        return df.drop(columns=X_DROP_COLUMNS).values
    
def getY(df):
    # return df['FVC'].values
    return df['FVC_best_diff'].values

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

def train(df_train_coef, df_train, random_states):
    preds = []
    train_maes = []
    maes = []
    train_sigmas = []
    sigmas = []
    clfs = []
    skbs = []

    scores = []

    for random_state in random_states:
        print(f'Seed: {random_state}')
        for fold, (train_df, val_df) in enumerate(train_val_split(df_train_coef, random_state=random_state)):
            print(f'Fold: {fold}')  
            # train_df = add_features_after_normalization(train_df, train_df)
            # val_df = add_features_after_normalization(val_df, train_df)
            train_df = row_per_patient(train_df)
            val_df = row_per_patient(val_df)

            train_df = train_df[['Patient', 'Currently smokes', 'Male', 'FVC', 'FVC_best_diff', 'Percent']]
            val_df = val_df[['Patient', 'Currently smokes', 'Male', 'FVC', 'FVC_best_diff', 'Percent']]

            # train_df.to_csv('train_df.csv')
            # val_df.to_csv('val_df.csv')
            # exit()

            X_train, y_train = get_X_y(train_df)
            X_val, y_val = get_X_y(val_df)

            train_fold_preds = []
            val_fold_preds = []

            _clfs = get_models(num_features=X_train.shape[1], random_state=random_state)
            for clf in _clfs:
                clf.fit(X_train, y_train)
                train_fold_preds.append(predict_confidence(clf, X_train))
                val_fold_preds.append(predict_confidence(clf, X_val))
            clfs.append(_clfs)
            
            train_fold_preds = np.mean(train_fold_preds, axis=0)
            val_fold_preds = np.mean(val_fold_preds, axis=0)

            with open(f'preds/seed_{random_state}_fold_{fold}.pkl', 'wb') as file:
                pickle.dump(train_fold_preds, file)
                pickle.dump(val_fold_preds, file)

            with open(f'clfs/tree_{random_state}_fold_{fold}.pkl', 'wb') as file:
                pickle.dump(_clfs[0], file)

            train_weeks_df = df_train[df_train['Patient'].isin(train_df['Patient'])]
            val_weeks_df = df_train[df_train['Patient'].isin(val_df['Patient'])]

            _tr_mae, _tr_sigma = validate(train_weeks_df, train_fold_preds[:, 0], train_fold_preds[:, 1], per_point=True)
            train_maes.append(_tr_mae)
            train_sigmas.append(_tr_sigma)

            _mae, _sigma = validate(val_weeks_df, val_fold_preds[:, 0], val_fold_preds[:, 1], per_point=True)
            maes.append(_mae)
            sigmas.append(_sigma)

            # _mae = validate(val_weeks_df, val_fold_preds, per_point=True)
            # maes.append(_mae)

            # _tr_mae = validate(train_weeks_df, train_fold_preds, per_point=True)
            # train_maes.append(_tr_mae)

    # print(f'MAES: {maes}') 
    print('********TRAIN********')
    per_fold_maes = [np.mean(fold_maes) for fold_maes in train_maes]
    val_mae = np.mean(per_fold_maes)
    print(f'MAE: {val_mae}')
    print(f'STD: {np.std(per_fold_maes)}')

    print('********VALIDATION********')
    per_fold_maes = [np.mean(fold_maes) for fold_maes in maes]
    val_mae = np.mean(per_fold_maes)
    print(f'MAE: {val_mae}')
    print(f'STD: {np.std(per_fold_maes)}')

    # print(f'Laplace: {_laplace(val_mae, val_mae * np.sqrt(2))}')
    print(f'Laplace: {np.mean(_laplace(np.concatenate(maes), np.concatenate(sigmas)))}')
    maes_flatten = np.array(list(itertools.chain(*maes)))
    print(f'Laplace opt.: {laplace_optimal(np.array(list(itertools.chain(*maes))))}')

    # import pickle
    # with open('scores.pkl', 'wb') as file:
    #     pickle.dump(train_df.drop(columns=['Patient', 'FVC', 'Percent']).columns, file)
    #     pickle.dump(scores, file)

    # return clfs, skbs, val_mae
    # return clfs, val_mae
    return clfs