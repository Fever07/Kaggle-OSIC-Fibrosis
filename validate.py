from utils import mae
import numpy as np
from constants import MIN_WEEK, MAX_WEEK, WEEKS
from preprocess import denormalize

def predict_by_coef(df, coefs):
    biases = df['FVC'].values - df['Weeks'].values * coefs
    patient_fvc = []
    for patient_id, coef, bias in zip(df['Patient'], coefs, biases):
        patient_fvc.append(coef * WEEKS + bias)
    return patient_fvc

def validate_predict(weeks_df, fvc_pred, per_point=False):
    y_true = []
    y_pred = []
    
    eps = 1e-4
    for patient_id, patient_preds in zip(weeks_df['Patient'].unique(), fvc_pred):
        patient_weeks = weeks_df[weeks_df['Patient'] == patient_id]['Weeks']
        patient_fvc = weeks_df[weeks_df['Patient'] == patient_id]['FVC']
        last_3 = patient_weeks[-3:]
        last_fvc = patient_fvc[-3:]

        y_true.extend(denormalize(last_fvc))
        for last_week, last_fvc in zip(last_3, last_fvc):
            idx = np.where(np.abs(WEEKS - last_week) < eps)[0][0]
            y_pred.append(denormalize(patient_preds[idx]))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if per_point:
        return np.abs(y_true - y_pred)
    else:
        return mae(y_true, y_pred)
    
def validate(weeks_df, initial_df, coefs, per_point=False):
    fvc_pred = predict_by_coef(initial_df, coefs)
    return validate_predict(weeks_df, fvc_pred, per_point=per_point)


