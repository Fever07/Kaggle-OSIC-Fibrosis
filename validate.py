from utils import mae, laplace
import numpy as np
from constants import MIN_WEEK, MAX_WEEK, WEEKS
from preprocess import denormalize, drop_first_point
from features import denormalize_diff

def get_fvc_by_patient(weeks_df, fvc_pred):
    df = weeks_df.copy()
    df['FVC_pred'] = fvc_pred    
    fvc_pred_by_patient = []
    for patient in df['Patient'].unique():
        pat_df = df[df['Patient'] == patient]
        fvc_pred_by_patient.append(pat_df['FVC_pred'].values)
    return fvc_pred_by_patient

def validate_predict(weeks_df, fvc_pred, per_point=False):
    y_true = []
    y_pred = []
    
    eps = 1e-4
    for patient_id, patient_preds in zip(weeks_df['Patient'].unique(), fvc_pred):
        patient_weeks = weeks_df[weeks_df['Patient'] == patient_id]['Weeks'].values
        patient_fvc = weeks_df[weeks_df['Patient'] == patient_id]['FVC'].values
        first_fvc = weeks_df[weeks_df['Patient'] == patient_id]['FVC_first'].values[0]

        last_fvc = patient_fvc[-3:]
        last_preds = patient_preds[-3:]

        y_true.extend(denormalize(last_fvc))
        y_pred.extend(denormalize(first_fvc) + denormalize_diff(last_preds))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if per_point:
        return np.abs(y_true - y_pred)
    else:
        return mae(y_true, y_pred)

def validate(weeks_df, fvc_pred, per_point=False):
    weeks_df = drop_first_point(weeks_df)
    fvc_pred_by_patient = get_fvc_by_patient(weeks_df, fvc_pred)
    return validate_predict(weeks_df, fvc_pred_by_patient, per_point=per_point)


