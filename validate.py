from utils import mae, laplace
import numpy as np
from constants import MIN_WEEK, MAX_WEEK, WEEKS
from preprocess import denormalize, drop_first_point

def get_fvc_by_patient(weeks_df, fvc_pred, sigma_pred):
    df = weeks_df.copy()    
    fvc_pred_by_patient = []
    sigma_pred_by_patient = []
    for i, patient in enumerate(df['Patient'].unique()):
        pat_df = df[df['Patient'] == patient]
        fvc_pred_by_patient.append(np.array([fvc_pred[i]] * len(pat_df)))
        sigma_pred_by_patient.append(np.array([sigma_pred[i]] * len(pat_df)))
    return fvc_pred_by_patient, sigma_pred_by_patient
    # return fvc_pred_by_patient

def validate_predict(weeks_df, fvc_pred, sigma_pred, per_point=False):
# def validate_predict(weeks_df, fvc_pred, per_point=False):
    y_true = []
    y_pred = []
    sigma = []

    eps = 1e-4
    for patient_id, patient_preds, sigma_preds in zip(weeks_df['Patient'].unique(), fvc_pred, sigma_pred):
    # for patient_id, patient_preds in zip(weeks_df['Patient'].unique(), fvc_pred):
        patient_weeks = weeks_df[weeks_df['Patient'] == patient_id]['Weeks'].values
        patient_fvc = weeks_df[weeks_df['Patient'] == patient_id]['FVC'].values

        first_fvc = patient_fvc[0]
        last_fvc = patient_fvc[-3:]
        last_preds = patient_preds[-3:]
        last_sigmas = sigma_preds[-3:]

        y_true.extend(last_fvc)
        y_pred.extend(first_fvc + last_preds)
        sigma.extend(last_sigmas)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)
    if per_point:
        return np.abs(y_true - y_pred), sigma
    else: 
        return mae(y_true, y_pred), mae(y_true, y_pred) * np.sqrt(2)
    # if per_point:
    #     return np.abs(y_true - y_pred)
    # else:
    #     return mae(y_true, y_pred)

def validate(weeks_df, fvc_pred, sigma_pred, per_point=False):
# def validate(weeks_df, fvc_pred, per_point=False):
    fvc_pred_by_patient, sigma_pred_by_patient = get_fvc_by_patient(weeks_df, fvc_pred, sigma_pred)
    return validate_predict(weeks_df, fvc_pred_by_patient, sigma_pred_by_patient, per_point=per_point)

    # fvc_pred_by_patient = get_fvc_by_patient(weeks_df, fvc_pred)
    # return validate_predict(weeks_df, fvc_pred_by_patient, per_point=per_point)


