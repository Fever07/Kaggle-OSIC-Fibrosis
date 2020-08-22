import numpy as np

DATA_DIR = './'
MIN_WEEK = -12
MAX_WEEK = 133
WEEKS = np.linspace(0, 1, MAX_WEEK - MIN_WEEK + 1)
CATEGORICAL_FEATURES = {
    'Sex': ['Male', 'Female'], 
    'SmokingStatus': ['Never smoked', 'Ex-smoker', 'Currently smokes']
}
NUMERICAL_FEATURES = ['Weeks', 'FVC', 'Percent', 'Age']
EPS = np.finfo(np.float32).eps