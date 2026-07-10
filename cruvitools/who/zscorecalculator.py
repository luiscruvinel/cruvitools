import pandas as pd

from pathlib import Path
from warnings import warn

'''
Following the same methodology applied to the WHO Child Growth Standards, a
restricted application of the LMS method was thus used for the 2007 WHO weight-based
indicators, limiting the Box-Cox normal distribution to the interval corresponding to z-scores
where empirical data were available (i.e. between -3 SD and 3 SD). Beyond these limits, the
standard deviation at each age was fixed to the distance between ±2 SD and ±3 SD, respectively.
This approach avoids making assumptions about the distribution of data beyond the limits of the
observed values (WHO Multicentre Growth Reference Study Group, 2006). 
'''

base_path = Path(__file__).resolve().parent / 'reference-data' / 'who-data-processed'

boys_height = pd.read_pickle(base_path / 'boys_height.pkl').set_index('Month')
boys_weight = pd.read_pickle(base_path / 'boys_weight.pkl').set_index('Month')
boys_bmi = pd.read_pickle(base_path / 'boys_bmi.pkl').set_index('Month')

girls_height = pd.read_pickle(base_path / 'girls_height.pkl').set_index('Month')
girls_weight = pd.read_pickle(base_path / 'girls_weight.pkl').set_index('Month')
girls_bmi = pd.read_pickle(base_path / 'girls_bmi.pkl').set_index('Month')

def _get_lms_parameters(sex, age, data_boys, data_girls):
    """Extract LMS parameters for given sex and age."""
    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228
    
    sex_lower = sex.lower()
    
    if sex_lower in ('male', 'm'):
        data = data_boys
    elif sex_lower in ('female', 'f'):
        data = data_girls
    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")
    
    return {
        'L': data.loc[age, 'L'],
        'M': data.loc[age, 'M'],
        'S': data.loc[age, 'S'],
        'StDevPos': data.loc[age, 'StDevPos'],
        'StDevNeg': data.loc[age, 'StDevNeg'],
        'SD3pos': data.loc[age, 'SD3'],
        'SD3neg': data.loc[age, 'SD3neg']
    }

def _calculate_z_score(value, params):
    """Calculate Z-score using WHO restricted LMS method."""
    L, M, S = params['L'], params['M'], params['S']
    Z = ((value / M) ** L - 1) / (L * S)
    
    # If the computed Z-score is between -3 and +3, return that value
    if -3 <= Z <= 3:
        return Z
    
    # If |Z| value is greater than 3, the recommendation is to compute the basic Z-score
    # SD should be defined as the distance between SD3 and SD2 (+ or -) for each age range
    elif Z > 3:
        return 3 + ((value - params['SD3pos']) / params['StDevPos'])
    else: # Z < -3
        return -3 + ((value - params['SD3neg']) / params['StDevNeg'])

def _reverse_z_score_with_restriction(Z, params):
    """Reverse Z-score to measurement using WHO restricted LMS method."""
    # If Z is between -3 and +3, use the standard LMS formula
    if -3 <= Z <= 3:
        L, M, S = params['L'], params['M'], params['S']
        return M * (1 + Z * L * S) ** (1 / L)
    
    # If |Z| > 3, use the reverse basic Z-score formula
    elif Z > 3:
        return params['SD3pos'] + (Z - 3) * params['StDevPos']
    else:  # Z < -3
        return params['SD3neg'] + (Z + 3) * params['StDevNeg']

def calculate_height_z_score(height, sex, age):
    """
    Height in centimeters, sex as "male/female" or "m/f", age in months.
    """
    params = _get_lms_parameters(sex, age, boys_height, girls_height)
    return _calculate_z_score(height, params)

def calculate_weight_z_score(weight, sex, age):
    """
    Weight in kg, sex as "male/female" or "m/f", age in months.
    """
    params = _get_lms_parameters(sex, age, boys_weight, girls_weight)
    return _calculate_z_score(weight, params)

def calculate_bmi_z_score(bmi, sex, age):
    """
    BMI in kg/m2, sex as "male/female" or "m/f", age in months.
    """
    params = _get_lms_parameters(sex, age, boys_bmi, girls_bmi)
    return _calculate_z_score(bmi, params)

## Reverse calculations (absolute numbers from Z-score)

def reverse_height_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """
    params = _get_lms_parameters(sex, age, boys_height, girls_height)
    return _reverse_z_score_with_restriction(Z, params)

def reverse_weight_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """
    params = _get_lms_parameters(sex, age, boys_weight, girls_weight)
    return _reverse_z_score_with_restriction(Z, params)

def reverse_bmi_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """
    params = _get_lms_parameters(sex, age, boys_bmi, girls_bmi)
    return _reverse_z_score_with_restriction(Z, params)