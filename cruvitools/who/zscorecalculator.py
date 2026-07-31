from pathlib import Path
from warnings import warn
from statistics import NormalDist
norm = NormalDist()

import pandas as pd


'''
Following the same methodology applied to the WHO Child Growth Standards, a
restricted application of the LMS method was thus used for the 2007 WHO weight-based
indicators, limiting the Box-Cox normal distribution to the interval corresponding to z-scores
where empirical data were available (i.e. between -3 SD and 3 SD). Beyond these limits, the
standard deviation at each age was fixed to the distance between ±2 SD and ±3 SD, respectively.
This approach avoids making assumptions about the distribution of data beyond the limits of the
observed values (WHO Multicentre Growth Reference Study Group, 2006). 
'''

base_path = Path(__file__).resolve().parent / 'reference-data'
_reference_data = None


# Helper functions for input validation and warnings
def _warning_extreme_results(z):
    '''
    If the result is extreme, remember the user that the functions expect age in months, weight in kg and height in cm.
    '''
    if abs(z) >= 7:
        warn(f'Calculated Z-score of {z} is extreme. Make sure that age is provided in months, height in cm, and weight in kg')

def _load_reference_data():
    global _reference_data

    if _reference_data is None:
        _reference_data = {
            'boys_height': pd.read_pickle(base_path / 'boys_height.pkl').set_index('Month'),
            'boys_weight': pd.read_pickle(base_path / 'boys_weight.pkl').set_index('Month'),
            'boys_bmi': pd.read_pickle(base_path / 'boys_bmi.pkl').set_index('Month'),
            'girls_height': pd.read_pickle(base_path / 'girls_height.pkl').set_index('Month'),
            'girls_weight': pd.read_pickle(base_path / 'girls_weight.pkl').set_index('Month'),
            'girls_bmi': pd.read_pickle(base_path / 'girls_bmi.pkl').set_index('Month'),
        }

    return _reference_data

def _get_lms_parameters(sex, age, data_boys, data_girls):
    """Extract LMS parameters for given sex and age."""    
    
    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()
    
    if sex_lower in ['male', 'm', 1]:
        data = data_boys
    elif sex_lower in ['female', 'f', 2]:
        data = data_girls
    else:
        raise ValueError(f"Invalid sex, expected m/male/1 or f/female/2, received {sex}")
    
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
    
    # If |Z| value is greater than 3, the recommendation is to compute the basic Z-score
    # SD should be defined as the distance between SD3 and SD2 (+ or -) for each age range
    if Z > 3:
        Z = 3 + ((value - params['SD3pos']) / params['StDevPos'])

    elif Z < -3:
        Z = -3 + ((value - params['SD3neg']) / params['StDevNeg'])

    _warning_extreme_results(Z)
    return Z

def _calculate_centile(z):
    '''Convert Z-score to centile using standard CDF.'''
    centile = norm.cdf(z) * 100
    return centile

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


# Z-score functions
def calculate_height_z_score(height, sex, age):
    """
    Height in centimeters, sex as "male/female" or "m/f", age in months.
    """
    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_height'], reference_data['girls_height'])
    return _calculate_z_score(height, params)

def calculate_weight_z_score(weight, sex, age):
    """
    Weight in kg, sex as "male/female" or "m/f", age in months.
    """
    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_weight'], reference_data['girls_weight'])
    return _calculate_z_score(weight, params)

def calculate_bmi_z_score(bmi, sex, age):
    """
    BMI in kg/m2, sex as "male/female" or "m/f", age in months.
    """
    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])
    return _calculate_z_score(bmi, params)

# Centile functions
def calculate_height_centile(height, sex, age):
    """
    Height in centimeters, sex as "male/female" or "m/f", age in months.
    """
    z = calculate_height_z_score(height, sex, age)
    centile = _calculate_centile(z)
    return centile

def calculate_weight_centile(weight, sex, age):
    """
    Weight in kg, sex as "male/female" or "m/f", age in months.
    """
    z = calculate_weight_z_score(weight, sex, age)
    centile = _calculate_centile(z)
    return centile

def calculate_bmi_centile(bmi, sex, age):
    """
    BMI in kg/m2, sex as "male/female" or "m/f", age in months.
    """
    z = calculate_bmi_z_score(bmi, sex, age)
    centile = _calculate_centile(z)
    return centile

## Reverse calculations (absolute numbers from Z-score)
def reverse_height_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """
    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_height'], reference_data['girls_height'])
    return _reverse_z_score_with_restriction(Z, params)

def reverse_weight_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """
    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_weight'], reference_data['girls_weight'])
    return _reverse_z_score_with_restriction(Z, params)

def reverse_bmi_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """
    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])
    return _reverse_z_score_with_restriction(Z, params)

## Reverse centile functions
def reverse_height_centile(centile, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    Takes a centile and returns corresponding height (cm) for sex / age.
    """
    z = norm.inv_cdf(centile / 100)
    height = reverse_height_z_score(z, sex, age)
    return height

def reverse_weight_centile(centile, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    Takes a centile and returns corresponding weight (kg) for sex / age.
    """
    z = norm.inv_cdf(centile / 100)
    weight = reverse_weight_z_score(z, sex, age)
    return weight

def reverse_bmi_centile(centile, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    Takes a centile and returns corresponding bmi (kg/m2) for sex / age.
    """
    z = norm.inv_cdf(centile / 100)
    bmi = reverse_bmi_z_score(z, sex, age)
    return bmi