import numpy as np
import pandas as pd

from statistics import NormalDist
norm = NormalDist()

from pathlib import Path
from warnings import warn

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
            'boys_height': pd.read_pickle(base_path / 'boys_height.pkl').set_index('Month', drop=True),
            'girls_height': pd.read_pickle(base_path / 'girls_height.pkl').set_index('Month', drop=True),
            'boys_weight': pd.read_pickle(base_path / 'boys_weight.pkl').set_index('Month', drop=True),
            'girls_weight': pd.read_pickle(base_path / 'girls_weight.pkl').set_index('Month', drop=True),
            'boys_bmi': pd.read_pickle(base_path / 'boys_bmi.pkl').set_index('Month', drop=True),
            'girls_bmi': pd.read_pickle(base_path / 'girls_bmi.pkl').set_index('Month', drop=True),
        }

    return _reference_data


def _get_lms_parameters(sex, age, data_boys, data_girls):
    '''
    Extract L, M, S parameters for given sex and age using linear interpolation.
    
    Args:
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months (can be fractional)
        data_boys: DataFrame with boys reference data
        data_girls: DataFrame with girls reference data
    
    Returns:
        dict with 'L', 'M', 'S' parameters (interpolated if age falls between data points)
    '''
    
    # Handle different sex inputs
    sex_normalized = sex.lower() if isinstance(sex, str) else sex
    
    if sex_normalized in ['male', 'm', 1]:
        data = data_boys
    elif sex_normalized in ['female', 'f', 2]:
        data = data_girls
    else:
        raise ValueError(f'Invalid sex value: {sex}. Must be "male/m/1" or "female/f/2"')
    
    # Handle different age inputs
    available_ages = data.index.values
    max_age = data.index.max()
    min_age = data.index.min()
    
    if age > max_age:
        warn(f'Age {age} months exceeds maximum age {max_age} months. Using maximum {max_age} for calculations.')
        age = max_age

    if age < min_age:
        raise ValueError(f'Age {age} months is below the minimum available of {min_age}')

    # If exact age exists in data, use it directly
    if age in available_ages:
        params = {
            'L': float(data.loc[age, 'L']),
            'M': float(data.loc[age, 'M']),
            'S': float(data.loc[age, 'S']),
            'P95': float(data.loc[age, 'P95'])
        }
        if 'sigma' in data.columns:
            params['sigma'] = float(data.loc[age, 'sigma'])
    else:
        # Linear interpolation between adjacent age points
        # Find the two ages that bracket the input age
        lower_age = available_ages[available_ages <= age].max()
        upper_age = available_ages[available_ages > age].min()
        
        # Calculate interpolation weight
        weight = (age - lower_age) / (upper_age - lower_age)
        
        # Interpolate each parameter
        params = {}
        for param in ['L', 'M', 'S', 'P95']:
            lower_val = float(data.loc[lower_age, param])
            upper_val = float(data.loc[upper_age, param])
            params[param] = lower_val + weight * (upper_val - lower_val)
        
        if 'sigma' in data.columns:
            lower_sigma = float(data.loc[lower_age, 'sigma'])
            upper_sigma = float(data.loc[upper_age, 'sigma'])
            params['sigma'] = lower_sigma + weight * (upper_sigma - lower_sigma)

    return params


def _calculate_z_score(measurement, sex, age, data_boys, data_girls):
    '''
    Calculate CDC Z-score using LMS method.
    
    Uses the CDC LMS formula:
    Z = ((X/M)^L - 1) / (L*S)  when L ≠ 0
    Z = ln(X/M) / S            when L = 0
    
    Args:
        measurement: The measurement value (height in cm or weight in kg)
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
        data_boys: DataFrame with boys reference data
        data_girls: DataFrame with girls reference data
    
    Returns:
        float: Z-score
    '''
    params = _get_lms_parameters(sex, age, data_boys, data_girls)
    
    L = params['L']
    M = params['M']
    S = params['S']
    X = measurement
    
    if abs(L) == 0:
        Z = np.log(X / M) / S
    else:
        Z = (((X / M) ** L) - 1) / (L * S)
    
    return Z


def _reverse_z_score(z_score, sex, age, data_boys, data_girls):
    '''
    Calculate measurement from CDC Z-score using LMS method.
    
    Args:
        z_score value
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
        data_boys: DataFrame with boys reference data
        data_girls: DataFrame with girls reference data
    
    Returns:
        float: Measurement value (height in cm or weight in kg)
    '''
    params = _get_lms_parameters(sex, age, data_boys, data_girls)
    
    L = params['L']
    M = params['M']
    S = params['S']
    Z = z_score
    
    if abs(L) == 0:
        X = M * np.exp(S * Z)
    else:
        X = M * ((1 + L * S * Z) ** (1 / L))
    
    return X

def _calculate_extreme_z_score(bmi, sex, age, data_boys, data_girls):
    '''
    Calculate BMI Z-score for extreme values using modified approach
    Following Wei et al. 2020 recommendations.
    
    Args:
        bmi: BMI value
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
        data_boys: DataFrame with boys BMI reference data
        data_girls: DataFrame with girls BMI reference data
    
    Returns:
        float: Modified Z-score for extreme BMI
    '''
    params = _get_lms_parameters(sex, age, data_boys, data_girls)
    
    sigma = params['sigma']
    P95 = params['P95']

    centile = 90 + 10 * norm.cdf((bmi - P95) / sigma)
    centile = np.clip(centile, 0, np.nextafter(100.0, 0.0))
    Z = norm.inv_cdf(centile/100)

    return Z

def calculate_height_z_score(height, sex, age):
    '''
    Calculate CDC height-for-age Z-score using LMS method.
    
    Args:
        height: Height in centimeters
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
    
    Returns:
        float: height-for-age Z-score
    '''
    reference_data = _load_reference_data()
    return _calculate_z_score(height, sex, age, reference_data['boys_height'], reference_data['girls_height'])


def calculate_weight_z_score(weight, sex, age):
    '''
    Calculate CDC weight-for-age Z-score using LMS method.
    
    Args:
        weight: Weight in kilograms
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
    
    Returns:
        float: Z-score for weight-for-age
    '''
    reference_data = _load_reference_data()
    return _calculate_z_score(weight, sex, age, reference_data['boys_weight'], reference_data['girls_weight'])


def reverse_height_z_score(z_score, sex, age):
    '''
    Calculate height from CDC height-for-age Z-score using LMS method.
    
    Args:
        z_score: Target Z-score
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
    
    Returns:
        float: Height in centimeters
    '''
    reference_data = _load_reference_data()
    return _reverse_z_score(z_score, sex, age, reference_data['boys_height'], reference_data['girls_height'])


def reverse_weight_z_score(z_score, sex, age):
    '''
    Calculate weight from CDC weight-for-age Z-score using LMS method.
    
    Args:
        z_score: Target Z-score
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
    
    Returns:
        float: Weight in kilograms
    '''
    reference_data = _load_reference_data()
    return _reverse_z_score(z_score, sex, age, reference_data['boys_weight'], reference_data['girls_weight'])


def calculate_bmi_z_score(bmi, sex, age, method='extended'):
    '''
    Calculate CDC BMI-for-age Z-score using LMS method.

    Args:
        bmi: BMI value (kg/m²)
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
        method: 'original' returns the standard LMS Z-score for any value;
            'extended' uses CDC extended bmi tables for values above 95th centile.

    Returns:
        float: BMI-for-age Z-score
    '''
    if method not in {'original', 'extended'}:
        raise ValueError("method must be 'original' or 'extended'")

    reference_data = _load_reference_data()
    Z = _calculate_z_score(bmi, sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])

    if method == 'original':
        return Z

    centile = norm.cdf(Z) * 100
    centile = np.clip(centile, 0, np.nextafter(100.0, 0.0))

    if centile <= 95:
        return Z

    return _calculate_extreme_z_score(bmi, sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])


def reverse_bmi_z_score(z_score, sex, age, method='extended'):
    '''
    Calculate BMI from CDC BMI-for-age Z-score using LMS method.

    Args:
        z_score: Target Z-score
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
        method: 'original' returns the standard inverse LMS BMI value for any z-score;
            'extended' uses the data from CDC extended BMI tables.
    
    Returns:
        float: BMI value (kg/m²)
    '''
    if method not in {'original', 'extended'}:
        raise ValueError("method must be 'original' or 'extended'")

    if method == 'original':
        reference_data = _load_reference_data()
        return _reverse_z_score(z_score, sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])

    centile = norm.cdf(z_score) * 100
    centile = np.clip(centile, 0, np.nextafter(100.0, 0.0))

    if centile <= 95:
        reference_data = _load_reference_data()
        return _reverse_z_score(z_score, sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])

    reference_data = _load_reference_data()
    params = _get_lms_parameters(sex, age, reference_data['boys_bmi'], reference_data['girls_bmi'])

    sigma = params['sigma']
    P95 = params['P95']
    bmi = P95 + sigma * norm.inv_cdf((centile - 90) / 10)

    return bmi