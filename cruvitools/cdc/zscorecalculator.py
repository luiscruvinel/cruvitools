import numpy as np
import pandas as pd
from scipy.stats import norm

from pathlib import Path
from warnings import warn

base_path = Path(__file__).resolve().parent / 'reference-data'

boys_height = pd.read_pickle(base_path / 'boys_height.pkl').set_index('Month', drop=True)
girls_height = pd.read_pickle(base_path / 'girls_height.pkl').set_index('Month', drop=True)
boys_weight = pd.read_pickle(base_path / 'boys_weight.pkl').set_index('Month', drop=True)
girls_weight = pd.read_pickle(base_path / 'girls_weight.pkl').set_index('Month', drop=True)
boys_bmi = pd.read_pickle(base_path / 'boys_bmi.pkl').set_index('Month', drop=True)
girls_bmi = pd.read_pickle(base_path / 'girls_bmi.pkl').set_index('Month', drop=True)


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
    
    sex_normalized = sex.lower() if isinstance(sex, str) else sex
    
    if sex_normalized in ['male', 'm', 1]:
        data = data_boys
    elif sex_normalized in ['female', 'f', 2]:
        data = data_girls
    else:
        raise ValueError(f'Invalid sex value: {sex}. Must be "male/m/1" or "female/f/2"')
        
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
            'p95': float(data.loc[age, 'P95'])
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
            params[param.lower() if param == 'P95' else param] = lower_val + weight * (upper_val - lower_val)
        
        # Rename P95 to p95 for consistency
        if 'P95' in params:
            params['p95'] = params.pop('P95')
        
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
    Z = ln(X/M) / S             when L = 0
    
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
    
    L = params['L']
    M = params['M']
    S = params['S']
    sigma = params['sigma']
    p95 = params['p95']

    centile = 90 + 10 * norm.cdf((bmi - p95) / sigma)
    Z = norm.ppf(centile/100)

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
    return _calculate_z_score(height, sex, age, boys_height, girls_height)


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
    return _calculate_z_score(weight, sex, age, boys_weight, girls_weight)


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
    return _reverse_z_score(z_score, sex, age, boys_height, girls_height)


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
    return _reverse_z_score(z_score, sex, age, boys_weight, girls_weight)


def calculate_bmi_z_score(bmi, sex, age):
    '''
    Calculate CDC BMI-for-age Z-score using LMS method with extreme value handling.
    
    For extreme BMI values (Z < -3 or Z > 3), uses modified calculation
    following Wei et al. 2020 recommendations to prevent unrealistic extrapolation.
    
    Args:
        bmi: BMI value (kg/m²)
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
    
    Returns:
        float: BMI-for-age Z-score
    '''
    Z = _calculate_z_score(bmi, sex, age, boys_bmi, girls_bmi)
    centile = norm.cdf(Z) * 100

    if centile <= 95:
        return Z
    
    else:
        return _calculate_extreme_z_score(bmi, sex, age, boys_bmi, girls_bmi)


def reverse_bmi_z_score(z_score, sex, age):
    '''
    Calculate BMI from CDC BMI-for-age Z-score using LMS method.

    Args:
        z_score: Target Z-score
        sex: 'male', 'female', 'M', 'F', 'm', 'f', 1, or 2
        age: Age in months
    
    Returns:
        float: BMI value (kg/m²)
    '''

    centile = norm.cdf(z_score) * 100
    
    if centile <= 95:
        return _reverse_z_score(z_score, sex, age, boys_bmi, girls_bmi)
    
    else:
        params = _get_lms_parameters(sex, age, boys_bmi, girls_bmi)
        
        sigma = params['sigma']
        p95 = params['p95']
        bmi = p95 + sigma * norm.ppf((centile - 90) / 10)
        
        return bmi