import pandas as pd

from pathlib import Path
from warnings import warn

base_path = Path(__file__).resolve().parent / 'who-data-processed' / 'complete-tables'

boys_bmi = pd.read_pickle(base_path / 'boys_bmi.pkl').set_index('Month')
boys_height = pd.read_pickle(base_path / 'boys_height.pkl').set_index('Month')
boys_weight = pd.read_pickle(base_path / 'boys_weight.pkl').set_index('Month')

girls_bmi = pd.read_pickle(base_path / 'girls_bmi.pkl').set_index('Month')
girls_height = pd.read_pickle(base_path / 'girls_height.pkl').set_index('Month')
girls_weight = pd.read_pickle(base_path / 'girls_weight.pkl').set_index('Month')

def calculate_height_z_score(height, sex, age):
    """
    Height in centimeters, sex as "male/female" or "m/f", age in months.
    """

    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()

    if sex_lower in ('male', 'm'):
        L = boys_height.loc[age, 'L']
        M = boys_height.loc[age, 'M']
        S = boys_height.loc[age, 'S']

    elif sex_lower in ('female', 'f'):
        L = girls_height.loc[age, 'L']
        M = girls_height.loc[age, 'M']
        S = girls_height.loc[age, 'S']

    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")

    Z = ((height / M) ** L - 1) / (L * S)
    return Z

def calculate_weight_z_score(weight, sex, age):
    """
    Weight in kg, sex as "male/female" or "m/f", age in months.
    """

    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()

    if sex_lower in ('male', 'm'):
        L = boys_weight.loc[age, 'L']
        M = boys_weight.loc[age, 'M']
        S = boys_weight.loc[age, 'S']

    elif sex_lower in ('female', 'f'):
        L = girls_weight.loc[age, 'L']
        M = girls_weight.loc[age, 'M']
        S = girls_weight.loc[age, 'S']

    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")

    Z = ((weight / M) ** L - 1) / (L * S)
    return Z

def calculate_bmi_z_score(bmi, sex, age):
    """
    BMI in kg/m2, sex as "male/female" or "m/f", age in months.
    """

    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()

    if sex_lower in ('male', 'm'):
        L = boys_bmi.loc[age, 'L']
        M = boys_bmi.loc[age, 'M']
        S = boys_bmi.loc[age, 'S']

    elif sex_lower in ('female', 'f'):
        L = girls_bmi.loc[age, 'L']
        M = girls_bmi.loc[age, 'M']
        S = girls_bmi.loc[age, 'S']

    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")

    Z = ((bmi / M) ** L - 1) / (L * S)
    return Z

## Reverse calculations (absolute numbers from Z-score)

def reverse_height_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """

    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()

    if sex_lower in ('male', 'm'):
        L = boys_height.loc[age, 'L']
        M = boys_height.loc[age, 'M']
        S = boys_height.loc[age, 'S']

    elif sex_lower in ('female', 'f'):
        L = girls_height.loc[age, 'L']
        M = girls_height.loc[age, 'M']
        S = girls_height.loc[age, 'S']

    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")

    height = M * (1 + Z * L * S) ** (1 / L)
    return height

def reverse_weight_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """

    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()

    if sex_lower in ('male', 'm'):
        L = boys_weight.loc[age, 'L']
        M = boys_weight.loc[age, 'M']
        S = boys_weight.loc[age, 'S']

    elif sex_lower in ('female', 'f'):
        L = girls_weight.loc[age, 'L']
        M = girls_weight.loc[age, 'M']
        S = girls_weight.loc[age, 'S']

    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")

    weight = M * (1 + Z * L * S) ** (1 / L)
    return weight

def reverse_bmi_z_score(Z, sex, age):
    """
    Sex as "male/female" or "m/f", age in months.
    """

    if age > 228:
        warn('Age value provided is above 19 years. Using 19 years for calculations.')
        age = 228

    sex_lower = sex.lower()

    if sex_lower in ('male', 'm'):
        L = boys_bmi.loc[age, 'L']
        M = boys_bmi.loc[age, 'M']
        S = boys_bmi.loc[age, 'S']

    elif sex_lower in ('female', 'f'):
        L = girls_bmi.loc[age, 'L']
        M = girls_bmi.loc[age, 'M']
        S = girls_bmi.loc[age, 'S']

    else:
        raise ValueError(f"Invalid sex, expected male/female or m/f, received {sex}")

    bmi = M * (1 + Z * L * S) ** (1 / L)
    return bmi

if __name__ == '__main__':
    print(round(calculate_height_z_score(176, 'm', 360), 2))
    print(calculate_height_z_score(163, 'f', 360))
    print(calculate_height_z_score(110, 'Male', 60))
    print(calculate_height_z_score(110, 'Female', 60))

    print(reverse_height_z_score(-0.07, 'm', 360))
    print(reverse_height_z_score(-0.02, 'f', 360))
    print(reverse_height_z_score(-2.15, 'Male', 60))
    print(reverse_height_z_score(-1.98, 'Female', 60))