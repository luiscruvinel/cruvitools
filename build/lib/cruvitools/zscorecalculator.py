import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent / 'who-data-processed' / 'complete-tables'

boys_bmi = pd.read_pickle(base_path / 'boys_bmi.pkl').set_index('Month')
boys_height = pd.read_pickle(base_path / 'boys_height.pkl').set_index('Month')
boys_weight = pd.read_pickle(base_path / 'boys_weight.pkl').set_index('Month')

girls_bmi = pd.read_pickle(base_path / 'girls_bmi.pkl').set_index('Month')
girls_height = pd.read_pickle(base_path / 'girls_height.pkl').set_index('Month')
girls_weight = pd.read_pickle(base_path / 'girls_weight.pkl').set_index('Month')

def calculate_height_z_score(age, height, sex):
    """
    Age in months, height in centimeters, sex as "male/female" or "m/f"
    """
    
    if sex.lower() == 'male' or sex.lower() == 'm':
        L = boys_height.query('Month == @age')['L'].values[0]
        M = boys_height.query('Month == @age')['M'].values[0]
        S = boys_height.query('Month == @age')['S'].values[0]

    if sex.lower() == 'female' or sex.lower() == 'f':
        L = girls_height.query('Month == @age')['L'].values[0]
        M = girls_height.query('Month == @age')['M'].values[0]
        S = girls_height.query('Month == @age')['S'].values[0]

    Z = ((height / M) ** L - 1) / (L * S)
    return Z

def calculate_weight_z_score(age, weight, sex):
    if sex.lower() == 'male' or sex.lower() == 'm':
        L = boys_weight.query('Month == @age')['L'].values[0]
        M = boys_weight.query('Month == @age')['M'].values[0]
        S = boys_weight.query('Month == @age')['S'].values[0]

    if sex.lower() == 'female' or sex.lower() == 'f':
        L = girls_weight.query('Month == @age')['L'].values[0]
        M = girls_weight.query('Month == @age')['M'].values[0]
        S = girls_weight.query('Month == @age')['S'].values[0]

    Z = ((weight / M) ** L - 1) / (L * S)
    return Z

def calculate_bmi_z_score(age, bmi, sex):
    if sex.lower() == 'male' or sex.lower() == 'm':
        L = boys_bmi.query('Month == @age')['L'].values[0]
        M = boys_bmi.query('Month == @age')['M'].values[0]
        S = boys_bmi.query('Month == @age')['S'].values[0]

    if sex.lower() == 'female' or sex.lower() == 'f':
        L = girls_bmi.query('Month == @age')['L'].values[0]
        M = girls_bmi.query('Month == @age')['M'].values[0]
        S = girls_bmi.query('Month == @age')['S'].values[0]

    Z = ((bmi / M) ** L - 1) / (L * S)
    return Z
