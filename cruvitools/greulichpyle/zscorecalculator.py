import math
from pathlib import Path
import pandas as pd


base_path = Path(__file__).resolve().parent / 'reference-data'
_reference_data = None


def _load_reference_data():
    global _reference_data

    if _reference_data is None:
        _reference_data = pd.read_pickle(base_path / 'greulich-pyle-reference.pkl').set_index('chronological_age_years')

    return _reference_data

def _round_to_quarter(x):
    return round(x * 4) / 4

def _round_to_half(x):
    return round(x * 2) / 2

def calculate_bone_age_z_score(chronological_age, bone_age, sex):
    """
    Chronological_age in years
    Bone age in years
    Sex as "male" or "female" or "M" / "F"

    Returns the Z-score for bone age in months
    """

    greulich_pyle_reference = _load_reference_data()

    if chronological_age < 1: rounded_age = _round_to_quarter(chronological_age)
    elif chronological_age < 5: rounded_age = _round_to_half(chronological_age)
    else: rounded_age = math.floor(chronological_age)
    
    if sex.lower() in ['male', 'm']:
        SD = greulich_pyle_reference.query('chronological_age_years == @rounded_age')['boys_sd_months'].values[0]
        Z = (bone_age - chronological_age) * 12 / SD
        return Z

    if sex.lower() in ['female', 'f']:
        SD = greulich_pyle_reference.query('chronological_age_years == @rounded_age')['girls_sd_months'].values[0]
        Z = (bone_age - chronological_age) * 12 / SD
        return Z
