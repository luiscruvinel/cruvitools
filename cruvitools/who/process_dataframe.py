import pandas as pd

from .zscorecalculator import (
    calculate_bmi_z_score,
    calculate_height_z_score,
    calculate_weight_z_score,
)

required_columns = {'age', 'height', 'weight', 'sex'}

def _calculate_bmi(height: pd.Series, weight: pd.Series) -> pd.Series:
    return weight / ((height / 100) ** 2)


def process_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a copy of the input dataframe with WHO Z-score columns added.

    Expected input columns:
        age, height, weight, sex

    Optional input column:
        bmi

    If bmi is missing, it is calculated from height and weight and included in
    the returned dataframe before the z-score columns are added.
    '''

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f'Missing required columns: {missing}')

    result = dataframe.copy()

    if 'bmi' not in result.columns:
        result['bmi'] = _calculate_bmi(result['height'], result['weight'])

    result['height_z_score'] = result.apply(
        lambda row: calculate_height_z_score(row['height'], row['sex'], row['age']),
        axis=1,
    )
    result['weight_z_score'] = result.apply(
        lambda row: calculate_weight_z_score(row['weight'], row['sex'], row['age']),
        axis=1,
    )
    result['bmi_z_score'] = result.apply(
        lambda row: calculate_bmi_z_score(row['bmi'], row['sex'], row['age']),
        axis=1,
    )

    return result