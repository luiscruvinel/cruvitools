import pandas as pd

from .zscorecalculator import calculate_bone_age_z_score


required_columns = {'chronological_age', 'bone_age', 'sex'}


def process_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a copy of the input dataframe with Greulich-Pyle Z-score columns added.

    Expected input columns:
        chronological_age, bone_age, sex

    Ages are expected in years, matching the underlying calculator.
    '''

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f'Missing required columns: {missing}')

    result = dataframe.copy()
    result['bone_age_z_score'] = result.apply(
        lambda row: calculate_bone_age_z_score(
            row['chronological_age'], row['bone_age'], row['sex']
        ),
        axis=1,
    )

    return result