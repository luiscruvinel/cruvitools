import pytest
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

import cruvitools.cdc.zscorecalculator as zsc


test_data_path = Path(__file__).parent / 'data' / 'testCasesCDC.csv'
test_answers = pd.read_csv(test_data_path)

tolerance_level = 0.05

height_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        row['sex'],
        row['height'],
        row['z_height'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
]

weight_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        row['sex'],
        row['weight'],
        row['z_weight'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
]

bmi_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        row['sex'],
        row['weight'] / ((row['height'] / 100) ** 2),
        row['z_bmi'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
]

bmi_extended_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        row['sex'],
        row['weight'] / ((row['height'] / 100) ** 2),
        row['z_bmi_extended'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
    if abs(row['z_bmi']) > 2
]

class TestCalculateZScores:
    '''Test forward Z-score calculations against reference data.'''

    @pytest.mark.parametrize('row_index, age, sex, height, expected_z', height_reference_cases)
    def test_height_z_scores(self, row_index, age, sex, height, expected_z):
        '''Test height Z-score calculations match reference values.'''
        calculated_z = zsc.calculate_height_z_score(height, sex, age)

        assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
            f'Row {row_index}: Height Z-score mismatch. ' \
            f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
            f'(height={height}cm, sex={sex}, age={age}m)'

    @pytest.mark.parametrize('row_index, age, sex, weight, expected_z', weight_reference_cases)
    def test_weight_z_scores(self, row_index, age, sex, weight, expected_z):
        '''Test weight Z-score calculations match reference values.'''
        calculated_z = zsc.calculate_weight_z_score(weight, sex, age)

        assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
            f'Row {row_index}: Weight Z-score mismatch. ' \
            f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
            f'(weight={weight}kg, sex={sex}, age={age}m)'

    @pytest.mark.parametrize('row_index, age, sex, bmi, expected_z', bmi_reference_cases)
    def test_bmi_z_scores(self, row_index, age, sex, bmi, expected_z):
        '''Test BMI Z-score calculations match the original reference values.'''
        calculated_z = zsc.calculate_bmi_z_score(bmi, sex, age, method='original')

        assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
            f'Row {row_index}: BMI Z-score mismatch. ' \
            f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
            f'(bmi={bmi:.2f}, sex={sex}, age={age}m)'

    @pytest.mark.parametrize('row_index, age, sex, bmi, expected_z', bmi_extended_reference_cases)
    def test_bmi_z_scores_extended(self, row_index, age, sex, bmi, expected_z):
        '''Test BMI Z-score calculations match the extended reference values for tail cases.'''
        calculated_z = zsc.calculate_bmi_z_score(bmi, sex, age, method='extended')

        assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
            f'Row {row_index}: Extended BMI Z-score mismatch. ' \
            f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
            f'(bmi={bmi:.2f}, sex={sex}, age={age}m)'


class TestReverseZScores:
    '''Test reverse Z-score calculations (Z-score to measurement).'''

    def test_height_reverse_round_trip(self):
        '''Test that height -> Z-score -> height gives original value.'''
        test_cases = [
            ('m', 24, 85),
            ('f', 36, 92),
            ('m', 60, 110),
            ('f', 120, 145),
        ]

        for sex, age, height in test_cases:
            z = zsc.calculate_height_z_score(height, sex, age)
            height_back = zsc.reverse_height_z_score(z, sex, age)

            assert height == pytest.approx(height_back, abs=0.01), \
                f'Height round-trip failed: {height}cm -> Z={z:.2f} -> {height_back:.2f}cm ' \
                f'(sex={sex}, age={age}m)'

    def test_weight_reverse_round_trip(self):
        '''Test that weight -> Z-score -> weight gives original value.'''
        test_cases = [
            ('m', 24, 12),
            ('f', 36, 14),
            ('m', 60, 18),
            ('f', 120, 35),
        ]

        for sex, age, weight in test_cases:
            z = zsc.calculate_weight_z_score(weight, sex, age)
            weight_back = zsc.reverse_weight_z_score(z, sex, age)

            assert weight == pytest.approx(weight_back, abs=0.01), \
                f'Weight round-trip failed: {weight}kg -> Z={z:.2f} -> {weight_back:.2f}kg ' \
                f'(sex={sex}, age={age}m)'

    def test_bmi_reverse_round_trip(self):
        '''Test that BMI -> Z-score -> BMI gives original value.'''
        test_cases = [
            ('m', 24, 16.5),
            ('f', 36, 15.8),
            ('m', 60, 17.2),
            ('f', 120, 19.5),
        ]

        for sex, age, bmi in test_cases:
            z = zsc.calculate_bmi_z_score(bmi, sex, age)
            bmi_back = zsc.reverse_bmi_z_score(z, sex, age)

            assert bmi == pytest.approx(bmi_back, abs=0.01), \
                f'BMI round-trip failed: BMI={bmi} -> Z={z:.2f} -> {bmi_back:.2f}'

    def test_extreme_bmi_reverse_round_trip(self):
        '''Test reverse calculations work for extreme BMI values.'''
        test_cases = [
            ('m', 60, 25.0),
            ('f', 120, 28.0),
        ]

        for sex, age, bmi in test_cases:
            z = zsc.calculate_bmi_z_score(bmi, sex, age)
            bmi_back = zsc.reverse_bmi_z_score(z, sex, age)

            assert bmi == pytest.approx(bmi_back, abs=0.1), \
                f'Extreme BMI round-trip failed: BMI={bmi} -> Z={z:.2f} -> {bmi_back:.2f}'


class TestSexFormatVariations:
    '''Test that different sex format inputs are handled correctly.'''

    def test_male_formats(self):
        '''Test various male sex format inputs.'''
        height = 100
        age = 60

        z_m = zsc.calculate_height_z_score(height, 'm', age)
        z_male = zsc.calculate_height_z_score(height, 'male', age)
        z_M = zsc.calculate_height_z_score(height, 'M', age)
        z_1 = zsc.calculate_height_z_score(height, 1, age)

        assert abs(z_m - z_male) < 0.001
        assert abs(z_m - z_M) < 0.001
        assert abs(z_m - z_1) < 0.001

    def test_female_formats(self):
        '''Test various female sex format inputs.'''
        height = 100
        age = 60

        z_f = zsc.calculate_height_z_score(height, 'f', age)
        z_female = zsc.calculate_height_z_score(height, 'female', age)
        z_F = zsc.calculate_height_z_score(height, 'F', age)
        z_2 = zsc.calculate_height_z_score(height, 2, age)

        assert abs(z_f - z_female) < 0.001
        assert abs(z_f - z_F) < 0.001
        assert abs(z_f - z_2) < 0.001

    def test_invalid_sex_raises_error(self):
        '''Test that invalid sex values raise appropriate errors.'''
        with pytest.raises((ValueError, AttributeError)):
            zsc.calculate_height_z_score(100, 'x', 60)

        with pytest.raises((ValueError, AttributeError)):
            zsc.calculate_height_z_score(100, 3, 60)


class TestAgeHandling:
    '''Test age range handling and edge cases.'''

    def test_valid_age_range(self):
        '''Test calculations work across the valid age range.'''
        ages = [0, 24, 36, 60, 120, 180, 240]

        for age in ages:
            z = zsc.calculate_height_z_score(100, 'm', age)
            assert isinstance(z, (int, float, np.number))
            assert not np.isnan(z)

    def test_age_above_240_months_warning(self):
        '''Test that ages above 240 months trigger a warning.'''
        with pytest.warns(UserWarning, match='exceeds maximum age'):
            zsc.calculate_height_z_score(170, 'm', 250)

    def test_fractional_ages_interpolated(self):
        '''Test that fractional ages are properly interpolated between data points.'''
        z_low = zsc.calculate_height_z_score(100, 'm', 36.5)
        z_mid = zsc.calculate_height_z_score(100, 'm', 36.7)
        z_high = zsc.calculate_height_z_score(100, 'm', 37.5)

        assert min(z_low, z_high) <= z_mid <= max(z_low, z_high), \
            'Interpolated Z-score should fall between adjacent data points'

        expected_z = z_low + 0.2 * (z_high - z_low)
        assert abs(z_mid - expected_z) < 0.001, 'Linear interpolation should be accurate'

    def test_special_age_values(self):
        '''Test special age values like 0.0, 24.0, 36.0, 240.0 that exist in data.'''
        special_ages = [0.0, 24.0, 36.0, 240.0]

        for age in special_ages:
            z = zsc.calculate_height_z_score(100, 'm', age)
            assert isinstance(z, (int, float, np.number))

    def test_negative_age_raises_error(self):
        '''Test that negative ages raise an error.'''
        with pytest.raises(ValueError, match='below the minimum'):
            zsc.calculate_height_z_score(50, 'm', -1)

    def test_bmi_minimum_age(self):
        '''Test that BMI calculations respect minimum age of 24 months.'''
        with pytest.raises(ValueError, match='below the minimum'):
            zsc.calculate_bmi_z_score(16, 'm', 12)


class TestDataIntegrity:
    '''Test that test data is loaded correctly.'''

    def test_test_data_loaded(self):
        '''Test that reference test data is loaded successfully.'''
        assert len(test_answers) > 0, 'Test data should not be empty'
        assert 'sex' in test_answers.columns
        assert 'age' in test_answers.columns
        assert 'height' in test_answers.columns
        assert 'weight' in test_answers.columns
        assert 'z_height' in test_answers.columns
        assert 'z_weight' in test_answers.columns
        assert 'z_bmi' in test_answers.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])