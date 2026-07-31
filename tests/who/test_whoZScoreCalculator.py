import pytest
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

import cruvitools.who.zscorecalculator as zsc

# Load test data
test_data_path = Path(__file__).parent / 'data' / 'testCasesWHO.csv'
test_answers = pd.read_csv(test_data_path)

tolerance_level = 0.05

height_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        'male' if row['sex'] == 1 else 'female',
        row['height'],
        row['z_height'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
    if row['age'] <= 228
]

weight_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        'male' if row['sex'] == 1 else 'female',
        row['weight'],
        row['z_weight'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
    if row['age'] <= 120
]

bmi_reference_cases = [
    pytest.param(
        row.name,
        row['age'],
        'male' if row['sex'] == 1 else 'female',
        row['bmi'],
        row['z_bmi'],
        id=f'row-{row.name}',
    )
    for _, row in test_answers.iterrows()
    if row['age'] <= 228
]

@pytest.mark.filterwarnings("ignore:Calculated Z-score of .* is extreme:UserWarning:cruvitools.who.zscorecalculator")
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
        '''Test BMI Z-score calculations match reference values.'''
        calculated_z = zsc.calculate_bmi_z_score(bmi, sex, age)

        assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
            f'Row {row_index}: BMI Z-score mismatch. ' \
            f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
            f'(bmi={bmi:.2f}, sex={sex}, age={age}m)'


class TestReverseZScores:
    '''Test reverse Z-score calculations (Z-score to measurement).'''
    
    def test_height_reverse_round_trip(self):
        '''Test that height -> Z-score -> height gives original value.'''
        test_cases = [
            (50, 'male', 0),
            (75, 'female', 12),
            (85, 'male', 24),
            (110, 'female', 60),
            (145, 'male', 120),
        ]
        
        for height, sex, age in test_cases:
            z = zsc.calculate_height_z_score(height, sex, age)
            height_back = zsc.reverse_height_z_score(z, sex, age)
            
            assert height == pytest.approx(height_back, abs=0.01), \
                f'Height round-trip failed: {height}cm -> Z={z:.2f} -> {height_back:.2f}cm ' \
                f'(sex={sex}, age={age}m)'
    
    def test_weight_reverse_round_trip(self):
        '''Test that weight -> Z-score -> weight gives original value.'''
        test_cases = [
            (3.5, 'male', 0),
            (9, 'female', 12),
            (12, 'male', 24),
            (18, 'female', 60),
            (32, 'male', 120),
        ]
        
        for weight, sex, age in test_cases:
            z = zsc.calculate_weight_z_score(weight, sex, age)
            weight_back = zsc.reverse_weight_z_score(z, sex, age)
            
            assert weight == pytest.approx(weight_back, abs=0.01), \
                f'Weight round-trip failed: {weight}kg -> Z={z:.2f} -> {weight_back:.2f}kg ' \
                f'(sex={sex}, age={age}m)'
    
    def test_bmi_reverse_round_trip(self):
        '''Test that BMI -> Z-score -> BMI gives original value.'''
        test_cases = [
            (16, 'male', 24),
            (15, 'female', 36),
            (15.5, 'male', 60),
            (17, 'female', 120),
        ]
        
        for bmi, sex, age in test_cases:
            z = zsc.calculate_bmi_z_score(bmi, sex, age)
            bmi_back = zsc.reverse_bmi_z_score(z, sex, age)
            
            assert bmi == pytest.approx(bmi_back), \
                f'BMI round-trip failed: {bmi:.1f} -> Z={z:.2f} -> {bmi_back:.2f} ' \
                f'(sex={sex}, age={age}m)'
    
    def test_extreme_z_scores_reverse(self):
        '''Test reverse calculations with extreme Z-scores (|Z| > 3).'''
        test_cases = [
            # (Z-score, sex, age, measurement_type)
            (4.0, 'male', 24, 'weight'),
            (-4.0, 'female', 36, 'weight'),
            (5.0, 'male', 60, 'height'),
            (-5.0, 'female', 60, 'height'),
            (4.5, 'male', 48, 'bmi'),
            (-4.5, 'female', 48, 'bmi'),
        ]
        
        for z, sex, age, measure_type in test_cases:
            if measure_type == 'weight':
                value = zsc.reverse_weight_z_score(z, sex, age)
                z_back = zsc.calculate_weight_z_score(value, sex, age)
            elif measure_type == 'height':
                value = zsc.reverse_height_z_score(z, sex, age)
                z_back = zsc.calculate_height_z_score(value, sex, age)
            else:  # bmi
                value = zsc.reverse_bmi_z_score(z, sex, age)
                z_back = zsc.calculate_bmi_z_score(value, sex, age)
            
            assert abs(z - z_back) < 0.01, \
                f'Extreme Z-score round-trip failed: Z={z} -> {value:.2f} -> Z={z_back:.2f} ' \
                f'({measure_type}, sex={sex}, age={age}m)'


class TestSexFormatVariations:
    '''Test that various sex format inputs are accepted.'''
    
    def test_male_formats(self):
        '''Test different ways to specify male sex.'''
        male_formats = ['male', 'Male', 'MALE', 'm', 'M']
        
        for sex_format in male_formats:
            z = zsc.calculate_height_z_score(75, sex_format, 12)
            assert isinstance(z, float), f'Failed with sex format: {sex_format}'
    
    def test_female_formats(self):
        '''Test different ways to specify female sex.'''
        female_formats = ['female', 'Female', 'FEMALE', 'f', 'F']
        
        for sex_format in female_formats:
            z = zsc.calculate_height_z_score(75, sex_format, 12)
            assert isinstance(z, float), f'Failed with sex format: {sex_format}'
    
    def test_invalid_sex_raises_error(self):
        '''Test that invalid sex values raise ValueError.'''
        invalid_formats = ['boy', 'girl', 'x', '1', '2', None]
        
        for invalid_sex in invalid_formats:
            with pytest.raises((ValueError, AttributeError)):
                zsc.calculate_height_z_score(75, invalid_sex, 12)


class TestAgeHandling:
    '''Test age-related functionality.'''
    
    def test_age_above_19_years_warning(self):
        '''Test that ages above 228 months (19 years) trigger warning.'''
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            z = zsc.calculate_height_z_score(165, 'male', 250)
            
            assert len(w) == 1, 'Expected a warning for age > 19 years'
            assert 'above 19 years' in str(w[0].message).lower()
            assert isinstance(z, float), 'Should still return a Z-score'
    
    def test_fractional_ages_not_supported(self):
        '''Test that fractional ages cause issues (WHO data is monthly).'''
        # WHO data only has integer month values
        with pytest.raises((KeyError, ValueError)):
            zsc.calculate_height_z_score(75, 'male', 12.5)

class TestDataIntegrity:
    '''Test that the test data itself is valid.'''
    
    def test_test_data_loaded(self):
        '''Test that test_answers CSV loaded correctly.'''
        assert len(test_answers) > 0, 'Test data should not be empty'
        assert 'age' in test_answers.columns
        assert 'sex' in test_answers.columns
        assert 'height' in test_answers.columns
        assert 'weight' in test_answers.columns
        assert 'bmi' in test_answers.columns
        assert 'z_height' in test_answers.columns
        assert 'z_weight' in test_answers.columns
        assert 'z_bmi' in test_answers.columns
    
    def test_test_data_has_variety(self):
        '''Test that test data covers various scenarios.'''
        assert test_answers['sex'].nunique() == 2, 'Should have both sexes'
        assert test_answers['age'].min() >= 0, 'Ages should be non-negative'
        assert test_answers['age'].max() <= 228, 'Ages should be ≤ 19 years'
        
        # Should have extreme Z-scores (both positive and negative)
        assert test_answers['z_height'].min() < -3, 'Should have extreme negative Z-scores'
        assert test_answers['z_height'].max() > 3, 'Should have extreme positive Z-scores'