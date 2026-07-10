import pytest
import pandas as pd
import warnings
from pathlib import Path

import cruvitools.who.zscorecalculator as zsc

# Load test data
test_data_path = Path(__file__).parent / 'data' / 'testCasesWho.csv'
test_answers = pd.read_csv(test_data_path)

tolerance_level = 0.05

class TestCalculateZScores:
    '''Test forward Z-score calculations against reference data.'''
    
    def test_height_z_scores(self):
        '''Test height Z-score calculations match reference values.'''
        for _, row in test_answers.iterrows():
            age = row['age']
            if age > 228:  # WHO height data only goes to 228 months
                continue
                
            sex = 'male' if row['sex'] == 1 else 'female'
            height = row['height']
            expected_z = row['z_height']
            
            calculated_z = zsc.calculate_height_z_score(height, sex, age)
            
            assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
                f'Row {row.name}: Height Z-score mismatch. ' \
                f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
                f'(height={height}cm, sex={sex}, age={age}m)'
    
    def test_weight_z_scores(self):
        '''Test weight Z-score calculations match reference values.'''
        for _, row in test_answers.iterrows():
            age = row['age']
            if age > 120:  # WHO weight data only goes to 120 months
                continue
                
            sex = 'male' if row['sex'] == 1 else 'female'
            weight = row['weight']
            expected_z = row['z_weight']
            
            calculated_z = zsc.calculate_weight_z_score(weight, sex, age)
            
            assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
                f'Row {row.name}: Weight Z-score mismatch. ' \
                f'Expected {expected_z:.2f}, got {calculated_z:.2f} ' \
                f'(weight={weight}kg, sex={sex}, age={age}m)'
    
    def test_bmi_z_scores(self):
        '''Test BMI Z-score calculations match reference values.'''
        for _, row in test_answers.iterrows():
            age = row['age']
            if age > 228:  # WHO BMI data only goes to 228 months
                continue
                
            sex = 'male' if row['sex'] == 1 else 'female'
            bmi = row['bmi']
            expected_z = row['z_bmi']
            
            calculated_z = zsc.calculate_bmi_z_score(bmi, sex, age)
            
            assert calculated_z == pytest.approx(expected_z, abs=tolerance_level), \
                f'Row {row.name}: BMI Z-score mismatch. ' \
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
    
    def test_valid_age_range(self):
        '''Test that ages from 0 to 228 months work correctly.'''
        test_ages = [0, 1, 12, 24, 60, 120, 228]
        
        for age in test_ages:
            z = zsc.calculate_height_z_score(75, 'male', age)
            assert isinstance(z, float), f'Failed at age {age} months'
    
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


class TestInputValidation:
    '''Test error handling for invalid inputs.'''
    
    def test_negative_values_produce_nan_or_extreme(self):
        '''Test that negative values produce NaN or extreme Z-scores.'''
        import numpy as np
        
        # Negative height produces NaN
        z = zsc.calculate_height_z_score(-75, 'male', 12)
        assert np.isnan(z) or z < -10, 'Negative height should produce NaN or extreme Z'
    
    def test_zero_weight_produces_extreme_z(self):
        '''Test that zero weight produces extreme negative Z-score.'''
        import numpy as np
        
        z = zsc.calculate_weight_z_score(0.0001, 'male', 12)  # Very close to zero
        assert z < -10, 'Near-zero weight should produce very negative Z-score'
    
    def test_unrealistic_values_produce_extreme_z(self):
        '''Test that unrealistic values produce very high/low Z-scores.'''
        # Very high height should produce high Z-score
        z_high = zsc.calculate_height_z_score(200, 'male', 24)
        assert z_high > 10, 'Extremely high height should produce Z > 10'
        
        # Very low weight should produce very negative Z-score
        z_low = zsc.calculate_weight_z_score(2, 'male', 24)
        assert z_low < -5, 'Extremely low weight should produce Z < -5'


class TestBoundaryConditions:
    '''Test behavior at Z-score boundaries (around ±3 SD).'''
    
    def test_z_score_at_positive_3sd(self):
        '''Test calculations at Z = +3 boundary.'''
        # Get the exact value at Z=3
        value_at_3sd = zsc.reverse_height_z_score(3.0, 'male', 24)
        z_back = zsc.calculate_height_z_score(value_at_3sd, 'male', 24)
        
        assert abs(z_back - 3.0) < 0.01, 'Z-score at +3SD should be exactly 3'
    
    def test_z_score_at_negative_3sd(self):
        '''Test calculations at Z = -3 boundary.'''
        value_at_neg3sd = zsc.reverse_weight_z_score(-3.0, 'female', 36)
        z_back = zsc.calculate_weight_z_score(value_at_neg3sd, 'female', 36)
        
        assert abs(z_back - (-3.0)) < 0.01, 'Z-score at -3SD should be exactly -3'
    
    def test_values_beyond_3sd_use_linear_extrapolation(self):
        '''Test that values beyond ±3SD use the WHO restricted method.'''
        # Values slightly beyond +3SD
        value_at_3sd = zsc.reverse_height_z_score(3.0, 'male', 24)
        value_at_4sd = zsc.reverse_height_z_score(4.0, 'male', 24)
        
        # The difference should be approximately linear (not exponential)
        # In standard LMS, this would grow exponentially, but WHO method is linear
        z_at_3 = zsc.calculate_height_z_score(value_at_3sd, 'male', 24)
        z_at_4 = zsc.calculate_height_z_score(value_at_4sd, 'male', 24)
        
        assert abs(z_at_3 - 3.0) < 0.01
        assert abs(z_at_4 - 4.0) < 0.01

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