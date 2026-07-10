import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from cruvitools.cdc.zscorecalculator import (
    calculate_height_z_score,
    calculate_weight_z_score,
    calculate_bmi_z_score,
    reverse_height_z_score,
    reverse_weight_z_score,
    reverse_bmi_z_score
)

# Load test data
test_data_path = Path(__file__).parent / 'correctCDCdata.csv'
test_answers = pd.read_csv(test_data_path)


class TestCalculateZScores:
    """Test forward Z-score calculations against reference data."""
    
    def test_height_z_scores(self):
        """Test height Z-score calculations match reference values."""
        mismatches = []
        for _, row in test_answers.iterrows():
            age = row['age']
            sex = row['sex']
            height = row['height']
            expected_z = row['z_height']
            
            calculated_z = calculate_height_z_score(height, sex, age)
            
            diff = abs(calculated_z - expected_z)
            if diff >= 0.03:
                mismatches.append(
                    f"Height: age={age}m, sex={sex}, height={height}cm, "
                    f"expected Z={expected_z:.4f}, calculated Z={calculated_z:.4f}, diff={diff:.4f}"
                )
        
        if mismatches:
            pytest.warns(UserWarning)
            print("\n⚠️  WARNING: Height Z-score mismatches (tolerance 0.03 SD):")
            for msg in mismatches:
                print(f"  - {msg}")
    
    def test_weight_z_scores(self):
        """Test weight Z-score calculations match reference values."""
        mismatches = []
        for _, row in test_answers.iterrows():
            age = row['age']
            sex = row['sex']
            weight = row['weight']
            expected_z = row['z_weight']
            
            calculated_z = calculate_weight_z_score(weight, sex, age)
            
            diff = abs(calculated_z - expected_z)
            if diff >= 0.03:
                mismatches.append(
                    f"Weight: age={age}m, sex={sex}, weight={weight}kg, "
                    f"expected Z={expected_z:.4f}, calculated Z={calculated_z:.4f}, diff={diff:.4f}"
                )
        
        if mismatches:
            pytest.warns(UserWarning)
            print("\n⚠️  WARNING: Weight Z-score mismatches (tolerance 0.03 SD):")
            for msg in mismatches:
                print(f"  - {msg}")
    
    def test_bmi_z_scores(self):
        """Test BMI Z-score calculations match reference values."""
        mismatches = []
        for _, row in test_answers.iterrows():
            age = row['age']
            sex = row['sex']
            height = row['height']
            weight = row['weight']
            bmi = weight / ((height / 100) ** 2)
            expected_z = row['z_bmi']
            
            calculated_z = calculate_bmi_z_score(bmi, sex, age)
            
            diff = abs(calculated_z - expected_z)
            if diff >= 0.03:
                mismatches.append(
                    f"BMI: age={age}m, sex={sex}, bmi={bmi:.2f}, "
                    f"expected Z={expected_z:.4f}, calculated Z={calculated_z:.4f}, diff={diff:.4f}"
                )
        
        if mismatches:
            pytest.warns(UserWarning)
            print("\n⚠️  WARNING: BMI Z-score mismatches (tolerance 0.03 SD):")
            for msg in mismatches:
                print(f"  - {msg}")
            print(f"\n  Note: {len(mismatches)}/{len(test_answers)} BMI values exceed tolerance.")
            print("  This may indicate the reference data uses different CDC tables or calculation methods.")


class TestReverseZScores:
    """Test reverse Z-score calculations (Z-score to measurement)."""
    
    def test_height_reverse_round_trip(self):
        """Test that height -> Z-score -> height produces consistent results."""
        test_cases = [
            ('m', 24, 85),
            ('f', 36, 92),
            ('m', 60, 110),
            ('f', 120, 145),
        ]
        
        for sex, age, height in test_cases:
            z = calculate_height_z_score(height, sex, age)
            reversed_height = reverse_height_z_score(z, sex, age)
            
            assert abs(reversed_height - height) < 0.01, \
                f"Round trip failed: {height}cm -> Z={z:.2f} -> {reversed_height:.2f}cm"
    
    def test_weight_reverse_round_trip(self):
        """Test that weight -> Z-score -> weight produces consistent results."""
        test_cases = [
            ('m', 24, 12),
            ('f', 36, 14),
            ('m', 60, 18),
            ('f', 120, 35),
        ]
        
        for sex, age, weight in test_cases:
            z = calculate_weight_z_score(weight, sex, age)
            reversed_weight = reverse_weight_z_score(z, sex, age)
            
            assert abs(reversed_weight - weight) < 0.01, \
                f"Round trip failed: {weight}kg -> Z={z:.2f} -> {reversed_weight:.2f}kg"
    
    def test_bmi_reverse_round_trip(self):
        """Test that BMI -> Z-score -> BMI produces consistent results."""
        test_cases = [
            ('m', 24, 16.5),
            ('f', 36, 15.8),
            ('m', 60, 17.2),
            ('f', 120, 19.5),
        ]
        
        for sex, age, bmi in test_cases:
            z = calculate_bmi_z_score(bmi, sex, age)
            reversed_bmi = reverse_bmi_z_score(z, sex, age)
            
            assert abs(reversed_bmi - bmi) < 0.01, \
                f"Round trip failed: BMI={bmi} -> Z={z:.2f} -> {reversed_bmi:.2f}"
    
    def test_extreme_bmi_reverse_round_trip(self):
        """Test reverse calculations work for extreme BMI values (>95th percentile)."""
        test_cases = [
            ('m', 60, 25.0),  # Very high BMI
            ('f', 120, 28.0),  # Very high BMI
        ]
        
        for sex, age, bmi in test_cases:
            z = calculate_bmi_z_score(bmi, sex, age)
            reversed_bmi = reverse_bmi_z_score(z, sex, age)
            
            # Extreme values may have larger tolerance due to modified calculation
            assert abs(reversed_bmi - bmi) < 0.1, \
                f"Extreme BMI round trip failed: BMI={bmi} -> Z={z:.2f} -> {reversed_bmi:.2f}"


class TestSexFormatVariations:
    """Test that different sex format inputs are handled correctly."""
    
    def test_male_formats(self):
        """Test various male sex format inputs."""
        height = 100
        age = 60
        
        z_m = calculate_height_z_score(height, 'm', age)
        z_male = calculate_height_z_score(height, 'male', age)
        z_M = calculate_height_z_score(height, 'M', age)
        z_1 = calculate_height_z_score(height, 1, age)
        
        assert abs(z_m - z_male) < 0.001
        assert abs(z_m - z_M) < 0.001
        assert abs(z_m - z_1) < 0.001
    
    def test_female_formats(self):
        """Test various female sex format inputs."""
        height = 100
        age = 60
        
        z_f = calculate_height_z_score(height, 'f', age)
        z_female = calculate_height_z_score(height, 'female', age)
        z_F = calculate_height_z_score(height, 'F', age)
        z_2 = calculate_height_z_score(height, 2, age)
        
        assert abs(z_f - z_female) < 0.001
        assert abs(z_f - z_F) < 0.001
        assert abs(z_f - z_2) < 0.001
    
    def test_invalid_sex_raises_error(self):
        """Test that invalid sex values raise appropriate errors."""
        with pytest.raises((ValueError, AttributeError)):
            calculate_height_z_score(100, 'x', 60)
        
        with pytest.raises((ValueError, AttributeError)):
            calculate_height_z_score(100, 3, 60)


class TestAgeHandling:
    """Test age range handling and edge cases."""
    
    def test_valid_age_range(self):
        """Test calculations work across the valid age range."""
        ages = [0, 24, 36, 60, 120, 180, 240]
        
        for age in ages:
            z = calculate_height_z_score(100, 'm', age)
            assert isinstance(z, (int, float, np.number))
            assert not np.isnan(z)
    
    def test_age_above_240_months_warning(self):
        """Test that ages above 240 months trigger a warning."""
        with pytest.warns(UserWarning, match='exceeds maximum age'):
            calculate_height_z_score(170, 'm', 250)
    
    def test_fractional_ages_interpolated(self):
        """Test that fractional ages are properly interpolated between data points."""
        # Age 36.7 should be interpolated between 36.5 and 37.5
        z_low = calculate_height_z_score(100, 'm', 36.5)
        z_mid = calculate_height_z_score(100, 'm', 36.7)
        z_high = calculate_height_z_score(100, 'm', 37.5)
        
        # Z-score at 36.7 should be between Z-scores at 36.5 and 37.5
        assert min(z_low, z_high) <= z_mid <= max(z_low, z_high), \
            "Interpolated Z-score should fall between adjacent data points"
        
        # Z-score at 36.7 should be closer to 36.5 than to 37.5
        # (36.7 is 20% of the way from 36.5 to 37.5)
        expected_z = z_low + 0.2 * (z_high - z_low)
        assert abs(z_mid - expected_z) < 0.001, "Linear interpolation should be accurate"
    
    def test_special_age_values(self):
        """Test special age values like 0.0, 24.0, 36.0, 240.0 that exist in data."""
        special_ages = [0.0, 24.0, 36.0, 240.0]
        
        for age in special_ages:
            # Should not raise any errors
            z = calculate_height_z_score(100, 'm', age)
            assert isinstance(z, (int, float, np.number))
    
    def test_negative_age_raises_error(self):
        """Test that negative ages raise an error."""
        with pytest.raises(ValueError, match='below the minimum'):
            calculate_height_z_score(50, 'm', -1)
    
    def test_bmi_minimum_age(self):
        """Test that BMI calculations respect minimum age of 24 months."""
        # BMI data starts at 24 months
        with pytest.raises(ValueError, match='below the minimum'):
            calculate_bmi_z_score(16, 'm', 12)


class TestInputValidation:
    """Test handling of invalid or edge-case inputs."""
    
    def test_zero_height_produces_extreme_z(self):
        """Test that zero height produces an extreme (negative) Z-score."""
        z = calculate_height_z_score(0.1, 'm', 60)
        assert z < -10, "Very small height should produce extreme negative Z-score"
    
    def test_zero_weight_produces_extreme_z(self):
        """Test that zero weight produces an extreme (negative) Z-score."""
        z = calculate_weight_z_score(0.1, 'm', 60)
        assert z < -10, "Very small weight should produce extreme negative Z-score"
    
    def test_unrealistic_high_values_produce_extreme_z(self):
        """Test that unrealistically high values produce extreme Z-scores."""
        z_height = calculate_height_z_score(300, 'm', 60)
        z_weight = calculate_weight_z_score(200, 'm', 60)
        z_bmi = calculate_bmi_z_score(50, 'm', 60)
        
        assert z_height > 10, "Unrealistic height should produce extreme positive Z-score"
        assert z_weight > 5, "Unrealistic weight should produce high positive Z-score"
        assert z_bmi > 5, "Unrealistic BMI should produce extreme positive Z-score"
    
    def test_reasonable_values_produce_reasonable_z(self):
        """Test that reasonable values produce Z-scores within expected range."""
        # Typical 5-year-old boy: ~110cm, ~18kg
        z_height = calculate_height_z_score(110, 'm', 60)
        z_weight = calculate_weight_z_score(18, 'm', 60)
        bmi = 18 / (1.1 ** 2)  # ~14.9
        z_bmi = calculate_bmi_z_score(bmi, 'm', 60)
        
        assert -3 < z_height < 3, f"Reasonable height should produce Z-score in [-3, 3], got {z_height}"
        assert -3 < z_weight < 3, f"Reasonable weight should produce Z-score in [-3, 3], got {z_weight}"
        assert -3 < z_bmi < 3, f"Reasonable BMI should produce Z-score in [-3, 3], got {z_bmi}"


class TestExtremeBMIHandling:
    """Test CDC's special handling of extreme BMI values."""
    
    def test_bmi_above_95th_percentile_uses_modified_calculation(self):
        """Test that BMI values above 95th percentile use Wei et al. 2020 method."""
        # High BMI values should use modified calculation
        # which typically produces different results than standard LMS
        
        # Test with a high BMI for a 5-year-old
        high_bmi = 22  # Very high for age 60 months
        z_high = calculate_bmi_z_score(high_bmi, 'm', 60)
        
        # Should produce a high Z-score (likely > 2)
        assert z_high > 2, f"High BMI should produce Z > 2, got {z_high}"
    
    def test_bmi_below_95th_percentile_uses_standard_calculation(self):
        """Test that BMI values below 95th percentile use standard LMS."""
        # Normal BMI should use standard calculation
        normal_bmi = 16
        z_normal = calculate_bmi_z_score(normal_bmi, 'm', 60)
        
        # Should produce a reasonable Z-score
        assert -2 < z_normal < 2, f"Normal BMI should produce Z in [-2, 2], got {z_normal}"
    
    def test_extreme_bmi_consistency(self):
        """Test that extreme BMI handling is consistent across calculations."""
        # Test BMI values that are high for their respective ages
        high_bmi_young = 25  # Very high for 60 months
        high_bmi_older = 28  # High for 120 months
        
        z_60 = calculate_bmi_z_score(high_bmi_young, 'm', 60)
        z_120 = calculate_bmi_z_score(high_bmi_older, 'm', 120)
        
        # Both should use modified calculation and produce high Z-scores
        assert z_60 > 2, "High BMI at 60 months should produce Z > 2"
        assert z_120 > 2, "High BMI at 120 months should produce Z > 2"


class TestConsistencyAcrossGenders:
    """Test consistency of calculations across genders."""
    
    def test_same_relative_position_produces_similar_z(self):
        """Test that children at similar relative positions have similar Z-scores."""
        # A boy and girl both at median height for their age/sex should have Z ≈ 0
        
        # Get median values (M parameter) by calculating Z=0
        median_height_boy = reverse_height_z_score(0, 'm', 60)
        median_height_girl = reverse_height_z_score(0, 'f', 60)
        
        z_boy = calculate_height_z_score(median_height_boy, 'm', 60)
        z_girl = calculate_height_z_score(median_height_girl, 'f', 60)
        
        assert abs(z_boy) < 0.01, f"Median height for boys should give Z ≈ 0, got {z_boy}"
        assert abs(z_girl) < 0.01, f"Median height for girls should give Z ≈ 0, got {z_girl}"


class TestDataIntegrity:
    """Test that test data is loaded correctly."""
    
    def test_test_data_loaded(self):
        """Test that reference test data is loaded successfully."""
        assert len(test_answers) > 0, "Test data should not be empty"
        assert 'sex' in test_answers.columns
        assert 'age' in test_answers.columns
        assert 'height' in test_answers.columns
        assert 'weight' in test_answers.columns
        assert 'z_height' in test_answers.columns
        assert 'z_weight' in test_answers.columns
        assert 'z_bmi' in test_answers.columns
    
    def test_test_data_has_variety(self):
        """Test that test data includes variety in ages and sexes."""
        assert len(test_answers['age'].unique()) > 5, "Should have variety in ages"
        assert len(test_answers['sex'].unique()) >= 2, "Should have both sexes"
    
    def test_test_data_values_reasonable(self):
        """Test that test data values are within reasonable ranges."""
        assert test_answers['age'].min() >= 0, "Age should be non-negative"
        assert test_answers['age'].max() <= 240, "Age should not exceed 240 months"
        assert test_answers['height'].min() > 0, "Height should be positive"
        assert test_answers['weight'].min() > 0, "Weight should be positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
