"""
CDC Growth Chart Z-Score Calculator

This module provides functions to calculate and reverse Z-scores for height, weight, and BMI
measurements using the CDC (Centers for Disease Control and Prevention) LMS method.

Available functions:
    - calculate_height_z_score: Calculate height-for-age Z-score
    - calculate_weight_z_score: Calculate weight-for-age Z-score
    - calculate_bmi_z_score: Calculate BMI-for-age Z-score (with extreme value handling)
    - reverse_height_z_score: Calculate height from Z-score
    - reverse_weight_z_score: Calculate weight from Z-score
    - reverse_bmi_z_score: Calculate BMI from Z-score (with extreme value handling)

Example:
    >>> from cruvitools.cdc import calculate_weight_z_score, calculate_bmi_z_score
    >>> z = calculate_weight_z_score(9.7, 'male', 9)
    >>> print(f"Z-score: {z:.3f}")
    Z-score: 0.207
"""

from .zscorecalculator import *

__all__ = [
    'calculate_height_z_score',
    'calculate_weight_z_score',
    'calculate_bmi_z_score',
    'reverse_height_z_score',
    'reverse_weight_z_score',
    'reverse_bmi_z_score'
]
