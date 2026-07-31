import sys
import warnings

import typer

# Using "centile_score" for the library functions to avoid conflict with the CLI definitions
from .zscorecalculator import (
    calculate_bmi_centile as calculate_bmi_centile_score,
    calculate_bmi_z_score,
    calculate_height_centile as calculate_height_centile_score,
    calculate_height_z_score,
    calculate_weight_centile as calculate_weight_centile_score,
    calculate_weight_z_score,
    reverse_bmi_centile as reverse_bmi_centile_score,
    reverse_bmi_z_score,
    reverse_height_centile as reverse_height_centile_score,
    reverse_height_z_score,
    reverse_weight_centile as reverse_weight_centile_score,
    reverse_weight_z_score,
)

app = typer.Typer(help="Calculate using WHO references and standards")

# This edits the warnings for the CLI so that they don't expose filepaths and duplicate messages
# Usual behavior (intended for developers) is reserved for the library part of the package
def cli_showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"Warning: {message}", file=sys.stderr)

warnings.showwarning = cli_showwarning

# Definition of CLI commands
@app.command()
def calculate_height_z(
    height: float = typer.Option(..., help="Height in cm"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO height-for-age Z-score.
    """

    z = calculate_height_z_score(height, sex, age)

    typer.echo(f"{z:.2f}")


@app.command()
def calculate_weight_z(
    weight: float = typer.Option(..., help="Weight in kg"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO weight-for-age Z-score.
    """

    z = calculate_weight_z_score(weight, sex, age)

    typer.echo(f"{z:.2f}")


@app.command()
def calculate_bmi_z(
    bmi: float = typer.Option(..., help="BMI in kg/m2"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO BMI-for-age Z-score.
    """

    z = calculate_bmi_z_score(bmi, sex, age)

    typer.echo(f"{z:.2f}")


@app.command()
def calculate_height_centile(
    height: float = typer.Option(..., help="Height in cm"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO height-for-age centile.
    """

    centile = calculate_height_centile_score(height, sex, age)

    typer.echo(f"{centile:.2f}")


@app.command()
def calculate_weight_centile(
    weight: float = typer.Option(..., help="Weight in kg"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO weight-for-age centile.
    """

    centile = calculate_weight_centile_score(weight, sex, age)

    typer.echo(f"{centile:.2f}")


@app.command()
def calculate_bmi_centile(
    bmi: float = typer.Option(..., help="BMI in kg/m2"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO BMI-for-age centile.
    """

    centile = calculate_bmi_centile_score(bmi, sex, age)

    typer.echo(f"{centile:.2f}")


@app.command()
def reverse_height_z(
    z_score: float = typer.Option(..., help="Z-score"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO height from Z-score.
    """

    height = reverse_height_z_score(z_score, sex, age)

    typer.echo(f"{height:.2f}")


@app.command()
def reverse_weight_z(
    z_score: float = typer.Option(..., help="Z-score"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO weight from Z-score.
    """

    weight = reverse_weight_z_score(z_score, sex, age)

    typer.echo(f"{weight:.2f}")


@app.command()
def reverse_bmi_z(
    z_score: float = typer.Option(..., help="Z-score"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO BMI from Z-score.
    """

    bmi = reverse_bmi_z_score(z_score, sex, age)

    typer.echo(f"{bmi:.2f}")


@app.command()
def reverse_height_centile(
    centile: float = typer.Option(..., help="Centile"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO height from centile.
    """

    height = reverse_height_centile_score(centile, sex, age)

    typer.echo(f"{height:.2f}")


@app.command()
def reverse_weight_centile(
    centile: float = typer.Option(..., help="Centile"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO weight from centile.
    """

    weight = reverse_weight_centile_score(centile, sex, age)

    typer.echo(f"{weight:.2f}")


@app.command()
def reverse_bmi_centile(
    centile: float = typer.Option(..., help="Centile"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate WHO BMI from centile.
    """

    bmi = reverse_bmi_centile_score(centile, sex, age)

    typer.echo(f"{bmi:.2f}")


__all__ = ["app"]