import sys
import warnings

import typer

from .zscorecalculator import (
    calculate_bmi_z_score,
    calculate_height_z_score,
    calculate_weight_z_score,
    reverse_bmi_z_score,
    reverse_height_z_score,
    reverse_weight_z_score,
)

app = typer.Typer(help="Calculate using CDC references and standards")

# This edits the warnings for the CLI so that they don't expose filepaths and duplicate messages
# Usual behavior (intended for developers) is reserved for the library part of the package
def cli_showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"Warning: {message}", file=sys.stderr)

warnings.showwarning = cli_showwarning

# Define CLI commands
@app.command()
def calculate_height_z(
    height: float = typer.Option(..., help="Height in cm"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate CDC height-for-age Z-score.
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
    Calculate CDC weight-for-age Z-score.
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
    Calculate CDC BMI-for-age Z-score.
    """

    z = calculate_bmi_z_score(bmi, sex, age)

    typer.echo(f"{z:.2f}")


@app.command()
def reverse_height_z(
    z_score: float = typer.Option(..., help="Z-score"),
    age: int = typer.Option(..., help="Age in months"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate CDC height from Z-score.
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
    Calculate CDC weight from Z-score.
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
    Calculate CDC BMI from Z-score.
    """

    bmi = reverse_bmi_z_score(z_score, sex, age)

    typer.echo(f"{bmi:.2f}")