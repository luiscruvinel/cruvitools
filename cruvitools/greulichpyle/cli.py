import typer

from .zscorecalculator import calculate_bone_age_z_score

app = typer.Typer(help="Calculate bone age Z-score using Greulich and Pyle references")


@app.command()
def calculate_boneage_z(
    chronological_age: float = typer.Option(..., help="Chronological age in years"),
    bone_age: float = typer.Option(..., help="Bone age in years"),
    sex: str = typer.Option(..., help="Sex (male/female or m/f)"),
):
    """
    Calculate bone age Z-score for sex and chronological age.
    """

    z = calculate_bone_age_z_score(chronological_age, bone_age, sex)

    typer.echo(f"{z:.2f}")


__all__ = ["app"]