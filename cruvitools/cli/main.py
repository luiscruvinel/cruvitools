import typer

from cruvitools.cdc.cli import app as cdc_app
from cruvitools.greulichpyle.cli import app as greulichpyle_app
from cruvitools.who.cli import app as who_app

app = typer.Typer(
    name='cruvitools',
    help='Anthropometric data processing'
)

app.add_typer(
    who_app,
    name='who',
    help='Calculate using World Health Organization references and standards'
)

app.add_typer(
    cdc_app,
    name='cdc',
    help='Calculate using CDC references'
)

app.add_typer(
    greulichpyle_app,
    name='greulichpyle',
    help='Calculate bone age Z-score using Greulich and Pyle references'
)
if __name__ == "__main__":
    app()