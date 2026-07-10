from setuptools import setup, find_packages

setup(
    name="cruvitools",
    version="0.1",
    description="A package to process anthropometric data",
    author="Luis Eduardo Cruvinel Pinto",
    author_email="luiscruvinel@gmail.com",
    packages=find_packages(exclude=["__pycache__", "*.pyc", "*.pyo", "*.ipynb_checkpoints"]),
    install_requires=[
        "pandas>=1.0.0",
        "openpyxl>=3.0.0",
        "pathlib; python_version<'3.4'",
    ],
    python_requires=">=3.6",
    include_package_data=True, 
)
