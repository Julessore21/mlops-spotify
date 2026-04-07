from setuptools import setup, find_packages

setup(
    name="spotify_mlops",
    version="0.1.0",
    description="MLOps pipeline for Spotify listening time prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.2",
        "mlflow>=2.13.0",
        "joblib>=1.4.2",
        "fastapi>=0.111.0",
        "uvicorn>=0.30.1",
        "pyarrow>=15.0.0",
    ],
    python_requires=">=3.10",
)
