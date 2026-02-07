"""
Setup script for claims-autoencoder package
"""

from setuptools import setup, find_packages

setup(
    name="claims-autoencoder",
    version="1.0.0",
    description="Claims Autoencoder for Anomaly Detection",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "fastparquet>=2023.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "mlflow>=2.5.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "streamlit>=1.25.0",
        "streamlit-option-menu>=0.3.6",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "optuna>=3.3.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    entry_points={
        'console_scripts': [
            'train-autoencoder=src.training:main',
            'score-claims=src.batch_scoring:main',
            'tune-hyperparameters=src.hyperparameter_tuning:tune_and_train',
        ],
    },
)
