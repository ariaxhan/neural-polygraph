from setuptools import setup, find_packages

setup(
    name="hallucination-detector",
    version="0.1.0",
    description="Detecting hallucinations via SAE spectral signatures",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformer-lens>=1.16.0",
        "sae-lens>=3.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)




