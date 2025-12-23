"""
Hallucination Detector Package

A package for detecting hallucinations in language models using
Sparse Autoencoder (SAE) spectral signatures.
"""

from .sae_utils import (
    initialize_model_and_sae,
    extract_features,
    decode_feature,
    get_loudest_unique_features,
    run_differential_diagnosis,
)

__all__ = [
    "initialize_model_and_sae",
    "extract_features",
    "decode_feature",
    "get_loudest_unique_features",
    "run_differential_diagnosis",
]

