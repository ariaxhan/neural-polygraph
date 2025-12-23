#!/usr/bin/env python3
"""Script to create the complete 03_feature_visualization.ipynb notebook."""

import json

# Read the existing notebook to preserve what we have
try:
    with open('03_feature_visualization.ipynb', 'r') as f:
        notebook = json.load(f)
except:
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# Clear existing cells and rebuild
notebook["cells"] = []

# Define all cells
cells_data = [
    # Title cell
    ("markdown", "# Tutorial 3: Feature Visualization and Geometric Distribution\n\n**Learning Objectives:**\n- Visualize feature activations as spectral signatures\n- Explore geometric distributions of features in high-dimensional space\n- Create comparison visualizations between different texts\n- Understand feature relationships through dimensionality reduction\n- Generate publication-quality visualizations\n\n**Estimated Time:** 20-30 minutes\n\n**Prerequisites:** Complete Tutorials 1 & 2: SAE Basics and Feature Extraction\n\n---"),
    
    # Introduction
    ("markdown", "## Introduction: Visualizing the Feature Space\n\nIn Tutorials 1 and 2, we learned to extract and compare features. Now we'll visualize:\n\n1. **Spectral Signatures**: Feature activations as wavelength-intensity plots\n2. **Geometric Distributions**: How features cluster in high-dimensional space\n3. **Feature Overlaps**: Heatmaps showing shared and unique features\n4. **Activation Statistics**: Distributions of activation magnitudes\n5. **Feature Decoding**: Visual representation of what features \"mean\"\n\nThese visualizations help us understand:\n- How different texts activate different feature patterns\n- The geometric structure of the feature space\n- Relationships between features and their meanings\n- Patterns that distinguish factual from hallucinated content"),
    
    # Setup markdown
    ("markdown", "## Setup: Import Libraries\n\nWe'll use:\n- `matplotlib` & `seaborn`: Static visualizations\n- `numpy`: Numerical operations\n- `sklearn`: Dimensionality reduction (t-SNE, PCA)\n- `umap-learn`: UMAP for better geometric visualization (optional)\n- `hallucination_detector`: Our feature extraction functions"),
    
    # Import code
    ("python", """import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP (optional, but recommended)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: UMAP not installed. Install with: pip install umap-learn")
    print("      Will use t-SNE instead (slower but works)")

# Import our feature extraction functions
from hallucination_detector import (
    initialize_model_and_sae,
    extract_features,
    decode_feature,
    run_differential_diagnosis,
)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("✓ Libraries loaded")
if HAS_UMAP:
    print("✓ UMAP available for geometric visualization")
else:
    print("⚠ Using t-SNE (slower) - consider installing umap-learn")"""),
    
    # Initialize markdown
    ("markdown", "## Initialize Model and SAE\n\nLoad the same model and SAE we've been using throughout the tutorials."),
    
    # Initialize code
    ("python", "model, sae, device = initialize_model_and_sae()\nprint(f\"\\n✓ Ready to visualize features on {device}\")"),
]

# Add cells
for cell_type, source in cells_data:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split('\n')
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

# Save notebook
with open('03_feature_visualization.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Created notebook with {len(notebook['cells'])} cells")




