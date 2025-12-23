#!/usr/bin/env python3
"""Generate complete 03_feature_visualization.ipynb with all sections."""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.split('\n')
    })

# Read the existing notebook to continue from where we left off
try:
    with open('03_feature_visualization.ipynb', 'r') as f:
        existing = json.load(f)
        # Keep first 6 cells (title through initialization)
        notebook["cells"] = existing["cells"][:6]
except:
    pass

# Continue adding from cell 6 onwards
# Section 1: Spectral Signatures
add_markdown("""## Section 1: Spectral Signatures

Visualize feature activations as spectral plots, where:
- **X-axis**: Feature indices (wavelengths)
- **Y-axis**: Activation magnitudes (intensities)

This shows which features are active and how strongly they activate.""")

add_code("""def plot_spectral_signature(
    features: Dict,
    title: str = "Feature Activation Spectrum",
    top_k: Optional[int] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    \"\"\"Plot feature activations as a spectral signature.\"\"\"
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    indices = np.array(features['indices'])
    magnitudes = np.array(features['magnitudes'])
    
    if top_k is not None and len(indices) > top_k:
        sort_idx = np.argsort(magnitudes)[::-1][:top_k]
        indices = indices[sort_idx]
        magnitudes = magnitudes[sort_idx]
    
    sort_order = np.argsort(indices)
    indices = indices[sort_order]
    magnitudes = magnitudes[sort_order]
    
    ax.stem(indices, magnitudes, basefmt=' ', linefmt='-', markerfmt='o')
    ax.set_xlabel('Feature Index (Wavelength)', fontsize=12)
    ax.set_ylabel('Activation Magnitude (Intensity)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    stats_text = (
        f"Active Features: {len(features['indices'])}\n"
        f"Total Energy: {features['energy']:.2f}\n"
        f"Max Intensity: {magnitudes.max():.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return ax

# Example: Visualize features from a simple text
text = "The cat sat on the mat"
features = extract_features(text, model, sae)

plot_spectral_signature(features, title=f"Spectral Signature: '{text}'")
plt.show()""")

add_markdown("### Compare Multiple Spectra Side-by-Side\n\nCompare spectral signatures from different texts to see how they differ.")

add_code("""def compare_spectra(
    texts: List[str],
    labels: Optional[List[str]] = None,
    model=None,
    sae=None
) -> None:
    \"\"\"Compare spectral signatures from multiple texts.\"\"\"
    if labels is None:
        labels = texts
    
    n_texts = len(texts)
    fig, axes = plt.subplots(n_texts, 1, figsize=(14, 4 * n_texts))
    
    if n_texts == 1:
        axes = [axes]
    
    all_features = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        features = extract_features(text, model, sae)
        all_features.append(features)
        
        plot_spectral_signature(
            features,
            title=f"Spectrum {i+1}: {label}",
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.show()
    
    return all_features

# Compare three different texts
texts = [
    "Paris is the capital of France",
    "The cat sat on the mat",
    "Machine learning uses neural networks"
]

all_features = compare_spectra(texts, model=model, sae=sae)""")

# Section 2: Feature Overlap
add_markdown("""## Section 2: Feature Overlap Heatmaps

Visualize which features are shared between texts and which are unique.""")

add_code("""def compute_feature_overlap(features_a: Dict, features_b: Dict) -> float:
    \"\"\"Compute Jaccard similarity (overlap) between two feature sets.\"\"\"
    set_a = set(features_a['indices'])
    set_b = set(features_b['indices'])
    
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0

def plot_feature_overlap_heatmap(
    texts: List[str],
    labels: Optional[List[str]] = None,
    model=None,
    sae=None
) -> None:
    \"\"\"Create a heatmap showing feature overlap between all pairs of texts.\"\"\"
    if labels is None:
        labels = [text[:30] + "..." if len(text) > 30 else text for text in texts]
    
    n = len(texts)
    overlap_matrix = np.zeros((n, n))
    
    # Extract features for all texts
    all_features = []
    for text in texts:
        features = extract_features(text, model, sae)
        all_features.append(features)
    
    # Compute pairwise overlaps
    for i in range(n):
        for j in range(n):
            overlap_matrix[i, j] = compute_feature_overlap(all_features[i], all_features[j])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Feature Overlap (Jaccard)'},
        ax=ax
    )
    ax.set_title('Feature Overlap Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return overlap_matrix, all_features

# Example: Compare fact vs hallucination
fact_text = "The Eiffel Tower is in Paris"
hall_text = "The Eiffel Tower is in Rome"

overlap_matrix, features_list = plot_feature_overlap_heatmap(
    [fact_text, hall_text],
    labels=["Fact: Paris", "Hallucination: Rome"],
    model=model,
    sae=sae
)""")

# Save
with open('03_feature_visualization.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook now has {len(notebook['cells'])} cells")
print("Note: This is a partial notebook. Continue adding sections as needed.")




