# Pokémon Region Classifier

This project focuses on training a CNN model to classify Pokémon based on their **region of origin** using image data.
The initial protoype will focus on the **Kanto region** using **EfficientNet-B0**.

The goal is to build a dataset of Pokémon images from multiple sources and evaluate the model performance on pokemon classification.

---

# Project Structure

```
pokemon-region-classifier/
│
├── pokemon_images/           # Main image dataset
│   ├── kanto/
│   │   ├── bulbasaur/
│   │   ├── ivysaur/
│   │   ├── venusaur/
│   │   └── ...
│   │
│   ├── johto/
│   ├── hoenn/
│   ├── sinnoh/
│   └── ...
│
├── utils/                    # Utility scripts
│   ├── scraping/             # Image scraping scripts
│   ├── image_retrieval/      # Scripts to download sprites/artworks
│   └── helpers/              # Misc helper functions
│
├── notebooks/                # Experiment notebooks
│   └── kanto_experiments.ipynb
│
├── models/                   # Model experiments (future)
│
├── outputs/                  # Saved results and checkpoints
│
├── requirements.txt
└── README.md
```

---

# Dataset

The dataset consists of Pokémon images organized by **region and Pokémon name**.

Example structure:

```
pokemon_images/001_Bulbasaur/
pokemon_images/152_Chikorita/

```

Each Pokémon folder contains multiple images such as:

* Game sprites
* Official artwork
* Other curated images

These images will later be used to train models.

---

# Utilities

The `utils` directory contains scripts used to build the dataset.

### Scraping

Scripts for retrieving Pokémon images from external sources.

### Image Retrieval

Tools for downloading sprites and official artwork from public repositories or APIs.

---

# Current Goal

The first experiment is to:

1. Use **EfficientNet-B0**
2. Train on **Kanto Pokémon images**
3. Evaluate classification performance

This serves as a baseline before scaling to additional regions.

---

# Future Work

Planned improvements include:

* Expanding the dataset with additional image sources
* Training additional CNN architectures
* Data augmentation and dataset balancing

---

# Setup

Install dependencies:

```
pip install -r requirements.txt
```

---

# Notes
This repository is currently under development. The dataset and utilities are actively being expanded.
