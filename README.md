# 🟣 News Topic Discovery Dashboard

## app: https://sharvani1357-hierarch-cluster-app-utimjn.streamlit.app/

- This Streamlit application uses Hierarchical Clustering to automatically group similar news articles based on textual similarity.

## Features

- Automatic text column detection
- TF-IDF vectorization with configurable parameters
- Hierarchical clustering with multiple linkage methods
- Dendrogram visualization
- PCA-based cluster visualization
- Cluster summary with top keywords
- Silhouette score validation
- Business interpretation view

## Project Structure

- news-topic-dashboard/
- │
- ├── app.py
- ├── requirements.txt
- ├── README.md
- └── data/
-      └── raw/
-           └── all-data.csv

## Run Locally

### 1. Install dependencies:
   - pip install -r requirements.txt

### 2. Run Streamlit:
   - streamlit run app.py

## Purpose

- Discover hidden themes in news articles without defining categories upfront.
- ✅ 4️⃣ data/raw/all-data.csv
- Place your dataset inside:

- data/raw/all-data.csv
